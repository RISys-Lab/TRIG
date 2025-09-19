from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn
import argparse
import json
import os
import time
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, in_dim=4096, out_dim=4096, hidden_dim=4096, out_dim1=768, use_residual=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.fc = nn.Linear(out_dim, out_dim1)
    def forward(self, x):
        x = self.layernorm(x)
        x = self.projector(x)
        x2 = nn.GELU()(x)
        x1 = self.fc(x2)
        x1 = torch.mean(x1,1)
        return x1,x2


parser = argparse.ArgumentParser("PEA Inference", add_help=True)
parser.add_argument('--flux_path', type=str, default="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/FLUX.1-schnell")
parser.add_argument('--t5_path', type=str, default='/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/byt5-xxl')
parser.add_argument('--proj_path', type=str, default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/MultilingualFLUX.1-adapter/diffusion_pytorch_model.bin")
parser.add_argument('--input_json', type=str, required=True, default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/dataset/TRIG-multilingual/text-to-image-multilingual.json")
parser.add_argument('--output_dir', type=str, required=True, default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/output/t2i_ml/PEA")
parser.add_argument('--start_idx', type=int, default=None)
parser.add_argument('--end_idx', type=int, default=None)
parser.add_argument('--num_steps', type=int, default=28)
parser.add_argument('--height', type=int, default=1024)
parser.add_argument('--width', type=int, default=1024)
parser.add_argument('--guidance_scale', type=float, default=3.0)
args = parser.parse_args()

dtype = torch.bfloat16
device = "cuda"
ckpt_id = args.flux_path
text_encoder_ckpt_id = args.t5_path

proj_t5 = MLP(in_dim=4672, out_dim=4096, hidden_dim=4096, out_dim1=768).to(device=device,dtype=dtype)
print("loading text encoder")
text_encoder_t5 = T5ForConditionalGeneration.from_pretrained(text_encoder_ckpt_id).get_encoder().to(device=device,dtype=dtype)
tokenizer_t5 = AutoTokenizer.from_pretrained(text_encoder_ckpt_id)


proj_t5_save_path = args.proj_path
print("loading proj_t5")
state_dict = torch.load(proj_t5_save_path, map_location="cpu")
state_dict_new = {}
for k,v in state_dict.items():
    k_new = k.replace("module.","")
    state_dict_new[k_new] = v

proj_t5.load_state_dict(state_dict_new)

print("loading Flux pipeline")
pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, vae=None,
    torch_dtype=torch.bfloat16
).to(device)

print("loading vae")
vae = AutoencoderKL.from_pretrained(
    ckpt_id, 
    subfolder="vae",
    torch_dtype=torch.bfloat16
).to(device)
vae_scale_factor = pipeline.vae_scale_factor
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

@torch.no_grad()
def generate_one(prompt_text: str, out_dir: str, filename: str):
    text_inputs = tokenizer_t5(
        prompt_text,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings = text_encoder_t5(text_inputs)[0]
    pooled_prompt_embeds, prompt_embeds = proj_t5(text_embeddings)

    latents = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        output_type="latent",
    ).images

    # 兼容不同 diffusers 版本输出：可能是打包的 3D [B,P,C] 或已解包的 4D [B,C,H,W]
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    if isinstance(latents, torch.Tensor):
        if latents.ndim == 3:
            latents = FluxPipeline._unpack_latents(latents, args.height, args.width, vae_scale_factor)
        elif latents.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")
    else:
        raise TypeError(f"Unsupported latent type: {type(latents)}")

    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    os.makedirs(out_dir, exist_ok=True)
    image[0].save(os.path.join(out_dir, f"{filename}.jpg"))


def json2image(input_json_path: str, out_dir: str):
    start_time = time.time()
    with open(input_json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if not isinstance(data_list, list):
        raise ValueError("Input JSON must be a list of objects with data_id and prompt")

    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else len(data_list)
    start = max(0, start)
    end = min(len(data_list), end)
    if start >= end:
        print(f"Empty range: start_idx={args.start_idx}, end_idx={args.end_idx}, total={len(data_list)}")
        return
    data_slice = data_list[start:end]

    generated_count = 0
    skipped_count = 0
    for item in tqdm(data_slice, desc=f"Generating [{start}:{end}]", unit="img"):
        data_id = item.get("data_id")
        prompt = item.get("prompt")
        if data_id is None or prompt is None:
            continue
        filename = str(data_id).replace("/", "_")
        target_path = os.path.join(out_dir, f"{filename}.jpg")
        if os.path.exists(target_path):
            skipped_count += 1
            continue
        generate_one(prompt, out_dir, filename)
        generated_count += 1

    print(f"\n🎉 Generation finished for range [{start}:{end}]")
    print(f"⏱️  Total time: {time.time() - start_time:.2f}s")
    print(f"📁 Images saved to: {out_dir}")
    print(f"✅ Generated: {generated_count} | ⏭️ Skipped existing: {skipped_count}")


if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    json2image(args.input_json, args.output_dir)
