from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import T5ForConditionalGeneration,AutoTokenizer
import torch 
import torch.nn as nn


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


dtype = torch.bfloat16
device = "cuda"
ckpt_id = "$WORK/fmohamma/TRIG/data/FLUX.1-schnell"
text_encoder_ckpt_id = '$WORK/fmohamma/TRIG/data/byt5-xxl'
proj_t5 = MLP(in_dim=4672, out_dim=4096, hidden_dim=4096, out_dim1=768).to(device=device,dtype=dtype)
text_encoder_t5 = T5ForConditionalGeneration.from_pretrained(text_encoder_ckpt_id).get_encoder().to(device=device,dtype=dtype)
tokenizer_t5 = AutoTokenizer.from_pretrained(text_encoder_ckpt_id)


proj_t5_save_path = f"$WORK/fmohamma/TRIG/data/MultilingualFLUX.1-adapter/diffusion_pytorch_model.bin"
state_dict = torch.load(proj_t5_save_path, map_location="cpu")
state_dict_new = {}
for k,v in state_dict.items():
    k_new = k.replace("module.","")
    state_dict_new[k_new] = v

proj_t5.load_state_dict(state_dict_new)

pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, vae=None,
    torch_dtype=torch.bfloat16
).to(device)

vae = AutoencoderKL.from_pretrained(
    ckpt_id, 
    subfolder="vae",
    torch_dtype=torch.bfloat16
).to(device)
vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

raw_text = "一个漂亮的女孩在海边散步"

with torch.no_grad():
    text_inputs = tokenizer_t5(
        raw_text,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings = text_encoder_t5(text_inputs)[0]
    pooled_prompt_embeds,prompt_embeds = proj_t5(text_embeddings)
    height, width = 1024, 1024
    latents = pipeline(
        prompt_embeds=prompt_embeds, 
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=4, guidance_scale=0, 
        height=height, width=width,
        output_type="latent",
    ).images

    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("ChineseFLUX.jpg")
