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
ckpt_id = "/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/FLUX.1-schnell"
text_encoder_ckpt_id = '/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/byt5-xxl'

proj_t5 = MLP(in_dim=4672, out_dim=4096, hidden_dim=4096, out_dim1=768).to(device=device,dtype=dtype)
print("loading text encoder")
text_encoder_t5 = T5ForConditionalGeneration.from_pretrained(text_encoder_ckpt_id).get_encoder().to(device=device,dtype=dtype)
tokenizer_t5 = AutoTokenizer.from_pretrained(text_encoder_ckpt_id)


proj_t5_save_path = f"/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/MultilingualFLUX.1-adapter/diffusion_pytorch_model.bin"
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

raw_text = "一只雄伟的大象优雅地站在阳光普照的草原上，它那纹理分明的灰色皮肤在午后温暖的金色阳光下闪闪发光。大象有着大而富有表现力的耳朵和微微弯曲的鼻子，正处于行走的姿态中，踢起一团尘土，向被茂密的绿色金合欢树环绕的波光粼粼的水坑走去。这幅场景以一种充满活力的印象派风格绘制，使用了丰富的泥土色调、柔和的绿色和温暖的黄色，唤起了一种宁静与自然相连的感觉。拍摄角度较低，捕捉到了大象在点缀着缕缕白云的广阔天空下的壮观景象。前景中，几朵色彩斑斓的野花盛开，增添了颜色的斑驳，而远处一群羚羊正在平静地吃草，增强了这片野外迷人时刻的宁静氛围。"

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

    # 兼容不同 diffusers 版本输出：可能是打包的 3D [B,P,C] 或已解包的 4D [B,C,H,W]
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    if isinstance(latents, torch.Tensor):
        if latents.ndim == 3:
            latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        elif latents.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")
    else:
        raise TypeError(f"Unsupported latent type: {type(latents)}")
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("ChineseFLUX.jpg")
