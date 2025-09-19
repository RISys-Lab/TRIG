from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import T5ForConditionalGeneration,AutoTokenizer
import torch 
import torch.nn as nn
import time


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


print("=" * 60)
print("🚀 开始加载 PEA Diffusion 模型")
print("=" * 60)

dtype = torch.bfloat16
device = "cuda"
ckpt_id = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/FLUX.1-schnell"
text_encoder_ckpt_id = '/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/byt5-xxl'

print(f"📋 模型配置:")
print(f"  - 设备: {device}")
print(f"  - 数据类型: {dtype}")
print(f"  - FLUX模型路径: {ckpt_id}")
print(f"  - T5编码器路径: {text_encoder_ckpt_id}")

# 1. 初始化MLP投影层
print("\n🔧 [1/6] 初始化MLP投影层...")
start_time = time.time()
proj_t5 = MLP(in_dim=4672, out_dim=4096, hidden_dim=4096, out_dim1=768).to(device=device,dtype=dtype)
print(f"  ✅ MLP投影层初始化完成 ({time.time() - start_time:.2f}s)")

# 2. 加载T5文本编码器
print("\n📝 [2/6] 加载T5文本编码器...")
start_time = time.time()
text_encoder_t5 = T5ForConditionalGeneration.from_pretrained(text_encoder_ckpt_id).get_encoder().to(device=device,dtype=dtype)
print(f"  ✅ T5文本编码器加载完成 ({time.time() - start_time:.2f}s)")

# 3. 加载T5分词器
print("\n🔤 [3/6] 加载T5分词器...")
start_time = time.time()
tokenizer_t5 = AutoTokenizer.from_pretrained(text_encoder_ckpt_id)
print(f"  ✅ T5分词器加载完成 ({time.time() - start_time:.2f}s)")


# 4. 加载投影层权重
print("\n⚖️  [4/6] 加载投影层权重...")
start_time = time.time()
proj_t5_save_path = f"/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/MultilingualFLUX.1-adapter/diffusion_pytorch_model.bin"
print(f"  📁 权重文件路径: {proj_t5_save_path}")

state_dict = torch.load(proj_t5_save_path, map_location="cpu")
print(f"  📊 原始权重包含 {len(state_dict)} 个参数")

state_dict_new = {}
for k,v in state_dict.items():
    k_new = k.replace("module.","")
    state_dict_new[k_new] = v

proj_t5.load_state_dict(state_dict_new)
print(f"  ✅ 投影层权重加载完成 ({time.time() - start_time:.2f}s)")

# 5. 加载FLUX Pipeline
print("\n🌊 [5/6] 加载FLUX Pipeline...")
start_time = time.time()
pipeline = FluxPipeline.from_pretrained(
    ckpt_id, text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, vae=None,
    torch_dtype=torch.bfloat16
).to(device)
print(f"  ✅ FLUX Pipeline加载完成 ({time.time() - start_time:.2f}s)")

# 6. 加载VAE和图像处理器
print("\n🖼️  [6/6] 加载VAE和图像处理器...")
start_time = time.time()
vae = AutoencoderKL.from_pretrained(
    ckpt_id, 
    subfolder="vae",
    torch_dtype=torch.bfloat16
).to(device)
vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
print(f"  ✅ VAE和图像处理器加载完成 ({time.time() - start_time:.2f}s)")

print("\n" + "=" * 60)
print("🎉 所有模型加载完成！开始图像生成...")
print("=" * 60)

raw_text = "一个漂亮的女孩在海边散步"
print(f"\n💬 输入文本: '{raw_text}'")

print("\n🎨 开始图像生成流程...")
total_start_time = time.time()

with torch.no_grad():
    # 文本编码阶段
    print("  📝 [1/5] 文本分词和编码...")
    step_start = time.time()
    text_inputs = tokenizer_t5(
        raw_text,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    print(f"      Token数量: {text_inputs.shape[1]}")
    
    text_embeddings = text_encoder_t5(text_inputs)[0]
    print(f"      文本嵌入维度: {text_embeddings.shape}")
    print(f"      ✅ 文本编码完成 ({time.time() - step_start:.2f}s)")
    
    # 投影阶段
    print("  🔄 [2/5] MLP投影...")
    step_start = time.time()
    pooled_prompt_embeds, prompt_embeds = proj_t5(text_embeddings)
    print(f"      池化嵌入维度: {pooled_prompt_embeds.shape}")
    print(f"      提示嵌入维度: {prompt_embeds.shape}")
    print(f"      ✅ 投影完成 ({time.time() - step_start:.2f}s)")
    
    # 扩散生成阶段
    print("  🌀 [3/5] FLUX扩散生成...")
    step_start = time.time()
    height, width = 1024, 1024
    num_steps = 4
    print(f"      图像尺寸: {height}x{width}")
    print(f"      推理步数: {num_steps}")
    
    latents = pipeline(
        prompt_embeds=prompt_embeds, 
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_steps, guidance_scale=0, 
        height=height, width=width,
        output_type="latent",
    ).images
    print(f"      Latent维度: {latents.shape}")
    print(f"      ✅ 扩散生成完成 ({time.time() - step_start:.2f}s)")

    # VAE解码阶段
    print("  🖼️  [4/5] VAE解码...")
    step_start = time.time()
    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    print(f"      ✅ VAE解码完成 ({time.time() - step_start:.2f}s)")
    
    # 后处理和保存
    print("  💾 [5/5] 图像后处理和保存...")
    step_start = time.time()
    image = image_processor.postprocess(image, output_type="pil")
    output_path = "ChineseFLUX.jpg"
    image[0].save(output_path)
    print(f"      ✅ 图像已保存到: {output_path} ({time.time() - step_start:.2f}s)")

print(f"\n🎉 图像生成完成！总用时: {time.time() - total_start_time:.2f}s")
print("=" * 60)
