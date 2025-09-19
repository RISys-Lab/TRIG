from diffusers import FluxPipeline
from transformers import T5ForConditionalGeneration, AutoTokenizer
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
        x1 = torch.mean(x1, 1)
        return x1, x2


print("=" * 60)
print("🚀 开始加载 PEA Diffusion 模型")
print("=" * 60)

dtype = torch.bfloat16
device = "cuda"
ckpt_id = "/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/FLUX.1-schnell"
text_encoder_ckpt_id = "/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/byt5-xxl"

print(f"📋 模型配置:")
print(f"  - 设备: {device}")
print(f"  - 数据类型: {dtype}")
print(f"  - FLUX模型路径: {ckpt_id}")
print(f"  - T5编码器路径: {text_encoder_ckpt_id}")

# 1. 初始化MLP投影层
print("\n🔧 [1/4] 初始化MLP投影层...")
start_time = time.time()
proj_t5 = MLP(in_dim=4672, out_dim=4096, hidden_dim=4096, out_dim1=768).to(device=device, dtype=dtype)
print(f"  ✅ MLP投影层初始化完成 ({time.time() - start_time:.2f}s)")

# 2. 加载T5文本编码器 & 分词器
print("\n📝 [2/4] 加载T5文本编码器与分词器...")
start_time = time.time()
text_encoder_t5 = T5ForConditionalGeneration.from_pretrained(text_encoder_ckpt_id).get_encoder().to(device=device, dtype=dtype)
tokenizer_t5 = AutoTokenizer.from_pretrained(text_encoder_ckpt_id)
print(f"  ✅ T5加载完成 ({time.time() - start_time:.2f}s)")

# 3. 加载投影层权重
print("\n⚖️  [3/4] 加载投影层权重...")
start_time = time.time()
proj_t5_save_path = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/MultilingualFLUX.1-adapter/diffusion_pytorch_model.bin"
print(f"  📁 权重文件路径: {proj_t5_save_path}")
state_dict = torch.load(proj_t5_save_path, map_location="cpu")
state_dict_new = {k.replace("module.", ""): v for k, v in state_dict.items()}
proj_t5.load_state_dict(state_dict_new)
print(f"  ✅ 投影层权重加载完成 ({time.time() - start_time:.2f}s)")

# 4. 加载FLUX Pipeline（让它负责解码）
print("\n🌊 [4/4] 加载FLUX Pipeline...")
start_time = time.time()
pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None, text_encoder_2=None,
    tokenizer=None, tokenizer_2=None, vae=None,
    torch_dtype=torch.bfloat16
).to(device)
print(f"  ✅ FLUX Pipeline加载完成 ({time.time() - start_time:.2f}s)")

print("\n" + "=" * 60)
print("🎉 所有模型加载完成！开始图像生成...")
print("=" * 60)

raw_text = "一个漂亮的女孩在海边散步"
print(f"\n💬 输入文本: '{raw_text}'")

print("\n🎨 开始图像生成流程...")
total_start_time = time.time()

with torch.no_grad():
    # 1) 文本编码
    print("  📝 [1/3] 文本分词和编码...")
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

    # 2) 适配器投影
    print("  🔄 [2/3] MLP投影...")
    step_start = time.time()
    pooled_prompt_embeds, prompt_embeds = proj_t5(text_embeddings)
    print(f"      池化嵌入维度: {pooled_prompt_embeds.shape}")
    print(f"      提示嵌入维度: {prompt_embeds.shape}")
    print(f"      ✅ 投影完成 ({time.time() - step_start:.2f}s)")

    # 3) 直接由 pipeline 生成最终图像（内部自动解码）
    print("  🌀 [3/3] FLUX生成(PIL)...")
    step_start = time.time()
    height, width = 1024, 1024
    num_steps = 4
    print(f"      图像尺寸: {height}x{width}")
    print(f"      推理步数: {num_steps}")

    images = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=0,
        height=height, width=width,
        output_type="pil",        # ✅ 关键改动：由管线完成解包+VAE解码
    ).images

    output_path = "ChineseFLUX.jpg"
    images[0].save(output_path)
    print(f"      ✅ 图像已保存到: {output_path} ({time.time() - step_start:.2f}s)")

print(f"\n🎉 图像生成完成！总用时: {time.time() - total_start_time:.2f}s")
print("=" * 60)
