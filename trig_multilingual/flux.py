import torch
from diffusers import FluxPipeline
import os
import argparse
from tqdm import tqdm
from PIL import Image

from data import DEFAULT_DATASET, TEXT_RENDERING_SPLIT, iter_range, load_text_rendering_data, replace_render_token

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate images using FLUX.1-dev model")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET,
                        help="Hugging Face dataset name or local dataset directory")
    parser.add_argument("--split", type=str, default=TEXT_RENDERING_SPLIT,
                        help="Dataset split to use for multilingual text rendering")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Optional legacy JSON file. Overrides dataset_name when set")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for generation (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index for generation (exclusive, -1 means all)")
    return parser.parse_args()

# 配置
OUTPUT_DIR = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/output/tr_ml/flux"
MODEL_NAME = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/FLUX.1-Krea-dev"

def check_existing_image(data_id, output_dir):
    """检查图片是否已存在"""
    output_path = os.path.join(output_dir, f"{data_id}.png")
    return os.path.exists(output_path)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print(f"📦 Loading FLUX model: {MODEL_NAME}")
print("⏳ This may take a few minutes for the first time...")

pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")



def generate_image(prompt, data_id, output_dir):
    """生成单张图片"""
    try:
        # FLUX推荐参数
        width, height = (1024, 1024)
        guidance_scale = 3.5
        num_inference_steps = 50
        max_sequence_length = 512
        
        # 生成图像
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cuda").manual_seed(42)  # 使用CPU generator以保持一致性
        ).images[0]
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{data_id}.png")
        image.save(output_path)
        
        return True, output_path
        
    except Exception as e:
        return False, str(e)

def main():
    # 解析参数
    args = parse_args()
    
    # 加载TRIG-Multilingual parquet数据
    data_list = load_text_rendering_data(args.dataset_name, args.split, args.data_file)
    
    print(f"📊 Loaded {len(data_list)} items from {args.data_file or args.dataset_name}:{args.split}")
    
    # 确定处理范围
    start_idx = args.start_idx
    end_idx = len(data_list) if args.end_idx == -1 else min(args.end_idx, len(data_list))
    
    print(f"🎯 Processing range: {start_idx} to {end_idx} ({end_idx - start_idx} items)")
    
    # 切片数据
    data_subset = iter_range(data_list, args.start_idx, args.end_idx)
    
    # 批量生成图片
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    print(f"🎨 Starting FLUX image generation...")
    print(f"⚙️  Parameters: 1024x1024, guidance_scale=3.5, steps=50")
    
    for item in tqdm(data_subset, desc="🎨 Generating FLUX images"):
        data_id = item["data_id"]
        prompt = item["prompt"]
        render_text = item.get("render_text")
        
        # 检查是否跳过已存在的图片
        if check_existing_image(data_id, OUTPUT_DIR):
            skipped_count += 1
            continue
        
        final_prompt = replace_render_token(prompt, render_text)
        
        # 生成图片
        success, result = generate_image(final_prompt, data_id, OUTPUT_DIR)
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"❌ Failed: {data_id} -> {result}")
        
    print(f"\n🎉 FLUX Generation completed!")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"💾 Total images generated: {success_count} PNG files")
    

if __name__ == "__main__":
    main()
