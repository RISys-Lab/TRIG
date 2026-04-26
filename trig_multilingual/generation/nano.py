from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import sys
import argparse
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import DEFAULT_DATASET, TEXT_RENDERING_SPLIT, iter_range, load_text_rendering_data, replace_render_token

OUTPUT_DIR = "/home/muzammal/Projects/TRIG/data/output/tr_ml/nanobanana"
MODEL_NAME = "gemini-2.5-flash-image-preview"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate images using Gemini image preview")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET,
                        help="Hugging Face dataset name or local dataset directory")
    parser.add_argument("--split", type=str, default=TEXT_RENDERING_SPLIT,
                        help="Dataset split to use for multilingual text rendering")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Optional legacy JSON file. Overrides dataset_name when set")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_existing_image(data_id, output_dir):
    """检查图片是否已存在"""
    output_path = os.path.join(output_dir, f"{data_id}.png")
    return os.path.exists(output_path)

# 初始化客户端
client = genai.Client()

def generate_image(prompt, data_id, output_dir):
    """生成单张图片"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        
        image_parts = [
            part.inline_data.data
            for part in response.candidates[0].content.parts
            if part.inline_data
        ]
        
        if image_parts:
            image = Image.open(BytesIO(image_parts[0]))
            output_path = os.path.join(output_dir, f"{data_id}.png")
            image.save(output_path)
            return True, output_path
        else:
            return False, "No image generated"
            
    except Exception as e:
        return False, str(e)

def main():
    args = parse_args()
    data_list = load_text_rendering_data(args.dataset_name, args.split, args.data_file)
    
    print(f"Loaded {len(data_list)} items from {args.data_file or args.dataset_name}:{args.split}")
    
    # 批量生成图片
    success_count = 0
    error_count = 0
    
    for item in tqdm(iter_range(data_list, args.start_idx, args.end_idx), desc="Generating images"):
        data_id = item["data_id"]
        prompt = item["prompt"]
        render_text = item.get("render_text")
        
        # 检查是否跳过已存在的图片
        if check_existing_image(data_id, OUTPUT_DIR):
            continue
        
        final_prompt = replace_render_token(prompt, render_text)
        
        # 生成图片
        success, result = generate_image(final_prompt, data_id, OUTPUT_DIR)
        
        if success:
            success_count += 1
            print(f"✅ Generated: {data_id} -> {result}")
        else:
            error_count += 1
            print(f"❌ Failed: {data_id} -> {result}")
    
    print(f"\n🎉 Generation completed!")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
