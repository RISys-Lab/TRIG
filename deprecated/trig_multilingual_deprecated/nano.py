from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import json
import os
from tqdm import tqdm

# 配置
JSON_FILE = "/home/muzammal/Projects/TRIG/dataset/TRIG-multilingual/trig_multilingual_tr.json"
OUTPUT_DIR = "/home/muzammal/Projects/TRIG/data/output/tr_ml/nanobanana"
MODEL_NAME = "gemini-2.5-flash-image-preview"

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
    # 加载JSON数据
    print(f"Loading JSON file: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Loaded {len(data_list)} items")
    
    # 批量生成图片
    success_count = 0
    error_count = 0
    
    for item in tqdm(data_list, desc="Generating images"):
        data_id = item["data_id"]
        prompt = item["prompt"]
        dimension_prompt = item["dimension_prompt"]
        
        # 检查是否跳过已存在的图片
        if check_existing_image(data_id, OUTPUT_DIR):
            continue
        
        # 替换 <sks1> 为 dimension_prompt[0]
        if dimension_prompt and len(dimension_prompt) > 0:
            final_prompt = prompt.replace("<sks1>", dimension_prompt[0])
        else:
            final_prompt = prompt.replace("<sks1>", "")
        
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