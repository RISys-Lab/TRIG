import json
import os
import subprocess
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# 配置
JSON_FILE = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/dataset/TRIG-multilingual/trig_multilingual_tr.json"
OUTPUT_DIR = "/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/output/tr_ml/seedream_image"
API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
API_KEY = "d02972a8-c26a-4dc3-a690-f55923af22fa"
MODEL_NAME = "seedream-4-0-250828"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_existing_image(data_id, output_dir):
    """检查图片是否已存在"""
    output_path = os.path.join(output_dir, f"{data_id}.png")
    return os.path.exists(output_path)

def download_image(url, output_path):
    """从URL下载图片并保存"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存图片
        image = Image.open(BytesIO(response.content))
        image.save(output_path)
        return True, output_path
    except Exception as e:
        return False, str(e)

def generate_image_curl(prompt, data_id, output_dir):
    """使用curl调用API生成图片"""
    try:
        # 构建curl命令
        curl_data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "sequential_image_generation": "disabled",
            "response_format": "url",
            "size": "2K",
            "stream": False,
            "watermark": True
        }
        
        curl_cmd = [
            "curl", "-X", "POST", API_URL,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {API_KEY}",
            "-d", json.dumps(curl_data)
        ]
        
        # 执行curl命令
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return False, f"Curl command failed: {result.stderr}"
        
        # 解析响应
        response_data = json.loads(result.stdout)
        
        # 检查API响应
        if "data" not in response_data or not response_data["data"]:
            return False, f"No image data in response: {response_data}"
        
        # 获取图片URL
        image_url = response_data["data"][0]["url"]
        
        # 下载图片
        output_path = os.path.join(output_dir, f"{data_id}.png")
        success, download_result = download_image(image_url, output_path)
        
        if success:
            return True, output_path
        else:
            return False, f"Download failed: {download_result}"
            
    except subprocess.TimeoutExpired:
        return False, "Request timeout"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, str(e)

def generate_image_requests(prompt, data_id, output_dir):
    """使用requests库调用API生成图片（备选方案）"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "sequential_image_generation": "disabled",
            "response_format": "url",
            "size": "2K",
            "stream": False,
            "watermark": True
        }
        
        # 发送请求
        response = requests.post(API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        # 解析响应
        response_data = response.json()
        
        # 检查API响应
        if "data" not in response_data or not response_data["data"]:
            return False, f"No image data in response: {response_data}"
        
        # 获取图片URL
        image_url = response_data["data"][0]["url"]
        
        # 下载图片
        output_path = os.path.join(output_dir, f"{data_id}.png")
        success, download_result = download_image(image_url, output_path)
        
        if success:
            return True, output_path
        else:
            return False, f"Download failed: {download_result}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, str(e)

def main():
    print("🎨 SeeDream Image Generation Script")
    print("="*50)
    
    # 加载JSON数据
    print(f"📂 Loading JSON file: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"📊 Loaded {len(data_list)} items")
    print(f"🔑 Using API: {API_URL}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # 批量生成图片
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for item in tqdm(data_list, desc="🎨 Generating images"):
        data_id = item["data_id"]
        prompt = item["prompt"]
        dimension_prompt = item["dimension_prompt"]
        
        # 检查是否跳过已存在的图片
        if check_existing_image(data_id, OUTPUT_DIR):
            skipped_count += 1
            continue
        
        # 替换 <sks1> 为 dimension_prompt[0]
        if dimension_prompt and len(dimension_prompt) > 0 and dimension_prompt[0]:
            final_prompt = prompt.replace("<sks1>", dimension_prompt[0])
        else:
            final_prompt = prompt.replace("<sks1>", "")
        
        # 生成图片 - 优先使用requests，失败时使用curl
        success, result = generate_image_requests(final_prompt, data_id, OUTPUT_DIR)
        
        # 如果requests失败，尝试使用curl
        if not success:
            print(f"⚠️  Requests failed for {data_id}, trying curl...")
            success, result = generate_image_curl(final_prompt, data_id, OUTPUT_DIR)
        
        if success:
            success_count += 1
            if success_count % 10 == 0:  # 每10张图片打印一次进度
                print(f"✅ Generated {success_count} images. Latest: {data_id}")
        else:
            error_count += 1
            print(f"❌ Failed: {data_id} -> {result}")
    
    print(f"\n🎉 Generation completed!")
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"💾 Total images generated: {success_count} PNG files")

if __name__ == "__main__":
    main()
