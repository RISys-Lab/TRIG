import os
import re
import base64
import io
import argparse
import json
import time
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


LANG_DICT = {
    "English": "en",
    "Chinese": "zh",
    "Hindi": "hi",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
}


FONT_DICT = {
    "English": "./font/NotoSans-Regular.ttf",
    "Chinese": "./font/NotoSansSC-Regular.ttf",
    "Hindi": "./font/NotoSansDevanagari-Regular.ttf",
    "Spanish": "./font/NotoSans-Regular.ttf",
    "Arabic": "./font/NotoSansArabic-Regular.ttf",
    "French": "./font/NotoSans-Regular.ttf",
    "Portuguese": "./font/NotoSans-Regular.ttf",
    "Russian": "./font/NotoSans-Regular.ttf",
    "Japanese": "./font/SourceHanSansJP-Regular.otf",
    "Korean": "./font/NotoSansKR-Regular.ttf",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path",default="/data/dataset_zoo", type=str)
    parser.add_argument("--raw_dataset", default="EasyText", type=str)
    parser.add_argument("--save_dataset", default="TRIGv1.5", type=str)


    return parser.parse_args()


def load_data(args):
    dataset = load_dataset("lllrrnn/EasyText")
    # dataset = load_dataset(args.raw_path, args.raw_dataset)
    
    subset = [
        item for item in dataset['train']
        if isinstance(item, dict)
        and '<sks1>' in str(item.get('text', ''))
        and '<sks2>' not in str(item.get('text', ''))
    ]
    
    return subset


def encode_image(img):
    """Encodes a PIL image to Base64 and detects its type."""
    image_type = img.format.lower() if img.format else "jpeg"

    if image_type not in ["jpeg", "png"]:
        raise ValueError(f"Unsupported image format: {image_type}")

    buffered = io.BytesIO()
    img.save(buffered, format=image_type.upper())
    image_data = buffered.getvalue()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    return base64_image, image_type


def create_ocr_message(base64_image, image_type):
    ocr_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an OCR system. Extract all text from the image, including any text within the image itself."
                        "You are given an image that contains text in black font on a white background.\n"
                        "1. First, accurately extract all the text from the image. Make sure the extracted content "
                        "is complete, forms valid words or phrases, and strictly follows the grammar and spelling "
                        "conventions of the detected language.\n"
                        "2. Then, translate the extracted content into the following 10 languages: "
                        "English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, Japanese, and Korean.\n"
                        "3. Return your result strictly in valid JSON format, with the keys in the following order: "
                        "English, Chinese, Hindi, Spanish, Arabic, French, Portuguese, Russian, Japanese, Korean.\n"
                        "4. Ensure that the translations convey exactly the same meaning as the extracted text, without "
                        "adding explanations or extra content.\n\n"
                        "The output must be **only** a valid JSON object following this format:\n"
                        "{\n"
                        "  \"English\": \"...\",\n"
                        "  \"Chinese\": \"...\",\n"
                        "  \"Hindi\": \"...\",\n"
                        "  \"Spanish\": \"...\",\n"
                        "  \"Arabic\": \"...\",\n"
                        "  \"French\": \"...\",\n"
                        "  \"Portuguese\": \"...\",\n"
                        "  \"Russian\": \"...\",\n"
                        "  \"Japanese\": \"...\",\n"
                        "  \"Korean\": \"...\"\n"
                        "}"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    return ocr_message


# --- Base mode ---

from tqdm import tqdm
from openai import OpenAI

def send_request(messages, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            print("Sending request...")
            api_key = '  '
            client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages
            )
            response_content = response.choices[0].message.content
            response_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            response_content = json.loads(response_match.group(0))

            return response_content
        
        except Exception as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"Retry {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retry limit reached, aborting request.")
                return None


def process_data(data_list):
    
    results = []    
    for idx, object in enumerate(tqdm(data_list[:300], desc="Processing object", unit="image")):
        text = object['text']
        
        image = object['condition_image']
        base64_image, image_type = encode_image(image)
        ocr_message = create_ocr_message(base64_image, image_type)
        ocr_contents = send_request(ocr_message)

        for lang, ocr_content in ocr_contents.items():
            data_id = f'TR_{LANG_DICT[lang].lower()}_{str(idx)}'            
            content = {
                "data_id": data_id,
                # "item": None,
                "prompt": text,
                "dimension_prompt": [ocr_content, ""],
                "parent_dataset": ["EasyText", "Origin"],
                "img_id": f'{data_id}.jpg',
                "dimensions": ["TR", LANG_DICT[lang].lower()],
                # "image": None,
            }
            results.append(content)
    
    save_path = os.path.join(args.raw_path, args.save_dataset)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "trig_multilingual.json"), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


##############################


# --- High Concurrency ---

import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

MAX_CONCURRENT_REQUESTS = 64
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def async_send_request(client, messages, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=messages
                )
            response_content = response.choices[0].message.content
            response_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            response_content = json.loads(response_match.group(0))
            return response_content

        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Request failed: {e}")
            if attempt + 1 < max_retries:
                wait_time = delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                return None


async def async_process_single_item(client, idx, obj):
    try:
        text = obj['text']
        
        image = obj['condition_image']
        base64_image, image_type = encode_image(image)
        ocr_message = create_ocr_message(base64_image, image_type)
        ocr_contents = await async_send_request(client, ocr_message)
        
        if ocr_contents is None:
            return []

        results = []
        for lang, ocr_content in ocr_contents.items():
            data_id = f'TR_{LANG_DICT[lang].lower()}_{str(idx)}'            
            content = {
                "data_id": data_id,
                # "item": None,
                "prompt": text,
                "dimension_prompt": [ocr_content, ""],
                "parent_dataset": ["EasyText", "Origin"],
                "img_id": f'{data_id}.jpg',
                "dimensions": ["TR", LANG_DICT[lang].lower()],
                # "image": None,
            }
            results.append(content)
        
        return results

    except Exception as e:
        return [{"data_id": f"error_{idx}", "error": str(e)}]


async def async_process_data(data_list):
    api_key = '  '
    client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    tasks = [
        async_process_single_item(client, idx, obj)
        for idx, obj in enumerate(data_list[:300])
    ]
    results_list = await tqdm_asyncio.gather(*tasks)

    # Flatten the results since each item now returns a list of content entries
    results = []
    for result_list in results_list:
        results.extend(result_list)

    save_path = os.path.join(args.raw_path, args.save_dataset)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "trig_multilingual.json"), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


##############################


# --- Text Rendering ---

def render_text(text, font_path, output_path, image_size=(512, 128), margin=10):
    """渲染文本到图片"""
    # 创建白底图像
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # 初始字体大小
    font_size = 60
    font = ImageFont.truetype(font_path, font_size)

    # 调整字体大小，确保文字能放下
    bbox = draw.textbbox((0, 0), text, font=font)
    while (bbox[2] - bbox[0] > image_size[0] - 2*margin) and font_size > 10:
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)

    # 居中对齐
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # 绘制文字
    draw.text((x, y), text, font=font, fill="black")
    img.save(output_path)
    print(f"保存成功: {output_path}")


def render_dimension_prompts(args):
    """从JSON文件读取数据并渲染dimension_prompt"""
    
    print("读取JSON文件...")
    
    data_path = os.path.join(args.raw_path, args.save_dataset)
    with open(os.path.join(data_path, "trig_multilingual.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共找到 {len(data)} 项数据")
    
    # 创建输出目录
    output_dir = os.path.join(data_path, "condition_image")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有数据
    print(f"开始渲染所有 {len(data)} 项数据...")
    print("=" * 60)
    
    success_count = 0
    
    for i in tqdm(range(len(data)), desc="渲染文本图片", unit="item"):
        item = data[i]
        
        # 获取基本信息
        data_id = item.get("data_id", f"item_{i}")
        dimensions = item.get("dimensions", [])
        dimension_prompt = item.get("dimension_prompt", [])
        img_id = item.get("img_id", f"{data_id}.jpg")
        
        # 获取语言代码
        lang_code = dimensions[1] if len(dimensions) > 1 else "en"
        
        # 通过LANG_DICT找到对应的语言名称
        lang_name = None
        for name, code in LANG_DICT.items():
            if code == lang_code:
                lang_name = name
                break
        
        if lang_name is None:
            print(f"  ❌ 不支持的语言代码: {lang_code}")
            continue
        
        # 获取要渲染的文本（dimension_prompt的第一个元素）
        text_to_render = dimension_prompt[0] if dimension_prompt and len(dimension_prompt) > 0 else "No text"
        
        print(f"\n处理第 {i+1} 项:")
        print(f"  Data ID: {data_id}")
        print(f"  语言: {lang_name} ({lang_code})")
        print(f"  文本: {text_to_render}")
        print(f"  IMG ID: {img_id}")
        
        try:
            # 获取字体路径
            font_path = FONT_DICT.get(lang_name, FONT_DICT["English"])
            
            # 检查字体文件是否存在
            if not os.path.exists(font_path):
                print(f"  ❌ 字体文件不存在: {font_path}")
                continue
            
            # 为每个语言创建单独的子目录
            lang_output_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_output_dir, exist_ok=True)
            
            # 使用img_id中的文件名，但改为.png扩展名
            img_filename = os.path.splitext(img_id)[0] + ".png"
            output_path = os.path.join(lang_output_dir, img_filename)
            
            # 渲染文本
            render_text(text_to_render, font_path, output_path)
            print(f"  ✅ 渲染成功: {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 渲染失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"渲染完成！成功: {success_count}/{len(data)}")
    print(f"图片保存在: {output_dir}")
    print("目录结构:")
    for lang_code in LANG_DICT.values():
        lang_dir = os.path.join(output_dir, lang_code)
        if os.path.exists(lang_dir):
            file_count = len([f for f in os.listdir(lang_dir) if f.endswith('.png')])
            print(f"  {lang_code}/: {file_count} 个文件")
    
    return success_count



if __name__ == "__main__":    
    args = get_args()
    
    data_list = load_data(args)
    async_mode = True
    if async_mode:
        print("Running in **async high-concurrency** mode with asyncio.")
        asyncio.run(async_process_data(data_list))
    else:
        print("Running in **sync single-threaded** mode.")
        process_data(data_list)
    
    # 主程序执行完成后，执行文本渲染
    print("\n" + "=" * 60)
    print("开始执行文本渲染...")
    render_dimension_prompts(args)
    