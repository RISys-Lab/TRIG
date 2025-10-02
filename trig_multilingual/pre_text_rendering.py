import os
import re
import base64
import io
import argparse
import json
import time
import random
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path",default="/data/dataset_zoo", type=str)
    parser.add_argument("--raw_dataset", default="EasyText", type=str)
    parser.add_argument("--save_dataset", default="TRIGv1.5", type=str)


    return parser.parse_args()


def load_data(args):
    # dataset = load_dataset("lllrrnn/EasyText")
    dataset = load_dataset(os.path.join(args.raw_path, args.raw_dataset))
    
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
        position = object['position']
        
        base64_image, image_type = encode_image(image)
        ocr_message = create_ocr_message(base64_image, image_type)
        ocr_contents = send_request(ocr_message)

        for lang, ocr_content in ocr_contents.items():
            data_id = f'TR_{LANG_DICT[lang].lower()}_{str(idx)}'            
            content = {
                "data_id": data_id,
                # "item": None,
                "prompt": text,
                "dimension_prompt": [ocr_content, position],
                "parent_dataset": ["EasyText", "Origin"],
                "img_id": f'{data_id}.jpg',
                "dimensions": ["TR", LANG_DICT[lang].lower()],
                # "image": None,
            }
            results.append(content)
    
    save_path = os.path.join(args.raw_path, args.save_dataset)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "trig_multilingual_tr.json"), 'w', encoding='utf-8') as file:
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
        position = obj['position']

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
                "dimension_prompt": [ocr_content, position],
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
    with open(os.path.join(save_path, "trig_multilingual_tr.json"), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


##############################


# --- Text Rendering ---

def render_text(text, font_path, output_path, image_size=(512, 128), margin=10):
    """渲染文本到图片"""
    # 首先计算文字的实际大小，如果图片太小则动态调整
    temp_img = Image.new("RGB", (1000, 1000), "white")  # 临时大图片用于测量
    temp_draw = ImageDraw.Draw(temp_img)
    
    # 初始字体大小
    font_size = 60
    font = ImageFont.truetype(font_path, font_size)
    
    # 测量文字实际大小
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 如果计算出的图片尺寸太小，动态调整
    required_width = text_width + 2 * margin
    required_height = text_height + 2 * margin
    
    # 使用较大的尺寸
    final_width = max(image_size[0], required_width)
    final_height = max(image_size[1], required_height)
    
    # 如果文字太大，调整字体大小以适应原始图片尺寸
    if text_width > image_size[0] - 2*margin or text_height > image_size[1] - 2*margin:
        # 计算合适的字体大小
        width_ratio = (image_size[0] - 2*margin) / text_width
        height_ratio = (image_size[1] - 2*margin) / text_height
        size_ratio = min(width_ratio, height_ratio)
        
        font_size = max(8, int(font_size * size_ratio))
        font = ImageFont.truetype(font_path, font_size)
        
        # 重新测量
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 使用原始图片尺寸
        final_width = image_size[0]
        final_height = image_size[1]
    
    # 创建最终图像
    img = Image.new("RGB", (final_width, final_height), "white")
    draw = ImageDraw.Draw(img)

    # 居中对齐
    x = (final_width - text_width) // 2
    y = (final_height - text_height) // 2

    # 绘制文字
    draw.text((x, y), text, font=font, fill="black")
    img.save(output_path)
    print(f"保存成功: {output_path} (尺寸: {final_width}x{final_height}, 字体: {font_size})")


def calculate_merged_positions(positions):
    """计算合并后的位置信息并返回新的位置和图片大小"""
    if not positions or len(positions) == 0:
        return None, (512, 128)  # 默认大小
    
    # 解析位置信息
    try:
        # 如果positions是字符串，先解析为列表
        if isinstance(positions, str):
            positions = eval(positions)
        
        # 计算所有第一部分的并集（插入位置）
        all_insert_x_coords = []
        all_insert_y_coords = []
        
        # 计算所有第二部分的并集（图片尺寸）
        all_image_x_coords = []
        all_image_y_coords = []
        
        for group in positions:
            if len(group) >= 2:
                # 第一部分是插入位置的4个点
                part1 = group[0]
                if len(part1) >= 4:
                    for point in part1:
                        all_insert_x_coords.append(point[0])
                        all_insert_y_coords.append(point[1])
                
                # 第二部分是图片的左上和右下坐标
                part2 = group[1]
                if len(part2) >= 2:
                    # part2[0] 是左上角 [x, y]
                    # part2[1] 是右下角 [x, y]
                    all_image_x_coords.extend([part2[0][0], part2[1][0]])
                    all_image_y_coords.extend([part2[0][1], part2[1][1]])
        
        # 计算合并后的插入位置边界框
        if all_insert_x_coords and all_insert_y_coords:
            insert_min_x, insert_max_x = min(all_insert_x_coords), max(all_insert_x_coords)
            insert_min_y, insert_max_y = min(all_insert_y_coords), max(all_insert_y_coords)
            
            # 创建合并后的插入位置（4个角点）
            merged_insert_pos = [
                [insert_min_x, insert_min_y],  # 左上
                [insert_max_x, insert_min_y],  # 右上
                [insert_max_x, insert_max_y],  # 右下
                [insert_min_x, insert_max_y]   # 左下
            ]
        else:
            merged_insert_pos = [[0, 0], [100, 0], [100, 50], [0, 50]]
        
        # 计算合并后的图片尺寸
        if all_image_x_coords and all_image_y_coords:
            image_min_x, image_max_x = min(all_image_x_coords), max(all_image_x_coords)
            image_min_y, image_max_y = min(all_image_y_coords), max(all_image_y_coords)
            
            # 计算宽度和高度
            width = image_max_x - image_min_x
            height = image_max_y - image_min_y
            
            # 确保最小尺寸，并增加一些边距以确保文字不被截断
            width = max(width, 200)  # 增加最小宽度
            height = max(height, 100)  # 增加最小高度
            
            # 创建合并后的图片尺寸坐标
            merged_image_size = [[0, 0], [width, height]]
        else:
            width, height = 512, 128
            merged_image_size = [[0, 0], [width, height]]
        
        # 创建新的合并位置信息
        merged_positions = [[merged_insert_pos, merged_image_size]]
        
        return merged_positions, (width, height)
            
    except Exception as e:
        print(f"  ⚠️ 解析位置信息失败: {e}")
        return None, (512, 128)  # 默认大小


def render_dimension_prompts(args):
    """从JSON文件读取数据并渲染dimension_prompt"""
    
    print("读取JSON文件...")
    
    data_path = os.path.join(args.raw_path, args.save_dataset)
    with open(os.path.join(data_path, "trig_multilingual_tr.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共找到 {len(data)} 项数据")
    
    # 创建输出目录
    output_dir = os.path.join(data_path, "condition_image")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有数据
    print(f"开始渲染所有 {len(data)} 项数据...")
    print("=" * 60)
    
    success_count = 0
    updated_data = []  # 存储更新后的数据
    
    for i in tqdm(range(len(data)), desc="渲染文本图片", unit="item"):
        item = data[i].copy()  # 创建副本以避免修改原始数据
        
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
            updated_data.append(item)  # 保留原始数据
            continue
        
        # 获取要渲染的文本（dimension_prompt的第一个元素）
        text_to_render = dimension_prompt[0] if dimension_prompt and len(dimension_prompt) > 0 else "No text"
        
        # 获取位置信息并计算合并后的位置和图片大小
        positions = dimension_prompt[1] if len(dimension_prompt) > 1 else None
        merged_positions, image_size = calculate_merged_positions(positions)
        
        # 更新 dimension_prompt[1] 为合并后的位置信息
        if merged_positions is not None:
            item["dimension_prompt"][1] = merged_positions
        
        print(f"\n处理第 {i+1} 项:")
        print(f"  Data ID: {data_id}")
        print(f"  语言: {lang_name} ({lang_code})")
        print(f"  文本: {text_to_render}")
        print(f"  原始位置: {positions}")
        print(f"  合并后位置: {merged_positions}")
        print(f"  图片大小: {image_size}")
        print(f"  IMG ID: {img_id}")
        
        try:
            # 获取字体路径
            font_path = "/data/TRIG/trig_multilingual/font/Arial_Unicode.ttf"
            
            # 检查字体文件是否存在
            if not os.path.exists(font_path):
                print(f"  ❌ 字体文件不存在: {font_path}")
                updated_data.append(item)  # 保留更新后的数据
                continue
            
            # 为每个语言创建单独的子目录
            lang_output_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_output_dir, exist_ok=True)
            
            # 使用img_id中的文件名，保持.jpg扩展名
            img_filename = img_id
            output_path = os.path.join(lang_output_dir, img_filename)
            
            # 渲染文本，使用计算出的图片大小
            print(f"  🎨 开始渲染文本: '{text_to_render[:50]}{'...' if len(text_to_render) > 50 else ''}'")
            print(f"  📏 目标图片尺寸: {image_size}")
            render_text(text_to_render, font_path, output_path, image_size=image_size)
            print(f"  ✅ 渲染成功: {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 渲染失败: {str(e)}")
        
        # 将更新后的数据添加到列表中
        updated_data.append(item)
    
    # 保存更新后的数据
    print("\n保存更新后的数据...")
    with open(os.path.join(data_path, "trig_multilingual_tr.json"), 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)
    print("✅ 数据已保存")
    
    print("\n" + "=" * 60)
    print(f"渲染完成！成功: {success_count}/{len(data)}")
    print(f"图片保存在: {output_dir}")
    print("目录结构:")
    for lang_code in LANG_DICT.values():
        lang_dir = os.path.join(output_dir, lang_code)
        if os.path.exists(lang_dir):
            file_count = len([f for f in os.listdir(lang_dir) if f.endswith('.jpg')])
            print(f"  {lang_code}/: {file_count} 个文件")
    
    return success_count


##############################


# --- Coarse Mask ---

def create_base_grid(image_size=(1024, 1024)):
    """创建基础网格图像，包含25个正方形格子和中心红色方块"""
    img = Image.new("RGB", image_size, (200, 200, 200))  # 背景色设为RGB(200,200,200)
    draw = ImageDraw.Draw(img)
    
    # 计算网格参数
    grid_size = 5  # 5x5网格
    cell_width = image_size[0] // grid_size
    cell_height = image_size[1] // grid_size
    
    # 先填充所有格子为RGB(200,200,200)
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = i * cell_width
            y1 = j * cell_height
            x2 = (i + 1) * cell_width
            y2 = (j + 1) * cell_height
            draw.rectangle([(x1, y1), (x2, y2)], fill=(200, 200, 200))
    
    # 绘制网格线，使用RGB(150,150,150)
    for i in range(grid_size + 1):
        x = int(i * cell_width)
        y = int(i * cell_height)
        # 垂直线
        draw.line([(x, 0), (x, image_size[1])], fill=(150, 150, 150), width=2)
        # 水平线
        draw.line([(0, y), (image_size[0], y)], fill=(150, 150, 150), width=2)
    
    # 在中心格子绘制红色正方形 - 测试时注释掉
    # center_x = 2 * cell_width
    # center_y = 2 * cell_height
    # center_size = min(cell_width, cell_height) // 8  # 缩减为原来的一半
    
    # # 确保坐标为整数，避免抗锯齿问题
    # red_square_x = int(center_x + (cell_width - center_size) // 2)
    # red_square_y = int(center_y + (cell_height - center_size) // 2)
    # red_square_size = int(center_size)
    
    # # 使用精确的红色RGB值，避免颜色混合
    # draw.rectangle([
    #     (red_square_x, red_square_y),
    #     (red_square_x + red_square_size, red_square_y + red_square_size)
    # ], fill=(255, 0, 0))  # 使用精确的红色RGB值
    
    return img


def add_position_scribbles(draw, positions, image_size):
    """在指定位置添加涂鸦，确保生成一个连续的黑色区域"""
    if not positions:
        return
    
    try:
        # 如果positions是字符串，先解析为列表
        if isinstance(positions, str):
            positions = eval(positions)
        
        # 收集所有区域的坐标
        all_x_coords = []
        all_y_coords = []
        
        for group in positions:
            if len(group) >= 2:
                # 第一部分是插入位置的4个点
                part1 = group[0]
                if len(part1) >= 4:
                    # 收集所有坐标点
                    x_coords = [point[0] for point in part1]
                    y_coords = [point[1] for point in part1]
                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
        
        if all_x_coords and all_y_coords:
            # 计算所有区域的合并边界框
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            min_y, max_y = min(all_y_coords), max(all_y_coords)
            
            # 确保坐标在图像范围内
            min_x = max(0, min_x)
            max_x = min(image_size[0], max_x)
            min_y = max(0, min_y)
            max_y = min(image_size[1], max_y)
            
            # 生成一个连续的黑色区域
            add_scribbles_in_area(draw, min_x, min_y, max_x, max_y)
                    
    except Exception as e:
        print(f"  ⚠️ 处理位置信息失败: {e}")


def add_scribbles_in_area(draw, min_x, min_y, max_x, max_y):
    """在指定区域内完全涂满黑色"""
    area_width = max_x - min_x
    area_height = max_y - min_y
    
    if area_width <= 0 or area_height <= 0:
        return
    
    # 直接填充整个区域为黑色
    draw.rectangle([
        (int(min_x), int(min_y)),
        (int(max_x), int(max_y))
    ], fill="black")


def create_coarse_mask(args):
    """为每个item生成coarse mask"""
    
    print("读取JSON文件...")
    
    data_path = os.path.join(args.raw_path, args.save_dataset)
    with open(os.path.join(data_path, "trig_multilingual_tr.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共找到 {len(data)} 项数据")
    
    # 创建输出目录
    output_dir = os.path.join(data_path, "coarse_mask")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有数据
    print(f"开始生成所有 {len(data)} 项coarse mask...")
    print("=" * 60)
    
    success_count = 0
    
    for i in tqdm(range(len(data)), desc="生成coarse mask", unit="item"):
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
        
        # 获取位置信息
        positions = dimension_prompt[1] if len(dimension_prompt) > 1 else None
        
        print(f"\n处理第 {i+1} 项:")
        print(f"  Data ID: {data_id}")
        print(f"  语言: {lang_name} ({lang_code})")
        print(f"  位置信息: {positions}")
        print(f"  IMG ID: {img_id}")
        
        try:
            # 为每个语言创建单独的子目录
            lang_output_dir = os.path.join(output_dir, lang_code)
            os.makedirs(lang_output_dir, exist_ok=True)
            
            # 使用img_id中的文件名，保持.jpg扩展名
            img_filename = img_id
            output_path = os.path.join(lang_output_dir, img_filename)
            
            # 创建基础网格图像
            base_img = create_base_grid()
            
            # 在指定位置添加涂鸦
            if positions:
                draw = ImageDraw.Draw(base_img)
                add_position_scribbles(draw, positions, base_img.size)
            
            # 将图像resize为512x512并保存（使用最临近插值）
            base_img_resized = base_img.resize((512, 512), Image.Resampling.NEAREST)
            base_img_resized.save(output_path)
            print(f"  ✅ 生成成功: {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 生成失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Coarse mask生成完成！成功: {success_count}/{len(data)}")
    print(f"图片保存在: {output_dir}")
    print("目录结构:")
    for lang_code in LANG_DICT.values():
        lang_dir = os.path.join(output_dir, lang_code)
        if os.path.exists(lang_dir):
            file_count = len([f for f in os.listdir(lang_dir) if f.endswith('.jpg')])
            print(f"  {lang_code}/: {file_count} 个文件")
    
    return success_count


if __name__ == "__main__":    
    args = get_args()
    random.seed(42)
    
    # 处理数据
    data_list = load_data(args)
    async_mode = True
    if async_mode:
        print("Running in **async high-concurrency** mode with asyncio.")
        asyncio.run(async_process_data(data_list))
    else:
        print("Running in **sync single-threaded** mode.")
        process_data(data_list)
    
    # 文本渲染
    print("\n" + "=" * 60)
    print("开始执行文本渲染...")
    render_dimension_prompts(args)
    
    # 粗略掩码
    print("\n" + "=" * 60)
    print("开始执行粗略掩码...")
    create_coarse_mask(args)
    