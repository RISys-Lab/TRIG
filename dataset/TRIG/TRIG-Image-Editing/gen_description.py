import os
from openai import OpenAI
from PIL import Image
import base64
import json
import io
import imghdr

# 设置 OpenAI API Key（替换为你的 API Key）
api_key = 'sk-mqUwZI8bhIv746rG6f3fE830D8B146E789Fd11717aD8C4B1'
client = OpenAI(api_key=api_key, base_url="https://api.bltcy.ai/v1")


def encode_left_image(image_path):
    """Encodes an image to Base64 and detects its type after cropping the left square part."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        _, height = img.size
        crop_size = height
        cropped_img = img.crop((0, 0, crop_size, crop_size))
        img_buffer = io.BytesIO()
        cropped_img.save(img_buffer, format="PNG")
        img_data = img_buffer.getvalue()

    image_type = imghdr.what(None, img_data)
    if image_type not in ["jpeg", "png"]:
        raise ValueError(f"Unsupported image format: {image_type}")
    base64_image = base64.b64encode(img_data).decode("utf-8")
    return base64_image, image_type


def generate_description(image_path):
    """调用 OpenAI API 生成图片描述"""
    try:
        # 读取图片
        base64_image, image_type = encode_left_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent image description assistant and your task is to generate a brief, short but detailed description based on an input image."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this image:"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}}
                    ]
                }
            ],

        )

        # 获取 GPT 生成的描述
        description = response.choices[0].message.content
        return description

    except Exception as e:
        print(f"❌ 处理 {image_path} 时出错: {e}")
        return None


def process_images(folder_path, output_json="image_descriptions.json"):
    """遍历文件夹的所有图片，并对每个图片生成描述，保存为 JSON 文件"""
    if not os.path.exists(folder_path):
        print(f"❌ 目录 {folder_path} 不存在！")
        return

    image_descriptions = {}
    with open(output_json, "r", encoding="utf-8") as f:
        image_descriptions = json.load(f)

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):  # 只处理图片
            if file in image_descriptions:
                print(f"⏩ 跳过已处理的图片: {file}")
                continue
            image_path = os.path.join(folder_path, file)
            print(f"📷 处理图片: {file} ...")

            # 生成描述
            description = generate_description(image_path)
            if description:
                image_descriptions[file] = description
                print(f"✅ {file} -> {description}")
            # break

    # 保存结果到 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(image_descriptions, f, ensure_ascii=False, indent=4)

    print(f"\n📜 所有图片描述已保存至 `{output_json}`")


# 示例调用
folder_path = r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\images"  # 替换为你的图片文件夹路径
process_images(folder_path)
