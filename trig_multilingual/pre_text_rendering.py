import os
import re
import base64
import io
import argparse
import json
import time
from datasets import load_dataset


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
    # parser.add_argument("--raw_path",default=r'H:\ProjectsPro\TRIG\dataset\raw_dataset\text-rendering', type=str)
    parser.add_argument("--raw_path",default=r"H:\ProjectsPro\TRIG\data\dataset_hub", type=str)
    parser.add_argument("--dataset", default="EasyText", type=str)

    return parser.parse_args()


def load_data(args):
    dataset = load_dataset("lllrrnn/EasyText")
    
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
    
    with open("trig_multilingual.json", 'w', encoding='utf-8') as file:
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

    with open("trig_multilingual.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


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
    
