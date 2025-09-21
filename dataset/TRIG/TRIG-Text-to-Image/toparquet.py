import pandas as pd
import json
from PIL import Image
import os
import io
import base64
from tqdm import tqdm

# 原始 JSON 数据
with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-without-m-e-v8.json", "r") as f:
    data = json.load(f)

need_item = ["data_id","item","prompt","dimension_prompt","parent_dataset","img_id","dimensions","image"]
image_dir = "/home/muzammal/Projects/TRIG/dataset/Trig/Trig-image-editing/image-editing-images"

# 处理数据，只保留需要的字段
processed_data = []
for item in tqdm(data):
    # 创建新的item，只包含需要的字段
    new_item = {}
    
    # 处理除image外的其他字段
    for field in need_item:
        if field == "dimensions":
            continue  # image字段单独处理
        new_item[field] = item.get(field, None)  # 如果字段不存在，填充None
        new_item["dimensions"] = item.get("dimension", None)
    
        # img_path = os.path.join(image_dir, item.get("img_id", ""))
        # new_item["image"] = img_path
    
    processed_data.append(new_item)

# 转为 DataFrame
df = pd.DataFrame(processed_data)

# 确保列的顺序与need_item一致
df = df[need_item]

with open("/home/muzammal/Projects/TRIG/dataset/Trig/HF/text_to_image.json", "w") as f:
    json.dump(processed_data, f)
