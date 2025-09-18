import os
import json
import numpy as np
import torch
import clip
# 注意：请确保 clipscore_main 模块在你的 PYTHONPATH 中或在相应目录下
from clipscore_main.clipscore import extract_all_images, extract_all_captions

def process_images_with_prompts(image_folder, json_path, batch_size=100, device='cuda'):
    device = device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化 CLIP 模型
    print("加载 CLIP 模型...")
    model, clip_preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("CLIP 模型加载完成！")
    
    # 从 JSON 文件中读取数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 构造批处理用的 prompt_data 列表（仅保留存在对应图片的样本）
    prompt_data = []
    for item in data:
        data_id = item.get('data_id')
        prompt = item.get('prompt')
        image_path = os.path.join(image_folder, f"{data_id}.png")
        if os.path.exists(image_path):
            prompt_data.append({
                "gen_image_path": image_path,
                "prompt": prompt,
                "data_id": data_id
            })
        else:
            print(f"图片不存在: {image_path}")
    
    # 批量计算 CLIPScore
    results = {}
    total = len(prompt_data)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(0, total, batch_size):
        batch_data = prompt_data[i:i+batch_size]
        batch_images = [item['gen_image_path'] for item in batch_data]
        batch_prompts = [item['prompt'] for item in batch_data]
        batch_ids = [item['data_id'] for item in batch_data]
        
        # 提取图像和文本特征
        image_features = extract_all_images(batch_images, model, device, num_workers=0)
        text_features = extract_all_captions(batch_prompts, model, device, num_workers=0)
        
        # 对特征进行归一化处理
        image_features = image_features / np.sqrt(np.sum(image_features**2, axis=1, keepdims=True))
        text_features = text_features / np.sqrt(np.sum(text_features**2, axis=1, keepdims=True))
        
        # 计算相似度得分，并进行剪裁和缩放
        similarities = np.sum(image_features * text_features, axis=1)
        batch_scores = 2.5 * np.clip(similarities, 0, None)
        
        # 将当前批次的结果以 data_id 为键保存到 results 字典中
        for data_id, score in zip(batch_ids, batch_scores):
            results[data_id] = score
        
        batch_num = (i // batch_size) + 1
        print(f"已处理 {batch_num}/{num_batches} 批，当前批次共 {len(batch_data)} 个样本")
    
    return results

# 示例调用：
if __name__ == "__main__":
    image_folder = "/path/to/your/images"  # 替换为实际的图片文件夹路径
    json_path = "/path/to/your/data.json"    # 替换为实际的 JSON 文件路径
    scores = process_images_with_prompts(image_folder, json_path, batch_size=100, device='cuda')
    print("所有样本的 CLIPScore 计算完成！")
    # 可将结果保存到文件：
    with open("clipscore_results.json", "w") as f:
        json.dump(scores, f, indent=2)
