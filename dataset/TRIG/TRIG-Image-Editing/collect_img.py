import json
import os
import shutil

# 指定相关路径
json_path = r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\p2p_t.json"
source_folder = r"H:\ProjectsPro\TRIG\dataset\raw_dataset\character"
destination_folder = r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\toxicity_images"  # 请替换成你想保存的目标文件夹

# 如果目标文件夹不存在，创建该文件夹
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 读取 JSON 文件
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))
# 遍历 JSON 数据，根据 img_id 构造源文件路径并复制到目标文件夹
# for item in data:
#     img_id = item["img_id"]
#     src_path = os.path.join(source_folder, img_id)
#     dst_path = os.path.join(destination_folder, img_id)
#
#     # 如果源文件存在，再进行复制
#     if os.path.exists(src_path):
#         shutil.copy(src_path, dst_path)
#         print(f"已复制: {src_path} 到 {dst_path}")
#     else:
#         print(f"源文件不存在: {src_path}")
