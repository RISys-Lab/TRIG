import json
import os

total = []
dir = "H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\OmniEdit-Filtered"
for file in os.listdir(dir):
    if file.endswith(".json"):
        with open(os.path.join(dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            total = total + data
for i in total:
    i['img_id'] = i.pop('image_path')
    i['img_id'] = i['img_id'].split("dev/src\\")[-1]
    i["parent_dataset"][1] = "Auto"
    i["dimensions"] = [i['data_id'].split("_")[0], i['data_id'].split("_")[1]]

print(len(total))
with open("H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\p2p_without_t.json", 'w', encoding='utf-8') as f:
    json.dump(total, f, indent=4, ensure_ascii=False)