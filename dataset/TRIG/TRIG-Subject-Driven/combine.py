import json
import os

total = []
# dir = "H:\ProjectsPro\TRIG\dataset\Trig\Trig-subject-driven\Subjects200K"
# for file in os.listdir(dir):
#     if file.endswith(".json"):
with open(os.path.join(r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\p2p_with_t.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)
    total = data
print(len(total))
for i in total:
    i['img_id'] = i.pop('image_path')
    i['img_id'] = i['img_id'].split("/")[-1]
    i["parent_dataset"] = [i["parent_dataset"], "Auto"]
    i["dimensions"] = [i['data_id'].split("_")[0], i['data_id'].split("_")[1]]

print(total[0])
with open("H:\ProjectsPro\TRIG\dataset\Trig\Trig-image-editing\p2p_t.json", 'w', encoding='utf-8') as f:
    json.dump(total, f, indent=4, ensure_ascii=False)