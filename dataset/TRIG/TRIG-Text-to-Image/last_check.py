import json
import random
with open(r'H:\ProjectsPro\TRIG\dataset\Trig\Trig-text-to-image\text-to-imgae-without-m-e.json', 'r') as f:
    data = json.load(f)

cnt = 0
for i in data:
    if i['parent_dataset'][0].startswith("<") and i['parent_dataset'][0].endswith(">") and i['parent_dataset'][1] == "Origin":
        continue
    else:
        cnt+=1

print(cnt)


# for i in data:
#     if i['dimension'] == ["IQ-R","TA-R"]:
#         id = i["data_id"].split("_")[-1]
#         if int(id) > 265:
#             replace_data = random.choice(docci)
#             i['prompt'] = replace_data['description']
#             i['dimension_prompt'] = ["","Complex spatial relation"]
#             i['img_id'] = replace_data['image_file']
#             i['parent_dataset'] = ["DOCCI","Origin"]
#
# with open(r'H:\ProjectsPro\TRIG\dataset\Trig\Trig-text-to-image\text-to-imgae-new1.json', 'w') as f:
#     json.dump(data, f, indent=4)

