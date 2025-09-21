import json

with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-without-m-e-v8.json", "r") as f:
    data = json.load(f)

new_data = []

for i in data:
    if i['dimension'] == ["D-K", "D-A"] or i['dimension'] == ["TA-R", "TA-S"] or i['dimension'] == ["IQ-R", "TA-C"]:
        new_data.append(i)

with open("/home/muzammal/Projects/TRIG/dataset/Trig/Trig-text-to-image/text-to-image-test.json", "w") as f:
    json.dump(new_data, f, indent=4)