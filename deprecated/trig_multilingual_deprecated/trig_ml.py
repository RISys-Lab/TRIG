import os
import csv
import json
import base64
from pathlib import Path

# ===== 引入上面两个函数 =====
import math
from openai import OpenAI

language_map = {'en': 'English','zh': 'Chinese','hi': 'Hindi','es': 'Spanish','ar': 'Arabic','fr': 'French','pt': 'Portuguese','ru': 'Russian','ja': 'Japanese','ko': 'Korean'}

def send_request(prompt, image_b64, model="gpt-4o-mini", endpoint="http://localhost:8000/v1/", api_key="EMPTY"):
    client = OpenAI(api_key=api_key, base_url=endpoint)
    messages = [
        {"role": "system", "content": "You are an evaluation assistant. Respond with one word: excellent, good, medium, bad, or terrible."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Please evaluate: {prompt}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        logprobs=True,
        max_tokens=1,
        temperature=0.0,
        top_logprobs=10,
    )
    return completion.choices[0].logprobs.content[0].top_logprobs


def logprobs_score(top_logprobs):
    classes = {
        "excellent": ("excellent", "ex"),
        "good": ("good",),
        "medium": ("medium", "med"),
        "bad": ("bad",),
        "terrible": ("terrible", "terr"),
    }
    weights = {"excellent": 1.0, "good": 0.75, "medium": 0.5, "bad": 0.25, "terrible": 0.0}

    pairs = []
    for item in (top_logprobs or []):
        tok = getattr(item, "token", None) or (isinstance(item, dict) and item.get("token"))
        lp = getattr(item, "logprob", None) or (isinstance(item, dict) and item.get("logprob"))
        if tok and lp is not None:
            pairs.append((tok.strip().lower(), float(lp)))
    if not pairs:
        return 0.0

    agg = {k: None for k in classes}
    for tok, lp in pairs:
        for cls, prefixes in classes.items():
            if any(tok.startswith(p) for p in prefixes):
                agg[cls] = lp if agg[cls] is None else max(agg[cls], lp) + math.log1p(math.exp(-abs(agg[cls] - lp)))

    vals = {k: (float("-inf") if v is None else v) for k, v in agg.items()}
    m = max(vals.values())
    probs = {k: math.exp(v - m) if v > float("-inf") else 0.0 for k, v in vals.items()}
    Z = sum(probs.values()) + 1e-12
    probs = {k: v / Z for k, v in probs.items()}

    return round(sum(weights[k] * probs[k] for k in weights), 3)


# ===== 主函数 =====
def main(json_file, image_dir, output_csv="results.csv", model="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen2.5-VL-72B-Instruct-AWQ", endpoint="http://localhost:8000/v1/"):
    results = []
    
    # 读取JSON文件
    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    
    image_dir = Path(image_dir)
    
    for data in data_list:
        data_id = data["data_id"]
        prompt = data["prompt"]
        language_id = data_id.split("_")[1]
        # text_render = data["dimension_prompt"][0]
        
        # 寻找对应的图片文件 (jpg或png)
        img_path = None
        for ext in [".jpg", ".png"]:
            candidate = image_dir / f"{data_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"❌ Image not found for data_id: {data_id}")
            results.append({"data_id": data_id, "score": 0.0})
            continue
        
        # 读取图片并编码
        with open(img_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        try:
            logprobs = send_request(prompt, image_b64, model=model, endpoint=endpoint)
            score = logprobs_score(logprobs)
        except Exception as e:
            print(f"❌ Error with {data_id}: {e}")
            score = 0.0

        results.append({"data_id": data_id, "score": score})
        print(f"✅ {data_id}: {score}")

    # 保存结果
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data_id", "score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved to {output_csv}")


if __name__ == "__main__":
    # 示例：读取JSON文件并对图片打分
    main(json_file="./data.json", image_dir="./images", output_csv="scores.csv")
