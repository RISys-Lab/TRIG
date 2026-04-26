import json
import os
import time
import traceback
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types

# 语言映射表
language_map = {
    'en': 'English','zh': 'Chinese','hi': 'Hindi','es': 'Spanish','ar': 'Arabic',
    'fr': 'French','pt': 'Portuguese','ru': 'Russian','ja': 'Japanese','ko': 'Korean'
}

# 加载数据
with open("/home/muzammal/Projects/TRIG/dataset/TRIG-multilingual/trig_multilingual_tr.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

system_prompt = """
You are a strict evaluator. You will be given an image and the rendered text inside it. The rendered text is in {language_name}.
No background description will be provided — you should only focus on the rendered text and its integration with the already generated background.

Evaluate the image-text pair based on the following three criteria:

1. Text Aesthetics: Evaluate only the visual presentation of the text, including font style, spacing, alignment, and readability.  
2. Text Quality: Assess whether the generated text in the image is clear, coherent, and consistent with the intended text.  
3. Text-Image Fusing: Assess how well the rendered text is integrated into the image — whether the text is appropriately placed and harmonizes with the scene context. You must carefully consider whether this text appears jarring in this context, rather than as if printed on plain white paper.

For each criterion, provide:  
- A score from 1 to 10, where 1 is very poor and 10 is excellent.  
- A very short and concise brief explanation in english justifying the score.  

Return your evaluation strictly in the following JSON format:  
{{
  "Text Aesthetics": {{"score": <score>, "comment": "<explanation>"}},
  "Text Quality": {{"score": <score>, "comment": "<explanation>"}},
  "Text-Image Fusing": {{"score": <score>, "comment": "<explanation>"}}
}}
"""

# 创建 Gemini 客户端
client = genai.Client()

def process_one(item, model_name, retries=3, delay=3):
    """处理单个样本"""
    data_id = item["data_id"]
    lang_code = data_id.split("_")[1]
    lang_name = language_map.get(lang_code, lang_code)
    render_text = item["dimension_prompt"][0]

    img_folder = os.path.join("/home/muzammal/Projects/TRIG/data/output/tr_ml", model_name)
    img_path = os.path.join(img_folder, f"{data_id}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(img_folder, f"{data_id}.png")

    if not os.path.exists(img_path):
        return {"data_id": data_id, "error": "image not found"}

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"data_id": data_id, "error": f"image open error: {e}"}

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.format(language_name=lang_name),
                    response_mime_type="application/json"
                ),
                contents=[
                    f"Language: {lang_name}\nRender Text: {render_text}",
                    img
                ]
            )
            raw = response.text
            scores = json.loads(raw)
            # print(scores)
            return {"data_id": data_id, "scores": scores, "lang_code": lang_code}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            return {"data_id": data_id, "error": f"{type(e).__name__}: {e}"}

def evaluate_model(model_name, base_output_dir, save_every=50):
    print(f"\n{'='*60}")
    print(f"正在串行评估模型 (Gemini): {model_name}")
    print(f"{'='*60}")

    output_file = os.path.join(base_output_dir, f"evaluation_results_{model_name}.json")
    os.makedirs(base_output_dir, exist_ok=True)

    # ====== 断点续跑 ======
    finished_ids = set()
    all_results = []
    if os.path.exists(output_file):
        print(f"🔄 检测到已有结果文件，将跳过已完成的数据: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
            all_results = prev_data.get("all_results", [])
            finished_ids = {r["data_id"] for r in all_results}
    # ======================

    results = []
    pending_items = [it for it in data_list if it["data_id"] not in finished_ids]
    pbar = tqdm(pending_items, desc="Processing", unit="sample")

    for idx, item in enumerate(pbar, 1):
        res = process_one(item, model_name)
        results.append(res)

        # ✅ 定期保存
        if idx % save_every == 0 or idx == len(pending_items):
            tmp_results = all_results + results
            lang_scores = defaultdict(lambda: {"Text Aesthetics": [], "Text Quality": [], "Text-Image Fusing": []})
            for r in tmp_results:
                if "scores" in r:
                    for key in ["Text Aesthetics", "Text Quality", "Text-Image Fusing"]:
                        lang_scores[r["lang_code"]][key].append(r["scores"][key]["score"])
            lang_summary = {
                lc: {k: sum(v)/len(v) if v else 0 for k, v in vals.items()}
                for lc, vals in lang_scores.items()
            }
            partial_output = {"model_name": model_name, "lang_summary": lang_summary, "all_results": tmp_results}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(partial_output, f, indent=2, ensure_ascii=False)
            print(f"💾 已保存进度: {len(tmp_results)} 条 -> {output_file}")

    pbar.close()

    print(f"✅ 模型 {model_name} 评估完成，结果保存到: {output_file}")
    return output_file

def main():
    base_output_dir = "/home/muzammal/Projects/TRIG/trig_multilingual"
    models = ["EasyText", "qwen_image", "AnyText2" ,"AnyText"]

    print("开始串行评估多个模型 (Gemini)...")

    for model_name in models:
        try:
            evaluate_model(model_name, base_output_dir, save_every=50)
        except Exception as e:
            print(f"❌ 评估模型 {model_name} 出错: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("🎉 所有模型评估完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
