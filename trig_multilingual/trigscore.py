# -*- coding: utf-8 -*-
"""
Minimal offline vLLM inference for TRIG visual scoring (no Ray, no server).
- Requires: vLLM >= 0.6 (建议 0.8.x), transformers, pillow, pandas
- Example: python trig_vllm_offline.py --model /path/to/Qwen2.5-VL-7B-Instruct-AWQ --image_dir /path/to/images
"""

import os
import json
import math
import argparse
from typing import Dict, Any, List

from PIL import Image
import pandas as pd

from vllm import LLM, SamplingParams


# --------------------------
# 评分函数（沿用你现在的逻辑）
# --------------------------
def logprobs_score(top_logprobs_dict: Dict[str, float], confidence: bool = False) -> float:
    weights = {
        "excellent": 1.0,
        "good": 0.75,
        "medium": 0.5,
        "bad": 0.25,
        "terrible": 0.0,
    }
    prefixes = {
        "excellent": ("excellent", "ex", "Ex", "Excellent"),
        "good": ("good", "Good"),
        "medium": ("medium", "med", "Medium"),
        "bad": ("bad", "Bad"),
        "terrible": ("terrible", "terr", "Terrible", "Terr"),
    }

    def norm_token(t: str) -> str:
        t = t.strip().lower()
        if t.startswith("▁") or t.startswith("Ġ"):
            t = t[1:]
        return t

    agg = {k: None for k in weights}

    for tok, lp in top_logprobs_dict.items():
        tk = norm_token(tok)
        for label, cands in prefixes.items():
            if any(tk == p or tk.startswith(p) for p in cands):
                if agg[label] is None:
                    agg[label] = float(lp)
                else:
                    a, b = agg[label], float(lp)
                    m = max(a, b)
                    agg[label] = m + math.log(math.exp(a - m) + math.exp(b - m))

    if all(v is None for v in agg.values()):
        return 0.0

    for k in agg:
        if agg[k] is None:
            agg[k] = float("-inf")

    m = max(agg.values())
    exps = {k: (0.0 if v == float("-inf") else math.exp(v - m)) for k, v in agg.items()}
    Z = sum(exps.values()) + 1e-10
    probs = {k: v / Z for k, v in exps.items()}

    score = sum(weights[k] * probs[k] for k in weights)
    if confidence:
        score *= max(probs.values())

    return round(score, 3)


# --------------------------
# 提示模板文本
# --------------------------
LANG_MAP = {
    'en': 'English', 'zh': 'Chinese', 'hi': 'Hindi', 'es': 'Spanish',
    'ar': 'Arabic', 'fr': 'French', 'pt': 'Portuguese', 'ru': 'Russian',
    'ja': 'Japanese', 'ko': 'Korean',
}
DIM_MSG = {
    'IQ-R': "Realism: Evaluate how realistic the image appears. Assess whether the visual elements, textures, lighting, and overall composition resemble real-world scenarios. Consider factors such as physical plausibility, natural variations, and absence of artificial distortions.",
    'IQ-O': "Originality: Evaluate the creativity and uniqueness of the image. Assess whether the composition, style, and content introduce novel elements or exhibit diversity in expression. Consider the distinctiveness of the concept and whether the image avoids clichés or repetitive patterns.",
    'IQ-A': "Aesthetics: Evaluate the overall aesthetic appeal of the image. Assess the harmony of colors, composition, balance, and visual impact. Consider factors such as artistic style, symmetry, contrast, and emotional resonance.",
    'TA-C': "Content Alignment: Evaluate how well the main objects and scenes in the image align with the elements specified in the prompt. Assess whether key subjects, their attributes, and the overall composition accurately reflect the given instructions.",
    'TA-R': "Relation Alignment: Evaluate how well the spatial or logical relationships between objects in the image align with the prompt. Assess whether elements are positioned, scaled, and arranged according to the specified instructions.",
    'TA-S': "Style Alignment: Evaluate how well the image's style aligns with the prompt. Assess whether the overall artistic and visual presentation matches the specified style in the prompt without deviation.",
    'D-K': "Knowledge: Evaluate the model’s ability to comprehend and accurately represent complex or specialized knowledge described in the prompt. Assess whether the image demonstrates a correct and nuanced understanding of intricate concepts, avoiding factual inaccuracies or oversimplifications.",
    'D-A': "Ambiguous: Evaluate the model’s ability to generate images that effectively capture the ambiguity, abstraction, or complexity described in the prompt. Assess whether the image maintains an intentional level of uncertainty or open interpretation without reducing the concept to a singular, overly simplistic form.",
    # 你之前过滤掉 R-T / R-B，就不跑它们
    'R-T': "Toxicity ...",
    'R-B': "Bias ..."
}

SYS_TMPL = """You are an evaluation assistant, I will give an AI generated image and a description (i.e. prompt) in {lang}, I need you to evaluate the performance of this generated image on a specific dimension based on this original description and evaluation criteria.
I will give you the definition of this dimension and the criteria for evaluation. You just need to evaluate the performance of this image on this dimension.
The information and evaluation criteria about the dimension is as follows:
{dim_msg}
1. You need to use this specific language prompt to assist you in your evaluation of the generated image.
2. You should evaluate the image in this dimension by a scale from: excellent, good, medium, bad, terrible.
Your grading scale should be uniform; Excellent for accuracy, Good for very good performance, Medium for acceptable, Bad for some errors, and Terrible for more errors.
3. You must give me one of these words as your evaluation, your answer should only be one word.
4. You need to directly assess how well the image aligns with this specific language prompt in this dimension, and understand the prompt directly without translating it into English for comprehension.
"""

USER_SUFFIX = "\nPlease give your evaluation of the generated image on this dimension with one of these words: excellent, good, medium, bad, terrible."


def build_messages(data_id: str, prompt_text: str):
    dim, lang = data_id.split("_")[0], data_id.split("_")[1]
    sys_msg = SYS_TMPL.format(lang=LANG_MAP.get(lang, "English"),
                              dim_msg=DIM_MSG[dim])
    # vLLM 用 tokenizer.apply_chat_template 识别图片消息
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": [
            {"type": "text", "text": "Prompt for generating this image: " + prompt_text + USER_SUFFIX},
            {"type": "image"}  # 占位；真正的 PIL.Image 通过 generate(images=[...]) 传入
        ]}
    ]
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  
                       default="/leonardo_scratch/fast/EUHPC_R04_192/fmohamma/fast_weights/Qwen2.5-VL-72B-Instruct-AWQ")
    parser.add_argument("--image_dir", type=str,  
                        default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/data/output/t2i_ml/flux")
    parser.add_argument("--ann_json", type=str,  
                        default="/leonardo_work/EUHPC_R04_192/fmohamma/TRIG/dataset/TRIG-multilingual/text-to-image-multilingual.json")
    parser.add_argument("--out_csv", type=str, default="trig_offline_results.csv")
    parser.add_argument("--max_tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)  # 打分建议 0.0 更稳
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=30, help="只跑前 N 张（0 表示全跑）")
    parser.add_argument("--max_model_len", type=int, default=2048)
    args = parser.parse_args()

    # ---- vLLM 离线实例 ----
    # 关键点：
    # - speculative_model 不用（避免 logprobs 在 spec 期间被跳过）
    # - disable_logprobs_during_spec_decoding=False（保守起见）
    # - max_model_len 充足
    # - 多卡的话 vLLM 会自动探测；你也可以传 tensor_parallel_size
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype="auto",
        gpu_memory_utilization=0.85,
    )

    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # vLLM 里要回 logprobs 用整数（前 K 个）
        logprobs=max(args.top_k, args.top_logprobs),
        # 如需首段 logprobs，也可以：prompt_logprobs=True
        stop=["\n"],
    )

    # ---- 读注释，做成字典 ----
    with open(args.ann_json, "r", encoding="utf-8") as f:
        ann_list = json.load(f)
    ann = {x["data_id"]: x for x in ann_list}

    # ---- 列出图片 ----
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    if args.limit > 0:
        files = files[:args.limit]

    rows: List[Dict[str, Any]] = []

    for fname in files:
        data_id = os.path.splitext(fname)[0]
        if data_id not in ann:
            continue
        # 过滤掉 R-T / R-B
        if data_id.split("_")[0] in ["R-T", "R-B"]:
            continue

        prompt_text = ann[data_id]["prompt"]
        image_path = os.path.join(args.image_dir, fname)

        # 构造 messages，并用 chat template 转换成字符串 prompt
        messages = build_messages(data_id, prompt_text)
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 打开图片（PIL）
        img = Image.open(image_path).convert("RGB")

        # 生成（离线）
        # 注意：多模态要把图片通过 images 参数传进去；vLLM 会把 messages 里的 {"type":"image"} 占位与之对齐
        outputs = llm.generate(
            prompts=[prompt_str],
            sampling_params=sampling,
            images=[[img]],   # 每个 prompt 对应一个 list[Image]
            use_tqdm=False
        )

        # 解析结果
        out = outputs[0].outputs[0]   # 单条
        generated_text = out.text or ""
        # out.logprobs 是一个 TokenLogprobs 列表；取第一个 token 的 top_logprobs 映射成 {token: logprob}
        top0_dict = {}
        if out.logprobs and len(out.logprobs) > 0:
            first_tok = out.logprobs[0]
            # vLLM 里 first_tok.top_logprobs 是 list[TopLogprob]，包含 token 和 logprob
            if getattr(first_tok, "top_logprobs", None):
                for cand in first_tok.top_logprobs:
                    top0_dict[cand.token] = float(cand.logprob)

        score = logprobs_score(top0_dict) if top0_dict else 0.0

        rows.append({
            "data_id": data_id,
            "image_path": image_path,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "score": score,
            "top_logprobs_str": json.dumps(top0_dict, ensure_ascii=False),
        })

        print(f"[OK] {data_id}: text='{generated_text}'  score={score}")

    # ---- 存结果（CSV 更省事；parquet 想存的话 top_logprobs 用字符串字段即可）----
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
