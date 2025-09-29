# pip install transformers==4.43.3  # 或更高版本
import json, re
from collections import defaultdict
from statistics import mean
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer

# ============ 可选：替换为你的数据加载 ============
with open("/home/muzammal/Projects/TRIG/dataset/TRIG-multilingual/trig_multilingual_tr.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ============ 辅助函数 ============
_word_re_en = re.compile(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+")  # 英文单词粗略提取

def is_english_record(rec: Dict[str, Any]) -> bool:
    """判断是否属于 xx_en_xx"""
    lang = (rec.get("lang") or "").strip().lower()
    if lang == "en":
        return True
    dim = rec.get("dimension") or []
    if any((str(x).lower() == "en") for x in dim):
        return True
    data_id = (rec.get("data_id") or "").lower()
    if "_en_" in data_id:
        return True
    return False

def extract_cls(rec: Dict[str, Any]) -> str:
    """提取 cls：取 data_id 下划线分割后的第二部分"""
    data_id = str(rec.get("data_id") or "")
    parts = data_id.split("_")
    if len(parts) >= 2:
        return parts[1]  # 第二段
    return "UNKNOWN"

def get_prompt(rec: Dict[str, Any]) -> str:
    return str(rec.get("dimension_prompt")[0] or "").strip()

def get_main_prompt(rec: Dict[str, Any]) -> str:
    """获取主要的 prompt 字段"""
    return str(rec.get("prompt") or "").strip()

def english_word_stats(prompts: List[str]) -> Tuple[float, float]:
    """
    返回:
      - avg_words_per_prompt: 平均每个 prompt 的英文词数
      - avg_chars_per_word: 合并后每个英文词的平均字符数
    """
    word_counts = []
    all_words = []
    for p in prompts:
        words = _word_re_en.findall(p)
        word_counts.append(len(words))
        all_words.extend(words)
    avg_words_per_prompt = mean(word_counts) if word_counts else 0.0
    if all_words:
        avg_chars_per_word = mean(len(w) for w in all_words)
    else:
        avg_chars_per_word = 0.0
    return avg_words_per_prompt, avg_chars_per_word

# ============ 主流程 ============
from transformers import AutoTokenizer
from collections import defaultdict
from statistics import mean

def main(records):
    # 只用 T5 tokenizer（t5-base / t5-xxl 都是同一个）
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    cls_token_lengths = defaultdict(list)
    all_token_lengths = []
    
    # 收集所有主要 prompt 用于英文词汇统计
    all_main_prompts = []

    for rec in records:
        prompt = get_prompt(rec)
        if not prompt:
            continue
        enc = tokenizer(
            prompt,
            truncation=False,   # 不截断
            return_attention_mask=True,
            add_special_tokens=True,
            padding=False
        )
        ids = enc["input_ids"]
        tok_len = len(ids)

        all_token_lengths.append(tok_len)
        cls_name = extract_cls(rec)
        cls_token_lengths[cls_name].append(tok_len)
        
        # 收集主要 prompt
        main_prompt = get_main_prompt(rec)
        if main_prompt:
            all_main_prompts.append(main_prompt)

    print("=== 按 cls 的平均 token 数 (T5 tokenizer, 不截断) ===")
    for cls_name, lens in sorted(cls_token_lengths.items(), key=lambda x: x[0]):
        avg_len = mean(lens) if lens else 0.0
        print(f"{cls_name}: n={len(lens)}, avg_tokens={avg_len:.4f}")

    overall_avg = mean(all_token_lengths) if all_token_lengths else 0.0
    print()
    print(f"=== 全部样本的平均 token 数 ===\noverall_avg_tokens={overall_avg:.4f}")
    
    # 对所有主要 prompt 进行英文词汇统计
    if all_main_prompts:
        avg_words, avg_chars = english_word_stats(all_main_prompts)
        print()
        print("=== 所有 prompt 字段的英文词汇统计 ===")
        print(f"总样本数: {len(all_main_prompts)}")
        print(f"平均每个 prompt 的英文词数: {avg_words:.4f}")
        print(f"每个英文词的平均字符数: {avg_chars:.4f}")
    else:
        print()
        print("=== 所有 prompt 字段的英文词汇统计 ===")
        print("没有找到有效的 prompt 数据")


if __name__ == "__main__":
    main(data)
