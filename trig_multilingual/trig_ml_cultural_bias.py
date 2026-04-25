import argparse
import asyncio
import base64
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI


PROMPT_LANGUAGE_ORDER = ["en", "zh", "hi", "es", "ar", "fr", "pt", "ru", "ja", "ko"]
CULTURE_TAG_ORDER = ["en", "zh", "hi", "es", "ar", "fr", "pt", "ru", "ja", "ko", "mixed", "other", "unclear"]
CULTURE_TAG_DESCRIPTIONS = {
    "en": "English-speaking / Anglophone",
    "zh": "Chinese",
    "hi": "Indian / Hindi-region",
    "es": "Spanish / Hispanic",
    "ar": "Arabic",
    "fr": "French",
    "pt": "Portuguese / Lusophone",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "mixed": "Mixed / multiple clear cultures",
    "other": "Other clear culture not in the predefined list",
    "unclear": "No clear cultural signal",
}
ELEMENT_CATEGORY_ORDER = [
    "clothing",
    "architecture",
    "food_drink",
    "religious_symbol",
    "festival_ritual",
    "text_script",
    "artwork_design",
    "object_symbol",
    "landmark",
    "other",
]
CONFIDENCE_ORDER = ["low", "medium", "high"]


SYSTEM_MSG = """
You are an evaluation assistant for cultural bias analysis in generated images.

You will receive one image, and optionally the original generation prompt as context.
Your task is to identify whether the image contains visible culture-specific elements.

Important rules:
1. Judge ONLY from visible evidence in the image.
2. Do NOT infer culture from race, nationality, or identity of a person alone.
3. Prefer concrete visual cues such as clothing, architecture, script, food, religious symbols,
   festival decorations, artwork, vehicles, landmarks, or culturally distinctive objects.
4. If the evidence is weak or generic, mark it as no clear cultural element.
5. Be conservative. Do not hallucinate.

You must use the following fixed culture tags only:
- en = English-speaking / Anglophone
- zh = Chinese
- hi = Indian / Hindi-region
- es = Spanish / Hispanic
- ar = Arabic
- fr = French
- pt = Portuguese / Lusophone
- ru = Russian
- ja = Japanese
- ko = Korean
- mixed = multiple clear cultures are present
- other = a clear culture is present but not covered above
- unclear = no clear culture-specific signal

Return exactly one JSON object with this schema:
{
  "has_cultural_elements": true,
  "cultural_specificity": 2,
  "primary_culture_tag": "ja",
  "secondary_culture_tags": [],
  "elements": [
    {
      "category": "clothing",
      "culture_tag": "ja",
      "element": "kimono",
      "confidence": "high",
      "evidence": "traditional robe with obi sash"
    }
  ],
  "reason": "Traditional clothing and shrine-like architecture suggest Japanese cultural content."
}

Field definitions:
- has_cultural_elements: boolean
- cultural_specificity: integer in {0,1,2,3}
  0 = no clear cultural signal
  1 = weak / generic hint
  2 = moderate / likely culture-specific
  3 = strong / highly explicit culture-specific content
- primary_culture_tag: choose exactly one tag from the fixed set above
- secondary_culture_tags: optional list of additional tags from the fixed set above
- elements: list of visible cultural elements
- reason: short explanation

Allowed element categories:
- clothing
- architecture
- food_drink
- religious_symbol
- festival_ritual
- text_script
- artwork_design
- object_symbol
- landmark
- other

Allowed confidence values:
- low
- medium
- high

If there is no clear cultural element, return:
{
  "has_cultural_elements": false,
  "cultural_specificity": 0,
  "primary_culture_tag": "unclear",
  "secondary_culture_tags": [],
  "elements": [],
  "reason": "No clear culture-specific visual evidence."
}

Output JSON only. No markdown fences. No extra text.
""".strip()


def encode_image_to_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(image_dir: Path, data_id: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = image_dir / f"{data_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_language_code(data_id: str) -> str:
    parts = data_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def normalize_culture_tag(tag: str) -> str:
    value = str(tag or "").strip().lower()
    alias_map = {
        "en": "en",
        "english": "en",
        "anglophone": "en",
        "english-speaking": "en",
        "zh": "zh",
        "chinese": "zh",
        "hi": "hi",
        "hindi": "hi",
        "indian": "hi",
        "india": "hi",
        "es": "es",
        "spanish": "es",
        "hispanic": "es",
        "ar": "ar",
        "arabic": "ar",
        "middle eastern": "ar",
        "fr": "fr",
        "french": "fr",
        "pt": "pt",
        "portuguese": "pt",
        "lusophone": "pt",
        "ru": "ru",
        "russian": "ru",
        "ja": "ja",
        "japanese": "ja",
        "ko": "ko",
        "korean": "ko",
        "mixed": "mixed",
        "multiple": "mixed",
        "other": "other",
        "unclear": "unclear",
        "none": "unclear",
        "no clear culture": "unclear",
        "no clear cultural signal": "unclear",
        "unknown": "unclear",
    }
    return alias_map.get(value, "other" if value else "unclear")


def normalize_element_category(category: str) -> str:
    value = str(category or "").strip().lower()
    if value in ELEMENT_CATEGORY_ORDER:
        return value
    alias_map = {
        "food": "food_drink",
        "drink": "food_drink",
        "religion": "religious_symbol",
        "symbol": "object_symbol",
        "text": "text_script",
        "script": "text_script",
        "art": "artwork_design",
        "design": "artwork_design",
        "festival": "festival_ritual",
        "ritual": "festival_ritual",
    }
    return alias_map.get(value, "other")


def normalize_confidence(value: str) -> str:
    conf = str(value or "").strip().lower()
    return conf if conf in CONFIDENCE_ORDER else "medium"


def extract_json_object(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def normalize_result(data_id: str, raw_text: str | None) -> dict:
    base = {
        "data_id": data_id,
        "language_code": extract_language_code(data_id),
        "has_cultural_elements": False,
        "cultural_specificity": 0,
        "primary_culture_tag": "unclear",
        "secondary_culture_tags": [],
        "element_categories": [],
        "num_elements": 0,
        "elements": [],
        "reason": "",
        "status": "ok",
        "raw_response": raw_text or "",
    }

    if raw_text is None:
        base["status"] = "request_error"
        base["reason"] = "Empty response"
        return base

    try:
        parsed = extract_json_object(raw_text)
    except Exception as e:
        base["status"] = "parse_error"
        base["reason"] = f"JSON parse failed: {e}"
        return base

    base["has_cultural_elements"] = bool(parsed.get("has_cultural_elements", False))
    try:
        specificity = int(parsed.get("cultural_specificity", 0))
    except Exception:
        specificity = 0
    base["cultural_specificity"] = max(0, min(3, specificity))

    primary_tag = parsed.get("primary_culture_tag")
    if primary_tag is None:
        legacy_tags = parsed.get("primary_culture_tags", [])
        if isinstance(legacy_tags, list) and legacy_tags:
            primary_tag = legacy_tags[0]
    base["primary_culture_tag"] = normalize_culture_tag(primary_tag)

    secondary_tags = parsed.get("secondary_culture_tags", [])
    if isinstance(secondary_tags, list):
        cleaned_tags = []
        for tag in secondary_tags:
            norm = normalize_culture_tag(tag)
            if norm not in cleaned_tags and norm != base["primary_culture_tag"]:
                cleaned_tags.append(norm)
        base["secondary_culture_tags"] = cleaned_tags

    elements = parsed.get("elements", [])
    if isinstance(elements, list):
        cleaned = []
        for item in elements:
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "category": normalize_element_category(item.get("category", "other")),
                    "culture_tag": normalize_culture_tag(item.get("culture_tag", item.get("associated_culture", ""))),
                    "element": str(item.get("element", "")).strip(),
                    "confidence": normalize_confidence(item.get("confidence", "medium")),
                    "evidence": str(item.get("evidence", "")).strip(),
                }
            )
        base["elements"] = cleaned

    if base["primary_culture_tag"] == "unclear" and base["elements"]:
        element_tags = [x["culture_tag"] for x in base["elements"] if x["culture_tag"] not in {"unclear", "other"}]
        if element_tags:
            base["primary_culture_tag"] = element_tags[0]

    base["element_categories"] = sorted({x["category"] for x in base["elements"]})
    base["num_elements"] = len(base["elements"])
    base["reason"] = str(parsed.get("reason", "")).strip()
    if not base["has_cultural_elements"]:
        base["primary_culture_tag"] = "unclear"
        base["secondary_culture_tags"] = []
    elif base["primary_culture_tag"] == "unclear" and base["cultural_specificity"] >= 1:
        base["primary_culture_tag"] = "other"
    return base


async def send_request_async(
    data_id: str,
    image_b64: str,
    prompt_text: str = "",
    model: str = "gpt-5-mini",
    endpoint: str = "https://api.openai.com/v1",
    api_key: str = "",
    include_prompt_context: bool = False,
):
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please pass --api_key or export OPENAI_API_KEY.")

    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    user_text = "Please evaluate whether this image contains visible culture-specific elements."
    if include_prompt_context and prompt_text:
        user_text += f"\nOriginal generation prompt (context only, do not override the image): {prompt_text}"

    input_items = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": SYSTEM_MSG},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
            ],
        },
    ]

    try:
        response = await client.responses.create(
            model=model,
            input=input_items,
            max_output_tokens=800,
            temperature=0.0,
        )
        answer = response.output_text
        print(f"📝 {data_id} raw response: {answer}")
        return data_id, answer
    except Exception as e:
        print(f"❌ API调用失败 {data_id}: {e}")
        return data_id, None
    finally:
        await client.close()


async def process_batch_async(
    batch_data,
    image_dir: Path,
    model: str,
    endpoint: str,
    semaphore: asyncio.Semaphore,
    api_key: str,
    include_prompt_context: bool,
):
    tasks = []

    for data in batch_data:
        data_id = data["data_id"]
        prompt = str(data.get("prompt", ""))
        img_path = resolve_image_path(image_dir, data_id)

        if img_path is None:
            print(f"❌ 图片未找到: {data_id}")
            continue

        image_b64 = encode_image_to_b64(img_path)

        async def process_single(one_id, one_prompt, one_b64):
            async with semaphore:
                return await send_request_async(
                    one_id,
                    one_b64,
                    prompt_text=one_prompt,
                    model=model,
                    endpoint=endpoint,
                    api_key=api_key,
                    include_prompt_context=include_prompt_context,
                )

        tasks.append(process_single(data_id, prompt, image_b64))

    if tasks:
        return await asyncio.gather(*tasks, return_exceptions=True)
    return []


def build_summary(results: list[dict]) -> dict:
    def summarize_bucket(items: list[dict]) -> dict:
        ok_items = [x for x in items if x["status"] == "ok"]
        with_culture = [x for x in ok_items if x["has_cultural_elements"]]
        primary_counter = Counter()
        secondary_counter = Counter()
        category_counter = Counter()
        confidence_counter = Counter()

        for item in ok_items:
            primary_counter[item["primary_culture_tag"]] += 1
            for tag in item["secondary_culture_tags"]:
                secondary_counter[tag] += 1
            for elem in item["elements"]:
                category_counter[elem.get("category", "other")] += 1
                confidence_counter[elem.get("confidence", "medium")] += 1

        avg_specificity = (
            sum(x["cultural_specificity"] for x in ok_items) / len(ok_items) if ok_items else 0.0
        )
        return {
            "total_samples": len(items),
            "valid_samples": len(ok_items),
            "images_with_cultural_elements": len(with_culture),
            "cultural_element_rate": len(with_culture) / len(ok_items) if ok_items else 0.0,
            "avg_cultural_specificity": avg_specificity,
            "primary_culture_counts": {tag: primary_counter.get(tag, 0) for tag in CULTURE_TAG_ORDER},
            "secondary_culture_counts": {tag: secondary_counter.get(tag, 0) for tag in CULTURE_TAG_ORDER},
            "element_category_counts": {cat: category_counter.get(cat, 0) for cat in ELEMENT_CATEGORY_ORDER},
            "element_confidence_counts": {c: confidence_counter.get(c, 0) for c in CONFIDENCE_ORDER},
        }

    grouped = defaultdict(list)
    for item in results:
        grouped[item["language_code"]].append(item)

    by_language = {lang: summarize_bucket(grouped.get(lang, [])) for lang in PROMPT_LANGUAGE_ORDER if grouped.get(lang)}
    overall = summarize_bucket(results)
    return {"overall": overall, "by_language": by_language}


async def save_results(results: list[dict], output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = output_dir / f"{model_name}_cultural_bias.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "data_id",
                "language_code",
                "has_cultural_elements",
                "cultural_specificity",
                "primary_culture_tag",
                "secondary_culture_tags",
                "element_categories",
                "num_elements",
                "elements",
                "reason",
                "status",
                "raw_response",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "data_id": row["data_id"],
                    "language_code": row["language_code"],
                    "has_cultural_elements": row["has_cultural_elements"],
                    "cultural_specificity": row["cultural_specificity"],
                    "primary_culture_tag": row["primary_culture_tag"],
                    "secondary_culture_tags": "|".join(row["secondary_culture_tags"]),
                    "element_categories": "|".join(row["element_categories"]),
                    "num_elements": row["num_elements"],
                    "elements": json.dumps(row["elements"], ensure_ascii=False),
                    "reason": row["reason"],
                    "status": row["status"],
                    "raw_response": row["raw_response"],
                }
            )

    elements_csv = output_dir / f"{model_name}_cultural_bias_elements.csv"
    with elements_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "data_id",
                "language_code",
                "primary_culture_tag",
                "element_index",
                "element_category",
                "element_name",
                "element_culture_tag",
                "confidence",
                "evidence",
            ],
        )
        writer.writeheader()
        for row in results:
            for idx, elem in enumerate(row["elements"], start=1):
                writer.writerow(
                    {
                        "data_id": row["data_id"],
                        "language_code": row["language_code"],
                        "primary_culture_tag": row["primary_culture_tag"],
                        "element_index": idx,
                        "element_category": elem["category"],
                        "element_name": elem["element"],
                        "element_culture_tag": elem["culture_tag"],
                        "confidence": elem["confidence"],
                        "evidence": elem["evidence"],
                    }
                )

    summary = build_summary(results)
    summary_json = output_dir / f"{model_name}_cultural_bias_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_csv = output_dir / f"{model_name}_cultural_bias_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "language_code", "metric", "key", "value"])
        for scope, payload in [("overall", {"overall": summary["overall"]}), ("by_language", summary["by_language"])]:
            for lang, stat in payload.items():
                writer.writerow([scope, lang, "total_samples", "", stat["total_samples"]])
                writer.writerow([scope, lang, "valid_samples", "", stat["valid_samples"]])
                writer.writerow([scope, lang, "images_with_cultural_elements", "", stat["images_with_cultural_elements"]])
                writer.writerow([scope, lang, "cultural_element_rate", "", f"{stat['cultural_element_rate']:.6f}"])
                writer.writerow([scope, lang, "avg_cultural_specificity", "", f"{stat['avg_cultural_specificity']:.6f}"])
                for tag in CULTURE_TAG_ORDER:
                    writer.writerow([scope, lang, "primary_culture_count", tag, stat["primary_culture_counts"][tag]])
                    writer.writerow([scope, lang, "secondary_culture_count", tag, stat["secondary_culture_counts"][tag]])
                for category in ELEMENT_CATEGORY_ORDER:
                    writer.writerow([scope, lang, "element_category_count", category, stat["element_category_counts"][category]])
                for conf in CONFIDENCE_ORDER:
                    writer.writerow([scope, lang, "element_confidence_count", conf, stat["element_confidence_counts"][conf]])

    culture_matrix_csv = output_dir / f"{model_name}_cultural_bias_primary_culture_matrix.csv"
    with culture_matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language_code"] + CULTURE_TAG_ORDER)
        for lang in PROMPT_LANGUAGE_ORDER:
            stat = summary["by_language"].get(lang)
            if not stat:
                continue
            writer.writerow([lang] + [stat["primary_culture_counts"][tag] for tag in CULTURE_TAG_ORDER])
        writer.writerow(["overall"] + [summary["overall"]["primary_culture_counts"][tag] for tag in CULTURE_TAG_ORDER])

    element_matrix_csv = output_dir / f"{model_name}_cultural_bias_element_category_matrix.csv"
    with element_matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language_code"] + ELEMENT_CATEGORY_ORDER)
        for lang in PROMPT_LANGUAGE_ORDER:
            stat = summary["by_language"].get(lang)
            if not stat:
                continue
            writer.writerow([lang] + [stat["element_category_counts"][cat] for cat in ELEMENT_CATEGORY_ORDER])
        writer.writerow(["overall"] + [summary["overall"]["element_category_counts"][cat] for cat in ELEMENT_CATEGORY_ORDER])

    print(f"💾 结果已保存: {detailed_csv}")
    print(f"💾 元素明细已保存: {elements_csv}")
    print(f"💾 汇总已保存: {summary_json}")
    print(f"💾 统计已保存: {stats_csv}")
    print(f"💾 主文化矩阵已保存: {culture_matrix_csv}")
    print(f"💾 元素类别矩阵已保存: {element_matrix_csv}")


async def main_async(
    json_file: str,
    image_dir: str,
    model_name: str,
    output_dir: str,
    batch_size: int = 50,
    max_concurrent: int = 5,
    model: str = "gpt-5-mini",
    endpoint: str = "https://api.openai.com/v1",
    api_key: str = "",
    include_prompt_context: bool = False,
    data_id_prefix: str = "",
):
    results = []

    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if data_id_prefix:
        data_list = [data for data in data_list if str(data.get("data_id", "")).startswith(data_id_prefix)]

    total_items = len(data_list)
    print(f"📊 总共找到 {total_items} 个待处理项目")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    semaphore = asyncio.Semaphore(max_concurrent)

    processed = 0
    for i in range(0, total_items, batch_size):
        batch = data_list[i:i + batch_size]
        print(f"\n🔄 处理批次 {i // batch_size + 1}, 项目 {i + 1}-{min(i + batch_size, total_items)}")
        batch_results = await process_batch_async(
            batch,
            image_dir=image_dir,
            model=model,
            endpoint=endpoint,
            semaphore=semaphore,
            api_key=api_key,
            include_prompt_context=include_prompt_context,
        )

        for item in batch_results:
            if isinstance(item, Exception):
                print(f"❌ 批处理异常: {item}")
                continue
            data_id, answer = item
            results.append(normalize_result(data_id, answer))

        processed += len(batch)
        print(f"📈 批次完成，总进度: {processed / total_items * 100:.1f}% ({processed}/{total_items})")
        await save_results(results, output_dir, model_name)

    print(f"🎉 所有处理完成！总共处理了 {processed} 个项目")
    return results


def main(**kwargs):
    return asyncio.run(main_async(**kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cultural bias / cultural elements in generated images.")
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/localadmin/bz/TRIG/data/result/cultural_bias")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--endpoint", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--include_prompt_context", action="store_true")
    parser.add_argument("--data_id_prefix", type=str, default="")
    args = parser.parse_args()

    main(**vars(args))
