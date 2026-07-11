import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


DEFAULT_DATASET = "RISys-Lab/LingT2I"
TEXT_RENDERING_SPLIT = "text_rendering"
CONTENT_GENERATION_SPLIT = "content_generation"
CONTENT_GENERATION_DATASET = "RISys-Lab/TRIG"


def _parse_layout(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_text_rendering_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict compatible with the legacy multilingual JSON scripts."""
    item = dict(sample)
    dimensions = item.get("dimensions") or [item.get("dimension"), item.get("lang")]
    lang = item.get("lang") or (dimensions[1] if len(dimensions) > 1 else None)
    render_text = item.get("render_text")
    render_layout = _parse_layout(item.get("render_layout"))

    legacy_dimension_prompt = item.get("dimension_prompt")
    if legacy_dimension_prompt:
        render_text = render_text or legacy_dimension_prompt[0]
        render_layout = render_layout if render_layout is not None else (
            legacy_dimension_prompt[1] if len(legacy_dimension_prompt) > 1 else None
        )

    item["dimensions"] = dimensions
    item["lang"] = lang
    item["render_text"] = render_text
    item["render_layout"] = render_layout
    item["dimension_prompt"] = [render_text, render_layout]
    return item


def load_text_rendering_data(
    dataset_name: str = DEFAULT_DATASET,
    split: str = TEXT_RENDERING_SPLIT,
    data_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load TRIG-Multilingual text-rendering data from parquet or legacy JSON."""
    source = data_file or dataset_name
    if source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            return [normalize_text_rendering_sample(x) for x in json.load(f)]

    dataset = load_dataset(source, split=split)
    return [normalize_text_rendering_sample(x) for x in dataset]


def load_content_generation_data(
    dataset_name: str = DEFAULT_DATASET,
    split: str = CONTENT_GENERATION_SPLIT,
    data_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load multilingual content-generation prompts from parquet or legacy JSON."""
    source = data_file or dataset_name
    if source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        rows = load_dataset(source, split=split)

    items = []
    for row in rows:
        item = dict(row)
        dimensions = item.get("dimensions") or item.get("dimension") or []
        if isinstance(dimensions, str):
            dimensions = [dimensions, item.get("lang")]
        item["dimensions"] = dimensions
        item["dimension"] = item.get("dimension") or (dimensions[0] if dimensions else None)
        item["lang"] = item.get("lang") or (dimensions[1] if len(dimensions) > 1 else None)
        items.append(item)
    return items


def replace_render_token(prompt: str, render_text: Optional[str], quote: bool = False) -> str:
    text = render_text or ""
    if quote and text:
        text = f'"{text}"'
    return prompt.replace("<sks1>", text)


def save_condition_image(sample: Dict[str, Any], output_dir: str, suffix: str = ".jpg") -> Optional[str]:
    """Materialize an embedded HF Image column to a temporary/working image path."""
    image = sample.get("condition_image")
    if image is None:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(sample.get("img_id") or sample["data_id"]).stem
    path = Path(output_dir) / f"{stem}{suffix}"
    image.convert("RGB").save(path)
    return str(path)


def iter_range(items: List[Dict[str, Any]], start_idx: int = 0, end_idx: int = -1) -> Iterable[Dict[str, Any]]:
    end = len(items) if end_idx == -1 or end_idx is None else min(end_idx, len(items))
    start = max(0, start_idx or 0)
    return items[start:end]
