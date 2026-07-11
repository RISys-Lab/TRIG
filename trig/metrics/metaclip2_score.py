import os
import json
import math
import numpy as np
import torch
from PIL import Image

try:
    from PIL import UnidentifiedImageError
except ImportError:  # 极老版本 Pillow
    UnidentifiedImageError = OSError  # type: ignore[misc, assignment]
from transformers import AutoProcessor, AutoModel
import csv   # 新增
from tqdm import tqdm  # 新增进度条
from datasets import load_dataset


DEFAULT_DATASET = "RISys-Lab/LingT2I"
CONTENT_GENERATION_SPLIT = "content_generation"

# ----------------------------
# 工具函数
# ----------------------------
def _l2_normalize(x: np.ndarray, axis=1, eps: float = 1e-12) -> np.ndarray:
    denom = np.sqrt(np.sum(x * x, axis=axis, keepdims=True)) + eps
    return x / denom

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _feature_tensor(x):
    if torch.is_tensor(x):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        val = getattr(x, attr, None)
        if torch.is_tensor(val):
            return val
    val = getattr(x, "last_hidden_state", None)
    if torch.is_tensor(val):
        return val[:, 0]
    if isinstance(x, (tuple, list)):
        for val in x:
            if torch.is_tensor(val):
                return val
    raise TypeError(f"cannot extract feature tensor from {type(x)!r}")

def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_prompt_items(
    json_path: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
    split: str = CONTENT_GENERATION_SPLIT,
) -> list[dict]:
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        print(f"[MetaCLIP2] loaded {len(items)} items from JSON: {json_path}")
        return items

    dataset = load_dataset(dataset_name, split=split)
    items = [dict(row) for row in dataset]
    print(f"[MetaCLIP2] loaded {len(items)} items from dataset: {dataset_name}/{split}")
    return items

# ----------------------------
# MetaCLIP2Embedder 同前
# ----------------------------
class MetaCLIP2Embedder:
    def __init__(
        self,
        model_name: str = "facebook/metaclip-2-worldwide-huge-quickgelu",
        device: str = "cuda",
        use_bf16: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        print(f"[MetaCLIP2] device = {self.device}")
        torch_dtype = torch.bfloat16 if (use_bf16 and self.device.type == "cuda") else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.has_feat_api = all(hasattr(self.model, attr) for attr in ["get_image_features", "get_text_features"])
        self.autocast_enabled = (self.device.type == "cuda" and torch_dtype == torch.bfloat16)

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        assert len(images) == len(texts), "images 与 texts 数量需一致"

        if self.has_feat_api:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled):
                img_inputs = self.processor(images=images, return_tensors="pt")
                img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
                img_feats = _feature_tensor(self.model.get_image_features(**img_inputs))

                txt_inputs = self.processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
                txt_feats = _feature_tensor(self.model.get_text_features(**txt_inputs))
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled):
                inputs = self.processor(images=images, text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                img_feats = _feature_tensor(getattr(outputs, "image_embeds", outputs))
                txt_feats = _feature_tensor(getattr(outputs, "text_embeds", outputs))

        return _to_numpy(img_feats), _to_numpy(txt_feats)

# ----------------------------
# 主函数
# ----------------------------
def process_images_with_prompts_metaclip2(
    image_folder: str,
    json_path: str | None = None,
    dataset_name: str = DEFAULT_DATASET,
    split: str = CONTENT_GENERATION_SPLIT,
    batch_size: int = 100,
    device: str = "cuda",
    use_bf16: bool = True,
    trust_remote_code: bool = False,
) -> dict:
    items = _load_prompt_items(json_path=json_path, dataset_name=dataset_name, split=split)

    samples = []
    for it in items:
        data_id = it.get("data_id")
        prompt = it.get("prompt")
        if data_id is None or prompt is None:
            print(f"[warn] skip item without data_id/prompt: {it}")
            continue
        img_path = os.path.join(image_folder, f"{data_id}.png")
        if os.path.exists(img_path):
            samples.append({"data_id": data_id, "prompt": prompt.replace("<sks1>", "some text"), "img_path": img_path})
        else:
            print(f"[warn] image not found: {img_path}")

    print(f"[MetaCLIP2] processed {len(samples)} samples")
    print(f"[MetaCLIP2] processing with batch size {batch_size}")
    print(f"[MetaCLIP2] device = {device}")

    print(f"[MetaCLIP2] loading embedder")
    embedder = MetaCLIP2Embedder(device=device, use_bf16=use_bf16, trust_remote_code=trust_remote_code)

    results: dict[str, float] = {}
    total = len(samples)
    num_batches = (total + batch_size - 1) // batch_size

    # 创建进度条
    pbar = tqdm(range(0, total, batch_size), desc="Processing batches", unit="batch")
    
    for i in pbar:
        batch = samples[i:i + batch_size]
        images: list[Image.Image] = []
        texts: list[str] = []
        ids: list = []
        for x in batch:
            try:
                images.append(_load_image(x["img_path"]))
                texts.append(x["prompt"])
                ids.append(x["data_id"])
            except (UnidentifiedImageError, OSError, ValueError) as e:
                print(f"[warn] skip unreadable image {x['img_path']}: {e}")

        if not images:
            pbar.set_postfix({"batch": f"{(i // batch_size) + 1}/{num_batches}", "samples": 0})
            continue

        img_feats, txt_feats = embedder.encode_batch(images, texts)
        img_feats = _l2_normalize(img_feats, axis=1, eps=1e-12)
        txt_feats = _l2_normalize(txt_feats, axis=1, eps=1e-12)

        sims = np.sum(img_feats * txt_feats, axis=1)
        scores = 2.5 * np.clip(sims, 0.0, None)

        for did, sc in zip(ids, scores):
            results[did] = float(sc)

        # 更新进度条描述信息
        pbar.set_postfix({
            'batch': f"{(i // batch_size) + 1}/{num_batches}",
            'samples': len(images),
        })

    return results

# ----------------------------
# 示例入口
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute MetaCLIP2 scores for TRIG content-generation outputs.")
    parser.add_argument("--image_folder", default="/home/localadmin/bz/TRIG/data/output/t2i_ml/omnidiffusion")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=CONTENT_GENERATION_SPLIT)
    parser.add_argument("--json_path", default=None, help="Optional legacy JSON prompt file.")
    parser.add_argument("--out_csv", default="/home/localadmin/bz/TRIG/data/result/metaclipscore_tr/metaclip2_omnidiffusion.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    scores = process_images_with_prompts_metaclip2(
        image_folder=args.image_folder,
        json_path=args.json_path,
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        use_bf16=True,
        trust_remote_code=args.trust_remote_code,
    )

    # 保存到 CSV：第一列 id，第二列 score（四位小数）
    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["data_id", "score"])  # 表头
        for did, sc in scores.items():
            writer.writerow([did, f"{sc:.4f}"])

    print("CSV saved")
