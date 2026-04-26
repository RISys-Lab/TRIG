# TRIG-Multilingual Generation

This folder contains the generation and evaluation scripts for the multilingual benchmark.

Building on TRIG, TRIG-multilingual was further designed to evaluate the cross-lingual consistency of text generation across different languages, and a text rendering task was added to address multilingual characteristics.

## Dataset
[🤗 RISys-Lab/TRIG-Multilingual](https://huggingface.co/datasets/RISys-Lab/TRIG-multilingual).


The dataset has two public splits:

- `content_generation`: multilingual content prompts. This follows the normal TRIG text-to-image generation flow; `pea.py` is the multilingual adapter script in this folder.
- `text_rendering`: multilingual text-rendering prompts. These samples contain `render_text`, `render_layout`, and an embedded `condition_image` PIL image for placement-aware models.

> [!NOTE]
> Legacy JSON is still supported with `--data_file path/to/trig_multilingual_tr.json` for generation scripts, but the default path is now the parquet dataset. If you need the old JSON files, download them from the Hugging Face dataset's `raw/` folder.


## Shared Loader

All updated scripts use `data.py`.

- `load_text_rendering_data(...)` reads `text_rendering` and exposes legacy-compatible `dimension_prompt = [render_text, render_layout]`.
- `load_content_generation_data(...)` reads `content_generation`.
- `condition_image` is read directly from parquet. AnyText/AnyText2 materialize it to a cache path only because those APIs require a file path.



## Usage

### Content Generation

Content generation uses the original TRIG YAML-driven generation pipeline. The multilingual config should use `task: "t2i_ml"` and the TRIG-Multilingual Hugging Face dataset:

```yaml
name: "trig-multilingual-content"
task: "t2i_ml"
dataset_name: "RISys-Lab/TRIG-Multilingual"

start_idx: 0
end_idx: 30000

generation:
  models: ["zimage"]
```

Run it from the repo root:

```bash
python main.py --config config/gen.yaml
```

Internally, `t2i_ml` maps to the `content_generation` split and calls the normal text-to-image generation interface with each multilingual prompt.

### Text Rendering
| Script | Split | Model family | Placement input |
| --- | --- | --- | --- |
| `flux.py` | `text_rendering` | FLUX | prompt only |
| `qwen.py` | `text_rendering` | Qwen-Image | prompt only |
| `nano.py` | `text_rendering` | NanoBanana | prompt only |
| `anytext.py` | `text_rendering` | AnyText | embedded `condition_image` materialized to path |
| `anytext2.py` | `text_rendering` | AnyText2 | embedded `condition_image` materialized to path |
| `easytext.py` | `text_rendering` | EasyText | embedded `condition_image` + `render_layout` |
Text rendering uses the scripts in this folder and reads the `text_rendering` split by default.

Prompt-only text-rendering models:

```bash
python flux.py --start_idx 0 --end_idx 10
python qwen.py --start_idx 0 --end_idx 10
python seedream.py --start_idx 0 --end_idx 10
python nano.py --start_idx 0 --end_idx 10
python imagen4.py --start_idx 0 --end_idx 10
```

Placement-aware text-rendering models:

```bash
python anytext.py --max_samples 10
python anytext2.py
python easytext.py --max_samples 10
```

To use a local dataset checkout instead of the Hub:

```bash
python flux.py \
  --dataset_name /home/localadmin/bz/TRIG/data/output/hf_reformat/TRIG-multilingual \
  --start_idx 0 \
  --end_idx 10
```

To use the old JSON temporarily:

```bash
python flux.py --data_file /path/to/trig_multilingual_tr.json --start_idx 0 --end_idx 10
```

## Evaluation

### Content Generation

Content-generation evaluation uses MetaCLIP2/CLIPScore in `trig/metrics/metaclip2_score.py`.

The script loads generated images by `data_id` from an image folder, pairs each image with the multilingual prompt, encodes both sides with `facebook/metaclip-2-worldwide-huge-quickgelu`, and writes one score per sample. The score is computed from the normalized image/text embedding similarity:

```text
score = 2.5 * max(cosine_similarity(image, text), 0)
```

By default, the metric helper loads the `content_generation` parquet split directly from Hugging Face:

```bash
python ../trig/metrics/metaclip2_score.py \
  --image_folder /home/localadmin/bz/TRIG/data/output/t2i_ml/zimage \
  --dataset_name RISys-Lab/TRIG-Multilingual \
  --split content_generation \
  --out_csv /home/localadmin/bz/TRIG/data/result/metaclipscore_tr/metaclip2_zimage.csv \
  --batch_size 64
```

For local legacy JSON files, pass `--json_path /path/to/text-to-image-multilingual.json`.

### Text Rendering

Text-rendering evaluation is OCR-first. The main entry point is `trig_ml_ocr.py`; it reads the `text_rendering` parquet split, finds generated images by language and `data_id`, runs OCR, then computes recognition metrics against the ground-truth render text.

Expected generated-image layout:

```text
MODEL_OUTPUT_DIR/
  ar/
    123.png
  zh/
    456.png
```

Run OCR and metrics:

```bash
python trig_ml_ocr.py \
  --model_path /data/experiments/TRIGv1.5/output/tr_ml/EasyText \
  --dataset_name RISys-Lab/TRIG-Multilingual \
  --split text_rendering \
  --ocr_mode gemini \
  --use_position \
  --output_file results.json
```

`--ocr_mode local` uses the local OCR recognizer and dictionary files under `eval_ocr/ocr_recog`; `--ocr_mode gemini` uses the Gemini OCR wrapper. `--use_position` crops text regions from the render layout before OCR, which is useful for placement-aware text-rendering models.

The output is saved inside `--model_path` with the OCR mode and parallel suffix, for example `results_gemini_parallel10.json`. The JSON contains `overall`, `by_language`, and per-sample `detailed_results`.

Metrics:

- `character_ned`: character-level normalized edit distance score.
- `token_ned`: token-level normalized edit distance score using the mT5 tokenizer.
- `sentence_accuracy`: exact sentence match after OCR.
- `word_accuracy`: word-level accuracy.
- `trig_score`: `0.4 * character_ned + 0.4 * token_ned + 0.2 * sentence_accuracy`.

If OCR results already exist and only the metrics need to be recalculated:

```bash
python trig_ml_ocr.py \
  --skip_ocr \
  --results_file /data/experiments/TRIGv1.5/output/tr_ml/EasyText/results_gemini_parallel10.json
```

To export a compact per-language summary, use `avg_precision.py`. It averages `character_ned`, `token_ned`, and `sentence_accuracy` for each language and writes a `.txt` file next to the result JSON:

```bash
python avg_precision.py /data/experiments/TRIGv1.5/output/tr_ml/EasyText/results_gemini_parallel10.json
```

> [!NOTE]
> Evaluation uses parquet by default. Legacy JSON is still supported with `--json_path` for content scoring and `--trig_json` for text-rendering OCR; those JSON files are available in the Hugging Face dataset's `raw/` folder.
