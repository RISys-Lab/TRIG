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

Evaluation scripts such as `trig_ml_ocr.py`, `trig_ml_bias.py`, and `trig_ml_nsfw.py` still contain legacy JSON loading paths. They are intentionally left for a later pass; this update only normalizes generation.
