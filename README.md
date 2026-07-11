# TRIG: Trade-offs in Image Generation
[![paper](https://img.shields.io/badge/cs.CV-2507.22100-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2507.22100)
[![Benchmark](https://img.shields.io/badge/Dataset-TRIG-orange)](https://huggingface.co/datasets/RISys-Lab/TRIG)
[![Collection](https://img.shields.io/badge/Collection-HF-blue)](https://huggingface.co/collections/RISys-Lab/trig-benchmark)

- **Trade-offs and Relationships in Image Generation: How Do Different Evaluation Dimensions Interact? (ICCV 2025)**  
For this TRIG benchmark, please check this main folder and [🤗 RISys-Lab/TRIG](https://huggingface.co/datasets/RISys-Lab/TRIG)
- **LingT2I: On the Limitations of Cross-Lingual Consistency in Multilingual Text-to-image Generation. (ACM MM 2026)**  
For this new multilingual benchmark, please check the [LingT2I Repository](https://github.com/RISys-Lab/LingT2I) and [🤗 RISys-Lab/LingT2I](https://huggingface.co/datasets/RISys-Lab/LingT2I).

## TODO

1. [x] Release the TRIG dataset and evaluation pipeline.
2. [x] Release the Finetune pipeline and experiments.
3. [x] Release the Multilingual Evaluation Benchmark.
3. [ ] Release the LingT2I paper.

## Quick Start
### TRIG Benchmark
Load from [🤗 Huggingface Link](https://huggingface.co/datasets/RISys-Lab/TRIG).
> [!NOTE]
> Legacy JSON is still supported for local experiments. The JSON files are kept in the Hugging Face dataset under `raw/`; use the `--data_file /path/to/file.json` argument in generation scripts when you need to bypass the parquet dataset.
```python
from datasets import load_dataset

ds_t2i = load_dataset("RISys-Lab/TRIG", split="text_to_image")
ds_p2p = load_dataset("RISys-Lab/TRIG", split="image_editing")
ds_s2p = load_dataset("RISys-Lab/TRIG", split="subject_driven")

sample = ds_t2i[0] # keys: (data_id, item, prompt, dimension_prompt, parent dataset, img_id, dimensions, image)
# Generation and Evaluation
```
### TRIG-Multilingual Benchmark
Load from [🤗 Huggingface Link](https://huggingface.co/datasets/RISys-Lab/LingT2I).


```python
from datasets import load_dataset

ds_cg = load_dataset("RISys-Lab/LingT2I", split="content_generation")
ds_tr = load_dataset("RISys-Lab/LingT2I", split="text_rendering")

sample_cg = ds_cg[0]
sample_tr = ds_tr[0]

print(sample_cg["prompt"])
print(sample_cg["dimension"], sample_cg["lang"])

print(sample_tr["prompt"])
print(sample_tr["render_text"])
print(sample_tr["condition_image"])  # PIL.Image.Image for text placement
```

Generation currently follows two paths:

- `content_generation` uses the standard TRIG text-to-image generation logic. For the multilingual FLUX adapter, see `trig_multilingual/generation/pea.py`.
- `text_rendering` uses the scripts in `trig_multilingual/generation/`. These read `render_text`, `render_layout`, and the embedded `condition_image` from parquet. Legacy JSON input is still available through `--data_file`, but parquet is the default.

Evaluation also loads the Hugging Face parquet splits by default. Content-generation scoring uses `trig/metrics/metaclip2_score.py` on the `content_generation` split, and multilingual text-rendering OCR evaluation uses `trig_multilingual/evaluation/trig_ml_ocr.py` on the `text_rendering` split. Legacy JSON files remain available in the dataset `raw/` folder for fallback use.

## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
We recommand to use TRIG score by [vllm](https://github.com/vllm-project/vllm). Please install with
```
# for Qwen2.5vl, please update your transformers
pip install transformers -U
pip install accelerate
pip install 'vllm>=0.7.2'
```
Then deploy the selected VLM models, currently the TRIG score support GPT series, Qwen2.5-VL series, and LLaVA-NeXT Series. For more information, please visit vllm document.
```
# use Qwen2.5-VL-7B
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --device cuda --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5

# or use Qwen2.5-VL-72B with quantize version
vllm serve Qwen/Qwen2-VL-72B-Instruct-AWQ --dtype float16 --port 8000 --gpu-memory-utilization 0.85 --tensor-parallel-size 2 --quantization awq --limit_mm_per_prompt image=4
```

## Getting Started
### Auto Evaluation pipeline on TRIG Benchmark
1. First Please set up a yaml file in config folder to run an experiment, as the format below:
```yaml
name: "test" # name for this experiment
task: "t2i" # chosen from t2i/p2p/s2p, support one task at a time
# load prompts from the Hugging Face dataset
dataset_name: "RISys-Lab/TRIG"

generation:
    # selected models
    models: ["flux",]
    
evaluation:
    image_dir: ["data/output/demo",]
    result_dir: "data/result"

dimensions:
    IQ-O:
        metrics: ["GPTLogitMetric"]
    TA-R:
        metrics: ["GPTLogitMetric"]
    TA-S:
        metrics: ["GPTLogitMetric", "AnotherMetric"]
    Other Dimensions You Want:
        metrics: ["OtherMetric"]

relation:
  models: ["flux"]
  res: "formatted_flux"
  metric: "spearman_corr"
  plot: true
  heatmap: true
  tsne: true
  tradeoff: true
  quadrant_analysis: true
  thresholds:
    synergy: 0.8 
    bottleneck: 0.5 
    
  insight_thresholds:
    synergy_density: 0.4
    bottleneck_density: 0.4
    dominance_ratio: 0.8
    tradeoff_corr: 0.6
```
More examples could be found in the config folder.

The evaluator maps `task` to the corresponding Hugging Face split automatically:
- `t2i` -> `text_to_image`
- `p2p` -> `image_editing`
- `s2p` -> `subject_driven`

For local legacy JSON files, you can still use `prompt_path` instead of `dataset_name`.

2. Run ```main.py```
```
python main.py --config your_config.yaml
```
3. Outputs:
- Generated images will be saved to ```data/output/your_task/your_model/```
- Evaluation result will be saved to ```data/output/your_task/your_model.json```
- Relation result will be saved to ```data/output/your_model/```


### Manual Evaluation by metrics toolkit
All the metrics could be used **independently**. For example:
```
metric_class = trig.metrics.import_metric("aesthetic_predictor")
metric_instance = metric_class()
# Single Evaluation
score = metric_instance.compute(image_path="/path/to/image", prompt="prompt")
# Batch Evaluation
score = metric_instance.compute_batch_manual(images=["/path/to/image"], prompts=["prompt"])
```


### Finetuning by DTM
1. Select the dimension and trade-off type you want to optimize. for example, in the paper, we choose Knowledge & Ambiguity, and try to balance these two dimensions.
2. Follow the TRIG principle, we create [a original set](https://huggingface.co/datasets/RISys-Lab/flux-ft-ds) which covers the two dim.
3. We generate images with this set, the ouput images are in [flux_ft_train.zip](https://huggingface.co/datasets/RISys-Lab/flux_ft_train/blob/main/flux_ft_train.zip).
4. Test these images, select [good samples](https://huggingface.co/datasets/RISys-Lab/flux_ft_train/blob/main/flux_ft_72B_filtered_ids.json) with trade-off as expected.
5. Use these selected image to do LoRA finetune on flux.
6. Then we got the [balanced flux model](https://huggingface.co/RISys-Lab/FLUX_FT_LoRA_TRIG_epoch10).

### Prompt Engineering by DTM
use model name 'sd35_dtm_dim', 'sana_dtm_dim', 'xflux_dtm_dim' and 'hqedit_dtm_dim' in the yaml config file to generate with Prompt Engineering.

## Acknowledgement
Many thanks to the great works in GenAI Models like [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), Benchmarks like [HEIM](https://crfm.stanford.edu/helm/heim/latest/), Metric like [VQAScore](https://github.com/linzhiqiu/t2v_metrics).

## Citation
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

```
@inproceedings{zhang2025trade,
  title={Trade-offs in image generation: How do different dimensions interact?},
  author={Zhang, Sicheng and Xie, Binzhu and Yan, Zhonghao and Zhang, Yuli and Zhou, Donghao and Chen, Xiaofei and Qiu, Shi and Liu, Jiaqi and Xie, Guoyang and Lu, Zhichao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17256--17267},
  year={2025}
}
```
