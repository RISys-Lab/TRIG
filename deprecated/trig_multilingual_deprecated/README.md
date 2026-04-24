# TRIGv1.5 Multilingual

## Usage

```bash
python anytext.py
python anytext2.py
python easytext.py
python flux.py
python nano.py
python qwen.py
```

## Evaluation

```bash
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/AnyText --ocr_mode gemini --use_position
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/AnyText2 --ocr_mode gemini --use_position
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/EasyText --ocr_mode gemini --use_position
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/Flux --ocr_mode gemini
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/NanoBanana --ocr_mode gemini
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/Qwen_Image --ocr_mode gemini
```

```bash
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/AnyText --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/AnyText/results_gemini_parallel10.json
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/AnyText2 --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/AnyText2/results_gemini_parallel10.json
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/EasyText --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/EasyText/results_gemini_parallel10.json
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/Flux --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/Flux/results_gemini_parallel10.json
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/NanoBanana --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/NanoBanana/results_gemini_parallel10.json
python trig_ml_ocr.py --model_path /data/experiments/TRIGv1.5/Qwen_Image --ocr_mode gemini --use_position --skip_ocr --results_file /data/experiments/TRIGv1.5/Qwen_Image/results_gemini_parallel10.json
```
