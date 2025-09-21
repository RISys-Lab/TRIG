python pre_image_editing.py --raw_path ../../raw_datasets/image-editing --dataset OmniEdit-Filtered --config prompt_dev.jsonl
python pre_image_editing.py --raw_path ../../raw_datasets/image-editing --dataset X2I-mm-instruction --config pix2pix.jsonl
python pre_image_editing.py --raw_path ../../raw_datasets/subject-driven --dataset X2I-subject-driven --config human.jsonl
python pre_image_editing.py --raw_path ../../raw_datasets/subject-driven --dataset X2I-subject-driven --config character.jsonl

python pre_toxicity.py --raw_path ../../raw_datasets/image-editing --dataset toxigen --config prompt.json --dim R-T
