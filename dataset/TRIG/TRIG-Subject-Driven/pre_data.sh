python pre_subjects_driven.py --raw_path ../../raw_datasets/subject-driven --dataset Subjects200K --config collection1.json
python pre_subjects_driven.py --raw_path ../../raw_datasets/subject-driven --dataset Subjects200K --config collection2.json
python pre_subjects_driven.py --raw_path ../../raw_datasets/subject-driven --dataset X2I-subject-driven --config human.jsonl
python pre_subjects_driven.py --raw_path ../../raw_datasets/subject-driven --dataset X2I-subject-driven --config character.jsonl

python pre_toxicity.py --raw_path ../../raw_datasets/image-editing --dataset toxigen --config prompt.json --dim R-T
