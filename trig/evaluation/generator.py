import os
import json
from trig.utils import load_config, base64_to_image
from trig.config import DIM_DICT_WITHOUT_M_E
from trig.models import import_model
from pathlib import Path
import shutil
import requests
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset

project_root = Path(__file__).resolve().parents[2]

# export PYTHONPATH=/home/muzammal/Projects/TRIG:$PYTHONPATH

TASK_TO_HF_SPLIT = {
    "t2i": "text_to_image",
    "p2p": "image_editing",
    "s2p": "subject_driven",
    "t2i_dtm": "text_to_image",
    "p2p_dtm": "image_editing",
    "s2p_dtm": "subject_driven",
}

class Generator:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        print("-" * 50)
        print("Experiment name:", self.config["name"])
        print("Task:", self.config["task"])
        print("Models:", self.config["generation"]["models"])
        self.prompts_data = self.load_prompts()
        if "description_path" in self.config:
            self.description_data = self.load_descriptions(self.config["description_path"])
        print("-" * 50)

    def instantiate_models(self):
        models = {}
        for model_name in self.config["generation"]["models"]:
            model_class = import_model(model_name)
            models[model_name] = model_class()

        print(f"Models loaded: {models.keys()}")
        return models

    def load_prompts(self):
        if "prompt_path" in self.config:
            # Not recommended, use datasets instead
            with open(self.config["prompt_path"], 'r', encoding='utf-8') as file:
                prompts_data = json.load(file)

            # check details
            total_len = len(prompts_data)
            print(total_len)

            return prompts_data
        else:
            task = self.config["task"]
            if task not in TASK_TO_HF_SPLIT:
                raise ValueError(f"Unsupported Hugging Face dataset task: {task}")

            dataset_name = self.config.get("dataset_name", "RISys-Lab/TRIG")
            split = TASK_TO_HF_SPLIT[task]
            print(f"Loading dataset from Hugging Face: {dataset_name}, split={split}")
            prompts_data = load_dataset(dataset_name, split=split)
            print("Dataset loaded")
            return prompts_data
    
    

    def load_descriptions(self, description_file):
        # deprecated
        with open(description_file, 'r', encoding='utf-8') as file:
            description_data = json.load(file)
        return description_data

    def generate_batch_models(self, start_idx=None, end_idx=None, batch_size=1):
        # FIXME: multiprocessing not working
        # with multiprocessing.Pool() as pool:
        #     pool.map(self.generate_single_model, self.config["generation"]["models"])
        for model_name in self.config["generation"]["models"]:
            self.generate_single_model(model_name, start_idx, end_idx, batch_size)

    def save_image(self, image, output_path, filename):
        file_path = os.path.join(output_path, f"{filename}.png")

        try:
            if isinstance(image, str):
                response = requests.get(image)
                response.raise_for_status()
                with open(file_path, "wb") as file:
                    file.write(response.content)
            else: 
                image.save(file_path)
        except Exception as e:
            print(f"Failed to save image: {filename}, Error: {e}")

    def generate_single_model(self, model_name, start_idx=None, end_idx=None, batch_size=1):
        model_class = import_model(model_name)
        model = model_class()
        task  = self.config["task"]
        
        output_path = os.path.join(project_root, 'data/output/', task, model_name)
        print(f"Output path: {output_path}")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_list = os.listdir(output_path)
        file_list = [os.path.splitext(f)[0] for f in file_list]

        if start_idx is not None and end_idx is not None:
            if end_idx > len(self.prompts_data):
                end_idx = len(self.prompts_data)
            self.prompts_data = self.prompts_data[start_idx:end_idx]

        # 如果batch_size > 1且任务是t2i，使用批处理
        if batch_size > 1 and task in ["t2i", "t2i_ml"]:
            self.generate_t2i_batch(model, output_path, file_list, batch_size)
        else:
            # 原有的单个处理逻辑
            for prompt_data in tqdm(self.prompts_data):
                # Align HF dataset format
                if prompt_data["data_id"] in file_list:
                    continue
                
                if "image_path" in self.config:
                    # if use json (not recommended), image is a path to the image
                    image = os.path.join(self.config["image_path"], prompt_data["img_id"])
                elif "image" in prompt_data:
                    # from HF dataset, image is a PIL Image object
                    image = prompt_data["image"].convert("RGB")
                else:
                    # no image available, leave empty
                    image = None

                prompt = prompt_data["prompt"]
                if "dimensions" in prompt_data:
                    dimensions = prompt_data["dimensions"]
                item = prompt_data["item"] if "item" in prompt_data else None

                #  deprecated
                # if model_name == "flowedit":
                #     src_prompt = description_data[prompt_data["img_id"]]

                task_mapping = {
                    "t2i": lambda: model.generate(prompt),
                    "p2p": lambda: model.generate_p2p(prompt, image),
                    "s2p": lambda: model.generate_s2p(prompt, item, image),
                    "t2i_dtm": lambda: model.generate(prompt, dimensions),
                    "p2p_dtm": lambda: model.generate_p2p(prompt, image, dimensions),
                    "s2p_dtm": lambda: model.generate_s2p(prompt, item, image, dimensions),
                    # for multilingual generation
                    "t2i_ml": lambda: model.generate(prompt),
                }
                image = task_mapping[task]()

                self.save_image(image, output_path, prompt_data["data_id"])

    def generate_t2i_batch(self, model, output_path, file_list, batch_size):
        """T2I任务的批处理生成"""
        batch_prompts = []
        batch_data_ids = []
        
        for prompt_data in tqdm(self.prompts_data):
            # 跳过已存在的文件
            if prompt_data["data_id"] in file_list:
                continue
                
            batch_prompts.append(prompt_data["prompt"])
            batch_data_ids.append(prompt_data["data_id"])
            
            # 当达到批处理大小或是最后一批时，进行生成
            if len(batch_prompts) == batch_size or prompt_data == self.prompts_data[-1]:
                try:
                    # 批处理生成
                    generated_images = model.generate(batch_prompts, batch_size=len(batch_prompts))
                    
                    # 保存生成的图像
                    if isinstance(generated_images, list):
                        for i, img in enumerate(generated_images):
                            self.save_image(img, output_path, batch_data_ids[i])
                    else:
                        # 单张图像的情况
                        self.save_image(generated_images, output_path, batch_data_ids[0])
                        
                except Exception as e:
                    print(f"Batch generation failed: {e}")
                    # 回退到单个处理
                    for i, prompt in enumerate(batch_prompts):
                        try:
                            img = model.generate(prompt)
                            self.save_image(img, output_path, batch_data_ids[i])
                        except Exception as single_e:
                            print(f"Single generation failed for {batch_data_ids[i]}: {single_e}")
                
                # 清空批处理缓存
                batch_prompts = []
                batch_data_ids = []


if __name__ == "__main__":
    generator = Generator(config_path=r"/home/zsc/TRIG/config/gen.yaml")
    generator.generate_batch_models()
