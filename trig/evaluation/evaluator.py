from collections import defaultdict
import yaml
import os
import importlib
from pathlib import Path
import json
from natsort import natsorted
from trig.metrics import import_metric
from tqdm import tqdm
from datasets import load_dataset
project_root = Path(__file__).resolve().parents[2]

TASK_TO_HF_SPLIT = {
    "t2i": "text_to_image",
    "p2p": "image_editing",
    "s2p": "subject_driven",
}


class Evaluator:
    def __init__(self, config_path="config/default.yaml"):
        self.prompt_dic = None
        self.config = self.load_config(config_path)
        self.prompt_dic = self.load_prompts()

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_prompts(self):
        if "dataset_name" in self.config:
            task = self.config["task"]
            if task not in TASK_TO_HF_SPLIT:
                raise ValueError(f"Unsupported Hugging Face dataset task: {task}")

            dataset_name = self.config["dataset_name"]
            split = TASK_TO_HF_SPLIT[task]
            print(f"Loading prompts from Hugging Face: {dataset_name}, split={split}")
            data = load_dataset(dataset_name, split=split)
        else:
            prompt_path = self.config["prompt_path"]
            with open(prompt_path, "r") as f:
                data = json.load(f)

        prompt_dic = {i['data_id']: {k: v for k, v in i.items()} for i in data}
        return prompt_dic

    def instantiate_metric(self, metric, dim):
        """
        Dynamically instantiate multiple metrics for each dimension.
        """
        if isinstance(metric, dict):
            metric_name = metric.get("name")
            params = {k: v for k, v in metric.items() if k != "name"}
            metric_class = import_metric(metric_name)
            metric_instance = metric_class(dimension=dim, **params)
            # get new metric name from the class
            metric_name = metric_instance.metric_name
            metric_name = metric_name
        else:
            metric_class = import_metric(metric)
            metric_instance = metric_class(dimension=dim)
            metric_name = metric
        return metric_instance, metric_name


    def parse_dimensions(self, filename):
        base_name = os.path.splitext(filename)[0]
        dimensions = base_name.split("_")[:-1]
        return dimensions

    def read_results(self, image_dir=None, save_file=None):
        if save_file is not None:
            output_path = os.path.join(project_root, save_file)
        else:
            save_name = image_dir.split("/")[-1]
            output_path = os.path.join(project_root, self.config['evaluation']['result_dir'], f"{save_name}.json")
        if not os.path.exists(output_path):
            return {}
        else:
            with open(output_path, "r") as f:
                return json.load(f)
        

    def save_results(self, results, image_dir=None, save_file=None):
        if save_file is not None:
            output_path = os.path.join(project_root, save_file)
        else:
            save_name = image_dir.split("/")[-1]
            output_path = os.path.join(project_root, self.config['evaluation']['result_dir'], self.config["task"], f"{save_name}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results Saved:'{output_path}'")

    def group_images_by_combination(self, image_dir):
        grouped = defaultdict(list)
        images = natsorted([f for f in os.listdir(os.path.join(project_root, image_dir))])
        print(f"Found {len(images)} images in '{image_dir}'")

        for filename in images:
            combination = "_".join(self.parse_dimensions(filename))
            image_path = os.path.join(image_dir, filename)
            data_id = os.path.splitext(filename)[0]
            # FIXME: logic for Toxicity dataset
            try:
                promp_data = self.prompt_dic[data_id]
            except Exception as e:
                promp_data = {}
                continue

            promp_data["gen_image_path"] = image_path
            
            if "image_path" in self.config:
                promp_data["img_id"] = os.path.join(self.config["image_path"], promp_data["img_id"])
            elif "image" in promp_data and promp_data["image"] is not None:
                promp_data["img_id"] = promp_data["image"]
            grouped[combination].append(promp_data)

        return grouped
    

    def evaluate_dim_pair(self, combination, prompt_data):
        combined_scores = defaultdict(lambda: defaultdict(list))
        for dim in combination.split("_"):
            if dim not in self.config["dimensions"]:
                print(f"Dimension '{dim}' not found in config file.")
                continue
            else:
                metrics_config = self.config["dimensions"][dim].get("metrics", [])
                for metric in metrics_config:
                    metric_instance, metric_name = self.instantiate_metric(metric, dim)
                    task = self.config["task"]

                    # FIXME: improve general metric interface
                    # scores = metric_instance.compute_batch(data["data_ids"], data["image_paths"], data["prompts"])
                    scores = metric_instance.compute_batch(task, prompt_data)

                    for key, value in scores.items():
                        if key not in combined_scores:
                            combined_scores[key][dim] = [{metric_name: value}]
                        else:
                            combined_scores[key][dim].append({metric_name: value})
                    del metric_instance
        return combined_scores


    def evaluate_all(self, image_dirs=None, result_files=None):
        if image_dirs is None:
            image_dirs = self.config["evaluation"]["image_dirs"]

        
        if isinstance(image_dirs, list):
            result_files = self.config["evaluation"]["result_files"]
            assert len(result_files) == len(image_dirs), "Number of result_files should be equal to number of image_dirs"
            for dir, result_file in zip(image_dirs, result_files):
                print(f"dir is a list, Evaluating images in '{dir}'...")
                self.evaluate_all(dir, result_file)
        elif isinstance(image_dirs, str):
            image_dir = os.path.join(project_root, image_dirs)
            grouped = self.group_images_by_combination(image_dir)

            final_results = self.read_results(image_dir=image_dir, save_file=result_files)
            exist_combination = {"_".join(key.split("_")[:-1]) for key in final_results}
            for combination, prompt_data in tqdm(grouped.items()):
                print(f"Combination: {combination}")
                if combination in exist_combination:
                    print(f"Combination '{combination}' already evaluated.")
                    continue
                print(f"Evaluating combination {combination} with {len(prompt_data)} images...")
                # print(f"Prompt data: {prompt_data[0]}")
                combined_scores = self.evaluate_dim_pair(combination, prompt_data)
                final_results.update(combined_scores)
                self.save_results(final_results,image_dir=image_dir, save_file=result_files)
            print("Evaluation complete!")
        else:
            raise ValueError("image_dir must be a list or a string.")



if __name__ == "__main__":
    evaluator = Evaluator(config_path=r"/home/muzammal/Projects/TRIG/config/demo.yaml")
    evaluator.evaluate_all()
