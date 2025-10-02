import os
import sys
import argparse
import json
from tqdm import tqdm
import cv2
import numpy as np

# 添加 AnyText 目录到 Python 路径，确保 ModelScope 可以找到我们的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
anytext_dir = os.path.join(current_dir, 'AnyText')
if anytext_dir not in sys.path:
    sys.path.insert(0, anytext_dir)
print(f"Added AnyText directory to Python path: {anytext_dir}")

from modelscope.pipelines import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='AnyText image generation for TRIGv1.5')
    parser.add_argument('--json_path', type=str, 
                        default="/data/dataset_zoo/TRIGv1.5/trig_multilingual_tr.json",
                        help='Path to TRIGv1.5 JSON file')
    parser.add_argument('--coarse_mask_path', type=str,
                        default="/data/dataset_zoo/TRIGv1.5/coarse_mask",
                        help='Path to coarse mask images directory')
    parser.add_argument('--output_path', type=str,
                        default="/data/experiments/TRIGv1.5/AnyText",
                        help='Output image path')
    parser.add_argument('--seed', type=int, default=66273235)
    parser.add_argument('--ddim_steps', type=int, default=20,
                        help='Number of DDIM steps')
    parser.add_argument('--image_count', type=int, default=1,
                        help='Number of images to generate per prompt')
    parser.add_argument('--show_debug', action='store_true', default=True,
                        help='Show debug information')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    return parser.parse_args()


def process_prompt(prompt: str, dimension_prompt: list) -> str:
    """Process prompt by replacing <sks1> with dimension_prompt[0] and handling quotes"""
    if not dimension_prompt or len(dimension_prompt) == 0:
        print("Warning: No dimension_prompt found, keeping original prompt")
        return prompt
    
    # Get the text from dimension_prompt[0]
    text_to_replace = dimension_prompt[0]
    
    # Count how many <sks1> tags are in the prompt
    sks1_count = prompt.count('<sks1>')
    print(f"Found {sks1_count} <sks1> tags in prompt")
    
    if sks1_count > 1:
        print("Warning: Multiple <sks1> tags found. Processing only the first one.")
        print(f"Original prompt: {prompt}")
        
        # Replace only the first <sks1> and remove the rest
        parts = prompt.split('<sks1>', 1)  # Split only on first occurrence
        if len(parts) == 2:
            processed_prompt = parts[0] + f'"{text_to_replace}"' + parts[1].replace('<sks1>', '')
            print("Applied: Replace only first <sks1>, remove others")
        else:
            processed_prompt = prompt.replace("<sks1>", f'"{text_to_replace}"')
    else:
        # Replace the single <sks1> with the text from dimension_prompt[0], wrapped in quotes
        processed_prompt = prompt.replace("<sks1>", f'"{text_to_replace}"')
    
    # Replace single quotes with double quotes
    processed_prompt = processed_prompt.replace("'", '"')
    
    print(f"Processed prompt: {processed_prompt}")
    
    return processed_prompt


def get_draw_pos_path(coarse_mask_path: str, language: str, img_id: str) -> str:
    """Construct the draw_pos path based on language and img_id"""
    # Remove file extension from img_id if present
    base_name = os.path.splitext(img_id)[0]
    
    # Try different extensions
    for ext in ['.png', '.jpg', '.jpeg']:
        draw_pos_path = os.path.join(coarse_mask_path, language, f"{base_name}{ext}")
        if os.path.exists(draw_pos_path):
            return draw_pos_path
    
    # If no file found, return the .png version (default)
    return os.path.join(coarse_mask_path, language, f"{base_name}.png")


def main():
    args = parse_args()
    
    # Initialize AnyText pipeline
    print("Loading AnyText pipeline...")
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    anytext_dir = os.path.join(original_cwd, 'AnyText')
    
    try:
        # 临时切换到 AnyText 目录，让 ModelScope 可以找到配置文件
        os.chdir(anytext_dir)
        print(f"Temporarily switched to directory: {anytext_dir}")
        
        pipe = pipeline('my-anytext-task', model='/data/model_zoo/AnyText', model_revision='v1.1.3')
        print("Pipeline loaded successfully!")
        
    finally:
        # 无论成功或失败，都切换回原始目录
        os.chdir(original_cwd)
        print(f"Switched back to directory: {original_cwd}")
    
    # Load JSON data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {args.json_path}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process samples
    processed_count = 0
    skipped_count = 0
    
    for i, sample in tqdm(enumerate(data), desc="Processing samples", unit="sample"):
        if args.max_samples and i >= args.max_samples:
            break
            
        data_id = sample['data_id']
        prompt = sample['prompt']
        img_id = sample['img_id']
        dimensions = sample['dimensions']
        dimension_prompt = sample.get('dimension_prompt', [])
        
        # Get language code (second element in dimensions)
        if len(dimensions) < 2:
            print(f"Skipping {data_id}: insufficient dimensions")
            skipped_count += 1
            continue
            
        language = dimensions[1]
        
        # Process prompt
        processed_prompt = process_prompt(prompt, dimension_prompt)
        
        # Get draw_pos path
        draw_pos_path = get_draw_pos_path(args.coarse_mask_path, language, img_id)
        
        # Check if draw_pos file exists
        if not os.path.exists(draw_pos_path):
            print(f"Warning: draw_pos file not found: {draw_pos_path}")
            print(f"Skipping {data_id}")
            skipped_count += 1
            continue
        
        
        print(f"Processing {data_id} (language: {language})...")
        print(f"Original prompt: {prompt[:100]}...")
        print(f"Processed prompt: {processed_prompt[:100]}...")
        print(f"Draw pos path: {draw_pos_path}")
        
        # Prepare input data for AnyText
        input_data = {
            "prompt": processed_prompt,
            "seed": args.seed,
            "draw_pos": draw_pos_path
        }
        
        # AnyText parameters
        params = {
            "show_debug": args.show_debug,
            "image_count": args.image_count,
            "ddim_steps": args.ddim_steps,
        }
        
        # Generate image
        results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode='text-generation', **params)
        
        # Save result
        output_filename = f"{data_id}.png"
        output_dir = os.path.join(args.output_path, language)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the generated image using cv2
        if results and len(results) > 0:
            # Get the first result image
            img = results[0]
            
            # Convert PIL Image to numpy array if needed
            if hasattr(img, 'convert'):
                # It's a PIL Image, convert to numpy array
                img_array = np.array(img)
            else:
                # It's already a numpy array
                img_array = img
            
            # Convert RGB to BGR for cv2 and save
            cv2.imwrite(output_path, img_array[..., ::-1])
            print(f"Successfully saved: {output_path}")
            processed_count += 1
        else:
            print(f"No results generated for {data_id}")
            skipped_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} samples...")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
