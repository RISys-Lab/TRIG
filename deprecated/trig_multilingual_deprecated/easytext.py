import os
import argparse
import json
import tempfile
from tqdm import tqdm
from PIL import Image

import torch
from EasyText.pipeline_pe_clone_multisample import FluxPipeline


# Load pipeline
print("Loading pipeline...")
pipeline = FluxPipeline.from_pretrained(
    "/data/model_zoo/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to('cuda')
seed = 42
torch.manual_seed(seed)
print("Loading pretrain LoRA weights...")
# Load and fuse pretrain LoRA weights
pipeline.load_lora_weights("/data/model_zoo/EasyText/pretrain.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()
print("Loading fine-tune LoRA weights...")
# Load fine-tune LoRA
pipeline.load_lora_weights("/data/model_zoo/EasyText/fine-tune.safetensors")
print("Pipeline loaded successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description='FLUX image generation with LoRA for TRIGv1.5')
    parser.add_argument('--json_path', type=str, 
                        default="/data/dataset_zoo/TRIGv1.5/trig_multilingual_tr.json",
                        help='Path to TRIGv1.5 JSON file')
    parser.add_argument('--condition_image_path', type=str,
                        default="/data/dataset_zoo/TRIGv1.5/condition_image",
                        help='Path to condition images directory')
    parser.add_argument('--output_path', type=str,
                        default="/data/experiments/TRIGv1.5/output/tr_ml/EasyText",
                        help='Output image path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    return parser.parse_args()


def process_condition_image(image_path: str) -> Image.Image:
    """Load and process condition image"""
    condition_image = Image.open(image_path).convert("RGB")
    return condition_image

def truncate_prompt(prompt: str, max_tokens: int = 70) -> str:
    """Truncate prompt to avoid CLIP token length limit"""
    # More aggressive truncation to ensure we stay well under 77 tokens
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt
    
    # Truncate and add ellipsis
    truncated_words = words[:max_tokens-1]
    return " ".join(truncated_words) + "..."

def main():
    args = parse_args()
    
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
        
        # Construct condition image path - try both .png and .jpg extensions
        condition_image_path = os.path.join(args.condition_image_path, language, img_id)
        # Load condition image
        condition_image = process_condition_image(condition_image_path)
        
        # Truncate prompt to avoid CLIP token length limit
        truncated_prompt = truncate_prompt(prompt)
        if truncated_prompt != prompt:
            print(f"Truncated prompt for {data_id} (was {len(prompt.split())} words)")
        
        print(f"Processing {data_id} (language: {language})...")
        print(f"Using prompt: {truncated_prompt[:100]}...")
        print(f"Condition image path: {condition_image_path}")
        
        # Get position information from dimension_prompt
        position_info = None
        if len(dimension_prompt) >= 2 and dimension_prompt[1]:
            position_info = dimension_prompt[1]
            print(f"Using position info: {position_info}")
            
            # Create a temporary position file for this sample
            temp_position_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            json.dump(position_info, temp_position_file)
            temp_position_file.close()
            position_file_path = temp_position_file.name
        else:
            print(f"No position info found for {data_id}, using default")
            position_file_path = None
        
        # Generate image with position information
        result = pipeline(
            prompt=prompt,
            condition_image=condition_image,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            position_file=position_file_path,
            max_sequence_length=512
        ).images[0]
        
        # Save result
        output_filename = f"{data_id}.png"
        output_path = os.path.join(args.output_path, language)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, output_filename)
        print(f"Saving result to: {output_path}")
        result.save(output_path)
        print(f"Successfully saved: {output_path}")
        processed_count += 1
        
        # Clean up temporary file if it exists
        if position_file_path and os.path.exists(position_file_path):
            os.unlink(position_file_path)
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} samples...")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()
