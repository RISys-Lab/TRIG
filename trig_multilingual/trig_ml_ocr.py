import sys
import os
import json
import argparse
import cv2
import torch
import numpy as np
import Levenshtein
import math
import random
from tqdm import tqdm
from collections import defaultdict
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from easydict import EasyDict as edict

# Add the current directory to Python path to ensure eval_ocr can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original OCR evaluation utilities
from eval_ocr.recognizer import TextRecognizer, crop_image

def get_ld(ls1, ls2):
    """Calculate normalized edit distance"""
    edit_dist = Levenshtein.distance(ls1, ls2)
    return 1 - edit_dist/(max(len(ls1), len(ls2)) + 1e-5)

def scale_polygon_coordinates(polygon, original_size, target_size):
    """
    Scale polygon coordinates from original size to target size
    
    Args:
        polygon: List of [x, y] coordinates
        original_size: (height, width) of original image
        target_size: (height, width) of target image
    
    Returns:
        List of scaled [x, y] coordinates as integers
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    scaled_polygon = []
    for point in polygon:
        if len(point) >= 2:
            # Scale and ensure integer coordinates
            scaled_x = int(round(point[0] * scale_x))
            scaled_y = int(round(point[1] * scale_y))
            # Clamp coordinates to image bounds
            scaled_x = max(0, min(scaled_x, target_w - 1))
            scaled_y = max(0, min(scaled_y, target_h - 1))
            scaled_polygon.append([scaled_x, scaled_y])
    
    return scaled_polygon

def find_image_file(base_dir, data_id, extensions=['.png', '.jpg', '.jpeg']):
    """
    Find image file with different extensions
    
    Args:
        base_dir: Directory to search in
        data_id: Base filename without extension
        extensions: List of extensions to try
    
    Returns:
        Full path to found image file, or None if not found
    """
    for ext in extensions:
        candidate_path = os.path.join(base_dir, data_id + ext)
        if os.path.exists(candidate_path):
            return candidate_path
    return None

def draw_pos(polygon, prob=1.0, img_size=(1024, 1024)):
    """Draw position mask from polygon coordinates"""
    img = np.zeros((*img_size, 1), dtype=np.uint8)
    if random.random() < prob:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
        
        if img.sum() == 0:
            print(f"❗Empty mask after fillPoly! Original polygon shape: {np.array(polygon).shape}")
            print(f"❗Converted polygon bounds: min={pts.min(axis=0)}, max={pts.max(axis=0)}")
            print(f"❗Image size: {img_size}")
    return img / 255.

def pre_process(img_list, shape):
    """Pre-process images for OCR recognition"""
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    
    return numpy_list

def load_trig_data(json_path):
    """Load TRIGv1.5 dataset"""
    print(f"Loading TRIGv1.5 dataset from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by language and data_id
    grouped_data = defaultdict(dict)
    for item in data:
        data_id = item['data_id']
        language = item['dimensions'][1]  # language code
        text = item['dimension_prompt'][0]  # ground truth text
        positions = item['dimension_prompt'][1]  # text positions
        img_id = item['img_id']
        
        grouped_data[language][data_id] = {
            'text': text,
            'positions': positions,
            'img_id': img_id
        }
    
    print(f"Loaded data for {len(grouped_data)} languages")
    for lang, items in grouped_data.items():
        print(f"  {lang}: {len(items)} samples")
    
    return grouped_data

def setup_ocr_recognizer(language='en'):
    """Setup OCR recognizer for the given language"""
    # Language-specific dictionary mapping
    language_dict_mapping = {
        'en': 'en_dict.txt',           # English
        'zh': 'zh_dict.txt',           # Chinese
        'hi': 'hi_dict.txt',           # Hindi
        'es': 'es_dict.txt',           # Spanish
        'ar': 'ar_dict.txt',           # Arabic
        'fr': 'fr_dict.txt',           # French
        'pt': 'pt_dict.txt',           # Portuguese
        'ru': 'ru_dict.txt',           # Russian
        'ja': 'ja_dict.txt',           # Japanese
        'ko': 'ko_dict.txt',           # Korean
    }
    
    # Model language mapping (determining which OCR model to use)
    model_lang_mapping = {
        'en': 'en',     # English - use English model
        'zh': 'ch',     # Chinese - use Chinese model
        'hi': 'ch',     # Hindi - use Chinese model (better for non-Latin scripts)
        'es': 'en',     # Spanish - use English model
        'ar': 'ch',     # Arabic - use Chinese model (better for complex scripts)
        'fr': 'en',     # French - use English model
        'pt': 'en',     # Portuguese - use English model
        'ru': 'ch',     # Russian - use Chinese model (Cyrillic script)
        'ja': 'ch',     # Japanese - use Chinese model
        'ko': 'ch',     # Korean - use Chinese model
    }
    
    # Get dictionary and model for the language
    dict_file = language_dict_mapping.get(language, 'en_dict.txt')
    model_lang = model_lang_mapping.get(language, 'en')
    
    rec_char_dict_path = os.path.join('/data/TRIG/trig_multilingual/eval_ocr/ocr_recog', dict_file)
    
    # Setup modelscope OCR pipeline
    predictor = pipeline(Tasks.ocr_recognition, model='/data/model_zoo/cv_convnextTiny_ocr-recognition-general_damo')
    
    # Setup text recognizer
    rec_image_shape = "3, 48, 320"
    args = edict()
    args.rec_image_shape = rec_image_shape
    args.rec_char_dict_path = rec_char_dict_path
    args.rec_batch_num = 1
    args.use_fp16 = False
    text_recognizer = TextRecognizer(args, None)
    
    return predictor, text_recognizer, rec_image_shape

def evaluate_language(model_path, language, lang_data, predictor, text_recognizer, rec_image_shape, json_coord_size=(1024, 1024)):
    """Evaluate a specific language"""
    print(f"\n📊 Evaluating language: {language}")
    
    sen_acc = []
    edit_dist = []
    missing_images = []
    processed_count = 0
    
    lang_output_dir = os.path.join(model_path, language)
    
    for data_id, item in tqdm(lang_data.items(), desc=f'Evaluating {language}'):
        # Try to find image with different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate_path = os.path.join(lang_output_dir, data_id + ext)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break
        
        if img_path is None:
            print(f"[WARNING] Image not found for {data_id} (tried .png, .jpg, .jpeg)")
            missing_images.append(os.path.join(lang_output_dir, data_id + '.png'))  # Use .png for logging
            continue
        
        # Load generated image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot read image: {img_path}")
            missing_images.append(img_path)
            continue
        
        H, W = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC->CHW
        
        # Get ground truth text and positions
        gt_text = item['text']
        positions = item['positions']
        
        # Check image resolution and scale coordinates if needed
        original_size = tuple(json_coord_size)  # JSON coordinates original resolution
        current_size = (H, W)
        
        # Process each text region in the image
        pred_texts = []
        
        if positions and len(positions) > 0:
            for pos_info in positions:
                if len(pos_info) > 0 and len(pos_info[0]) > 0:
                    # Extract polygon coordinates
                    original_polygon = pos_info[0]
                    
                    # Scale polygon coordinates to current image resolution
                    scaled_polygon = scale_polygon_coordinates(
                        original_polygon, original_size, current_size
                    )
                    
                    if len(scaled_polygon) > 0:
                        # Create position mask with scaled coordinates
                        pos_mask = draw_pos(scaled_polygon, 1.0, (H, W))
                        np_pos = (pos_mask * 255.).astype(np.uint8)
                        
                        # Crop and recognize text
                        pred_text_img = crop_image(img, np_pos)
                        pred_texts.append(pred_text_img)
        
        if len(pred_texts) > 0:
            # Pre-process for OCR
            pred_texts_processed = pre_process(pred_texts, rec_image_shape)
            
            # Run OCR recognition
            all_predictions = []
            for pt in pred_texts_processed:
                rst = predictor(pt)
                if 'text' in rst and len(rst['text']) > 0:
                    all_predictions.append(rst['text'][0])
                else:
                    all_predictions.append("")
            
            # Combine all predicted texts (join with space)
            pred_text = ' '.join(all_predictions).strip()
        else:
            pred_text = ""
        
        # Calculate metrics
        # Sentence accuracy
        sen_acc.append(int(pred_text == gt_text))
        
        # Normalized edit distance
        if text_recognizer:
            gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in gt_text]
            pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in pred_text]
            edit_dist.append(get_ld(pred_order, gt_order))
        else:
            # Fallback to character-level edit distance if no recognizer
            edit_dist.append(get_ld(list(gt_text), list(pred_text)))
        
        processed_count += 1
        
        # Debug output for first few samples
        if processed_count <= 5:
            print(f'  Sample {data_id}: pred="{pred_text}" | gt="{gt_text}" | acc={sen_acc[-1]} | ned={edit_dist[-1]:.4f}')
    
    # Calculate metrics
    if len(sen_acc) > 0:
        avg_sen_acc = np.array(sen_acc).mean()
        avg_ned = np.array(edit_dist).mean()
    else:
        avg_sen_acc = 0.0
        avg_ned = 0.0
    
    print(f"  📈 Processed: {processed_count}/{len(lang_data)} samples")
    print(f"  📈 Sentence Accuracy: {avg_sen_acc:.4f}")
    print(f"  📉 Normalized Edit Distance: {avg_ned:.4f}")
    print(f"  ⚠️ Missing images: {len(missing_images)}")
    
    return {
        'sentence_accuracy': float(avg_sen_acc),
        'normalized_edit_distance': float(avg_ned),
        'total_samples': len(lang_data),
        'processed_samples': processed_count,
        'missing_samples': len(missing_images)
    }

def evaluate_model(model_path, trig_data, json_coord_size=(1024, 1024)):
    """Evaluate a specific model"""
    print(f"\n🔍 Evaluating model: {model_path}")
    
    results = {
        'model_path': model_path,
        'overall': {
            'sentence_accuracy': 0.0,
            'normalized_edit_distance': 0.0,
            'total_samples': 0,
            'processed_samples': 0
        },
        'by_language': {}
    }
    
    all_sen_acc = []
    all_ned = []
    total_processed = 0
    total_samples = 0
    
    # Process each language
    for language, lang_data in trig_data.items():
        # Skip if no output directory for this language
        lang_output_dir = os.path.join(model_path, language)
        if not os.path.exists(lang_output_dir):
            print(f"[WARNING] No output directory found for language {language}: {lang_output_dir}")
            continue
        
        # Setup language-specific OCR
        predictor, text_recognizer, rec_image_shape = setup_ocr_recognizer(language)
        
        # Evaluate this language
        lang_results = evaluate_language(model_path, language, lang_data, predictor, text_recognizer, rec_image_shape, json_coord_size)
        
        results['by_language'][language] = lang_results
        
        # Accumulate for overall metrics
        lang_samples = lang_results['processed_samples']
        if lang_samples > 0:
            # Weight by number of samples
            all_sen_acc.extend([lang_results['sentence_accuracy']] * lang_samples)
            all_ned.extend([lang_results['normalized_edit_distance']] * lang_samples)
            total_processed += lang_samples
        
        total_samples += lang_results['total_samples']
    
    # Calculate overall metrics
    if len(all_sen_acc) > 0:
        results['overall']['sentence_accuracy'] = float(np.mean(all_sen_acc))
        results['overall']['normalized_edit_distance'] = float(np.mean(all_ned))
    
    results['overall']['total_samples'] = total_samples
    results['overall']['processed_samples'] = total_processed
    
    print(f"\n✅ Model evaluation complete!")
    print(f"📊 Overall Results:")
    print(f"  📈 Sentence Accuracy: {results['overall']['sentence_accuracy']:.4f}")
    print(f"  📉 Normalized Edit Distance: {results['overall']['normalized_edit_distance']:.4f}")
    print(f"  📁 Processed: {total_processed}/{total_samples} samples")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text recognition models on TRIGv1.5 dataset')
    parser.add_argument('--model_path', type=str, 
                        default='/data/experiments/TRIGv1.5/EasyText',
                        help='Path to model output directory')
    parser.add_argument('--trig_json', type=str,
                        default='/data/dataset_zoo/TRIGv1.5/trig_multilingual.json',
                        help='Path to TRIGv1.5 JSON file')
    parser.add_argument('--output_file', type=str,
                        default='results.json',
                        help='Output results file name')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=None,
                        help='Specific languages to evaluate (default: all)')
    parser.add_argument('--json_coord_size', type=int, nargs=2,
                        default=[1024, 1024],
                        help='Original resolution of coordinates in JSON file (height width)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Starting TRIGv1.5 Text Recognition Evaluation")
    print(f"📂 Model path: {args.model_path}")
    print(f"📄 TRIG JSON: {args.trig_json}")
    
    # Load TRIGv1.5 dataset
    trig_data = load_trig_data(args.trig_json)
    
    # Filter languages if specified
    if args.languages:
        filtered_data = {lang: data for lang, data in trig_data.items() if lang in args.languages}
        if not filtered_data:
            print(f"❌ No data found for specified languages: {args.languages}")
            return
        trig_data = filtered_data
        print(f"🔍 Evaluating only specified languages: {list(trig_data.keys())}")
    
    # Evaluate the model
    results = evaluate_model(args.model_path, trig_data, args.json_coord_size)
    
    # Save results
    output_path = os.path.join(args.model_path, args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()
