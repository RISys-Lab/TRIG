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
from transformers import AutoTokenizer
from easydict import EasyDict as edict
from multiprocessing import Pool, Manager
import multiprocessing as mp

# Add the current directory to Python path to ensure eval_ocr can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the original OCR evaluation utilities
from data import DEFAULT_DATASET, TEXT_RENDERING_SPLIT, load_text_rendering_data
from eval_ocr.recognizer import TextRecognizer, crop_image
from eval_ocr.gemini_ocr import GeminiOCRRecognizer, crop_image_for_gemini

# Fixed coordinate size for JSON files
POSITION_COORD_SIZE = (1024, 1024)

# Global tokenizer for mT5 (initialized once to avoid reloading in parallel processes)
_mt5_tokenizer = None

# =============================================================================
# METRICS CALCULATION FUNCTIONS
# =============================================================================

def cal_character_ned(pred_text, gt_text, text_recognizer=None):
    """
    Calculate normalized edit distance (NED) between predicted and ground truth text.
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
        text_recognizer: Optional text recognizer with char2id mapping
    
    Returns:
        Normalized edit distance (0.0 to 1.0, higher is better)
    """
    # Handle edge cases
    if len(pred_text) == 0 and len(gt_text) > 0:
        return 0.0
    elif len(gt_text) == 0 and len(pred_text) > 0:
        return 0.0
    elif len(pred_text) == 0 and len(gt_text) == 0:
        return 1.0  # Both empty - perfect match
    
    # Calculate edit distance
    if text_recognizer and hasattr(text_recognizer, 'char2id'):
        # Use character-to-ID mapping for local models
        gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in gt_text]
        pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1) for m in pred_text]
        edit_dist = Levenshtein.distance(pred_order, gt_order)
        max_len = max(len(pred_order), len(gt_order))
    else:
        # Fallback to character-level edit distance
        edit_dist = Levenshtein.distance(list(pred_text), list(gt_text))
        max_len = max(len(pred_text), len(gt_text))
    
    # Normalize by maximum length
    if max_len == 0:
        return 1.0
    return 1 - edit_dist / max_len

def cal_token_ned(pred_text, gt_text, tokenizer=None):
    """
    Calculate token-level normalized edit distance (NED) using mT5 tokenizer.
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
        tokenizer: mT5 tokenizer instance (if None, will use global tokenizer)
    
    Returns:
        Token-level normalized edit distance (0.0 to 1.0, higher is better)
    """
    # Handle edge cases
    if len(pred_text) == 0 and len(gt_text) > 0:
        return 0.0
    elif len(gt_text) == 0 and len(pred_text) > 0:
        return 0.0
    elif len(pred_text) == 0 and len(gt_text) == 0:
        return 1.0  # Both empty - perfect match
    
    # Use provided tokenizer or global tokenizer
    if tokenizer is None:
        global _mt5_tokenizer
        if _mt5_tokenizer is None:
            # Fallback: load tokenizer if global one is not initialized
            print("⚠️ Global tokenizer not initialized, loading tokenizer on demand...")
            tokenizer = AutoTokenizer.from_pretrained("/data/model_zoo/mt5-base")
        else:
            tokenizer = _mt5_tokenizer
    
    # Tokenize texts to token IDs
    pred_tokens = tokenizer(pred_text, add_special_tokens=False)["input_ids"]
    gt_tokens = tokenizer(gt_text, add_special_tokens=False)["input_ids"]
    
    # Calculate Levenshtein edit distance on token IDs
    edit_dist = Levenshtein.distance(pred_tokens, gt_tokens)
    
    # Normalize by maximum token length
    max_len = max(len(pred_tokens), len(gt_tokens))
    if max_len == 0:
        return 1.0
    
    return 1 - edit_dist / max_len

def cal_sentence_acc(pred_text, gt_text):
    """
    Calculate sentence-level accuracy (exact match).
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
    
    Returns:
        Sentence accuracy (0 or 1)
    """
    return int(pred_text == gt_text)

def cal_word_acc(pred_text, gt_text, language):
    """
    Calculate word-level accuracy based on language type.
    
    For alphabetic languages (en, es, fr, pt, ru, ar): split by spaces
    For character-based languages (zh, hi, ja, ko): character-level matching
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
        language: Language code (e.g., 'en', 'zh')
    
    Returns:
        Word accuracy score (0.0 to 1.0)
    """
    # Define language categories
    alphabetic_languages = {'en', 'es', 'fr', 'pt', 'ru', 'ar'}  # Space-separated words
    character_languages = {'zh', 'hi', 'ja', 'ko'}  # Character-level matching
    
    # Handle empty cases
    if len(pred_text) == 0 and len(gt_text) == 0:
        return 1.0  # Both empty - perfect match
    if len(pred_text) == 0 or len(gt_text) == 0:
        return 0.0  # One empty, one not - no match
    
    if language in alphabetic_languages:
        # Split by spaces for alphabetic languages
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate similarity score for each word using normalized edit distance
        total_similarity = 0.0
        for i, gt_word in enumerate(gt_words):
            if i < len(pred_words):
                # Calculate normalized edit distance for this word pair
                pred_word = pred_words[i]
                word_similarity = cal_character_ned(pred_word, gt_word)
                total_similarity += word_similarity
            else:
                # No corresponding predicted word - similarity is 0
                total_similarity += 0.0
        
        # Normalize by ground truth word count
        word_acc = total_similarity / len(gt_words)
        
    elif language in character_languages:
        # Character-level matching for character-based languages
        if len(gt_text) == 0:
            return 1.0 if len(pred_text) == 0 else 0.0
        
        # Count matching characters at corresponding positions
        matched_chars = 0
        min_len = min(len(pred_text), len(gt_text))
        
        for i in range(min_len):
            if pred_text[i] == gt_text[i]:
                matched_chars += 1
        
        # Normalize by ground truth length
        word_acc = matched_chars / len(gt_text)
        
    else:
        # Default: treat as alphabetic language
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate similarity score for each word using normalized edit distance
        total_similarity = 0.0
        for i, gt_word in enumerate(gt_words):
            if i < len(pred_words):
                # Calculate normalized edit distance for this word pair
                pred_word = pred_words[i]
                word_similarity = cal_character_ned(pred_word, gt_word)
                total_similarity += word_similarity
            else:
                # No corresponding predicted word - similarity is 0
                total_similarity += 0.0
        
        word_acc = total_similarity / len(gt_words)
    
    return float(word_acc)

# =============================================================================
# END OF METRICS CALCULATION FUNCTIONS
# =============================================================================

def load_existing_results(results_file):
    """
    Load existing results file and extract detailed results for metrics calculation.
    
    Args:
        results_file: Path to existing results JSON file
    
    Returns:
        Dictionary containing results by language with detailed_results
    """
    print(f"📂 Loading existing results from: {results_file}")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    # Extract detailed results by language
    results_by_language = {}
    
    if 'by_language' in results_data:
        for language, lang_data in results_data['by_language'].items():
            if 'detailed_results' in lang_data:
                results_by_language[language] = {
                    'detailed_results': lang_data['detailed_results'],
                    'total_samples': lang_data.get('total_samples', 0),
                    'processed_samples': lang_data.get('processed_samples', 0),
                    'missing_samples': lang_data.get('missing_samples', 0)
                }
                print(f"  📊 {language}: {len(lang_data['detailed_results'])} detailed results")
    
    print(f"✅ Loaded results for {len(results_by_language)} languages")
    return results_by_language

def calculate_metrics_from_detailed_results(detailed_results, language, total_samples=None, processed_samples=None, missing_samples=None):
    """
    Recalculate metrics from detailed results using ground_truth and predicted_text.
    
    Args:
        detailed_results: List of detailed results from existing results file
        language: Language code for word accuracy calculation
        total_samples: Original total samples count (if None, will use len(detailed_results))
        processed_samples: Original processed samples count (if None, will use len(detailed_results))
        missing_samples: Original missing samples count (if None, will use 0)
    
    Returns:
        Dictionary containing recalculated metrics
    """
    if not detailed_results:
        return {
            'character_ned': 0.0,
            'token_ned': 0.0,
            'sentence_accuracy': 0.0,
            'word_accuracy': 0.0,
            'trig_score': 0.0,
            'total_samples': 0,
            'processed_samples': 0,
            'missing_samples': 0
        }
    
    # Recalculate metrics for each sample
    sen_acc = []
    character_ned = []
    word_acc = []
    token_ned = []
    trig_score = []
    recalculated_detailed_results = []
    
    print(f"  🔄 Recalculating metrics for {len(detailed_results)} samples...")
    
    for i, result in enumerate(detailed_results):
        # Extract ground truth and predicted text
        gt_text = result.get('ground_truth', '')
        pred_text = result.get('predicted_text', '')
        
        # Recalculate metrics using our unified functions
        current_character_ned = cal_character_ned(pred_text, gt_text)
        current_token_ned = cal_token_ned(pred_text, gt_text)
        current_sen_acc = cal_sentence_acc(pred_text, gt_text)
        current_word_acc = cal_word_acc(pred_text, gt_text, language)
        current_trig_score = 0.4 * current_character_ned + 0.4 * current_token_ned + 0.2 * current_sen_acc
        
        # Store recalculated metrics
        character_ned.append(current_character_ned)
        token_ned.append(current_token_ned)
        sen_acc.append(current_sen_acc)
        word_acc.append(current_word_acc)
        trig_score.append(current_trig_score)
        
        # Create new detailed result with only necessary fields
        new_result = {
            'data_id': result.get('data_id', ''),
            'img_id': result.get('img_id', ''),
            'ground_truth': gt_text,
            'predicted_text': pred_text,
            'character_ned': current_character_ned,
            'token_ned': current_token_ned,
            'sentence_accuracy': current_sen_acc,
            'word_accuracy': current_word_acc,
            'trig_score': current_trig_score
        }
        recalculated_detailed_results.append(new_result)
        
        # Print detailed metrics for each sample
        print(f"    Sample {result.get('data_id', i+1)}: pred='{pred_text}' | gt='{gt_text}' | character_ned={current_character_ned:.4f} | token_ned={current_token_ned:.4f} | acc={current_sen_acc} | word_acc={current_word_acc:.4f} | trig_score={current_trig_score:.4f}")
    
    # Calculate averages
    avg_sen_acc = np.array(sen_acc).mean() if sen_acc else 0.0
    avg_character_ned = np.array(character_ned).mean() if character_ned else 0.0
    avg_word_acc = np.array(word_acc).mean() if word_acc else 0.0
    avg_token_ned = np.array(token_ned).mean() if token_ned else 0.0
    avg_trig_score = np.array(trig_score).mean() if trig_score else 0.0
    
    return {
        'character_ned': float(avg_character_ned),
        'token_ned': float(avg_token_ned),
        'sentence_accuracy': float(avg_sen_acc),
        'word_accuracy': float(avg_word_acc),
        'trig_score': float(avg_trig_score),
        'total_samples': total_samples if total_samples is not None else len(detailed_results),
        'processed_samples': processed_samples if processed_samples is not None else len(detailed_results),
        'missing_samples': missing_samples if missing_samples is not None else 0,
        'detailed_results': recalculated_detailed_results
    }

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

def load_trig_data(
    dataset_name=DEFAULT_DATASET,
    split=TEXT_RENDERING_SPLIT,
    data_file=None,
):
    """Load TRIG-Multilingual text-rendering data from parquet or legacy JSON."""
    if data_file:
        print(f"Loading TRIG-Multilingual text-rendering data from JSON: {data_file}")
    else:
        print(f"Loading TRIG-Multilingual text-rendering data from dataset: {dataset_name}/{split}")
    data = load_text_rendering_data(dataset_name=dataset_name, split=split, data_file=data_file)
    
    # Group by language and data_id
    grouped_data = defaultdict(dict)
    for item in data:
        data_id = item['data_id']
        language = item.get('lang') or item['dimensions'][1]  # language code
        text = item.get('render_text') or item['dimension_prompt'][0]  # ground truth text
        positions = item.get('render_layout') or item['dimension_prompt'][1]  # text positions
        img_id = item.get('img_id')
        
        grouped_data[language][data_id] = {
            'text': text,
            'positions': positions,
            'img_id': img_id
        }
    
    print(f"Loaded data for {len(grouped_data)} languages")
    for lang, items in grouped_data.items():
        print(f"  {lang}: {len(items)} samples")
    
    return grouped_data

def setup_ocr_recognizer(language='en', use_gemini=False):
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
    
    rec_char_dict_path = os.path.join(ROOT_DIR, 'eval_ocr', 'ocr_recog', dict_file)
    
    # Setup modelscope OCR pipeline
    predictor = pipeline(Tasks.ocr_recognition, model='/data/model_zoo/cv_convnextTiny_ocr-recognition-general_damo')
    
    # Setup text recognizer based on mode
    rec_image_shape = "3, 48, 320"
    
    if use_gemini:
        # Use Gemini API for OCR
        gemini_recognizer = GeminiOCRRecognizer()
        return None, gemini_recognizer, rec_image_shape
    else:
        # Use local models
        args = edict()
        args.rec_image_shape = rec_image_shape
        args.rec_char_dict_path = rec_char_dict_path
        args.rec_batch_num = 1
        args.use_fp16 = False
        text_recognizer = TextRecognizer(args, None)
        return predictor, text_recognizer, rec_image_shape

def setup_mt5_tokenizer():
    """
    Initialize the global mT5 tokenizer once to avoid reloading in parallel processes.
    This function should be called before starting parallel processing.
    
    Returns:
        The initialized mT5 tokenizer instance
    """
    global _mt5_tokenizer
    if _mt5_tokenizer is None:
        print("🔄 Loading mT5 tokenizer...")
        _mt5_tokenizer = AutoTokenizer.from_pretrained("/data/model_zoo/mt5-base")
        print("✅ mT5 tokenizer loaded successfully")
    return _mt5_tokenizer

def evaluate_language(model_path, language, lang_data, predictor, text_recognizer, rec_image_shape, use_gemini=False, use_position=False):
    """Evaluate a specific language"""
    print(f"\n📊 Evaluating language: {language}")
    
    sen_acc = []
    character_ned = []
    word_acc = []  # Add word accuracy list
    token_ned = []  # Add token NED list
    trig_score = []  # Add TRIG score list
    missing_images = []
    processed_count = 0
    detailed_results = []  # Store detailed results for each sample
    
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
        original_size = POSITION_COORD_SIZE  # position coordinates original resolution
        current_size = (H, W)
        
        # Process each text region in the image
        pred_texts = []
        scaled_polygons = []  # Store scaled polygons for Gemini OCR
        
        if use_position and positions and len(positions) > 0:
            # Use position information to crop specific text regions
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
                        scaled_polygons.append(scaled_polygon)
        else:
            # No position information - use the whole image
            pred_texts.append(img)
            scaled_polygons.append(None)
        
        if len(pred_texts) > 0:
            if use_gemini:
                # Use Gemini OCR for recognition
                all_predictions = []
                for i, pred_text_img in enumerate(pred_texts):
                    # Convert tensor to OpenCV format for Gemini
                    scaled_polygon = scaled_polygons[i] if i < len(scaled_polygons) else None
                    cropped_cv_img = crop_image_for_gemini(pred_text_img, None)
                    
                    # Use Gemini API with or without position information
                    if use_position and scaled_polygon is not None:
                        position_info = {"polygon": scaled_polygon, "image_size": (H, W)}
                    else:
                        position_info = None
                    
                    recognized_text = text_recognizer.recognize_text(cropped_cv_img, position_info)
                    all_predictions.append(recognized_text)
                
                # Combine all predicted texts (join with space)
                pred_text = ' '.join(all_predictions).strip()
            else:
                # Use local models for OCR
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
        
        # Calculate metrics using unified functions
        # Character NED
        current_character_ned = cal_character_ned(pred_text, gt_text, text_recognizer)
        character_ned.append(current_character_ned)
        
        # Token NED
        current_token_ned = cal_token_ned(pred_text, gt_text)
        token_ned.append(current_token_ned)
        
        # Sentence accuracy
        current_sen_acc = cal_sentence_acc(pred_text, gt_text)
        sen_acc.append(current_sen_acc)
        
        # Word accuracy
        current_word_acc = cal_word_acc(pred_text, gt_text, language)
        word_acc.append(current_word_acc)
        
        # TRIG Score: weighted combination of metrics
        current_trig_score = 0.4 * current_character_ned + 0.4 * current_token_ned + 0.2 * current_sen_acc
        trig_score.append(current_trig_score)
        
        # Store detailed result for this sample
        detailed_results.append({
            'data_id': data_id,
            'img_id': item['img_id'],
            'ground_truth': gt_text,
            'predicted_text': pred_text,
            'character_ned': float(current_character_ned),
            'token_ned': float(current_token_ned),
            'sentence_accuracy': current_sen_acc,
            'word_accuracy': float(current_word_acc),
            'trig_score': float(current_trig_score)
        })
        
        processed_count += 1
        
        # Debug output for all samples
        print(f'  Sample {data_id}: pred="{pred_text}" | gt="{gt_text}" | character_ned={current_character_ned:.4f} | token_ned={current_token_ned:.4f} | acc={current_sen_acc} | word_acc={current_word_acc:.4f} | trig_score={current_trig_score:.4f}')
    
    # Calculate metrics
    if len(sen_acc) > 0:
        avg_sen_acc = np.array(sen_acc).mean()
        avg_character_ned = np.array(character_ned).mean()
        avg_word_acc = np.array(word_acc).mean()
        avg_token_ned = np.array(token_ned).mean()
        avg_trig_score = np.array(trig_score).mean()
    else:
        avg_sen_acc = 0.0
        avg_character_ned = 0.0
        avg_word_acc = 0.0
        avg_token_ned = 0.0
        avg_trig_score = 0.0
    
    print(f"  📈 Processed: {processed_count}/{len(lang_data)} samples")
    print(f"  📉 Character NED: {avg_character_ned:.4f}")
    print(f"  🔤 Token NED: {avg_token_ned:.4f}")
    print(f"  📈 Sentence Accuracy: {avg_sen_acc:.4f}")
    print(f"  📊 Word Accuracy: {avg_word_acc:.4f}")
    print(f"  🎯 TRIG Score: {avg_trig_score:.4f}")
    print(f"  ⚠️ Missing images: {len(missing_images)}")
    
    return {
        'character_ned': float(avg_character_ned),
        'token_ned': float(avg_token_ned),
        'sentence_accuracy': float(avg_sen_acc),
        'word_accuracy': float(avg_word_acc),
        'trig_score': float(avg_trig_score),
        'total_samples': len(lang_data),
        'processed_samples': processed_count,
        'missing_samples': len(missing_images),
        'detailed_results': detailed_results  # Add detailed results for each sample
    }

def evaluate_language_parallel(args_tuple):
    """Wrapper function for parallel language evaluation"""
    model_path, language, lang_data, ocr_mode, use_position = args_tuple
    
    # Initialize tokenizer in each subprocess to avoid sharing issues
    global _mt5_tokenizer
    if _mt5_tokenizer is None:
        _mt5_tokenizer = AutoTokenizer.from_pretrained("/data/model_zoo/mt5-base")
    
    # Skip if no output directory for this language
    lang_output_dir = os.path.join(model_path, language)
    if not os.path.exists(lang_output_dir):
        print(f"[WARNING] No output directory found for language {language}: {lang_output_dir}")
        return language, None
    
    # Setup language-specific OCR
    use_gemini = (ocr_mode == 'gemini')
    predictor, text_recognizer, rec_image_shape = setup_ocr_recognizer(language, use_gemini)
    
    # Evaluate this language
    lang_results = evaluate_language(model_path, language, lang_data, predictor, text_recognizer, rec_image_shape, use_gemini, use_position)
    
    return language, lang_results

def evaluate_model(model_path, trig_data, ocr_mode, use_position=False, num_processes=None):
    """Evaluate a specific model with parallel processing"""
    print(f"\n🔍 Evaluating model: {model_path}")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(len(trig_data), mp.cpu_count())
    
    print(f"🚀 Using {num_processes} parallel processes")
    
    results = {
        'model_path': model_path,
        'overall': {
            'character_ned': 0.0,
            'token_ned': 0.0,
            'sentence_accuracy': 0.0,
            'word_accuracy': 0.0,
            'trig_score': 0.0,
            'total_samples': 0,
            'processed_samples': 0
        },
        'by_language': {}
    }
    
    all_sen_acc = []
    all_character_ned = []
    all_word_acc = []
    all_token_ned = []
    all_trig_score = []
    total_processed = 0
    total_samples = 0
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for language, lang_data in trig_data.items():
        parallel_args.append((model_path, language, lang_data, ocr_mode, use_position))
    
    # Process languages in parallel
    if num_processes > 1 and len(parallel_args) > 1:
        print(f"🔄 Processing {len(parallel_args)} languages in parallel...")
        with Pool(processes=num_processes) as pool:
            # Use imap for better progress tracking
            language_results = list(tqdm(
                pool.imap(evaluate_language_parallel, parallel_args),
                total=len(parallel_args),
                desc="Processing languages"
            ))
    else:
        # Fallback to sequential processing
        print("🔄 Processing languages sequentially...")
        language_results = []
        for args in tqdm(parallel_args, desc="Processing languages"):
            language_results.append(evaluate_language_parallel(args))
    
    # Collect results
    for language, lang_results in language_results:
        if lang_results is None:
            continue
            
        results['by_language'][language] = lang_results
        
        # Accumulate for overall metrics
        lang_samples = lang_results['processed_samples']
        if lang_samples > 0:
            # Weight by number of samples
            all_sen_acc.extend([lang_results['sentence_accuracy']] * lang_samples)
            all_character_ned.extend([lang_results['character_ned']] * lang_samples)
            all_word_acc.extend([lang_results['word_accuracy']] * lang_samples)
            all_token_ned.extend([lang_results['token_ned']] * lang_samples)
            all_trig_score.extend([lang_results['trig_score']] * lang_samples)
            total_processed += lang_samples
        
        total_samples += lang_results['total_samples']
    
    # Calculate overall metrics
    if len(all_sen_acc) > 0:
        results['overall']['character_ned'] = float(np.mean(all_character_ned))
        results['overall']['token_ned'] = float(np.mean(all_token_ned))
        results['overall']['sentence_accuracy'] = float(np.mean(all_sen_acc))
        results['overall']['word_accuracy'] = float(np.mean(all_word_acc))
        results['overall']['trig_score'] = float(np.mean(all_trig_score))
    
    results['overall']['total_samples'] = total_samples
    results['overall']['processed_samples'] = total_processed
    
    print(f"\n✅ Model evaluation complete!")
    print(f"📊 Overall Results:")
    print(f"  📉 Character NED: {results['overall']['character_ned']:.4f}")
    print(f"  🔤 Token NED: {results['overall']['token_ned']:.4f}")
    print(f"  📈 Sentence Accuracy: {results['overall']['sentence_accuracy']:.4f}")
    print(f"  📊 Word Accuracy: {results['overall']['word_accuracy']:.4f}")
    print(f"  🎯 TRIG Score: {results['overall']['trig_score']:.4f}")
    print(f"  📁 Processed: {total_processed}/{total_samples} samples")
    
    return results

def evaluate_model_skip_ocr(results_file):
    """
    Evaluate model by loading existing results and recalculating metrics.
    
    Args:
        results_file: Path to existing results JSON file
    
    Returns:
        Dictionary containing recalculated results
    """
    print(f"\n🔍 Loading existing results from: {results_file}")
    
    # Initialize mT5 tokenizer for token-level metrics recalculation
    print("🔧 Initializing mT5 tokenizer for token-level metrics...")
    setup_mt5_tokenizer()
    
    # Load existing results
    results_by_language = load_existing_results(results_file)
    
    results = {
        'model_path': os.path.dirname(results_file),
        'overall': {
            'character_ned': 0.0,
            'token_ned': 0.0,
            'sentence_accuracy': 0.0,
            'word_accuracy': 0.0,
            'trig_score': 0.0,
            'total_samples': 0,
            'processed_samples': 0
        },
        'by_language': {}
    }
    
    all_sen_acc = []
    all_character_ned = []
    all_word_acc = []
    all_token_ned = []
    all_trig_score = []
    total_processed = 0
    total_samples = 0
    
    # Process each language
    for language, lang_data in results_by_language.items():
        print(f"\n📊 Processing language: {language}")
        
        # Recalculate metrics from detailed results
        lang_results = calculate_metrics_from_detailed_results(
            lang_data['detailed_results'], 
            language,
            total_samples=lang_data.get('total_samples'),
            processed_samples=lang_data.get('processed_samples'),
            missing_samples=lang_data.get('missing_samples')
        )
        
        results['by_language'][language] = lang_results
        
        # Accumulate for overall metrics
        lang_samples = lang_results['processed_samples']
        if lang_samples > 0:
            # Weight by number of samples
            all_sen_acc.extend([lang_results['sentence_accuracy']] * lang_samples)
            all_character_ned.extend([lang_results['character_ned']] * lang_samples)
            all_word_acc.extend([lang_results['word_accuracy']] * lang_samples)
            all_token_ned.extend([lang_results['token_ned']] * lang_samples)
            all_trig_score.extend([lang_results['trig_score']] * lang_samples)
            total_processed += lang_samples
        
        total_samples += lang_results['total_samples']
        
        print(f"  📉 Character NED: {lang_results['character_ned']:.4f}")
        print(f"  🔤 Token NED: {lang_results['token_ned']:.4f}")
        print(f"  📈 Sentence Accuracy: {lang_results['sentence_accuracy']:.4f}")
        print(f"  📊 Word Accuracy: {lang_results['word_accuracy']:.4f}")
        print(f"  🎯 TRIG Score: {lang_results['trig_score']:.4f}")
        print(f"  📁 Processed: {lang_results['processed_samples']}/{lang_results['total_samples']} samples")
    
    # Calculate overall metrics
    if len(all_sen_acc) > 0:
        results['overall']['character_ned'] = float(np.mean(all_character_ned))
        results['overall']['token_ned'] = float(np.mean(all_token_ned))
        results['overall']['sentence_accuracy'] = float(np.mean(all_sen_acc))
        results['overall']['word_accuracy'] = float(np.mean(all_word_acc))
        results['overall']['trig_score'] = float(np.mean(all_trig_score))
    
    results['overall']['total_samples'] = total_samples
    results['overall']['processed_samples'] = total_processed
    
    print(f"\n✅ Metrics recalculation complete!")
    print(f"📊 Overall Results (Recalculated):")
    print(f"  📉 Character NED: {results['overall']['character_ned']:.4f}")
    print(f"  🔤 Token NED: {results['overall']['token_ned']:.4f}")
    print(f"  📈 Sentence Accuracy: {results['overall']['sentence_accuracy']:.4f}")
    print(f"  📊 Word Accuracy: {results['overall']['word_accuracy']:.4f}")
    print(f"  🎯 TRIG Score: {results['overall']['trig_score']:.4f}")
    print(f"  📁 Processed: {total_processed}/{total_samples} samples")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate text recognition models on TRIGv1.5 dataset')
    parser.add_argument('--model_path', type=str, 
                        default='/data/experiments/TRIGv1.5/output/tr_ml/EasyText',
                        help='Path to model output directory')
    parser.add_argument('--dataset_name', type=str,
                        default=DEFAULT_DATASET,
                        help='Hugging Face dataset name or local dataset directory')
    parser.add_argument('--split', type=str,
                        default=TEXT_RENDERING_SPLIT,
                        help='Dataset split to evaluate')
    parser.add_argument('--trig_json', type=str,
                        default=None,
                        help='Optional legacy TRIG JSON file. If set, this overrides --dataset_name/--split.')
    parser.add_argument('--output_file', type=str,
                        default='results.json',
                        help='Output results file name')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=None,
                        help='Specific languages to evaluate (default: all)')
    parser.add_argument('--ocr_mode', type=str, choices=['local', 'gemini'], 
                        default='local',
                        help='OCR mode: local (use local models) or gemini (use Gemini API)')
    parser.add_argument('--use_position', action='store_true',
                        help='Use position information for text recognition (default: False)')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of parallel processes (default: min(languages, CPU cores))')
    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel processing (force sequential)')
    parser.add_argument('--skip_ocr', action='store_true',
                        help='Skip OCR processing and use existing results file for metrics calculation')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to existing results file (required when --skip_ocr is used)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Starting TRIGv1.5 Text Recognition Evaluation")
    
    # Initialize mT5 tokenizer for token-level metrics
    print("🔧 Initializing mT5 tokenizer for token-level metrics...")
    setup_mt5_tokenizer()
    
    # Check if skip_ocr mode is enabled
    if args.skip_ocr:
        print("⏭️ Skip OCR mode enabled - using existing results file")
        
        # Validate results file
        if not args.results_file:
            print("❌ Error: --results_file is required when --skip_ocr is used")
            return
        
        if not os.path.exists(args.results_file):
            print(f"❌ Error: Results file not found: {args.results_file}")
            return
        
        print(f"📂 Results file: {args.results_file}")
        
        # Process existing results
        results = evaluate_model_skip_ocr(args.results_file)
        
        # Save recalculated results back to original file
        with open(args.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Recalculated results saved to: {args.results_file}")
        print("🎉 Metrics recalculation completed successfully!")
        return
    
    # Normal OCR evaluation mode
    print(f"📂 Model path: {args.model_path}")
    if args.trig_json:
        print(f"📄 Legacy TRIG JSON: {args.trig_json}")
    else:
        print(f"🤗 Dataset: {args.dataset_name}/{args.split}")
    print(f"🔧 OCR Mode: {'Gemini API' if args.ocr_mode == 'gemini' else 'Local Models'}")
    print(f"📍 Position Info: {'Enabled' if args.use_position else 'Disabled'}")
    
    # Determine parallel processing settings
    if args.no_parallel:
        num_processes = 1
        print("🔄 Parallel processing disabled (sequential mode)")
    else:
        num_processes = args.num_processes
        if num_processes is None:
            num_processes = min(len(trig_data) if 'trig_data' in locals() else 10, mp.cpu_count())
        print(f"🚀 Parallel processing enabled with {num_processes} processes")
    
    # Load TRIGv1.5 dataset
    trig_data = load_trig_data(
        dataset_name=args.dataset_name,
        split=args.split,
        data_file=args.trig_json,
    )
    
    # Filter languages if specified
    if args.languages:
        filtered_data = {lang: data for lang, data in trig_data.items() if lang in args.languages}
        if not filtered_data:
            print(f"❌ No data found for specified languages: {args.languages}")
            return
        trig_data = filtered_data
        print(f"🔍 Evaluating only specified languages: {list(trig_data.keys())}")
    
    # Update num_processes based on actual number of languages
    if not args.no_parallel and num_processes is None:
        num_processes = min(len(trig_data), mp.cpu_count())
        print(f"🚀 Using {num_processes} parallel processes for {len(trig_data)} languages")
    
    # Evaluate the model
    results = evaluate_model(args.model_path, trig_data, args.ocr_mode, args.use_position, num_processes)
    
    # Save results with OCR mode in filename
    # Extract base filename and extension
    base_name, ext = os.path.splitext(args.output_file)
    if not ext:
        ext = '.json'  # Default extension if none provided
    
    # Create filename with OCR mode and parallel info
    parallel_suffix = f"_parallel{num_processes}" if num_processes > 1 else "_sequential"
    output_filename = f"{base_name}_{args.ocr_mode}{parallel_suffix}{ext}"
    output_path = os.path.join(args.model_path, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    main()
