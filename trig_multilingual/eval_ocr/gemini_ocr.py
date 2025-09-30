import base64
import io
import json
import re
import time
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI


def encode_opencv_image(cv_img):
    """Encodes an OpenCV image (numpy array) to Base64."""
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # Encode to base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    image_data = buffered.getvalue()
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    return base64_image, "jpeg"


def create_simple_ocr_message(base64_image, image_type, positions=None):
    """Create OCR message for Gemini API with position information if available."""
    
    if positions is not None:
        # With position information - focus on specific regions
        base_prompt = (
            "You are an OCR system. Extract text from the image accurately.\n"
            "The image contains text that needs to be recognized.\n"
            f"The text is located at the following approximate positions in the image: {positions}\n"
            "Please focus on extracting text from these specific regions.\n"
            "Extract the text exactly as it appears, maintaining proper spelling and grammar.\n"
            "\nReturn only the extracted text as a simple string, without any additional formatting, "
            "explanations, or JSON structure. Just the pure text content."
        )
    else:
        # Without position information - extract the most important text
        base_prompt = (
            "You are an OCR system. Extract the most important text from the image.\n"
            "The image may contain various text elements. Please identify and extract only the most "
            "prominent, central, or key text content in the image.\n"
            "Focus on the main text that appears to be the primary content, ignoring any "
            "watermarks, labels, or secondary text elements.\n"
            "Extract the text exactly as it appears, maintaining proper spelling and grammar.\n"
            "\nReturn only ONE piece of text - the most important text content from the image. "
            "Do not return multiple text segments. Just return a single, clean text string without "
            "any additional formatting, explanations, or JSON structure."
        )
    
    ocr_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": base_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    return ocr_message


def send_gemini_request(messages, max_retries=5, delay=2):
    """Send request to Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Sending OCR request to Gemini (attempt {attempt + 1})...")
            # Note: You should set your actual API key here
            api_key = 'AIzaSyA7_SqnQBqjN1Z3vJersGd6mhYMmt1EiuU'  # Replace with actual API key
            client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages
            )
            response_content = response.choices[0].message.content
            
            # Clean up the response - remove any markdown formatting, extra whitespace
            response_content = response_content.strip()
            if response_content.startswith('```'):
                # Remove code block markers if present
                lines = response_content.split('\n')
                response_content = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_content
            
            return response_content
        
        except Exception as e:
            print(f"Gemini API request failed: {e}")
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"Retry {attempt + 1}/{max_retries}, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retry limit reached, returning empty result.")
                return ""


class GeminiOCRRecognizer:
    """OCR recognizer using Gemini API."""
    
    def __init__(self):
        """Initialize Gemini OCR recognizer."""
        self.model_name = "gemini-2.5-flash"
        print(f"Initialized Gemini OCR recognizer with model: {self.model_name}")
    
    def recognize_text(self, cv_image, positions=None):
        """
        Recognize text from an OpenCV image using Gemini API.
        
        Args:
            cv_image: OpenCV image (numpy array)
            positions: Optional position information for text location
            
        Returns:
            Recognized text as string
        """
        try:
            # Encode image to base64
            base64_image, image_type = encode_opencv_image(cv_image)
            
            # Create OCR message
            ocr_message = create_simple_ocr_message(base64_image, image_type, positions)
            
            # Send request to Gemini
            recognized_text = send_gemini_request(ocr_message)
            
            return recognized_text if recognized_text else ""
        
        except Exception as e:
            print(f"Error in Gemini OCR recognition: {e}")
            return ""
    
    def batch_recognize(self, cv_images, positions_list=None):
        """
        Recognize text from multiple images.
        
        Args:
            cv_images: List of OpenCV images
            positions_list: Optional list of position information for each image
            
        Returns:
            List of recognized texts
        """
        results = []
        positions_list = positions_list or [None] * len(cv_images)
        
        for i, cv_image in enumerate(cv_images):
            positions = positions_list[i] if i < len(positions_list) else None
            text = self.recognize_text(cv_image, positions)
            results.append(text)
            
            # Add small delay to avoid rate limiting
            if i < len(cv_images) - 1:
                time.sleep(0.1)
        
        return results


def crop_image_for_gemini(img_tensor, pos_mask):
    """
    Crop image based on position mask for Gemini OCR.
    
    Args:
        img_tensor: Image tensor (CHW format)
        pos_mask: Position mask (numpy array)
        
    Returns:
        Cropped OpenCV image
    """
    try:
        # Convert tensor to numpy array (CHW -> HWC)
        if len(img_tensor.shape) == 3:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
        
        # Ensure uint8 format
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255.0).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        # Find bounding box from position mask
        if pos_mask is None:
            # If no mask, return the whole image
            return img_np
            
        mask_coords = np.where(pos_mask > 0)
        if len(mask_coords[0]) == 0:
            # If no mask, return the whole image
            return img_np
        
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        # Add some padding
        padding = 5
        h, w = img_np.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Crop the image
        cropped = img_np[y_min:y_max, x_min:x_max]
        
        return cropped
    
    except Exception as e:
        print(f"Error cropping image for Gemini: {e}")
        # Return original image if cropping fails
        if len(img_tensor.shape) == 3:
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.cpu().numpy()
        
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255.0).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        return img_np
