import os
import sys
import cv2
import numpy as np
import argparse
from PIL import ImageColor

# 添加AnyText2目录到Python路径，解决ldm模块导入问题
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
anytext2_dir = os.path.join(ROOT_DIR, 'AnyText2')
if anytext2_dir not in sys.path:
    sys.path.insert(0, anytext2_dir)

from data import DEFAULT_DATASET, TEXT_RENDERING_SPLIT, load_text_rendering_data, save_condition_image
from AnyText2.ms_wrapper import AnyText2Model


class TextGenerator:
    def __init__(self, model_dir='/data/model_zoo/AnyText2', model_path='/data/model_zoo/AnyText2/anytext_v2.0.ckpt', 
                 use_fp16=True, use_translator=True):
        """初始化文字生成器"""
        # 构建字体文件路径
        font_path = os.path.join(ROOT_DIR, 'font', 'Arial_Unicode.ttf')
        infer_params = {
            "use_fp16": use_fp16,
            "use_translator": use_translator,
            "font_path": font_path,
        }
        if model_path:
            infer_params['model_path'] = model_path
            
        print("Loading AnyText2 model...")
        self.model = AnyText2Model(model_dir=model_dir, **infer_params).cuda(0)
        print("Model loaded!")
        
        # 可用字体
        self.fonts = {
            "Arial_Unicode": font_path,
        }

    def generate(self, img_prompt, text_prompt, draw_pos=None, 
                width=512, height=512, seed=-1, 
                fonts=None, colors=None, steps=20, cfg_scale=7.5, 
                count=1, save_folder='SaveImages'):
        """
        生成带文字的图像
        
        Args:
            img_prompt: 图像描述，如 "A coffee shop sign"
            text_prompt: 文字内容，如 '"Hello" "World"'
            draw_pos: 文字位置图像路径或numpy数组
            width, height: 图像尺寸
            seed: 随机种子
            fonts: 字体列表，如 ['Arial_Unicode', 'IndieFlower']
            colors: 颜色列表，如 ['rgba(255,0,0,1)', '#00FF00']
            steps: 推理步数
            cfg_scale: CFG强度
            count: 生成图片数量
            save_folder: 保存文件夹
        """
        
        # 处理位置图像
        if draw_pos is None:
            pos_imgs = np.zeros((width, height, 1))
        elif isinstance(draw_pos, str):
            pos_img = cv2.imread(draw_pos)
            if pos_img is None:
                raise ValueError(f"Cannot read position image: {draw_pos}")
            pos_imgs = 255 - pos_img[..., :3]
        else:
            pos_imgs = draw_pos
            
        # 处理字体
        font_paths = ['None'] * 5
        if fonts:
            for i, font in enumerate(fonts[:5]):
                if font in self.fonts:
                    font_paths[i] = self.fonts[font]
        
        # 处理颜色
        text_colors = ' '.join(['500,500,500'] * 5)  # 默认随机色
        if colors:
            color_strs = text_colors.split()
            for i, color in enumerate(colors[:5]):
                if color:
                    if 'rgba' in color:
                        rgb = [int(float(x)) for x in color.split('(')[1].split(')')[0].split(',')[:3]]
                    else:
                        rgb = ImageColor.getcolor(color, "RGB")
                    if list(rgb) not in [[0,0,0], [255,255,255]]:  # 避免纯黑白
                        color_strs[i] = ','.join(map(str, rgb))
            text_colors = ' '.join(color_strs)
        
        # 构建参数
        params = {
            "mode": "gen",
            "sort_priority": "↕",
            "show_debug": False,
            "revise_pos": True,
            "image_count": count,
            "ddim_steps": steps,
            "image_width": width,
            "image_height": height,
            "strength": 1.0,
            "attnx_scale": 1.0,
            "font_hollow": True,
            "cfg_scale": cfg_scale,
            "eta": 0.0,
            "a_prompt": "best quality, extremely detailed, clear text",
            "n_prompt": "low quality, blurred text, unreadable",
            "base_model_path": "",
            "lora_path_ratio": "",
            "glyline_font_path": font_paths,
            "font_hint_image": [None] * 5,
            "font_hint_mask": [None] * 5,
            "text_colors": text_colors
        }
        
        input_data = {
            "img_prompt": img_prompt,
            "text_prompt": text_prompt,
            "seed": seed,
            "draw_pos": pos_imgs,
            "ori_image": None,
        }
        
        # 推理
        results, rtn_code, rtn_warning, debug_info = self.model(input_data, **params)
        
        if rtn_code >= 0:
            print(f'Generated {len(results)} images')
            if rtn_warning:
                print(f'Warning: {rtn_warning}')
            return results
        else:
            raise RuntimeError(f'Generation failed: {rtn_warning}')
        

def create_simple_layout(width=512, height=512, num_lines=2):
    """创建简单的文字布局"""
    pos_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if num_lines == 1:
        # 单行居中
        cv2.rectangle(pos_img, (width//4, height//2-25), (3*width//4, height//2+25), (255,255,255), -1)
    elif num_lines == 2:
        # 两行垂直排列
        cv2.rectangle(pos_img, (width//4, height//3-25), (3*width//4, height//3+25), (255,255,255), -1)
        cv2.rectangle(pos_img, (width//4, 2*height//3-25), (3*width//4, 2*height//3+25), (255,255,255), -1)
    
    return pos_img


def process_text_prompt(text_prompt):
    """处理文本提示，确保引号格式正确"""
    # 如果文本已经包含引号，直接返回
    if text_prompt.startswith('"') and text_prompt.endswith('"'):
        return text_prompt
    
    # 否则添加引号
    return f'"{text_prompt}"'


def save_generated_images(img_list, save_folder, img_id):
    """
    直接保存生成的图片到指定路径
    
    Args:
        img_list: 生成的图片列表
        save_folder: 保存文件夹
        img_id: 图片ID作为文件名
    
    Returns:
        List of saved file paths
    """
    save_paths = []
    
    # 确保保存文件夹存在
    os.makedirs(save_folder, exist_ok=True)
    
    for idx, img in enumerate(img_list):
        if len(img_list) == 1:
            # 单张图片直接使用img_id作为文件名
            filename = img_id
        else:
            # 多张图片添加序号
            base_name = os.path.splitext(img_id)[0]
            ext = os.path.splitext(img_id)[1] or '.jpg'
            filename = f"{base_name}_{idx+1}{ext}"
        
        save_path = os.path.join(save_folder, filename)
        
        # 保存图片 (注意OpenCV使用BGR格式，需要转换为RGB)
        cv2.imwrite(save_path, img[..., ::-1])
        save_paths.append(save_path)
        
    return save_paths


def process_dataset_data(dataset_name=DEFAULT_DATASET, split=TEXT_RENDERING_SPLIT, data_file=None,
                         base_mask_path=None, base_save_path="/data/experiments/TRIGv1.5/output/tr_ml/AnyText2",
                         condition_cache_dir=None):
    """
    处理TRIG-Multilingual parquet数据，兼容旧JSON文件
    
    Args:
        dataset_name: Hugging Face dataset name or local dataset directory
        split: dataset split
        data_file: optional legacy JSON file
        base_mask_path: optional legacy mask图像基础路径
        base_save_path: 保存图像基础路径
    
    Returns:
        Generator yielding (item_data, pos_image_path, save_folder) tuples
    """
    data = load_text_rendering_data(dataset_name, split, data_file)
    
    for item in data:
        # 提取数据
        data_id = item['data_id']
        prompt = item['prompt']
        text_prompt = item['render_text']
        img_id = item['img_id']
        language_code = item['lang']
        
        # 替换 <sks1> 为 text_prompt
        img_prompt = prompt.replace('<sks1>', text_prompt)
        
        # 处理 text_prompt 格式
        formatted_text_prompt = process_text_prompt(text_prompt)
        
        cache_dir = condition_cache_dir or os.path.join(base_save_path, "_condition_image_cache", language_code)
        pos_image_path = save_condition_image(item, cache_dir)
        if pos_image_path is None and base_mask_path:
            pos_image_path = os.path.join(base_mask_path, language_code, img_id)
        
        # 构建保存文件夹路径
        save_folder = os.path.join(base_save_path, language_code)
        
        # 确保保存文件夹存在
        os.makedirs(save_folder, exist_ok=True)
        
        yield {
            'data_id': data_id,
            'img_prompt': img_prompt,
            'text_prompt': formatted_text_prompt,
            'pos_image_path': pos_image_path,
            'save_folder': save_folder,
            'language_code': language_code,
            'img_id': img_id
        }


def main():
    parser = argparse.ArgumentParser(description='AnyText2 TRIG-Multilingual Batch Processing')
    
    parser.add_argument('--dataset_name', default=DEFAULT_DATASET, help='Hugging Face dataset name or local dataset directory')
    parser.add_argument('--split', default=TEXT_RENDERING_SPLIT, help='Dataset split for batch processing')
    parser.add_argument('--data_file', default=None, help='Optional legacy JSON file. Overrides dataset_name when set')
    parser.add_argument('--base_mask_path', default=None, help='Optional legacy mask image directory')
    parser.add_argument('--base_save_path', default='/data/experiments/TRIGv1.5/output/tr_ml/AnyText2', help='Output image base path')
    parser.add_argument('--condition_cache_dir', default=None, help='Directory for materialized parquet condition images')
    
    # 可选参数
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--steps', type=int, default=20, help='DDIM steps')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--count', type=int, default=1, help='Number of images')
    parser.add_argument('--model_path', default='/data/model_zoo/AnyText2/anytext_v2.0.ckpt', help='Model path')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = TextGenerator(model_path=args.model_path)
    
    print(f"Processing dataset: {args.data_file or args.dataset_name}:{args.split}")
    
    total_processed = 0
    total_successful = 0
    
    for i, item_data in enumerate(process_dataset_data(
        dataset_name=args.dataset_name,
        split=args.split,
        data_file=args.data_file,
        base_mask_path=args.base_mask_path,
        base_save_path=args.base_save_path,
        condition_cache_dir=args.condition_cache_dir,
    )):

        total_processed += 1
        
        try:
            print(f"\nProcessing [{i}] {item_data['data_id']} ({item_data['language_code']})")
            print(f"  Text: {item_data['text_prompt']}")
            
            # 检查位置图像是否存在
            if not os.path.exists(item_data['pos_image_path']):
                print(f"  Warning: Position image not found: {item_data['pos_image_path']}")
                # 创建简单布局作为替代
                pos_image = create_simple_layout(args.width, args.height, 1)
            else:
                pos_image = item_data['pos_image_path']
            
            # 生成图像
            results = generator.generate(
                img_prompt=item_data['img_prompt'],
                text_prompt=item_data['text_prompt'],
                draw_pos=pos_image,
                width=args.width,
                height=args.height,
                seed=args.seed,
                fonts=None,
                colors=None,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                count=args.count,
                save_folder=item_data['save_folder']
            )
            
            # 保存图像
            save_paths = save_generated_images(
                results, 
                item_data['save_folder'], 
                item_data['img_id']
            )
            
            total_successful += 1
            print(f"  Success: Generated {len(results)} images, saved to: {save_paths[0]}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    print(f"\nBatch processing completed!")
    print(f"Processed: {total_processed}/{total_successful} successful")


if __name__ == "__main__":
    main()
