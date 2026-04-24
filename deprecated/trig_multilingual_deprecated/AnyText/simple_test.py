#!/usr/bin/env python3
"""
简单测试 AnyText 目录中的文件结构
"""
import os
import sys

def test_file_structure():
    """测试文件结构是否完整"""
    print("Checking AnyText directory structure...")
    
    # 检查关键文件
    required_files = [
        'bert_tokenizer.py',
        'dataset_util.py', 
        'lora_util.py',
        't3_dataset.py',
        'util.py'
    ]
    
    # 检查关键目录
    required_dirs = [
        'cldm',
        'ldm', 
        'models_yaml',
        'ocr_recog'
    ]
    
    print("\n=== Checking required files ===")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - MISSING")
    
    print("\n=== Checking required directories ===")
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"✓ {dir}/")
            # 检查目录内容
            files = os.listdir(dir)
            py_files = [f for f in files if f.endswith('.py')]
            print(f"  └─ Contains {len(py_files)} Python files")
        else:
            print(f"❌ {dir}/ - MISSING")
    
    print("\n=== Checking specific key files ===")
    key_files = [
        'cldm/cldm.py',
        'cldm/recognizer.py', 
        'ldm/util.py',
        'ocr_recog/RecModel.py',
        'models_yaml/anytext_sd15.yaml'
    ]
    
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - MISSING")
    
    # 检查字体路径引用
    print("\n=== Checking font path references ===")
    font_path = '../font/Arial_Unicode.ttf'
    if os.path.exists(font_path):
        print(f"✓ Font file accessible at {font_path}")
    else:
        print(f"❌ Font file NOT accessible at {font_path}")
        # 检查其他可能的位置
        alt_paths = ['../../font/Arial_Unicode.ttf', 'font/Arial_Unicode.ttf']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"  → Found font at {alt_path}")

if __name__ == "__main__":
    test_file_structure()
