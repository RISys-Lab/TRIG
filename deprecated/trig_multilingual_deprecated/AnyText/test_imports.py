#!/usr/bin/env python3
"""
测试 AnyText 目录中模块的导入是否正常
"""
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有关键模块的导入"""
    try:
        print("Testing basic utility imports...")
        import bert_tokenizer
        import dataset_util
        import lora_util
        import util
        print("✓ Basic utilities imported successfully")
        
        print("\nTesting ldm imports...")
        from ldm import util as ldm_util
        print("✓ LDM utilities imported successfully")
        
        print("\nTesting ocr_recog imports...")
        from ocr_recog.RecModel import RecModel
        import ocr_recog.common
        print("✓ OCR recognition modules imported successfully")
        
        print("\nTesting cldm imports...")
        from cldm.recognizer import TextRecognizer, create_predictor
        print("✓ CLDM recognizer imported successfully")
        
        # Note: cldm.cldm requires torch and other heavy dependencies
        # We'll test it separately if needed
        
        print("\nTesting t3_dataset imports...")
        from t3_dataset import T3DataSet
        print("✓ T3Dataset imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
