#!/usr/bin/env python3
"""
测试 AnyText 目录中 Python 文件的基本语法
"""
import ast
import os

def test_python_syntax():
    """测试所有 Python 文件的语法是否正确"""
    print("Testing Python syntax in AnyText directory...")
    
    errors = []
    success = []
    
    def check_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
    # 遍历所有 Python 文件
    for root, dirs, files in os.walk('.'):
        # 跳过 __pycache__ 目录
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                is_valid, error = check_file(filepath)
                
                if is_valid:
                    success.append(filepath)
                    print(f"✓ {filepath}")
                else:
                    errors.append((filepath, error))
                    print(f"❌ {filepath}: {error}")
    
    print(f"\n=== Summary ===")
    print(f"✓ Valid files: {len(success)}")
    print(f"❌ Files with errors: {len(errors)}")
    
    if errors:
        print("\nFiles with syntax errors:")
        for filepath, error in errors:
            print(f"  - {filepath}: {error}")
        return False
    else:
        print("\n🎉 All Python files have valid syntax!")
        return True

if __name__ == "__main__":
    test_python_syntax()
