# AnyText 模块组织结构

这个目录包含了运行 AnyText 相关代码所需的所有依赖模块。

## 目录结构

```
AnyText/
├── cldm/                      # ControlLDM 相关模块
│   ├── cldm.py               # 主要的 ControlLDM 模型
│   ├── recognizer.py         # 文本识别器
│   ├── ddim_hacked.py        # 修改过的 DDIM 采样器
│   ├── embedding_manager.py  # 嵌入管理器
│   ├── hack.py               # 辅助工具
│   ├── logger.py             # 日志工具
│   └── model.py              # 基础模型
├── ldm/                       # Latent Diffusion Model 相关模块
│   ├── util.py               # LDM 工具函数
│   ├── data/                 # 数据处理模块
│   ├── models/               # 模型定义
│   └── modules/              # 各种模块组件
├── models_yaml/               # 模型配置文件
│   ├── anytext_sd15.yaml
│   ├── anytext_sd15_conv.yaml
│   ├── anytext_sd15_perloss.yaml
│   └── anytext_sd15_vit.yaml
├── ocr_recog/                 # OCR 识别模块
│   ├── RecModel.py           # 识别模型
│   ├── RNN.py                # RNN 网络
│   ├── RecCTCHead.py         # CTC 头部
│   ├── RecMv1_enhance.py     # 增强版识别模型
│   ├── RecSVTR.py            # SVTR 模型
│   ├── common.py             # 通用函数
│   ├── en_dict.txt           # 英文字典
│   └── ppocr_keys_v1.txt     # 中文字符字典
├── bert_tokenizer.py          # BERT 分词器
├── dataset_util.py            # 数据集工具
├── lora_util.py               # LoRA 相关工具
├── t3_dataset.py              # T3 数据集类
└── util.py                    # 通用工具函数
```

## 使用方法

### 1. 从 trig_multilingual 目录运行 anytext.py

原始的 `anytext.py` 文件仍在 `/data/TRIG/trig_multilingual/` 目录中，可以正常运行：

```bash
cd /data/TRIG/trig_multilingual
python anytext.py --help
```

### 2. 如果需要在 AnyText 目录中运行代码

所有依赖模块都已经复制到这个目录中，字体路径也已经修正为相对路径 `../font/Arial_Unicode.ttf`。

## 路径修正

已进行的路径修正：
- `t3_dataset.py`: 字体路径改为 `../font/Arial_Unicode.ttf`
- `dataset_util.py`: 字体路径改为 `../font/Arial_Unicode.ttf`  
- `ldm/util.py`: 字体路径改为 `../font/Arial_Unicode.ttf`

## 验证

运行以下命令验证文件结构：

```bash
cd /data/TRIG/trig_multilingual/AnyText
python simple_test.py          # 检查文件结构
python test_basic_syntax.py    # 检查语法正确性
```

## 注意事项

1. 原始文件保持不变，不会影响现有的工作流程
2. 字体文件仍在 `/data/TRIG/trig_multilingual/font/` 目录中，通过相对路径访问
3. 所有 Python 文件都通过了语法检查
4. OCR 字典文件（`ppocr_keys_v1.txt`, `en_dict.txt`）已包含在内

这样的组织结构使得 AnyText 相关的代码模块化且独立，同时保持了与原始代码的兼容性。
