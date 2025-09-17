#!/usr/bin/env python3
# zip_fast_progress.py
import sys, os
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from tqdm import tqdm   # pip install tqdm

def zip_folder_fast(src_dir: str, zip_path: str):
    src = Path(src_dir).resolve()
    if not src.is_dir():
        raise ValueError(f"Source '{src}' is not a directory.")

    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)

    # 统计总文件数，用于进度条
    total_files = sum(len(files) for _, _, files in os.walk(src))

    with ZipFile(zip_path, mode="w", compression=ZIP_STORED, allowZip64=True) as zf:
        with tqdm(total=total_files, unit="file", desc="Zipping") as pbar:
            for root, _, files in os.walk(src):
                root_p = Path(root)
                for name in files:
                    fp = root_p / name
                    arcname = fp.relative_to(src)
                    zf.write(fp, arcname)
                    pbar.update(1)

    print(f"✅ Done: {zip_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python zip_fast_progress.py <src_dir> <out.zip>")
        sys.exit(1)
    zip_folder_fast(sys.argv[1], sys.argv[2])
