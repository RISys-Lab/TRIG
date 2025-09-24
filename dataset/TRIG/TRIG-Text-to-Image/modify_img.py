import os
from PIL import Image


def filter_large_images(folder_path, size_limit_mb=2, res_limit=1600):
    """
    遍历文件夹内的所有 PNG 图片，并打印：
    - 文件大小大于 `size_limit_mb` MB
    - 分辨率长或宽大于 `res_limit`

    :param folder_path: 图片所在的文件夹路径
    :param size_limit_mb: 文件大小阈值（MB，默认 2MB）
    :param res_limit: 分辨率阈值（默认 1600）
    """
    if not os.path.exists(folder_path):
        print(f"❌ 目录 {folder_path} 不存在！")
        return

    size_limit_bytes = size_limit_mb * 1024 * 1024  # MB 转换为字节

    # 遍历文件夹所有 PNG 图片
    for file in os.listdir(folder_path):
        if file.lower().endswith(".png"):  # 只处理 PNG 图片
            file_path = os.path.join(folder_path, file)

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 获取图片分辨率
            try:
                with Image.open(file_path) as img:
                    width, height = img.size  # 获取宽高
            except Exception as e:
                print(f"❌ 读取图片 {file} 失败: {e}")
                continue

            # 判断是否超出大小或分辨率限制
            if file_size > size_limit_bytes or width > res_limit or height > res_limit:
                print(f"📷 图片: {file}")
                print(f"   📏 分辨率: {width}x{height}")
                print(f"   📦 大小: {file_size / (1024 * 1024):.2f} MB")
                print("-" * 40)


# 示例调用
folder_path = r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-subject-driven\images"  # 替换为你的图片文件夹路径
filter_large_images(folder_path)
