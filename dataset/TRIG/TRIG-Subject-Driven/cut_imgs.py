from PIL import Image
import os


def crop_right_half(input_folder, output_folder):
    """
    遍历 input_folder 里的所有 PNG 图片，只保留右半部分，并保存到 output_folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(input_path) as img:
                width, height = img.size
                right_half = img.crop((width // 2, 0, width, height))
                right_half.save(output_path)


# 调用示例
crop_right_half(r"H:\ProjectsPro\TRIG\dataset\Trig\Trig-subject-driven\images",
                "H:\ProjectsPro\TRIG\dataset\Trig\Trig-subject-driven\images_right_half")
