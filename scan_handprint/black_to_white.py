import os
from PIL import Image, ImageOps

# 输入和输出文件夹路径
input_folder = 'data'
output_folder = 'output'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有 bmp 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.bmp'):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)
        
        # 打开图像
        with Image.open(file_path) as img:
            # 反转图像颜色
            inverted_image = ImageOps.invert(img)
            
            # 保存反转后的图像到输出文件夹
            inverted_image.save(os.path.join(output_folder, filename))