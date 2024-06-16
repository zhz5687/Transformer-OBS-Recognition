import cv2
import numpy as np

def sharpen_image(image):
    # 定义锐化内核
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # 应用锐化滤波器
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def process_image(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 锐化处理
    sharpened = sharpen_image(gray)
    
    # 二值化处理
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪处理
    denoised = cv2.medianBlur(binary, 3)
    
    # 反转颜色
    inverted = cv2.bitwise_not(denoised)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, inverted)
    print(f"Processed image saved to {output_path}")

# 示例用法
# image_path = 'OBC306/new_recognition_images_new/001000/001000d00039-1.bmp'
image_path = '001000h00009-1.bmp'
output_path = 'output_001000h00009.bmp'

process_image(image_path, output_path)