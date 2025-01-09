import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

# 指定模型类型和模型权重文件的路径
model_type = "vit_h"  # 例如：vit_h, vit_l, vit_b
checkpoint = r'D:\NewTechWood\sam_vit_h_4b8939.pth'

# 加载模型
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')  # 如果使用GPU，可以将模型移动到GPU上

# 创建预测器
predictor = SamPredictor(sam)

# 加载图像
image = cv2.imread(r'D:\NewTechWood\sam1\segment_anything\dog001.png')

# 设置图像
predictor.set_image(image)

# 使用文本提示进行分割
text_prompt = "dog"

# Segment the image based on the text prompt
masks, _, _ = predictor.predict(mask_input=text_prompt)

# Get the segmented image
segmented_image = predictor.get_segmented_image(masks)

# Save the segmented image
cv2.imwrite('segmented_image.jpg', segmented_image)