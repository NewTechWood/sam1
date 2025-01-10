import torch
import numpy as np
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPProcessor, CLIPModel


class SAMObjectCropper:
    def __init__(self, sam_checkpoint=r'D:\NewTechWood\sam_vit_h_4b8939.pth'):
        """初始化SAM和CLIP模型"""
        # 初始化SAM模型
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.sam)

        # 初始化CLIP模型用于文本-图像匹配
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

    def crop_object(self, image_path, target_path, prompt, conf_threshold=0.2):
        """根据提示词裁剪图片中的目标对象"""
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                return False

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 打印图片信息
            print(f"图片尺寸: {image.shape}")

            # 设置SAM的图像
            self.predictor.set_image(image)

            # 生成候选掩码
            masks = self._generate_masks(image)
            print(f"生成了 {len(masks)} 个候选掩码")

            # 使用CLIP评估每个掩码
            best_mask, best_score = self._find_best_mask(image, masks, prompt)
            print(f"最佳匹配分数: {best_score:.4f}")

            if best_mask is not None and best_score > conf_threshold:
                # 应用掩码并裁剪
                cropped_image = self._apply_mask_and_crop(image, best_mask)

                # 检查裁剪结果
                if cropped_image.shape == image.shape:
                    print("警告：裁剪后的图片与原图大小相同")
                    return False

                # 保存结果
                cv2.imwrite(target_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                print(f"成功裁剪出目标对象: {prompt}, 置信度: {best_score:.2f}")
                print(f"原图尺寸: {image.shape}, 裁剪后尺寸: {cropped_image.shape}")
                return True
            else:
                print(f"未找到匹配的对象: {prompt}")
                return False

        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return False

    def _generate_masks(self, image):
        """生成候选掩码"""
        h, w = image.shape[:2]
        masks = []

        # 生成网格点
        grid_points = [
            (w // 4, h // 4), (w // 2, h // 4), (3 * w // 4, h // 4),
            (w // 4, h // 2), (w // 2, h // 2), (3 * w // 4, h // 2),
            (w // 4, 3 * h // 4), (w // 2, 3 * h // 4), (3 * w // 4, 3 * h // 4)
        ]

        for cx, cy in grid_points:
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            # 获取每个点的掩码预测
            point_masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            masks.extend([mask for mask in point_masks])

        return masks

    def _find_best_mask(self, image, masks, prompt):
        """找到最匹配提示词的掩码"""
        best_score = -1
        best_mask = None

        # 使用多个提示词模板
        templates = [
            f"a photo of a {prompt}",
            f"a {prompt} in the image",
            f"a close up of a {prompt}",
            f"this is a {prompt}",
            f"{prompt}"
        ]

        # 准备文本特征
        text_inputs = self.clip_processor(
            text=templates,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features.mean(dim=0, keepdim=True)  # 平均所有模板的特征

        for mask in masks:
            # 提取掩码区域
            masked_image = image.copy()
            masked_image[~mask] = 0

            # 处理掩码图像
            image_inputs = self.clip_processor(
                images=[Image.fromarray(masked_image)],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            image_features = self.clip_model.get_image_features(**image_inputs)

            # 归一化特征向量
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            similarity = (text_features @ image_features.t()).mean().item()

            if similarity > best_score:
                best_score = similarity
                best_mask = mask

        return best_mask, best_score

    def _apply_mask_and_crop(self, image, mask):
        """应用掩码并裁剪图像"""
        try:
            # 使用形态学操作改进掩码
            kernel = np.ones((5, 5), np.uint8)
            mask = mask.astype(np.uint8) * 255  # 确保掩码是二值图像
            mask = cv2.dilate(mask, kernel, iterations=1)

            # 保存掩码用于调试
            cv2.imwrite("debug_mask.png", mask)

            # 获取掩码的边界框
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("未找到有效轮廓")
                return image

            # 获取最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # 打印裁剪区域信息
            print(f"裁剪区域: x={x}, y={y}, width={w}, height={h}")

            # 如果裁剪区域太小或太大，可能是检测错误
            image_area = image.shape[0] * image.shape[1]
            crop_area = w * h
            if crop_area < image_area * 0.01 or crop_area > image_area * 0.9:
                print("裁剪区域异常，可能是检测错误")
                return image

            # 添加padding
            padding = int(min(w, h) * 0.1)  # 使用相对padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # 裁剪图像
            cropped_image = image[y:y + h, x:x + w].copy()  # 使用copy()确保创建新的图像

            # 保存调试图像
            debug_image = image.copy()
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("debug_bbox.png", cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

            return cropped_image

        except Exception as e:
            print(f"裁剪过程出错: {str(e)}")
            return image


def main():
    # 创建SAMObjectCropper实例
    cropper = SAMObjectCropper()

    # 测试图片路径
    image_path = "test.jpg"  # 替换为您的输入图片路径
    output_path = "cropped_object.jpg"  # 替换为您想要保存的输出路径
    prompt = "dog"  # 要检测的对象

    # 裁剪指定对象
    success = cropper.crop_object(image_path, output_path, prompt)
    if success:
        print(f"已将裁剪结果保存到: {output_path}")
    else:
        print("裁剪失败")


if __name__ == "__main__":
    main()