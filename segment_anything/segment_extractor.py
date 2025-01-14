import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt


class SegmentExtractor:
    def __init__(self, sam_checkpoint=r'D:\NewTechWood\sam_vit_h_4b8939.pth'):
        """
        初始化SAM和CLIP模型
        """
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化SAM模型
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # 初始化CLIP模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)

        print("Models initialized successfully")

    def extract_segment(self, image_path, prompt, output_path, debug=True):
        """
        根据提示词提取图片中的目标区域
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 设置SAM的输入图像
            self.predictor.set_image(image)

            # 生成分割掩码
            masks = self._generate_masks(image)
            print(f"Generated {len(masks)} candidate masks")

            if debug:
                self._save_all_masks(image, masks, "debug_masks")

            # 找到最匹配的掩码
            best_mask, best_score = self._find_best_mask(image, masks, prompt)
            print(f"Best match score: {best_score:.4f}")

            if best_mask is not None:
                # 提取目标区域
                result = self._apply_mask(image, best_mask)

                # 保存结果
                plt.imsave(output_path, result)
                print(f"Result saved to {output_path}")

                if debug:
                    # 保存掩码用于调试
                    plt.imsave("debug_best_mask.png", best_mask)

                    # 保存带边界框的原图
                    self._save_debug_image(image, best_mask, "debug_bbox.png")

                return True
            else:
                print("No matching segment found")
                return False

        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            return False

    def _generate_masks(self, image):
        """
        生成候选掩码
        """
        h, w = image.shape[:2]
        masks = []

        # 生成网格点作为提示点
        points = []
        grid_size = 3
        for i in range(grid_size):
            for j in range(grid_size):
                x = w * (i + 1) // (grid_size + 1)
                y = h * (j + 1) // (grid_size + 1)
                points.append([x, y])

        points = np.array(points)
        labels = np.ones(len(points))

        # 为每个点生成掩码
        for i in range(len(points)):
            point_coords = points[i:i + 1]
            point_labels = labels[i:i + 1]

            mask_predictions, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            masks.extend([mask for mask in mask_predictions])

        return masks

    def _find_best_mask(self, image, masks, prompt):
        """
        找到最匹配提示词的掩码
        """
        best_score = -float('inf')
        best_mask = None

        # 准备提示词模板
        templates = [
            f"a photo of a {prompt}",
            f"a {prompt} in the image",
            f"this is a {prompt}",
            prompt
        ]

        # 获取文本特征
        text_inputs = self.clip_processor(
            text=templates,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        text_features = self.clip_model.get_text_features(**text_inputs)
        text_features = text_features.mean(dim=0, keepdim=True)

        # 评估每个掩码
        for mask in masks:
            # 应用掩码到图像
            masked_image = image.copy()
            masked_image[~mask] = 0

            # 转换为PIL Image
            masked_pil = Image.fromarray(masked_image)

            # 获取图像特征
            image_inputs = self.clip_processor(
                images=[masked_pil],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            image_features = self.clip_model.get_image_features(**image_inputs)

            # 计算相似度
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (text_features @ image_features.t()).item()

            if similarity > best_score:
                best_score = similarity
                best_mask = mask

        return best_mask, best_score

    def _apply_mask(self, image, mask):
        """
        应用掩码到图像
        """
        # 创建透明背景
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

        # 复制RGB通道
        result[..., :3] = image

        # 设置alpha通道
        result[..., 3] = mask * 255

        return result

    def _save_debug_image(self, image, mask, filename):
        """
        保存带边界框的调试图像
        """
        debug_image = image.copy()

        # 找到掩码的边界
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 绘制边界框
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imsave(filename, debug_image)

    def _save_all_masks(self, image, masks, output_dir):
        """
        保存所有候选掩码用于调试
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            masked_image = image.copy()
            masked_image[~mask] = 0
            plt.imsave(f"{output_dir}/mask_{i}.png", masked_image)


def main():
    # 创建提取器实例
    extractor = SegmentExtractor()

    # 设置路径
    image_path = "input.jpg"  # 替换为您的输入图片路径
    output_path = "output.png"  # 输出将保存为PNG以支持透明背景
    prompt = "dog"  # 要提取的目标

    # 执行提取
    success = extractor.extract_segment(
        image_path=image_path,
        prompt=prompt,
        output_path=output_path,
        debug=True  # 设置为True以生成调试图像
    )

    if success:
        print("提取完成！")
    else:
        print("提取失败！")


if __name__ == "__main__":
    main()
