import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'D:\NewTechWood\sam1')
from segment_anything import sam_model_registry, SamPredictor

# 可更改的模型参数

# 输入图片
sam_image = cv2.imread(r'D:\NewTechWood\sam1\segment_anything\5a1ffee96be88e160a317a4dec2be845.png')
sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
# 输入模型
sam_checkpoint = r'D:\NewTechWood\sam_vit_h_4b8939.pth'
# 输入模型类型
sam_model_type = 'vit_h'
# 输入模型所需设备类型：'cuda'代表使用GPU
sam_device = 'cpu'
# 输入预标记点
sam_point_x, sam_point_y = 500, 375


# 用SamPredict类来实现对图片的预测

class SamPredict:
    # SamPredict类构造函数：图片（any）、模型（any）、模型类型（str）、模型设备（str）、预标记x坐标（int）、预标记y坐标（int）、是否选择随机颜色覆盖（bool）
    def __init__(self, image: any, checkpoint: any, model_type: str, model_device: str, point_x: int, point_y: int,
                 random_color: bool):
        self.image = image
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.model_device = model_device
        self.point_x = point_x
        self.point_y = point_y
        self.marker_size = 300
        self.random_color = random_color

    # SamPredict类方法：展示标记点（该类方法用于人工判断自己设置的预标记点位是否在预测主体内，可以在类外部调用）
    def show_pre(self) -> None:
        plt.imshow(self.image)
        plt.scatter(self.point_x, self.point_y, color='red', marker='*', s=self.marker_size, edgecolor='white',
                    linewidth=1.25)
        plt.axis('off')
        plt.show()

    # SamPredict私有类方法：展示标记点
    def __show_points(self, points: np.array, labels: np.array) -> None:
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=self.marker_size,
                    edgecolor='white', linewidth=1.25)
        plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=self.marker_size, edgecolor='white',
                    linewidth=1.25)

    # SamPredict私有类方法：展示覆盖
    def __show_mask(self, masks: any) -> None:
        if self.random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * color.reshape((1, 1, -1))
        plt.imshow(mask_image)

    # SamPredict类方法：使用SAM模型进行预测（可以在类外部调用）
    def get_result(self) -> None:
        # 设置模型
        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        model.to(device=self.model_device)
        predictor = SamPredictor(model)
        # 传入图片
        predictor.set_image(self.image)
        # 传设置预标记点
        points = np.array([[self.point_x, self.point_y]])
        # 设置label（需要预测的主体一般设置为1，背景一般设置为0）
        labels = np.array([1])
        # 使用SAM_predictor返回覆盖、置信度及logits
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        print('原始图片高度Height为:', masks.shape[1])
        print('原始图片宽度Width为:', masks.shape[2])
        print('识别主体Mask次数为:', masks.shape[0])

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)
            # 显示覆盖
            self.__show_mask(mask)
            # 显示标记点
            self.__show_points(points=points, labels=labels)
            plt.title(f"Mask_Times:{i + 1},Mask_Scores:{score:2f}", fontsize=18)
            plt.axis('off')
            plt.show()


model_one = SamPredict(image=sam_image, checkpoint=sam_checkpoint, model_type=sam_model_type, model_device=sam_device,
                       point_x=sam_point_x, point_y=sam_point_y, random_color=False)
model_one.get_result()
