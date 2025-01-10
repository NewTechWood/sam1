from segment_anything.sam_cropper import SAMObjectCropper


def main():
    # 创建SAMObjectCropper实例
    cropper = SAMObjectCropper()

    # 测试图片路径
    image_path = r'D:\NewTechWood\sam1_1\segment_anything\dog001.png'
    # image_path = r'D:\NewTechWood\sam1_1\segment_anything\dog002.jpg'
    output_path = "cropped_object.jpg"
    prompt = "dog"

    # 降低置信度阈值
    success = cropper.crop_object(
        image_path=image_path,
        target_path=output_path,
        prompt=prompt,
        conf_threshold=0.15  # 降低阈值
    )

    if success:
        print(f"已将裁剪结果保存到: {output_path}")
    else:
        print("裁剪失败")


if __name__ == "__main__":
    main()
