import logging
from pathlib import Path
from ultralytics import YOLO


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义参数
MODEL_PATH = "yolo11n.pt"
DATA_PATH = "E:/Desktop/ultralytics/red_seka.v2i.yolov11/data.yaml"
EPOCHS = 100
IMGSZ = 640
DEVICE = "0"  # 或者 "cpu" 如果没有 GPU
IMAGE_PATH = "path/to/image.jpg"
EXPORT_FORMAT = "onnx"

def main():
    try:
        # 加载模型
        logger.info("加载模型中<<<<<")
        model = YOLO(MODEL_PATH)

        # 训练模型
        logger.info("开始训练<<<<<")
        train_results = model.train(
            data=DATA_PATH,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            device=DEVICE
        )
        logger.info("Training completed.")

        # 评估模型性能
        logger.info("评估模型中...")
        metrics = model.val()
        logger.info(f"Evaluation metrics: {metrics}")

        # 对图像进行目标检测
        logger.info("对图像进行目标检测...")
        results = model(IMAGE_PATH)
        results[0].show()

        # 导出模型
        logger.info("导出模型中...")
        export_path = model.export(format=EXPORT_FORMAT)
        logger.info(f"Model exported to {export_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()