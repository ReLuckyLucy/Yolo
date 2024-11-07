import comet_ml
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
import time

# 初始化 Comet 实验并传入 API 密钥
def initialize_comet():
    try:
        experiment = comet_ml.Experiment(api_key="********", project_name="yolo")#api_key请填写自己的
        logging.info("Comet Experiment initialized successfully.")
        return experiment
    except Exception as e:
        logging.error(f"Comet initialization failed: {e}")
        return None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义参数
MODEL_PATH = "yolo11n.pt"
DATA_PATH = "E:/Desktop/ultralytics/red_seka.v2i.yolov11/data.yaml"#存放数据集
EPOCHS = 100
IMGSZ = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
IMAGE_PATH = "path/to/image.jpg"
EXPORT_FORMAT = "onnx"

def train_model(model):
    """ 训练模型并返回结果 """
    logger.info("开始训练<<<<<")
    try:
        return model.train(
            data=DATA_PATH,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            device=DEVICE
        )
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise

def evaluate_model(model):
    """ 评估模型 """
    logger.info("评估模型中...")
    try:
        metrics = model.val()
        logger.info(f"Evaluation metrics: {metrics}")
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        raise

def detect_objects(model):
    """ 对图像进行目标检测 """
    logger.info("对图像进行目标检测...")
    try:
        results = model(IMAGE_PATH)
        results[0].show()
    except Exception as e:
        logger.error(f"目标检测过程中出错: {e}")
        raise

def export_model(model):
    """ 导出模型 """
    logger.info("导出模型中...")
    try:
        export_path = model.export(format=EXPORT_FORMAT)
        logger.info(f"Model exported to {export_path}")
    except Exception as e:
        logger.error(f"导出模型过程中出错: {e}")
        raise

def main():
    # 初始化 Comet 实验
    experiment = initialize_comet()

    # 检查设备
    logger.info(f"使用设备: {DEVICE}")
    
    try:
        # 加载模型
        logger.info("加载模型中...")
        model = YOLO(MODEL_PATH)

        # 训练模型
        train_results = train_model(model)

        # 评估模型性能
        evaluate_model(model)

        # 对图像进行目标检测
        detect_objects(model)

        # 导出模型
        export_model(model)

    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")

    # 结束 Comet 实验
    if experiment:
        try:
            experiment.end()
            logger.info("Comet experiment ended successfully.")
        except Exception as e:
            logger.error(f"Error ending Comet experiment: {e}")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
