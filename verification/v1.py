import cv2
from PIL import Image
from ultralytics import YOLO
import logging

# 配置日志显示
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def camera_predict(model):
    """摄像头实时检测（带错误处理）"""
    try:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ConnectionError("无法访问摄像头，请检查：1. 摄像头权限 2. 设备号是否正确 3. 是否被其他程序占用")
        cap.release()  # 先释放测试用的摄像头
        
        # 使用DShow后端避免MSMF问题（Windows专用）
        cv2.CAP_DSHOW = 0  # 尝试切换视频捕获后端
        logger.info("正在启动摄像头检测...")
        return model.predict(
            source="0",
            show=True,  # 显示实时画面
            verbose=False,  # 关闭冗余输出
            stream=True,  # 启用实时流模式
            # conf=0.5,  # 可选：设置置信度阈值
        )
    except Exception as e:
        logger.error(f"摄像头检测失败: {str(e)}")
        logger.info("建议：1. 尝试改用 'source=1' 2. 检查杀毒软件/防火墙设置")
        return None

def main():
    # 加载模型（带错误处理）
    try:
        model = YOLO("model/yolo11n.pt")  # 注意路径使用正斜杠
        logger.info("模型加载成功，类别列表：%s", model.names)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.info("请检查：1. 模型文件路径 2. 文件完整性")
        return

    # ========== 执行不同输入源的检测 ==========
    # 注意：建议分开运行不同检测模式，不要一次性全部启用
    
    # 模式1：摄像头检测（需要注释其他模式）
    # results = camera_predict(model)
    
    # 模式2：图片文件检测（取消注释以下区块）
    try:
        img_path = "img/test.png"
        logger.info(f"正在检测图片：{img_path}")
        im = Image.open(img_path)
        results = model.predict(
            source=im,
            save=True,  # 保存结果图片
            save_txt=True,  # 保存标签文件
            conf=0.5,  # 置信度阈值
            show=True  # 显示检测结果
        )
        logger.info(f"检测完成，结果保存在：{results[0].save_dir}")
    except Exception as e:
        logger.error(f"图片检测失败: {str(e)}")

    # 模式3：批量检测文件夹（需要真实文件夹路径）
    # try:
    #     results = model.predict(
    #         source="your_images_folder",  # 替换为真实路径
    #         show=False,
    #         save=True
    #     )
    # except Exception as e:
    #     logger.error(f"文件夹检测失败: {str(e)}")

if __name__ == "__main__":
    # Windows多进程支持
    from multiprocessing import freeze_support
    freeze_support()
    
    # 设置OpenCV环境变量（解决MSMF错误）
    import os
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    
    main()