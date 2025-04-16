import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO, YOLOE
import os
import numpy as np
import time

# 映射模型名称和模型文件路径
model_mapping = {
    # 普通YOLO模型 - 用于图片/视频模式
    "YOLOv11n": "model/yolo11n.pt",
    "YOLOv11m": "model/yolo11m.pt",
    # YOLOE模型 - 用于分割模式
    "YOLOev11m-pf": "model/yoloe-11m-seg-pf.pt",
    # 单独分割模式用的模型
    "YOLOev11m": "model/yoloe-11m-seg.pt",
}

# YOLOE常见类别名称 (从COCO数据集)
COMMON_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse"
]

# 为不同模式准备不同的模型选项
yolo_models = ["YOLOv11n", "YOLOv11m"]
yoloe_pf_models = ["YOLOev11m-pf"]
yoloe_seg_models = ["YOLOev11m"]

def yolo_inference(image, video, model_name, image_size, conf_threshold):
    model_path = model_mapping[model_name]
    model = YOLO(model_path)
    if image is not None:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    elif video is not None:
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".mp4")
        # 使用mp4格式和H.264编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        # 确保文件写入完成
        time.sleep(1)
        
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            return None, output_video_path
        else:
            return None, None

def yoloe_segmentation(image, model_name, image_size, conf_threshold, classes=None):
    """普通分割模式 - 使用YOLOev11-pf模型"""
    model_path = model_mapping[model_name]
    model = YOLOE(model_path)
    
    # 执行预测
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1]

def yoloe_separate_segmentation(image, model_name, image_size, conf_threshold, classes):
    """单独分割模式 - 使用YOLOev11模型，使用文本提示进行开放词汇检测和分割"""
    model_path = model_mapping[model_name]
    
    try:
        # 初始化YOLOE模型 - 不需要检查模型文件是否存在，YOLOE会自动下载
        print(f"正在加载YOLOE模型: {model_name}，路径: {model_path}")
        model = YOLOE(model_path)
        print(f"YOLOE模型加载成功")
        
        # 设置分类类别（文本提示）
        if not classes or not classes.strip():
            # 如果用户没有输入类别，使用一个默认类别列表
            class_list = ["person", "car", "dog", "cat", "chair", "table", "bottle"]
            debug_info = "使用默认类别: " + ", ".join(class_list)
        else:
            # 处理用户输入的类别
            class_list = [c.strip() for c in classes.split(',') if c.strip()]
            if not class_list:
                class_list = ["person", "car", "dog", "cat"]
            debug_info = "使用文本提示检测类别: " + ", ".join(class_list)
        
        # 设置文本提示 - 这是YOLOE的核心功能
        text_embeddings = model.get_text_pe(class_list)
        model.set_classes(class_list, text_embeddings)
        
        # 降低置信度阈值以增加检测几率
        actual_conf = min(conf_threshold, 0.15)
        
        # 执行预测
        results = model.predict(source=image, imgsz=image_size, conf=actual_conf, verbose=True)
        
        # 如果没有检测结果，返回调试信息
        if len(results[0].boxes) == 0:
            debug_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            y_pos = 30
            cv2.putText(debug_img, debug_info, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
            cv2.putText(debug_img, f"未检测到任何物体! 请尝试其他类别或降低置信度阈值", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for i, common_class in enumerate(COMMON_CLASSES[:6]):
                y_pos += 30
                cv2.putText(debug_img, f"- {common_class}", 
                           (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return debug_img[:, :, ::-1]
        
        # YOLOE自带的结果可视化包含边框和分割掩码
        # 使用plot()方法获取带有边框和掩码的完整可视化结果
        result_image = results[0].plot()
        
        # 添加调试信息
        cv2.putText(result_image, debug_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image[:, :, ::-1]
        
    except Exception as e:
        # 处理异常情况
        print(f"YOLOE错误: {str(e)}")
        error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, f"YOLOE错误: {str(e)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 提示可能的解决方案
        y_pos = 100
        cv2.putText(error_img, "可能的解决方案:", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(error_img, "1. 请确保网络连接正常，模型将自动下载", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(error_img, "2. 检查模型路径是否正确: " + model_path, 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(error_img, "3. 确保使用英文类别名称", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(error_img, "4. 尝试这些常见类别:", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for i, common_class in enumerate(COMMON_CLASSES[:5]):
            y_pos += 30
            cv2.putText(error_img, f"   - {common_class}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return error_img

# ==================== 应用入口 ====================
def app():
    # 示例图片路径
    test_image_path = os.path.abspath("img/test.png")
    example_samples = []
    segmentation_examples = []
    separate_seg_examples = []
    
    if os.path.exists(test_image_path):
        example_samples.append([test_image_path, "YOLOv11n", 640, 0.25])
        segmentation_examples.append([test_image_path, "YOLOev11m-pf", 640, 0.25])
        separate_seg_examples.append([test_image_path, "YOLOev11m", 640, 0.25, "person,car"])
    
    # 使用Soft主题构建界面
    with gr.Blocks(title="YOLOv11目标检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 YOLOv11 目标检测与分割系统")
        gr.Markdown("上传图像或视频进行目标检测和实例分割，支持参数调节和自定义类别")
        
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                input_type = gr.Radio(
                    choices=["图片", "视频", "分割", "单独分割"],
                    value="图片",
                    label="输入类型",
                    info="选择处理的媒体类型和任务"
                )
                
                # 各种输入组件
                with gr.Group(visible=True) as image_group:
                    image = gr.Image(label="输入图像", type="pil")
                
                with gr.Group(visible=False) as video_group:
                    video = gr.Video(label="输入视频")
                
                with gr.Group(visible=False) as segmentation_group:
                    segmentation_image = gr.Image(label="分割输入图像", type="pil")
                
                with gr.Group(visible=False) as separate_seg_group:
                    separate_seg_image = gr.Image(label="单独分割输入图像", type="pil")
                    classes = gr.Textbox(
                        label="检测类别",
                        placeholder="person,car,dog",
                        info="输入要检测的类别，用逗号分隔，必须使用英文逗号和英文名称"
                    )
                    gr.Markdown("""
                    ### 常用类别参考：
                    - 人物：person
                    - 交通工具：car, bicycle, motorcycle, bus, truck
                    - 动物：dog, cat, bird, horse, sheep, cow
                    - 物品：bottle, cup, chair, couch, bed, dining table
                    """)
                
                with gr.Accordion("模型参数", open=False):
                    with gr.Group(visible=True) as yolo_model_group:
                        yolo_model = gr.Dropdown(
                            label="YOLO模型选择",
                            choices=yolo_models,
                            value="YOLOv11n",
                            info="选择要使用的YOLOv11模型版本"
                        )
                    
                    with gr.Group(visible=False) as yoloe_pf_model_group:
                        yoloe_pf_model = gr.Dropdown(
                            label="分割模型选择",
                            choices=yoloe_pf_models,
                            value="YOLOev11m-pf",
                            info="选择要使用的YOLOE分割模型"
                        )
                    
                    with gr.Group(visible=False) as yoloe_seg_model_group:
                        yoloe_seg_model = gr.Dropdown(
                            label="单独分割模型选择",
                            choices=yoloe_seg_models,
                            value="YOLOev11m",
                            info="选择要使用的YOLOE单独分割模型"
                        )
                    
                    image_size = gr.Slider(
                        minimum=320, maximum=1280, value=640, step=32,
                        label="图像尺寸",
                        info="更大的尺寸通常能提高精度，但会降低速度"
                    )
                    
                    conf_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                        label="置信度阈值",
                        info="调整检测的置信度阈值，较低的值会增加检测数量但可能增加误报"
                    )
                
                submit_btn = gr.Button("🚀 开始检测", variant="primary")
            
            # 右侧结果展示
            with gr.Column(scale=2):
                output_image = gr.Image(label="检测结果", type="numpy", visible=True)
                output_video = gr.Video(label="检测结果", visible=False)
                output_segmentation = gr.Image(label="分割结果", type="numpy", visible=False)
                output_separate_seg = gr.Image(label="单独分割结果", type="numpy", visible=False)
        
        # 示例区块 - 图片检测
        if example_samples:
            with gr.Accordion("图片检测示例", open=False, visible=True) as image_examples_ui:
                gr.Examples(
                    examples=example_samples,
                    inputs=[image, yolo_model, image_size, conf_threshold],
                    outputs=output_image,
                    fn=lambda img, model, size, conf: yolo_inference(img, None, model, size, conf)[0],
                    cache_examples=True,
                    label="图片检测示例"
                )
        
        # 示例区块 - 分割
        if segmentation_examples:
            with gr.Accordion("分割示例", open=False, visible=False) as segmentation_examples_ui:
                gr.Examples(
                    examples=segmentation_examples,
                    inputs=[segmentation_image, yoloe_pf_model, image_size, conf_threshold],
                    outputs=output_segmentation,
                    fn=yoloe_segmentation,
                    cache_examples=True,
                    label="分割示例"
                )
        
        # 示例区块 - 单独分割
        if separate_seg_examples:
            with gr.Accordion("单独分割示例", open=False, visible=False) as separate_seg_examples_ui:
                gr.Examples(
                    examples=separate_seg_examples,
                    inputs=[separate_seg_image, yoloe_seg_model, image_size, conf_threshold, classes],
                    outputs=output_separate_seg,
                    fn=yoloe_separate_segmentation,
                    cache_examples=True,
                    label="单独分割示例"
                )
        
        # 交互逻辑
        def update_visibility(input_type):
            # 更新输入组
            image_group_visibility = input_type == "图片"
            video_group_visibility = input_type == "视频"
            segmentation_group_visibility = input_type == "分割"
            separate_seg_group_visibility = input_type == "单独分割"
            
            # 更新模型选择组
            yolo_model_visibility = input_type in ["图片", "视频"]
            yoloe_pf_model_visibility = input_type == "分割"
            yoloe_seg_model_visibility = input_type == "单独分割"
            
            # 更新输出显示
            output_image_visibility = input_type == "图片"
            output_video_visibility = input_type == "视频"
            output_segmentation_visibility = input_type == "分割"
            output_separate_seg_visibility = input_type == "单独分割"
            
            # 更新示例区域显示
            image_examples_visibility = input_type in ["图片", "视频"]
            segmentation_examples_visibility = input_type == "分割"
            separate_seg_examples_visibility = input_type == "单独分割"
            
            return (
                # 输入组可见性
                gr.update(visible=image_group_visibility),
                gr.update(visible=video_group_visibility),
                gr.update(visible=segmentation_group_visibility),
                gr.update(visible=separate_seg_group_visibility),
                
                # 模型选择组可见性
                gr.update(visible=yolo_model_visibility),
                gr.update(visible=yoloe_pf_model_visibility),
                gr.update(visible=yoloe_seg_model_visibility),
                
                # 输出显示可见性
                gr.update(visible=output_image_visibility),
                gr.update(visible=output_video_visibility),
                gr.update(visible=output_segmentation_visibility),
                gr.update(visible=output_separate_seg_visibility),
                
                # 示例区域可见性
                gr.update(visible=image_examples_visibility),
                gr.update(visible=segmentation_examples_visibility),
                gr.update(visible=separate_seg_examples_visibility),
            )

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[
                # 输入组
                image_group, video_group, segmentation_group, separate_seg_group,
                # 模型选择组
                yolo_model_group, yoloe_pf_model_group, yoloe_seg_model_group,
                # 输出显示
                output_image, output_video, output_segmentation, output_separate_seg,
                # 示例区域
                image_examples_ui, segmentation_examples_ui, separate_seg_examples_ui,
            ],
        )

        def run_inference(input_type, 
                         image, video, segmentation_image, separate_seg_image, classes,
                         yolo_model, yoloe_pf_model, yoloe_seg_model,
                         image_size, conf_threshold):
            if input_type == "图片":
                output_img, _ = yolo_inference(image, None, yolo_model, image_size, conf_threshold)
                return output_img, None, None, None
            elif input_type == "视频":
                _, output_vid = yolo_inference(None, video, yolo_model, image_size, conf_threshold)
                return None, output_vid, None, None
            elif input_type == "分割":
                seg_result = yoloe_segmentation(segmentation_image, yoloe_pf_model, image_size, conf_threshold)
                return None, None, seg_result, None
            elif input_type == "单独分割":
                separate_seg_result = yoloe_separate_segmentation(separate_seg_image, yoloe_seg_model, image_size, conf_threshold, classes)
                return None, None, None, separate_seg_result

        submit_btn.click(
            fn=run_inference,
            inputs=[
                input_type, 
                image, video, segmentation_image, separate_seg_image, classes,
                yolo_model, yoloe_pf_model, yoloe_seg_model,
                image_size, conf_threshold
            ],
            outputs=[output_image, output_video, output_segmentation, output_separate_seg],
        )
        
        return demo

# ==================== 启动应用 ====================
if __name__ == '__main__':
    demo = app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
