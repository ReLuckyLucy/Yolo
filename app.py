import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO, YOLOE
import os
import numpy as np
import time
import json
from datetime import datetime

# 确保结果目录存在
os.makedirs("results", exist_ok=True)

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

def save_results_to_json(results, output_path, mode="detection"):
    """保存检测结果到JSON文件"""
    detection_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "detections": []
    }
    
    try:
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue
                
            boxes = result.boxes
            for i, box in enumerate(boxes):
                try:
                    # 安全地获取类别索引和名称
                    class_idx = int(box.cls[0]) if len(box.cls) > 0 else 0
                    class_name = result.names[class_idx] if class_idx in result.names else "unknown"
                    
                    # 安全地获取置信度
                    confidence = float(box.conf[0]) if len(box.conf) > 0 else 0.0
                    
                    # 安全地获取边界框坐标
                    if hasattr(box, 'xyxy') and box.xyxy is not None and len(box.xyxy) > 0:
                        xyxy = box.xyxy[0]
                        x1 = float(xyxy[0]) if len(xyxy) > 0 else 0.0
                        y1 = float(xyxy[1]) if len(xyxy) > 1 else 0.0
                        x2 = float(xyxy[2]) if len(xyxy) > 2 else 0.0
                        y2 = float(xyxy[3]) if len(xyxy) > 3 else 0.0
                    else:
                        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                    
                    detection = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    }
                    
                    # 安全地添加分割掩码信息
                    if hasattr(result, "masks") and result.masks is not None and i < len(result.masks):
                        try:
                            mask = result.masks[i]
                            mask_data = {}
                            
                            # 添加形状信息
                            if hasattr(mask, 'shape'):
                                # 将numpy shape转为普通列表
                                shape = [int(s) for s in mask.shape]
                                mask_data["shape"] = shape
                            
                            # 添加像素坐标（如果存在）
                            if hasattr(mask, 'xy') and mask.xy is not None and len(mask.xy) > i:
                                xy_points = mask.xy[i]
                                if xy_points is not None:
                                    # 将numpy数组转为普通列表，确保所有值是原生Python类型
                                    mask_data["xy"] = [[float(x), float(y)] for x, y in xy_points.tolist()]
                            
                            # 添加归一化坐标（如果存在）
                            if hasattr(mask, 'xyn') and mask.xyn is not None and len(mask.xyn) > i:
                                xyn_points = mask.xyn[i]
                                if xyn_points is not None:
                                    # 将numpy数组转为普通列表，确保所有值是原生Python类型
                                    mask_data["xyn"] = [[float(x), float(y)] for x, y in xyn_points.tolist()]
                            
                            if mask_data:  # 只有当有掩码数据时才添加
                                detection["mask"] = mask_data
                        except Exception as mask_error:
                            print(f"处理掩码数据时出错: {str(mask_error)}")
                            # 在出错时添加简化的掩码信息
                            detection["mask"] = {"error": str(mask_error)}
                    
                    detection_data["detections"].append(detection)
                except Exception as box_error:
                    print(f"处理检测框时出错: {str(box_error)}")
                    # 记录错误但继续处理其他检测框
    except Exception as e:
        print(f"处理检测结果时出错: {str(e)}")
        # 添加错误信息到JSON
        detection_data["error"] = str(e)
    
    # 确保输出目录存在
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as dir_error:
        print(f"创建目录时出错: {str(dir_error)}")
        # 使用临时目录作为备选
        output_path = os.path.join(tempfile.gettempdir(), os.path.basename(output_path))
    
    # 保存JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as json_error:
        print(f"保存JSON文件时出错: {str(json_error)}")
        # 创建一个简化的错误JSON
        error_path = os.path.join(tempfile.gettempdir(), f"error_{int(time.time())}.json")
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(json_error)}, f)
        return error_path

def yolo_inference(image, video, model_name, image_size, conf_threshold):
    model_path = model_mapping[model_name]
    try:
        model = YOLO(model_path)
        
        if image is not None:
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
            if len(results) == 0:
                # 处理空结果
                empty_image = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(empty_image, "未检测到任何物体", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 保存空结果
                json_path = os.path.join("results", f"detection_empty_{int(time.time())}.json")
                save_results_to_json([], json_path, "image_detection")
                
                return empty_image, None, json_path
            
            annotated_image = results[0].plot()
            
            # 保存检测结果到JSON
            json_path = os.path.join("results", f"detection_{int(time.time())}.json")
            save_results_to_json(results, json_path, "image_detection")
            
            return annotated_image[:, :, ::-1], None, json_path
        elif video is not None:
            try:
                video_path = tempfile.mktemp(suffix=".mp4")
                with open(video_path, "wb") as f:
                    with open(video, "rb") as g:
                        f.write(g.read())

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise Exception(f"无法打开视频: {video_path}")
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_video_path = tempfile.mktemp(suffix=".mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                
                all_results = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
                    annotated_frame = frame_results[0].plot()
                    out.write(annotated_frame)
                    
                    # 只保存有检测结果的帧
                    if len(frame_results[0].boxes) > 0:
                        all_results.extend(frame_results)
                    frame_count += 1

                cap.release()
                out.release()

                # 保存视频检测结果到JSON
                json_path = os.path.join("results", f"video_detection_{int(time.time())}.json")
                save_results_to_json(all_results, json_path, "video_detection")
                
                return None, output_video_path, json_path
            except Exception as video_error:
                # 处理视频处理错误
                print(f"视频处理错误: {str(video_error)}")
                error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                cv2.putText(error_img, f"视频处理错误: {str(video_error)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 保存错误信息到JSON
                json_path = os.path.join("results", f"video_error_{int(time.time())}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({"error": str(video_error)}, f, ensure_ascii=False, indent=2)
                
                return error_img, None, json_path
    except Exception as e:
        # 处理模型加载或预测错误
        print(f"YOLO错误: {str(e)}")
        error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, f"YOLO错误: {str(e)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存错误信息到JSON
        json_path = os.path.join("results", f"error_{int(time.time())}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f, ensure_ascii=False, indent=2)
        
        return error_img, None, json_path

def yoloe_segmentation(image, model_name, image_size, conf_threshold, classes=None):
    """普通分割模式 - 使用YOLOev11-pf模型"""
    model_path = model_mapping[model_name]
    
    try:
        model = YOLOE(model_path)
        
        # 执行预测
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            # 处理空结果
            empty_image = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(empty_image, "未检测到任何物体", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存空结果
            json_path = os.path.join("results", f"segmentation_empty_{int(time.time())}.json")
            save_results_to_json([], json_path, "segmentation")
            
            return empty_image, json_path
            
        annotated_image = results[0].plot()
        
        # 保存分割结果到JSON
        json_path = os.path.join("results", f"segmentation_{int(time.time())}.json")
        save_results_to_json(results, json_path, "segmentation")
        
        return annotated_image[:, :, ::-1], json_path
    
    except Exception as e:
        # 处理错误
        print(f"分割错误: {str(e)}")
        error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, f"分割错误: {str(e)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存错误信息到JSON
        json_path = os.path.join("results", f"segmentation_error_{int(time.time())}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f, ensure_ascii=False, indent=2)
        
        return error_img, json_path

def yoloe_separate_segmentation(image, model_name, image_size, conf_threshold, classes):
    """单独分割模式 - 使用YOLOev11模型，使用文本提示进行开放词汇检测和分割"""
    model_path = model_mapping[model_name]
    
    try:
        print(f"正在加载YOLOE模型: {model_name}，路径: {model_path}")
        model = YOLOE(model_path)
        print(f"YOLOE模型加载成功")
        
        if not classes or not classes.strip():
            class_list = ["person", "car", "dog", "cat", "chair", "table", "bottle"]
            debug_info = "使用默认类别: " + ", ".join(class_list)
        else:
            class_list = [c.strip() for c in classes.split(',') if c.strip()]
            if not class_list:
                class_list = ["person", "car", "dog", "cat"]
            debug_info = "使用文本提示检测类别: " + ", ".join(class_list)
        
        text_embeddings = model.get_text_pe(class_list)
        model.set_classes(class_list, text_embeddings)
        
        actual_conf = min(conf_threshold, 0.15)
        results = model.predict(source=image, imgsz=image_size, conf=actual_conf, verbose=True)
        
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
            
            # 保存空结果到JSON
            json_path = os.path.join("results", f"separate_segmentation_{int(time.time())}.json")
            save_results_to_json([], json_path, "separate_segmentation")
            
            return debug_img[:, :, ::-1], json_path
        
        result_image = results[0].plot()
        cv2.putText(result_image, debug_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存分割结果到JSON
        json_path = os.path.join("results", f"separate_segmentation_{int(time.time())}.json")
        save_results_to_json(results, json_path, "separate_segmentation")
        
        return result_image[:, :, ::-1], json_path
        
    except Exception as e:
        print(f"YOLOE错误: {str(e)}")
        error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, f"YOLOE错误: {str(e)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
        
        # 保存错误信息到JSON
        json_path = os.path.join("results", f"error_{int(time.time())}.json")
        error_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "separate_segmentation",
            "error": str(e),
            "model_path": model_path
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        return error_img, json_path

# ==================== 应用入口 ====================
def app():
    # 确保模型和结果目录存在
    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("img", exist_ok=True)
    
    # 示例图片路径
    test_image_path = os.path.abspath("img/test.png")
    example_samples = []
    segmentation_examples = []
    separate_seg_examples = []
    
    # 只在示例图片存在时添加示例
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
                json_output = gr.File(label="检测结果JSON文件")
        
        # 示例区块 - 图片检测
        if example_samples:
            with gr.Accordion("图片检测示例", open=False, visible=True) as image_examples_ui:
                gr.Examples(
                    examples=example_samples,
                    inputs=[image, yolo_model, image_size, conf_threshold],
                    outputs=output_image,
                    fn=lambda img, model, size, conf: (
                        yolo_inference(img, None, model, size, conf)[0] 
                        if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
                    ),
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
                    fn=lambda img, model, size, conf: (
                        yoloe_segmentation(img, model, size, conf)[0] 
                        if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
                    ),
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
                    fn=lambda img, model, size, conf, classes: (
                        yoloe_separate_segmentation(img, model, size, conf, classes)[0] 
                        if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
                    ),
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
            try:
                if input_type == "图片":
                    if image is None:
                        # 处理空输入
                        empty_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(empty_img, "请上传图片", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        return empty_img, None, None, None, None
                        
                    output_img, _, json_path = yolo_inference(image, None, yolo_model, image_size, conf_threshold)
                    return output_img, None, None, None, json_path
                elif input_type == "视频":
                    if video is None:
                        # 处理空输入
                        empty_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(empty_img, "请上传视频", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        return empty_img, None, None, None, None
                        
                    _, output_vid, json_path = yolo_inference(None, video, yolo_model, image_size, conf_threshold)
                    return None, output_vid, None, None, json_path
                elif input_type == "分割":
                    if segmentation_image is None:
                        # 处理空输入
                        empty_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(empty_img, "请上传图片", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        return None, None, empty_img, None, None
                        
                    seg_result, json_path = yoloe_segmentation(segmentation_image, yoloe_pf_model, image_size, conf_threshold)
                    return None, None, seg_result, None, json_path
                elif input_type == "单独分割":
                    if separate_seg_image is None:
                        # 处理空输入
                        empty_img = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.putText(empty_img, "请上传图片", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        return None, None, None, empty_img, None
                        
                    separate_seg_result, json_path = yoloe_separate_segmentation(separate_seg_image, yoloe_seg_model, image_size, conf_threshold, classes)
                    return None, None, None, separate_seg_result, json_path
            except Exception as e:
                # 处理运行时错误
                print(f"运行错误: {str(e)}")
                error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                cv2.putText(error_img, f"运行错误: {str(e)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 根据当前模式决定返回哪个输出
                if input_type == "图片":
                    return error_img, None, None, None, None
                elif input_type == "视频":
                    return error_img, None, None, None, None
                elif input_type == "分割":
                    return None, None, error_img, None, None
                elif input_type == "单独分割":
                    return None, None, None, error_img, None

        submit_btn.click(
            fn=run_inference,
            inputs=[
                input_type, 
                image, video, segmentation_image, separate_seg_image, classes,
                yolo_model, yoloe_pf_model, yoloe_seg_model,
                image_size, conf_threshold
            ],
            outputs=[output_image, output_video, output_segmentation, output_separate_seg, json_output],
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
