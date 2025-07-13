import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import json
import tempfile

def save_results_to_json(results, output_path):
    detection_data = []
    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue
        boxes = result.boxes
        for i, box in enumerate(boxes):
            try:
                class_idx = int(box.cls[0]) if len(box.cls) > 0 else 0
                class_name = result.names[class_idx] if class_idx in result.names else "unknown"
                confidence = float(box.conf[0]) if len(box.conf) > 0 else 0.0
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
                detection_data.append(detection)
            except Exception as box_error:
                continue
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        return None

model_mapping = {
    "YOLOv11n": "model/yolo11n.pt",
    "YOLOv11m": "model/yolo11m.pt",
}
yolo_models = list(model_mapping.keys())

def yolo_video_inference(video, model_name, conf_threshold):
    model_path = model_mapping[model_name]
    try:
        model = YOLO(model_path)
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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_results = model.predict(source=frame, conf=conf_threshold)
            annotated_frame = frame_results[0].plot()
            out.write(annotated_frame)
            if len(frame_results[0].boxes) > 0:
                all_results.extend(frame_results)
        cap.release()
        out.release()
        json_path = os.path.join("results", f"video_detection_{int(time.time())}.json")
        save_results_to_json(all_results, json_path)
        return output_video_path, json_path
    except Exception as e:
        return None, None

def app():
    with gr.Blocks(title="YOLOv11目标检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 YOLOv11 视频目标检测系统")
        gr.Markdown("上传视频进行目标检测，支持模型选择和置信度调节")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="输入视频")
                yolo_model = gr.Dropdown(
                    label="YOLO模型选择",
                    choices=yolo_models,
                    value=yolo_models[0],
                    info="选择要使用的YOLOv11模型版本"
                )
                conf_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                    label="置信度阈值",
                    info="调整检测的置信度阈值，较低的值会增加检测数量但可能增加误报"
                )
                # 单独一行最大宽度的开始识别按钮
                start_btn = gr.Button("🚀 开始识别", variant="primary", scale=2)
                # 下一行并排显示清除和终止
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清除", variant="secondary")
                    stop_btn = gr.Button("⏹️ 终止", variant="stop")
            with gr.Column(scale=2):
                output_video = gr.Video(label="检测结果")
                json_output = gr.File(label="检测结果JSON文件")
        def run_inference(video, yolo_model, conf_threshold):
            if video is None:
                return None, None
            return yolo_video_inference(video, yolo_model, conf_threshold)
        start_btn.click(
            fn=run_inference,
            inputs=[video, yolo_model, conf_threshold],
            outputs=[output_video, json_output],
        )
        def clear_all():
            return None, None
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[output_video, json_output]
        )
        def stop_all():
            gr.Info("已终止推理任务。请刷新页面或重新上传视频后再试。")
            return None, None
        stop_btn.click(
            fn=stop_all,
            inputs=[],
            outputs=[output_video, json_output]
        )
        return demo

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    demo = app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,
        share=False,
    ) 