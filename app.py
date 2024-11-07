import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO  # 改为导入 YOLO 类

# 映射模型名称和模型文件路径
model_mapping = {
    "这里是训好的模型路径": r"E:\Desktop\redCard_train\weights\best.pt"  # 这里填写训好的模型路径
}

def yolov10_inference(image, video, model_name, image_size, conf_threshold):
    model_path = model_mapping[model_name]  # 从映射中获取模型路径
    model = YOLO(model_path)  # 使用本地路径加载模型，改为 YOLO 类
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov10_inference_for_examples(image, model_name, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_name, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="上传图片", visible=True)
                video = gr.Video(label="上传视频", visible=False)
                input_type = gr.Radio(
                    choices=["图片", "视频"],
                    value="图片",
                    label="输入类型",
                )
                model_name = gr.Dropdown(
                    label="选择模型",
                    choices=list(model_mapping.keys()),  # 界面显示模型名称
                    value="中文名字",  # 默认选择模型
                )
                image_size = gr.Slider(
                    label="图片尺寸",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="置信度阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="开始识别")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="结果", visible=True)
                output_video = gr.Video(label="结果", visible=False)

        def update_visibility(input_type):
            image_visibility = input_type == "图片"
            video_visibility = input_type == "视频"
            
            return (
                gr.update(visible=image_visibility),
                gr.update(visible=video_visibility),
                gr.update(visible=image_visibility),
                gr.update(visible=video_visibility),
            )

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_name, image_size, conf_threshold, input_type):
            if input_type == "图片":
                return yolov10_inference(image, None, model_name, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_name, image_size, conf_threshold)


        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_name, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[[
                r"top.JPG",  # 放例子图片
                "中文名字",  # 界面显示的模型名称
                640,
                0.25,
            ]],
            fn=yolov10_inference_for_examples,
            inputs=[image, model_name, image_size, conf_threshold],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    识别集群
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()
