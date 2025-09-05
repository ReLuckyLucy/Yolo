from ultralytics import YOLO
# 加载预训练模型
model = YOLO('model\yolo11n.pt')

# 进行视频检测（会自动保存结果）
results = model(
    source = 'img/test.png',
    show = True,    # 实时显示检测窗口
    save = True,    # 保存检测结果视频
)