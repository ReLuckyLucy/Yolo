import cv2
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('model/yolo11n.pt')

# 进行屏幕检测（全屏捕获）
results = model(
    source='screen',
    stream=True,        # 启用实时流模式
    show=False,         # 关闭默认显示（我们将自定义显示）
    conf=0.5,          # 置信度阈值
    verbose=False       # 关闭冗余日志
)

# 创建可调节窗口
cv2.namedWindow('YOLO Screen Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Screen Detection', 1280, 720)  # 初始窗口大小

# 逐帧处理并显示
for r in results:
    # 获取带检测框的渲染图像（BGR格式）
    frame = r.plot()  # 自动绘制boxes/labels
    
    # 显示处理后的帧
    cv2.imshow('YOLO Screen Detection', frame)
    
    # 按ESC退出循环
    if cv2.waitKey(1) == 27:
        break

# 清理资源
cv2.destroyAllWindows()