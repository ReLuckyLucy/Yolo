<div align="center">
 <img alt="logo" height="200px" src="img\logo.png">
</div>

<h1 align="center">Yolo框架整合</h1>

<div align="center">
 <img alt="logo"  src="img\yolo.png">
</div>


> 现已加入yoloe模块

## 💫环境下载
### pip
使用 `pip` 安装 Ultralytics，请执行以下命令：
```bash
pip install ultralytics
```

或者，您可以直接从 GitHub 安装最新的开发版本：
```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```
### conda 
使用 `conda` 安装 Ultralytics YOLO 
```bash
conda install -c conda-forge ultralytics
```

**此方法是 pip 的绝佳替代方案，可确保与环境中的：ultralytics pytorch pytorch-cuda与其他包兼容。对于 CUDA 环境，最好安装，其能同时解决任何冲突**

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

### 源码编译
克隆 Ultralytics 存储库并设置开发环境
```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```
 <br />

## 🦄ReLucy运行代码
### 运行训练代码
```bash
cd train 
python train.py
```

### 运行有comet可视化的训练代码
```bash
python train_comet.py
```

### 验证可视化界面移植
>### 超值体验
可视化界面由python库`gradio`实现
```bash
python app.py
```

## ✍️训练
使用`CLI`

从 YAML 文件构建新模型并从头开始训练
```bash
yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640
```

从预训练的 *.pt 模型开始训练
```bash
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
```

从 YAML 文件构建新模型，将预训练权重转移到新模型上并开始训练
```bash
yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
```

使用`python`
```python
# 导入Ultralytics YOLO库
from ultralytics import YOLO

# 加载/构建模型的三种方式：

# 方式1：从YAML配置文件构建新模型（初始化随机权重）
# 适用场景：需要自定义模型结构时使用
# 注意：yolo11n.yaml文件需包含完整的模型架构定义
model = YOLO("yolo11n.yaml")  # 📌 从YAML文件构建全新模型

# 方式2：直接加载预训练模型（.pt文件包含结构与权重）
# 推荐方式：利用迁移学习加速训练收敛
# 注意：.pt文件需与当前YOLO版本兼容
model = YOLO("yolo11n.pt")  # 🚀 加载预训练模型（含权重）

# 方式3：从YAML构建结构后加载预训练权重
# 适用场景：修改了YAML结构但仍想使用预训练权重
# 注意：YAML定义的网络结构需与.pt权重兼容
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 🔄 架构迁移学习

# 训练模型
results = model.train(
    data="coco8.yaml",    # 📂 数据集配置文件路径
    epochs=100,           # 🔄 训练总轮次（典型值100-300）
    imgsz=640,            # 🖼️ 输入图像尺寸（像素）
    
    # 可选常用参数（示例）：
    # batch=16,           # 🧠 批次大小（根据显存调整）
    # lr0=0.01,           # 📉 初始学习率
    # device=0,            # ⚡ 使用GPU设备（0表示第一块GPU）
    # pretrained=True,     # 🎯 是否使用预训练权重
    # cache=True,          # 🚀 缓存数据集加速训练
    # resume=True,         # ⏯️ 恢复中断的训练
)
```
## ⚠️警告
1.一般来说，报错多数是因为路径问题

2.若出现训练慢/使用了 DEVICE = "0" 参数后报错，一般来说是由于pytorch没有下载好相对应的版本,可以运行 pytorch_test.py 进行测试
```
python pytorch_test.py
```
 <br />

## 🎃验证
在训练后验证 YOLO 模型。在此模式下，将在验证集上评估模型，以测量其准确性和泛化性能。此模式可用于调整模型的超参数以提高其性能
```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate on training data
model.val()
```

若使用其他验证集进行评估，则为
```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate on separate data
model.val(data="path/to/separate/data.yaml")
```
<br/>

## 🔮预测
预测用于使用经过训练的 YOLO 模型对新图像或视频进行预测。在此模式下，模型是从检查点文件加载的，用户可以提供图像或视频来执行推理。该模型预测输入图像或视频中对象的类别和位置

> ## 这里的代码都可以在verification文件夹内找到

```python
from ultralytics import YOLO
# 加载预训练模型
model = YOLO('model/yolo11n.pt')

# 进行视频检测（会自动保存结果）
results = model(
    source = 'img/test.png',
    show = True,    # 实时显示检测窗口
    save = True,    # 保存检测结果视频
)
```
检测完后，图片会存放到runs/detect/predict


### 检测视频
> 在yolo中，会将视频裁剪成一帧一帧，进而逐帧去学习
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('model/yolo11n.pt')

# 进行视频检测（会自动保存结果）
results = model(
    source = 'img/test.mp4',
    show = True,    # 实时显示检测窗口
    save = True,    # 保存检测结果视频
)
```

### 检测屏幕
需要安装库 mss
```bash
pip install mss
```
> 这时候我们会发现会有警告
>
> “WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
> errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.”
>
> 本质：进行连续屏幕检测时，若不启用 `stream=True` 参数，所有检测结果会直接存储在内存中。对于长时间运行的屏幕流或高分辨率输入，会导致内存持续增长直至溢出

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('model/yolo11n.pt')

# 进行视频检测（会自动保存结果）
results = model(
    source = 'screen',
    stream = True
)
# 逐帧处理结果
for r in results:
    boxes = r.boxes.xyxy  # 获取当前帧的边界框坐标（Tensor格式）
    cls_probs = r.probs    # 分类任务的概率（若适用）
    if hasattr(r, 'masks'):
        masks = r.masks    # 实例分割的掩膜（若适用）
```

#### 检测电脑摄像头

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('model/yolo11n.pt')

# 进行视频检测（会自动保存结果）
results = model(
    source = 0,
)
```

<br/>



<br/>

## 🏛️模型简介
YOLO11 检测、分割和姿态模型在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 数据集上进行预训练，这些模型可在此处获得，此外还有在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 数据集上预训练的 YOLO11 分类 模型。


所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时自动从最新的 Ultralytics [发布](https://github.com/ultralytics/assets/releases)下载。

<details open><summary>检测 (COCO)</summary>

请参阅 [检测文档](https://docs.ultralytics.com/tasks/detect/) 以获取使用这些在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 数据集上训练的模型的示例，其中包含 80 个预训练类别。

| 模型                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>T4 TensorRT10<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | ------------------- | -------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                 | 39.5                 | 56.1 ± 0.8                    | 1.5 ± 0.0                          | 2.6              | 6.5               |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                 | 47.0                 | 90.0 ± 1.2                    | 2.5 ± 0.0                          | 9.4              | 21.5              |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                 | 51.5                 | 183.2 ± 2.0                   | 4.7 ± 0.1                          | 20.1             | 68.0              |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                 | 53.4                 | 238.6 ± 1.4                   | 6.2 ± 0.1                          | 25.3             | 86.9              |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                 | 54.7                 | 462.8 ± 6.7                   | 11.3 ± 0.2                         | 56.9             | 194.9             |

- **mAP<sup>val</sup>** 值针对单模型单尺度在 [COCO val2017](https://cocodataset.org/) 数据集上进行。 <br>复制命令 `yolo val detect data=coco.yaml device=0`
- **速度**在使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例的 COCO 验证图像上平均。 <br>复制命令 `yolo val detect data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>分割 (COCO)</summary>

请参阅 [分割文档](https://docs.ultralytics.com/tasks/segment/) 以获取使用这些在 [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/) 数据集上训练的模型的示例，其中包含 80 个预训练类别。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>T4 TensorRT10<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | ------------------- | -------------------- | --------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                 | 38.9                 | 32.0                  | 65.9 ± 1.1                    | 1.8 ± 0.0                          | 2.9              | 10.4              |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                 | 46.6                 | 37.8                  | 117.6 ± 4.9                   | 2.9 ± 0.0                          | 10.1             | 35.5              |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                 | 51.5                 | 41.5                  | 281.6 ± 1.2                   | 6.3 ± 0.1                          | 22.4             | 123.3             |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                 | 53.4                 | 42.9                  | 344.2 ± 3.2                   | 7.8 ± 0.2                          | 27.6             | 142.2             |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                 | 54.7                 | 43.8                  | 664.5 ± 3.2                   | 15.8 ± 0.7                         | 62.1             | 319.0             |

- **mAP<sup>val</sup>** 值针对单模型单尺度在 [COCO val2017](https://cocodataset.org/) 数据集上进行。 <br>复制命令 `yolo val segment data=coco-seg.yaml device=0`
- **速度**在使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例的 COCO 验证图像上平均。 <br>复制命令 `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu`

</details>

<details><summary>分类 (ImageNet)</summary>

请参阅 [分类文档](https://docs.ultralytics.com/tasks/classify/) 以获取使用这些在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 数据集上训练的模型的示例，其中包含 1000 个预训练类别。

| 模型                                                                                         | 尺寸<br><sup>(像素) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>T4 TensorRT10<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | ------------------- | ---------------- | ---------------- | ----------------------------- | ---------------------------------- | ---------------- | ------------------------ |
| [YOLO11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) | 224                 | 70.0             | 89.4             | 5.0 ± 0.3                     | 1.1 ± 0.0                          | 1.6              | 3.3                      |
| [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) | 224                 | 75.4             | 92.7             | 7.9 ± 0.2                     | 1.3 ± 0.0                          | 5.5              | 12.1                     |
| [YOLO11m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt) | 224                 | 77.3             | 93.9             | 17.2 ± 0.4                    | 2.0 ± 0.0                          | 10.4             | 39.3                     |
| [YOLO11l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-cls.pt) | 224                 | 78.3             | 94.3             | 23.2 ± 0.3                    | 2.8 ± 0.0                          | 12.9             | 49.4                     |
| [YOLO11x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt) | 224                 | 79.5             | 94.9             | 41.4 ± 0.9                    | 3.8 ± 0.0                          | 28.4             | 110.4                    |

- **acc** 值为在 [ImageNet](https://www.image-net.org/) 数据集验证集上的模型准确率。 <br>复制命令 `yolo val classify data=path/to/ImageNet device=0`
- **速度**在使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例的 ImageNet 验证图像上平均。 <br>复制命令 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

<details><summary>姿态 (COCO)</summary>

请参阅 [姿态文档](https://docs.ultralytics.com/tasks/pose/) 以获取使用这些在 [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/) 数据集上训练的模型的示例，其中包含 1 个预训练类别（人）。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>T4 TensorRT10<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | ------------------- | --------------------- | ------------------ | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                | 78.4                  | 117.6 ± 0.8        | 4.4 ± 0.0                     | 2.7                                | 17.2             |
| [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                | 79.5                  | 219.4 ± 4.0        | 5.1 ± 0.0                     | 9.7                                | 57.5             |
| [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                | 80.9                  | 562.8 ± 2.9        | 10.1 ± 0.4                    | 20.9                               | 183.5            |
| [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                | 81.0                  | 712.5 ± 5.0        | 13.5 ± 0.6                    | 26.2                               | 232.0            |
| [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                | 81.3                  | 1408.6 ± 7.7       | 28.6 ± 1.0                    | 58.8                               | 520.2            |

- **mAP<sup>val</sup>** 值针对单模型单尺度在 [COCO Keypoints val2017](https://cocodataset.org/) 数据集上进行。 <br>复制命令 `yolo val pose data=coco-pose.yaml device=0`
- **速度**在使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例的 COCO 验证图像上平均。 <br>复制命令 `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

</details>

<br/>

## 📋︎使用Comet ML进行可视化面板

<div align="center">
 <img alt="comet"  src="img\comet.png">
</div>

记录关键训练细节（如参数、指标、图像预测和模型检查点）在机器学习中至关重要，它可以保持项目的透明度、进度的可衡量性和结果的可重复性。

Ultralytics YOLO11 与 Comet ML 无缝集成，可有效捕获和优化 YOLO11 对象检测模型训练过程的各个方面。在本指南中，我们将介绍安装过程、Comet ML 设置、实时洞察、自定义日志记录和离线使用，确保您的 YOLO11 培训得到全面记录和微调，以获得出色的结果。

### 下载
```
pip install ultralytics comet_ml torch torchvision
```
安装所需的软件包后，您需要注册、获取 [Comet API Key](https://www.comet.com/signup) 密钥并对其进行配置。

```
export COMET_API_KEY=<Your API Key>
```
> 警告，export为Linux的命令，要在window下运行，有两种选择
> + 临时设置 API 密钥
> + 永久设置 API 密钥
### 在 Windows 中设置 `COMET_API_KEY`

1. **临时设置 API 密钥**:
   在命令行中使用 `set` 命令临时设置环境变量：
   ```bash
   set COMET_API_KEY=BV7SlLzug7TSvVqv4PMmFNpCT
   ```
   这会将 `COMET_API_KEY` 设置为当前会话中的环境变量，但关闭命令行窗口后会失效。

2. **永久设置 API 密钥**:
   如果希望永久保存 API 密钥，您可以通过以下步骤：
   - **打开环境变量设置**:
     1. 右键点击“此电脑”（或“我的电脑”）图标，选择“属性”。
     2. 点击“高级系统设置”。
     3. 在“系统属性”窗口中，点击“环境变量”。
   - **添加新的系统环境变量**:
     1. 在“环境变量”窗口中，点击“系统变量”区域的“新建”按钮。
     2. 设置变量名为 `COMET_API_KEY`，并将变量值设置为您的 API 密钥 `********************`。
     3. 点击“确定”保存设置。

3. **验证 API 密钥是否生效**:
   在命令行中运行以下命令来检查 Comet ML 是否能够成功识别您的 API 密钥：
   ```bash
   comet-cli check
   ```
   如果 API 密钥有效，您应该看到相关的确认信息。

### 使用 Comet ML 登录

一旦 API 密钥设置完成，您就可以使用 Comet ML 提供的命令行工具进行登录。例如：

```bash
comet login
```

它会要求您输入 API 密钥，如果环境变量已经配置正确，应该不需要再次输入。
 
 <br />