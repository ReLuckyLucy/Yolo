import numpy as np
import cv2

class DistanceEstimator:
    """
    单目视觉距离估计器
    使用已知物体大小和相机参数来估计物体到相机的距离
    """
    
    # 常见物体的平均高度（单位：米）
    OBJECT_HEIGHTS = {
        'person': 1.7,        # 成年人平均身高
        'car': 1.5,          # 轿车高度
        'truck': 3.5,        # 卡车高度
        'bus': 3.0,          # 公交车高度
        'bicycle': 1.1,      # 自行车高度
        'motorcycle': 1.2,   # 摩托车高度
        'cat': 0.3,          # 猫高度
        'dog': 0.6,          # 狗高度
        'chair': 0.9,        # 椅子高度
        'bottle': 0.25,      # 瓶子高度
        'default': 1.0       # 默认物体高度
    }
    
    def __init__(self, focal_length=None, sensor_height_mm=None, image_height_px=1080):
        """
        初始化距离估计器
        
        参数:
            focal_length: 相机焦距（像素），如果为None则使用默认值
            sensor_height_mm: 相机传感器高度（毫米），用于计算比例
            image_height_px: 图像高度（像素）
        """
        self.image_height_px = image_height_px
        
        # 如果没有提供焦距，使用常见值（以像素为单位）
        # 对于1080p图像，典型的焦距值在1000-3000像素之间
        self.focal_length = focal_length if focal_length is not None else 1500
        
        # 如果没有提供传感器高度，使用常见值（毫米）
        # 典型智能手机传感器的物理高度约为3-5mm
        self.sensor_height_mm = sensor_height_mm if sensor_height_mm is not None else 4.2
    
    def get_object_height(self, class_name):
        """获取已知物体的高度"""
        return self.OBJECT_HEIGHTS.get(class_name.lower(), self.OBJECT_HEIGHTS['default'])
    
    def estimate_distance(self, bbox, class_name, image_height=None):
        """
        估计物体到相机的距离
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2]
            class_name: 物体类别名称
            image_height: 图像高度（像素），如果为None则使用初始化时的值
            
        返回:
            distance: 估计的距离（米）
        """
        if image_height is None:
            image_height = self.image_height_px
            
        # 获取物体高度（米）
        object_height = self.get_object_height(class_name)
        
        # 计算边界框高度（像素）
        bbox_height = bbox[3] - bbox[1]
        
        # 确保边界框高度有效
        if bbox_height < 5:  # 最小高度限制
            return float('inf')
            
        try:
            # 使用相似三角形计算距离: distance = (object_height * focal_length) / bbox_height
            # 这里假设焦距是以像素为单位的
            distance = (object_height * self.focal_length) / bbox_height
            
            # 添加一些限制，避免不合理的距离值
            min_distance = 0.1  # 最小距离0.1米
            max_distance = 100.0  # 最大距离100米
            
            # 确保距离在合理范围内
            if distance < min_distance:
                return min_distance
            elif distance > max_distance:
                return max_distance
                
            return distance
            
        except Exception as e:
            print(f"距离计算错误: {e}")
            return float('inf')
    
    def draw_distance(self, image, bbox, class_name, distance, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制距离信息
        
        参数:
            image: 输入图像
            bbox: 边界框 [x1, y1, x2, y2]
            class_name: 物体类别
            distance: 距离（米）
            color: 文本颜色 (B, G, R)
            thickness: 文本线宽
            
        返回:
            绘制了距离信息的图像
        """
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备文本
        text = f"{class_name}: {distance:.2f}m"
        
        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - 25), (x1 + text_width + 5, y1), color, -1)
        
        # 绘制文本
        cv2.putText(
            image, text, (x1, y1 - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        return image
