# src/models/model.py
import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2

class CIFAR10ResNet(nn.Module):
    """
    针对 32x32 小尺寸图片进行轻量化微调的 ResNet-18 模型
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 加载官方基础模型（不使用 ImageNet 预训练权重，因为尺寸和分布差异较大）
        self.model = resnet18(weights=None)
        
        # 核心微调 1：修改首层卷积，适配 32x32 输入（缩小 kernel，去掉 stride=2）
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除最大池化层，防止小图特征过早丢失
        
        # 核心微调 2：修改最终分类全连接层，输出类别数设为 10
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class CIFAR10MobileNet(nn.Module):
    """针对 32x32 小尺寸图片微调的 MobileNetV2（更省显存、轻量）"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = mobilenet_v2(weights=None)
        # MobileNetV2 的首层特征提取器
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 🛠️ 模型工厂函数（核心扩展点）
# ==========================================
def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    模型工厂：根据字符串名字动态创建并返回对应的模型实例
    """
    model_name = model_name.lower()
    if model_name == "resnet18":
        print("--> [Model Factory] 正在加载模型: CIFAR10ResNet (ResNet-18)")
        return CIFAR10ResNet(num_classes=num_classes)
    elif model_name == "mobilenet":
        print("--> [Model Factory] 正在加载模型: CIFAR10MobileNet (MobileNetV2)")
        return CIFAR10MobileNet(num_classes=num_classes)
    else:
        raise ValueError(f"❌ 不支持的模型名称: '{model_name}'。可选值: ['resnet18', 'mobilenet']")