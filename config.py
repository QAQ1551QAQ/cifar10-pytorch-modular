# config.py
"""
项目全局超参数配置文件：统一管理所有可调参数，拒绝硬编码
"""
SEED = 42                 # 核心：固定随机种子，确保实验可复现
BATCH_SIZE = 128          # 批次大小（可根据显存大小调整）
NUM_WORKERS = 2           # 数据加载子进程数（笔记本建议设 2~4）
LR = 1e-3                 # 初始学习率 (Learning Rate)
EPOCHS = 5               # 训练总轮次

MODEL_NAME = "resnet18"   # 🎛️ 可以在这里自由切换模型："resnet18" 或 "mobilenet"

CHECKPOINT_DIR = "./checkpoints"  # 权重保存路径
DATA_DIR = "./data"       # 数据集存放路径