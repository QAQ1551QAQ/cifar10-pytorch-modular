# 🚀 PyTorch Modular CIFAR-10 Framework

> 一个规范化、可复现、易维护的 PyTorch 深度学习工程化实战模板，以经典的 CIFAR-10 图像分类任务为例，专为高效实验与项目迭代设计。

---

## ✨ 核心工程亮点

1. **结构清晰的模块化设计**：彻底告别“单文件巨型脚本”，采用标准的 `src/` 分层架构，各模块职责分明。
2. **绝对可复现（Reproducibility）**：内置全局随机种子固定函数（涵盖 Python、NumPy、PyTorch 及 CUDA/MPS 确定性配置）。
3. **双文件存档机制（支持断点续训）**：
   - `last_checkpoint.pth`：每轮无条件覆盖保存，记录完整的优化器状态、轮次与历史最佳分数，支持意外中断后**无缝断点续训（Resume Training）**。
   - `best_model.pth`：仅在验证准确率打破历史记录时保存模型权重，专供后续部署测试。
4. **灵活的模型工厂模式（Model Factory）**：通过统一的 `get_model` 接口，支持在 `config.py` 中一行配置动态切换 ResNet-18、MobileNet 等不同架构。

---

## 📂 项目目录结构

```text
cifar10_project/
├── main.py                   # 启动总指挥部（串联流程、断点续训读档）
├── config.py                 # 全局超参数与模型配置中心（杜绝硬编码）
├── requirements.txt          # 项目依赖包说明
├── README.md                 # 项目说明文档
├── data/                     # 数据集存放目录（代码自动下载，不纳入版本控制）
├── checkpoints/              # 权重与存档存放目录 (.pth)
└── src/                      # 核心源码包
    ├── __init__.py           # 包标识
    ├── trainer.py            # 训练引擎（训练/验证闭环、双文件存档）
    ├── dataset/
    │   ├── __init__.py
    │   └── dataset.py        # 数据加载与增强管道
    ├── models/
    │   ├── __init__.py       # 统一中转导出
    │   └── model.py          # 模型架构定义与模型工厂（Model Factory）
    └── utils/
        ├── __init__.py       # 前台接待（统一暴露底层工具）
        ├── common.py         # 随机种子固定、计算设备自动识别
        └── metrics.py        # 评价指标计算与 AverageMeter 平滑计数器