# src/utils/metrics.py
import torch

class AverageMeter:
    """
    滑动平均计数器：用于平滑统计每个 Epoch 中不断变化的 Loss 和 Accuracy，
    防止显存或内存因为直接记录历史列表而暴涨。
    """
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """计算分类任务的 Top-1 准确率 (Accuracy)"""
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)