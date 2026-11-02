# src/trainer.py
import os
import torch
from src.utils import AverageMeter, calculate_accuracy

class Trainer:
    """
    训练引擎类：封装单轮训练、验证闭环与双文件存档机制（支持断点续训）
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir="./checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_acc = 0.0

    def train_one_epoch(self, epoch: int):
        """单轮标准训练逻辑"""
        self.model.train()
        loss_meter = AverageMeter("Train Loss")
        acc_meter = AverageMeter("Train Acc")
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 1. 梯度清零
            self.optimizer.zero_grad()
            
            # 2. 前向传播与计算损失
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
                
            # 3. 反向传播与优化器更新
            loss.backward()
            self.optimizer.step()
            
            # 4. 统计并平滑指标
            acc = calculate_accuracy(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            
        print(f"Epoch [{epoch}] 训练集 -> Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg*100:.2f}%")

    def evaluate(self, epoch: int):
        """单轮验证逻辑、指标计算与双文件存档机制"""
        self.model.eval()
        loss_meter = AverageMeter("Val Loss")
        acc_meter = AverageMeter("Val Acc")
        
        # 验证阶段关闭梯度计算，节省显存并加快评估
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                acc = calculate_accuracy(outputs, targets)
                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update(acc, inputs.size(0))
                
        val_loss = loss_meter.avg
        val_acc = acc_meter.avg
        print(f"Epoch [{epoch}] 验证集 -> Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
        
        # ==========================================
        # 📁 双文件存档机制
        # ==========================================
        
        # 1. 保存 last_checkpoint.pth（每轮无条件覆盖，用于意外崩溃后的断点续训）
        # 1. 先判断并更新最佳准确率（确保 best_acc 是最新的）
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"--> 🎯 [Checkpoint] 刷新最佳准确率 ({self.best_acc*100:.2f}%)，最优权重已安全保存！")

        # 2. 再保存 last_checkpoint.pth（此时写入的 self.best_acc 已经包含了当前的最新最高记录）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc 
        }
        last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")
        torch.save(checkpoint, last_checkpoint_path)