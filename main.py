# main.py
import os
import torch
import torch.nn as nn
import config
from src.utils import set_seed, get_device
from src.dataset.dataset import get_cifar10_dataloaders
# from src.models import CIFAR10ResNet
from src.trainer import Trainer
from src.models import get_model  # 引入模型工厂

def main():
    # 2. 基础环境初始化（全局固定随机种子，确保可复现）
    set_seed(config.SEED)
    device = get_device()
    
    # 3. 加载 CIFAR-10 高效数据加载器（自带训练增强与验证集划分）
    train_loader, val_loader = get_cifar10_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    # 4. 通过模型工厂动态实例化模型（想换哪个模型，去 config.py 里改名字即可）
    model = get_model(model_name=config.MODEL_NAME, num_classes=10)
    # 4. 实例化轻量化微调模型、损失函数与优化器
    # model = CIFAR10ResNet(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    
    # 5. 组装 Trainer 训练引擎
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    
    # ==========================================
    # 6. 断点续训逻辑（Resume Training 读档机制）
    # ==========================================
    start_epoch = 1
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "last_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"\n--> 📂 检测到历史存档，正在加载断点续训: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 恢复模型权重、优化器状态、起始轮次与历史最佳准确率
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1      # 从中断的下一轮开始继续学
        trainer.best_acc = checkpoint['best_acc']  # 恢复历史最高准确率
        
        print(f"--> 🎉 读档成功！将从第 {start_epoch} 轮继续训练（当前历史最佳 Acc: {trainer.best_acc*100:.2f}%）。")
    else:
        print("\n--> 🌱 未检测到历史存档，开始全新训练。")
    
    # ==========================================
    # 7. 开启多轮训练闭环
    # ==========================================
    print(f"\n========== 🚀 开始执行 CIFAR-10 模块化训练 (目标总轮次: {config.EPOCHS}) ==========")
    for epoch in range(start_epoch, config.EPOCHS + 1):
        trainer.train_one_epoch(epoch)
        trainer.evaluate(epoch)
        
    print("========== 🎉 训练全部完成！模型权重已安全保存在 checkpoints/ 目录下。==========")

if __name__ == "__main__":
    main()