# src/dataset/dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2):
    """
    构建 CIFAR-10 数据加载器工厂函数
    包含：训练集数据增强（提升泛化能力）与 DataLoader 提速配置
    """
    # 1. 训练集数据增强：随机裁剪、水平翻转，防止过拟合
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 官方统计均值与方差
    ])
    
    # 2. 验证集无需增强，仅做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 3. 自动下载并加载官方数据集
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # 4. 高效 DataLoader 封装（pin_memory=True 加速 CPU 到 GPU 的锁页内存拷贝）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader