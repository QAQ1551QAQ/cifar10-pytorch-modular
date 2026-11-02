import os

folders = ["data", "checkpoints", "logs", "src/dataset", "src/models", "src/utils"]
files = [
    "src/__init__.py", "src/dataset/__init__.py", "src/dataset/dataset.py",
    "src/models/__init__.py", "src/models/model.py", "src/utils/__init__.py",
    "src/utils/common.py", "src/utils/metrics.py", "src/utils/logger.py",
    "trainer.py", "main.py", "config.py", "requirements.txt", "README.md"
]

for f in folders:
    os.makedirs(f, exist_ok=True)
for f in files:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as file: pass

print("--> 项目模块化目录结构初始化成功！")