import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import random
import numpy as np
import os
from model.basemodel import ECoGResNet, ECoGTransformerClassifier, ECoGEEGNet
from data import MillerFingersDataset, prepare_dataloaders, set_seed

    
set_seed(0)
fingerflexLABEL_MAP = {
    0: "Inter-stimulus interval", # -> GT Token ID: 3306
    1: "thumb", # -> GT Token ID: 25036
    2: "index finger", #  -> GT Token ID: 1252
    3: "middle finger", #  -> GT Token ID: 19656
    4: "ring finger", #  -> GT Token ID: 12640
    5: "little finger" # -> GT Token ID: 55392
}
BCILABEL_MAP = fingerflexLABEL_MAP
Gesture_MAP = fingerflexLABEL_MAP
Motor_MAP = {
    0: "blank screen",  # Global ID 10189 (blank screen)
    11: "tongue movement", # Global ID 83 (tongue movement)
    12: "hand movement" # Global ID 10661 (hand movement)
}
LABEL_MAP = fingerflexLABEL_MAP  # Motor_MAP # BCILABEL_MAP  # 

# -------------------------
# 参数配置
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 42  # 
window_size = 1000
stride = 256  #
subset = "fingerflex"  # or 'gestures', 'motor_basic'
datadir = '/disk2/user1/dataset/BCI-Standford/%s/%s/data/' % (subset, subset)
# save_path = "./checkpoints/Stanford/Transformer/%s/" % (subset)
# save_path = "./checkpoints/Stanford/ResNet/%s/" % (subset)
save_path = "./checkpoints/Stanford/EEGNet/%s/" % (subset)

os.makedirs(save_path, exist_ok=True)
model_path = f"{save_path}best.pth"
# -------------------------
# 数据加载
# -------------------------
full_dataset = MillerFingersDataset(datadir, window_size=window_size, stride=stride,single_wind=True,
                        single_channel=True, channel_idx=None, subset=subset)  # 单通道 channel_idx=0,

train_loader, val_loader, test_data = prepare_dataloaders(full_dataset, BATCH_SIZE, num_classes=len(LABEL_MAP), ratio=0.9)
    
# # 打乱数据（安全方式）
# indices = list(range(len(dataset)))
# random.shuffle(indices)
# train_size = int(0.8 * len(indices))
# train_indices = indices[:train_size]
# val_indices = indices[train_size:]
# from torch.utils.data import Subset
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# -------------------------
# 模型 & 优化器
# -------------------------
num_channels = full_dataset[0][0].shape[0]  # e.g., 64
num_classes = len(set(full_dataset.labels))

# model = ECoGTransformerClassifier(
#     num_channels=num_channels,
#     seq_len=window_size,
#     embed_dim=128,
#     nhead=8,
#     num_layers=4,
#     num_classes=num_classes,
#     dropout=0.1
# ).to(device)

# model = ECoGResNet(
#     num_channels=num_channels,
#     num_classes=num_classes,
#     layers=[2, 2, 2, 2]  # ResNet-18
# ).to(device)

model = ECoGEEGNet(num_channels=num_channels, num_classes=num_classes, input_time_length=window_size).to(device)
# Epoch 1000 | Train Loss: 0.0421 | Train Acc: 0.2167 | Val Acc: 0.2063 | Best Val Acc: 0.2594 at Epoch 490LR: 4.00e-05

                
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=200, steps_per_epoch=len(train_loader))
# -------------------------
# 训练循环
# -------------------------
best_acc = 0.0
bestepoch = 0
for epoch in range(1000):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # x: [B, C, T]
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)

    # 验证
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    
    if val_acc > best_acc:
        best_acc = val_acc
        bestepoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), model_path)
        print(f"⭐ New Best Model Saved (Acc: {best_acc*100:.2f}%)")

    scheduler.step()

    print(f"Epoch {epoch+1:2d} | Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_acc:.4f} at Epoch {bestepoch}"
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    torch.save(model.state_dict(), model_path.replace('best.pth', 'model.pth'))
    
