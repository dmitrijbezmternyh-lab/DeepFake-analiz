import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import Config
from models.deepfake_detector import DeepFakeDetector
from dataset import DeepFakeDataset
from utils import set_seed

config = Config()
set_seed(config.seed)

device = config.device
print(f"Device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
])

df = pd.read_csv(config.train_csv, header=None, names=['Id', 'target_feature'])
print(f"Total: {len(df)}, Real: {(df['target_feature']==0).sum()}, Fake: {(df['target_feature']==1).sum()}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.seed, stratify=df['target_feature'])

train_ds = DeepFakeDataset(train_df, config.train_img_dir, train_transform)
val_ds = DeepFakeDataset(val_df, config.train_img_dir, val_transform)

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

model = DeepFakeDetector(dropout=config.dropout).to(device)
print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight]).to(device))
scaler = GradScaler(enabled=config.use_amp)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_f1 = 0
patience_counter = 0

for epoch in range(config.epochs):
    print(f"\n Epoch {epoch+1}/{config.epochs}")

    # Train
    model.train()
    train_loss, train_preds, train_labels = 0, [], []
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > config.threshold).int()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds, zero_division=0)
    train_recall = recall_score(train_labels, train_preds, zero_division=0)

    # Validation
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > config.threshold).int()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    val_recall = recall_score(val_labels, val_preds, zero_division=0)
    cm = confusion_matrix(val_labels, val_preds)

    print(f"Train — Loss: {train_loss/len(train_loader):.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}")
    print(f"Val   — Loss: {val_loss/len(val_loader):.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    scheduler.step(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), "weights/best_model.pth")
        print(" Model saved!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.patience:
            print(" Early stopping")
            break

print(f"\n Best F1: {best_f1:.4f}")
