# two_stage_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

# ✅ 설정
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PASSENGER_BRANDS = [
    '현대자동차', '기아자동차', 'GM', 'BMW', '벤츠',
    '아우디', '테슬라', '쌍용자동차', '도요타', '혼다'
]
NON_PASSENGER_TYPES = ['트럭', '버스', '이륜차', '킥보드']

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# ✅ 데이터셋 정의
class VehicleDataset(Dataset):
    def __init__(self, df, label_col, transform=None, augment_transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augment_transform = augment_transform
        self.label_col = label_col
        self.augment_flags = self.df['augment'] if 'augment' in df.columns else [False]*len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.augment_flags[idx] and self.augment_transform:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return image, label

# ✅ 모델 정의
class SimpleResNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ✅ 밸런싱 함수
def create_balanced_df(df, column_name, classes, target_per_class):
    balanced = []
    for cls in classes:
        subset = df[df[column_name] == cls].copy()
        if len(subset) >= target_per_class:
            sampled = subset.sample(n=target_per_class, random_state=42)
            sampled['augment'] = False
        else:
            duplicated = resample(subset, replace=True, n_samples=target_per_class - len(subset), random_state=42)
            subset['augment'] = False
            duplicated['augment'] = True
            sampled = pd.concat([subset, duplicated])
        balanced.append(sampled)
    return pd.concat(balanced).reset_index(drop=True)

# ✅ 학습 함수
def train_classifier(model, loader, name):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"{name} | Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"✅ {name} 평균 Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f"{name}_classifier.pth")

# ✅ main
if __name__ == '__main__':
    df = pd.read_csv("full_training_data_label_1.csv")
    df['is_passenger'] = df['brand'].apply(lambda x: 0 if x in PASSENGER_BRANDS else 1)

    # 1단계: 승용차 여부 분류 (전체 사용)
    bin_df = df.copy()
    bin_df['augment'] = False
    bin_dataset = VehicleDataset(bin_df, 'is_passenger', transform, augment_transform)
    bin_loader = DataLoader(bin_dataset, batch_size=BATCH_SIZE, shuffle=True)
    bin_model = SimpleResNet(2).to(DEVICE)
    train_classifier(bin_model, bin_loader, "binary")

    # 2단계-a: 승용차면 브랜드 분류
    brand_df = df[(df['is_passenger'] == 0) & (df['brand'].isin(PASSENGER_BRANDS))].copy()
    brand_map = {b: i for i, b in enumerate(PASSENGER_BRANDS)}
    brand_df['brand_label'] = brand_df['brand'].map(brand_map)
    brand_df = create_balanced_df(brand_df, 'brand', PASSENGER_BRANDS, 2000)

    brand_dataset = VehicleDataset(brand_df, 'brand_label', transform, augment_transform)
    brand_loader = DataLoader(brand_dataset, batch_size=BATCH_SIZE, shuffle=True)
    brand_model = SimpleResNet(len(PASSENGER_BRANDS)).to(DEVICE)
    train_classifier(brand_model, brand_loader, "brand")

    # 2단계-b: 비승용차면 차종 분류
    type_df = df[(df['is_passenger'] == 1) & (df['brand'].isin(NON_PASSENGER_TYPES))].copy()
    type_map = {t: i for i, t in enumerate(NON_PASSENGER_TYPES)}
    type_df['type_label'] = type_df['brand'].map(type_map)
    type_df = create_balanced_df(type_df, 'brand', NON_PASSENGER_TYPES, 2000)

    type_dataset = VehicleDataset(type_df, 'type_label', transform, augment_transform)
    type_loader = DataLoader(type_dataset, batch_size=BATCH_SIZE, shuffle=True)
    type_model = SimpleResNet(len(NON_PASSENGER_TYPES)).to(DEVICE)
    train_classifier(type_model, type_loader, "type")
