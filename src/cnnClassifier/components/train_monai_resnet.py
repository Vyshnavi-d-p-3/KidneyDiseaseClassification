from monai.transforms import (
    Compose, LoadImaged, ResizeD, ScaleIntensityD, ToTensorD, EnsureChannelFirstD
)
# train_monai_resnet.py

# 📌 Step 1: Imports
import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from monai.transforms import (
    Compose, LoadImage, Resize, ScaleIntensity, ToTensor
)
from monai.data import Dataset
from monai.networks.nets import resnet18
from sklearn.metrics import accuracy_score
from monai.transforms import EnsureChannelFirstd


# 📌 Step 2: Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/Users/spartan/Desktop/desktop/USA/2 Sem/258/project/KidneyDiseaseClassification/artifacts/data_ingestion/kidney-ct-scan-image"

# 📌 Step 3: Load Images and Labels
normal_images = sorted(glob(os.path.join(data_dir, "Normal", "*.jpg")))
tumor_images = sorted(glob(os.path.join(data_dir, "Tumor", "*.jpg")))

image_files = normal_images + tumor_images
labels = [0] * len(normal_images) + [1] * len(tumor_images)

print(f"✅ Loaded {len(image_files)} images (Normal={len(normal_images)}, Tumor={len(tumor_images)})")

# 📌 Step 4: Define Transforms
transforms = Compose([
    LoadImaged(keys=["img"]),
    EnsureChannelFirstD(keys=["img"]),
    ResizeD(keys=["img"], spatial_size=(224, 224)),
    ScaleIntensityD(keys=["img"]),
    ToTensorD(keys=["img"])
])


# 📌 Step 5: Create Dataset and Split
data_dicts = [{"img": img, "label": label} for img, label in zip(image_files, labels)]
dataset = Dataset(data=data_dicts, transform=transforms)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# 📌 Step 6: Define Model
print("define model");
model = resnet18(pretrained=False, spatial_dims=2, n_input_channels=3, num_classes=2).to(device)

# 📌 Step 7: Training Setup
print("training setup");

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("training loop");
# 📌 Step 8: Training Loop
num_epochs = 2
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}");
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch_data = batch["img"].to(device)
        batch_labels = batch["label"].to(device)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

    # 📌 Validation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            val_data = batch["img"].to(device)
            val_labels = batch["label"].to(device)
            outputs = model(val_data)
            pred_classes = torch.argmax(outputs, dim=1)
            preds.extend(pred_classes.cpu().numpy())
            trues.extend(val_labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    print(f"✅ Epoch {epoch+1}/{num_epochs} | Validation Accuracy: {acc:.4f}")

# 📌 Save the model
torch.save(model.state_dict(), "monai_resnet18_kidney_2class.pt")
