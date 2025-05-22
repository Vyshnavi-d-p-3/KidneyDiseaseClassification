import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, random_split
import time
import random
import numpy as np

# === Helper Functions ===
def set_all_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_accu(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100

# === Model ===
def get_effnet_model(seed, num_classes):
    set_all_seeds(seed)
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms_ = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model, transforms_

# === Training ===
def train_model(model, train_loader, val_loader, test_loader, epochs, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max')

    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:03d}/{epochs} | Batch {batch_idx:03d}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = compute_accu(model, train_loader, device)
        val_acc = compute_accu(model, val_loader, device)
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_acc)

    test_acc = compute_accu(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "efficientnetb2_final_2classes.pt")

# === Main Execution ===
if __name__ == "__main__":
    dataset_path = "/Users/spartan/Desktop/desktop/USA/2 Sem/258/project/KidneyDiseaseClassification/artifacts/data_ingestion/kidney-ct-scan-image"

    model, transforms_ = get_effnet_model(seed=42, num_classes=2)

    full_dataset = datasets.ImageFolder(dataset_path, transform=transforms_)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    print(full_dataset.class_to_idx)


    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, test_loader, epochs=1, device=device)
