#!/usr/bin/env python
import sys
import subprocess
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from monai.networks.nets import resnet18
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.config.configuration_old import ConfigurationManager

# 1) Run existing VGG pipeline
ret = subprocess.call([sys.executable, "main.py"])
if ret != 0:
    sys.exit(ret)

# 2) Load configuration
cfg_mgr = ConfigurationManager()

# Load VGG model
vgg_path = cfg_mgr.get_training_config().trained_model_path
vgg_model = tf.keras.models.load_model("artifacts/training/model.h5", compile=False)

# Load EfficientNetB2 PyTorch model
effnetb2_model = torch.hub.load('pytorch/vision:v0.14.0', 'efficientnet_b2', weights=None)
effnetb2_model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=2)
effnetb2_model.load_state_dict(torch.load("artifacts/training/efficientnetb2_final_2classes.pt", map_location=torch.device("cpu")))
effnetb2_model.eval()

# Load MONAI model
monai_model = resnet18(spatial_dims=2, n_input_channels=3, num_classes=2)
monai_model.load_state_dict(torch.load("artifacts/training/monai_resnet18_kidney_2class.pt", map_location=torch.device("cpu")))
monai_model.eval()

# Torch transform
torch_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(cfg_mgr.get_evaluation_config().params_image_size[:-1]),
])

# Load validation generator
eval_cfg = cfg_mgr.get_evaluation_config()
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.30
)
val_gen = val_datagen.flow_from_directory(
    directory=eval_cfg.training_data,
    target_size=eval_cfg.params_image_size[:-1],
    batch_size=eval_cfg.params_batch_size,
    interpolation="bilinear",
    shuffle=False,
    subset="validation"
)

# Inference loop
y_true, y_pred = [], []
y_vgg, y_b2, y_monai = [], [], []
for i, (x_batch, y_batch) in enumerate(val_gen):
    print(f"Batch {i + 1}/{len(val_gen)}")

    # VGG prediction
    pv = vgg_model.predict(x_batch, verbose=0)
    y_vgg.extend(np.argmax(pv, axis=1))

    # EfficientNetB2 (PyTorch)
    torch_batch = torch.stack([torch_tf(Image.fromarray((img * 255).astype(np.uint8))) for img in x_batch])
    with torch.no_grad():
        pb2 = effnetb2_model(torch_batch)
        pb2 = F.softmax(pb2, dim=1).numpy()
    y_b2.extend(np.argmax(pb2, axis=1))

    # MONAI (PyTorch)
    with torch.no_grad():
        pm = monai_model(torch_batch)
        pm = F.softmax(pm, dim=1).numpy()
    y_monai.extend(np.argmax(pm, axis=1))

    # Weighted Ensemble: VGG (15%), EffNetB2 (55%), MONAI (30%)
    p_ens = 0.15 * pv + 0.55 * pb2 + 0.30 * pm

    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(p_ens, axis=1))

# Final evaluation
acc = accuracy_score(y_true, y_pred)
acc_vgg = accuracy_score(y_true, y_vgg)
acc_b2 = accuracy_score(y_true, y_b2)
acc_monai = accuracy_score(y_true, y_monai)

print(f"VGG16 accuracy: {acc_vgg:.4%}")
print(f"EffNetB2 accuracy: {acc_b2:.4%}")
print(f"MONAI ResNet18 accuracy: {acc_monai:.4%}")
print(f"Ensemble (VGG + EffNetB2 + MONAI) accuracy: {acc:.4%}")

# Save to scores.json
import json
scores = {
    "ensemble_accuracy": acc,
    "vgg_accuracy": acc_vgg,
    "effnetb2_accuracy": acc_b2,
    "monai_accuracy": acc_monai
}
with open(eval_cfg.scores_file, "w") as f:
    json.dump(scores, f, indent=2)

sys.exit(0)
