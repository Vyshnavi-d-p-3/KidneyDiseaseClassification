#!/usr/bin/env python
import sys, subprocess
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1) Run your existing pipeline (VGG only)
ret = subprocess.call([sys.executable, "main.py"])
if ret != 0:
    sys.exit(ret)


# 2) Prepare & train EfficientNet-B0 using your existing components
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

cfg_mgr = ConfigurationManager()

# 2a) Prepare EfficientNet
prep_eff_cfg = cfg_mgr.get_prepare_effnet_config()    # new getter you already added
prep = PrepareBaseModel(prep_eff_cfg, backbone="efficientnet_b0")
prep.prepare_model()

# 2b) Train EfficientNet
train_eff_cfg = cfg_mgr.get_training_effnet_config()  # new getter you already added
trainer = ModelTraining(train_eff_cfg)
trainer.train()


# 3) Ensemble evaluation
#    we’ll load both models and re-use your Stage-04 logic only for metrics
#    (skipping MLflow here, but you can integrate if you like)
vgg_path   = cfg_mgr.get_training_config().trained_model_path
eff_path   = cfg_mgr.get_training_effnet_config().trained_model_path

vgg_model  = tf.keras.models.load_model(vgg_path, compile=False)
eff_model  = tf.keras.models.load_model(eff_path, compile=False)

# build a small validation generator
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

# val_gen = EvaluationPipeline._build_validation_generator(eval_cfg)  # or reuse whatever you have

y_true, y_pred = [], []
for i, (x_batch, y_batch) in enumerate(val_gen):
    print(f"Batch {i + 1}/{len(val_gen)}")  # Add this
    pv = vgg_model.predict(x_batch, verbose=0)
    pe = eff_model.predict(x_batch, verbose=0)
    p_ens = 0.4 * pv + 0.6 * pe
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(p_ens,  axis=1))

acc = accuracy_score(y_true, y_pred)
print(f"Ensemble (40% VGG + 60% EfficientNet-B0) accuracy: {acc:.4%}")

# Optionally write back into your scores.json
import json
scores = {"ensemble_accuracy": acc}
with open(eval_cfg.scores_file, "w") as f:
    json.dump(scores, f, indent=2)

sys.exit(0)
