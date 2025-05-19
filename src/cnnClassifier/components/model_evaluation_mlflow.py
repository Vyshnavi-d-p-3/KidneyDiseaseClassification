import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score
import time
import json

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.vgg_model = None
        self.eff_model = None
        self.valid_generator = None
        self.scores = {}

    def _valid_generator(self):
        """
        Builds the validation data generator.
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)


    def evaluate(self):
        """
        Runs evaluation for VGG, EfficientNet, and their ensemble.
        Logs timings and batch-level progress to diagnose hangs.
        """
        print("🧪 [eval] Preparing validation generator...")
        self._valid_generator()
        print("✅ [eval] Validation generator ready.")

        print("📦 [eval] Loading VGG model...")
        self.vgg_model = self.load_model(self.config.path_of_model)
        print("✅ [eval] VGG loaded.")

        print("📦 [eval] Loading EfficientNet model...")
        self.eff_model = self.load_model(self.config.path_of_effnet_model)
        print("✅ [eval] EfficientNet loaded.")

        # Sanity: reset generator
        self.valid_generator.reset()
        steps = len(self.valid_generator)

        y_true, y_pred_vgg, y_pred_eff, y_pred_ens = [], [], [], []

        print(f"📊 [eval] Running inference over {steps} batches...")
        for i in range(steps):
            x_batch, y_batch = self.valid_generator[i]

            t0 = time.time()
            pv = self.vgg_model.predict(x_batch, verbose=0)
            pe = self.eff_model.predict(x_batch, verbose=0)
            t1 = time.time()

            p_ens = 0.4 * pv + 0.6 * pe

            labels = np.argmax(y_batch, axis=1)
            y_true.extend(labels)
            y_pred_vgg.extend(np.argmax(pv, axis=1))
            y_pred_eff.extend(np.argmax(pe, axis=1))
            y_pred_ens.extend(np.argmax(p_ens, axis=1))

            print(f"✅ Batch {i+1}/{steps} done — {t1 - t0:.2f}s")

        acc_vgg = accuracy_score(y_true, y_pred_vgg)
        acc_eff = accuracy_score(y_true, y_pred_eff)
        acc_ens = accuracy_score(y_true, y_pred_ens)

        self.scores = {
            "vgg_accuracy": acc_vgg,
            "effnet_accuracy": acc_eff,
            "ensemble_accuracy": acc_ens
        }

        print("✅ [eval] Evaluation complete.")
        print(json.dumps(self.scores, indent=2))

        save_json(path=Path("scores.json"), data=self.scores)


    def log_into_mlflow(self):
        """
        Logs parameters and metrics for all models to MLflow.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.scores)

            # Log models
            mlflow.keras.log_model(
                self.vgg_model, "vgg_model", registered_model_name="VGG16Model"
                if tracking_uri != "file" else None
            )
            mlflow.keras.log_model(
                self.eff_model, "effnet_model", registered_model_name="EfficientNetB0Model"
                if tracking_uri != "file" else None
            )
            # You may also log the ensemble as a custom artifact if desired

    def save_score(self):
        from cnnClassifier.utils.common import save_json
        save_json(path=Path("scores.json"), data=self.scores)
