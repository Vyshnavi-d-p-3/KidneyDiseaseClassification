import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score

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
        """
        # Prepare validation data
        self._valid_generator()

        # Load models
        self.vgg_model = self.load_model(self.config.path_of_model)
        self.eff_model = self.load_model(self.config.path_of_effnet_model)

        # Collect predictions and ground truth
        y_true, y_pred_vgg, y_pred_eff, y_pred_ens = [], [], [], []
        for x_batch, y_batch in self.valid_generator:
            pv = self.vgg_model.predict(x_batch, verbose=0)
            pe = self.eff_model.predict(x_batch, verbose=0)
            p_ens = 0.4 * pv + 0.6 * pe

            labels = np.argmax(y_batch, axis=1)
            y_true.extend(labels)
            y_pred_vgg.extend(np.argmax(pv, axis=1))
            y_pred_eff.extend(np.argmax(pe, axis=1))
            y_pred_ens.extend(np.argmax(p_ens, axis=1))

        # Compute accuracies
        acc_vgg = accuracy_score(y_true, y_pred_vgg)
        acc_eff = accuracy_score(y_true, y_pred_eff)
        acc_ens = accuracy_score(y_true, y_pred_ens)

        # Save scores
        self.scores = {
            "vgg_accuracy": acc_vgg,
            "effnet_accuracy": acc_eff,
            "ensemble_accuracy": acc_ens
        }
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
