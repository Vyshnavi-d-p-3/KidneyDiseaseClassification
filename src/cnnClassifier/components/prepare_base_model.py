import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.applications import VGG16, EfficientNetB0

class PrepareBaseModel:
    """
    Prepares and updates a base model (VGG16 or EfficientNetB0) for classification.

    Args:
        config (PrepareBaseModelConfig): Configuration dataclass with paths and params.
        backbone (str, optional): Name of the backbone to use ('vgg16' or 'efficientnet_b0').
                                  Falls back to config.backbone or 'vgg16'.
    """
    def __init__(self, config: PrepareBaseModelConfig, backbone: str = None):
        self.config = config
        # Determine which backbone to use
        self.backbone = (backbone or getattr(config, 'backbone', 'vgg16')).lower()
        self.model = None

    def get_base_model(self):
        """
        Loads the pretrained backbone model and saves it to base_model_path.
        """
        if self.backbone == 'vgg16':
            base_model = VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.backbone == 'efficientnet_b0':
            base_model = EfficientNetB0(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        self.model = base_model
        self._ensure_directory(self.config.base_model_path)
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model,
                             classes: int,
                             freeze_all: bool,
                             freeze_till: int | None,
                             learning_rate: float) -> tf.keras.Model:
        """
        Attaches a Flatten + Dense head, freezes layers as specified, compiles the model.
        """
        # Freeze layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Build classifier head
        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dense(units=classes, activation='softmax')(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=x)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
            run_eagerly=True  
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Applies the classification head to the loaded backbone, freezes layers, and saves the updated model.
        """
        # Ensure the backbone is loaded
        if self.model is None:
            self.get_base_model()

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self._ensure_directory(self.config.updated_base_model_path)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model to the given path.
        """
        model.save(path, include_optimizer=False)

    @staticmethod
    def _ensure_directory(path: Path):
        """
        Ensures that the directory for the given file path exists.
        """
        parent_dir = path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
