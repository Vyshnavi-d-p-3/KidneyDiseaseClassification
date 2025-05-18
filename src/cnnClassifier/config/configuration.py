from cnnClassifier.constants import *
import os
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)




class ConfigurationManager:

    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        tc     = self.config.training_effnet
        params = self.params.EFFICIENTNET

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    



    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate    = params.LEARNING_RATE
        )

        return training_config
    


    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model         = Path(self.config.training.trained_model_path),
            path_of_effnet_model  = Path(self.config.training_effnet.trained_model_path),  # ← add this
            training_data         = Path(self.config.data_ingestion.unzip_dir) / "kidney-ct-scan-image",
            all_params            = self.params,
            mlflow_uri            = self.config.mlflow_uri if hasattr(self.config, 'mlflow_uri') else "",
            params_image_size     = list(self.params.IMAGE_SIZE),
            params_batch_size     = self.params.BATCH_SIZE
        )
    
    def get_prepare_effnet_config(self) -> PrepareBaseModelConfig:
        from pathlib import Path

        cfg    = self.config.prepare_effnet
        params = self.params.EFFICIENTNET

        return PrepareBaseModelConfig(
            root_dir                = Path(cfg.root_dir),                    # NEW
            base_model_path         = Path(cfg.base_model_path),
            updated_base_model_path = Path(cfg.updated_base_model_path),
            params_image_size       = list(params.IMAGE_SIZE),
            params_learning_rate    = params.LEARNING_RATE,
            params_include_top      = False,
            params_weights          = "imagenet",
            params_classes          = self.params.CLASSES,                  # FIXED
            backbone                = "efficientnet_b0"
        )

    def get_training_effnet_config(self) -> TrainingConfig:
        import os
        from pathlib import Path
        from cnnClassifier.utils.common import create_directories

        tc     = self.config.training_effnet
        params = self.params.EFFICIENTNET

        # mirror your VGG path logic:
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "kidney-ct-scan-image"
        )
        create_directories([Path(tc.root_dir)])

        return TrainingConfig(
            root_dir                = Path(tc.root_dir),
            trained_model_path      = Path(tc.trained_model_path),
            updated_base_model_path = Path(self.config.prepare_effnet.updated_base_model_path),  # NEW
            training_data           = Path(training_data),                                       # NEW
            params_epochs           = params.EPOCHS,
            params_batch_size       = params.BATCH_SIZE,
            params_is_augmentation  = self.params.AUGMENTATION,                                 # NEW
            params_image_size       = list(params.IMAGE_SIZE),
            params_learning_rate    = params.LEARNING_RATE
        )
