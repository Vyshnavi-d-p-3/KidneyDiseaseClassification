from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training      # ← correct class
from cnnClassifier import logger

STAGE_NAME = "Training EfficientNet-B0"

class ModelTrainingEffnetPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Load our EffNet-specific training config
            cfg = ConfigurationManager()
            train_cfg = cfg.get_training_effnet_config()
            
            # Instantiate and prepare the model
            trainer = Training(config=train_cfg)
            trainer.get_base_model()
            trainer.train_valid_generator()
            
            # Run the training loop and save
            trainer.train()
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}")
            raise e

if __name__ == '__main__':
    try:
        logger.info("*********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<")
        ModelTrainingEffnetPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e
