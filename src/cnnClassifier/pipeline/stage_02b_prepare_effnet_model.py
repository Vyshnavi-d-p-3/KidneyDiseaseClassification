from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare EfficientNet-B0 Base Model"

class PrepareEffnetModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            cfg_mgr = ConfigurationManager()
            prep_cfg = cfg_mgr.get_prepare_effnet_config()
            preparer = PrepareBaseModel(config=prep_cfg)    # config.backbone=='efficientnet_b0'
            preparer.get_base_model()
            preparer.update_base_model()
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        PrepareEffnetModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=====x")
    except Exception as e:
        logger.exception(e)
        raise e
