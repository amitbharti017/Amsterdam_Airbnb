from config.configuration import ConfigurationManager
from components.model_trainer import ModelTrainer
from logger import logger

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_trainer_config()
        model_training = ModelTrainer(config=model_trainer_config)
        model_training.optimize_xgboost()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>stage {STAGE_NAME} started<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>stage{STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e