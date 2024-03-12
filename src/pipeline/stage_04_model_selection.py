from config.configuration import ConfigurationManager
from components.model_selection import ModelSelection
from logger import logger

STAGE_NAME = "Model Selection stage"

class ModelSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_selection_config = config.get_model_selection_config()
        model_selection_config = ModelSelection(config=model_selection_config)
        model_selection_config.selection_process()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>stage {STAGE_NAME} started<<<<<<<<<")
        obj = ModelSelectionPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>stage{STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e