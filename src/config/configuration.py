from constant import *
from utils.common import read_yaml, create_directories
from entity.config_entity import DataIngestionConfig
from entity.config_entity import DataTransformationConfig


class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file=config.local_data_file,
            local_data_file_csv = config.local_data_file_csv,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            data_train_path= config.data_train_path,
            data_val_path = config.data_val_path,
            data_test_path = config.data_test_path,
            transformer = config.transformer)
        
        return data_transformation_config

