from constant import *
from utils.common import read_yaml, create_directories
from entity.config_entity import DataIngestionConfig
from entity.config_entity import DataTransformationConfig
from entity.config_entity import DataTrainerConfig
from entity.config_entity import ModelSelectionConfig
from entity.config_entity import ModelEvaluationConfig



class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)

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
            data_val_path= config.data_val_path,
            data_test_path= config.data_test_path,
            transformer = config.transformer)
        
        return data_transformation_config
    
    def get_trainer_config(self) ->DataTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = DataTrainerConfig(
            root_dir= config.root_dir,
            data_X_train_path= config.data_X_train_path,
            data_y_train_path= config.data_y_train_path,
            data_X_val_path= config.data_X_val_path,
            data_y_val_path= config.data_y_val_path,
            best_xgboost_model = config.best_xgboost_model,
            best_lightgbm_model = config.best_lightgbm_model,
            best_linear_model = config.best_linear_model)
        
        return model_trainer_config
    
    def get_model_selection_config(self)->ModelSelectionConfig:
        config = self.config.model_selection

        create_directories([config.root_dir])
        model_selection_config = ModelSelectionConfig(
            root_dir = config.root_dir,
            val_data_X_path = config.val_data_X_path,
            val_data_y_path = config.val_data_y_path,
            best_xgboost_model = config.best_xgboost_model,
            best_lightgbm_model = config.best_lightgbm_model,
            best_linear_model = config.best_linear_model,
            best_model = config.best_model,
            val_metric_file_name = config.val_metric_file_name)
        
        return model_selection_config
    
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
                root_dir = config.root_dir,
                test_data_X_path = config.test_data_X_path,
                test_data_y_path = config.test_data_y_path,
                model_path = config.model_path,
                metric_file_name = config.metric_file_name)
        return model_evaluation_config




