from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    local_data_file_csv: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_train_path: Path
    data_val_path: Path
    data_test_path: Path
    transformer: Path

@dataclass(frozen=True)
class DataTrainerConfig:
    root_dir: Path
    data_X_train_path: Path
    data_y_train_path: Path
    data_X_val_path: Path
    data_y_val_path: Path
    best_xgboost_model: Path
    best_lightgbm_model: Path
    best_linear_model: Path

@dataclass(frozen=True)
class ModelSelectionConfig:
    root_dir: Path
    val_data_X_path: Path
    val_data_y_path: Path
    best_xgboost_model: Path
    best_lightgbm_model: Path
    best_linear_model: Path
    best_model: Path
    val_metric_file_name: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_X_path: Path
    test_data_y_path: Path
    model_path: Path
    metric_file_name: Path