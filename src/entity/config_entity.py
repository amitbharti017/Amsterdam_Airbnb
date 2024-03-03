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
    model_name: str