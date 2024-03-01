from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    local_data_file_csv: Path
    unzip_dir: Path
    root_train_dir: Path
    root_val_dir: Path
    root_test_dir: Path
    

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_train_path: Path
    data_val_path: Path