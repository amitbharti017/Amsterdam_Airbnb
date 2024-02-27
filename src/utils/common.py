import os
from box.exceptions import BoxValueError
import yaml
from logger import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns the corresponding Python data structure.

    Args:
        path_to_yaml (Path): Path of the yaml file

    Raises:
        ValueError: if yaml file is empty
        e: empty file
    
    Returns:
        ConfigBox: ConfigBox type    
    
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    create directories from the list
    
    Args:
        path_to_directories (str): list of path of directories
        ignore_log (bool, optional): ignore if multiple directories needs to be created. Defaults to True
    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    saves data in json file 
    Args:
        path (Path): path where json file needs to be saved
        data (dict): data to be saved in json file
    """
    with open(path,"w") as f:
        json.dump(data,f, indent=4)
    
    logger.info(f"json file is saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    load the json file in ConfigBox format
    Args:
        path (Path): path where the json file is saved

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        with open(path) as f:
            content = json.load(f)
        
        logger.info(f"json file loaded successfully from: {f}")
        return ConfigBox(content)
    except Exception as e:
        raise e
    
@ensure_annotations
def get_size(path: Path) -> str:
    """
    get the size of the file in KB
    Args:
        path (Path): path of the file
    
        Returns:
            str: size of the file in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"




