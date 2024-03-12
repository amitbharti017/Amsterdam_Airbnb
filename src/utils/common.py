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
import mlflow
from sklearn.metrics import root_mean_squared_error

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

def rmse_metrics(true,pred) ->float:
    """
    get the true and predict values to calculate the root mean squared error
    Args:
        true: ground truth price of the listing
        pred: predicted price of the listing
    
    Returns:
        float: root mean square error in the pricing after transforming the data   
    """
    true_value = np.expm1(true)
    predict_value = np.expm1(pred)
    return root_mean_squared_error(true_value,predict_value)


# def create_mlflow_experiment(experiment_name:str,artifact_location:str, tags:dict[str, Any]) -> str:
#     """
#     create a new mlflow experiment with the given name and artifact location
#     Args:
#         experiment_name (str): The name of the experiment to create
#         artifact_location (str): The artifact location of the experiment to create.
#         tags (dict[str,Any]): The tags of the experiment to create.
    
#     Returns:
#         experiment_id (str): The id of the created experiment.
#     """
#     try:
#         experiment_id = mlflow.create_experiment(name=experiment_name,artifact_location=artifact_location,tags=tags)
#     except:
#         print(f"Experiment {experiment_name} already exists.")
#         experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

#     mlflow.set_experiment(experiment_name=experiment_name)

#     return experiment_id

# @ensure_annotations
# def get_mlflow_experiment( experiment_id: str = None, experiment_name:str = None) -> mlflow.entities.Experiment:
#     """
#     Retrieve the mlflow experiment with the given id or name
#     Args:
#         experiment_id(str): The id of the experiment to retrieve
#         experiment_name(str): The name of the experiment to retrieve.

#     Returns:
#         experiment(mlflow.entities.Experiment): The mlflow experiment with the given id or name 
#     """
#     if experiment_id is not None:
#         experiment = mlflow.get_experiment(experiment_id)
#     elif experiment_name is not None:
#         experiment = mlflow.get_experiment_by_name(experiment_name)
#     else:
#         raise ValueError("Either experiment_id or experiment_name must be provided.")
#     return experiment
    


