import pandas as pd
import os
from logger import logger
import joblib
from entity.config_entity import DataTrainerConfig


class ModelTrainer:
    def __init__(self,config: DataTrainerConfig):
        self.config = config
    
    def train(self):
        
