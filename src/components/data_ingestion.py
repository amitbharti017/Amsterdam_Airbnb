import os
import urllib.request as request
import gzip
import shutil
from logger import logger
from utils.common import get_size
from entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,config: DataIngestionConfig,random_state=810,test_size=0.15):
        self.config = config
        self.random_state = random_state
        self.test_size=test_size

    def download_file(self):
        """
        function downloads the file from the source url
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"file already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_gzip_file(self):
        """
        this function extracts the zip file downloaded
        zip_file_path: str
        returns: None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with gzip.open(self.config.local_data_file,"r") as f_in:
            with open(self.config.local_data_file_csv,"wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def data_spliting(self):
        data = pd.read_csv(self.config.local_data_file_csv)
        
        #spliting the data into train, val and test (0.7, 0.15 0.15) split.
        train, test = train_test_split(data, test_size=self.test_size, random_state = self.random_state)
        train, val = train_test_split(train, test_size=self.test_size, random_state=self.random_state)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index = False)
        val.to_csv(os.path.join(self.config.root_dir,"val.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)

        logger.info("Data splitting done into train, val and test sets")





