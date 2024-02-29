import os
import urllib.request as request
import zipfile
from logger import logger
from utils.common import get_size
from entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

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

    def extract_zip_fil(self):
        """
        this function extracts the zip file downloaded
        zip_file_path: str
        returns: None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.Zipfile(self.config.local_data_file,"r") as zip_ref:
            zip_ref.extractall(unzip_path)




