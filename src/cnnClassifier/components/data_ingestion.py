import os
import zipfile
import gdown
from src.cnnClassifier import logger
from src.cnnClassifier.utils.common import get_size
from src.cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''
        zip_path   = Path(self.config.local_data_file)
        extract_to = Path(self.config.unzip_dir)
        # if zip exists AND unzip dir has files, skip
        if zip_path.exists() and extract_to.exists() and any(extract_to.iterdir()):
            logger.info("Artifacts already present; skipping download.")
            return str(zip_path)
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = Path(self.config.unzip_dir)
        # if already extracted, skip
        if unzip_path.exists() and any(unzip_path.iterdir()):
            logger.info("Data already extracted; skipping unzip.")
            return
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)