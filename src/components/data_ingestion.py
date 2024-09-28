# adding src to the system path
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger.custom_logging import logging
from src.exception import customexception

class DataIngestionConfig:
    #TODO define path for train, test and raw csv artifacts
    raw_data_path=os.path.join("../../artifacts","raw.csv") 
    train_data_path=os.path.join("../../artifacts","train.csv")
    test_data_path=os.path.join("../../artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            #TODO Load train 
            raw_data=pd.read_csv(os.path.join("../../experiment", "train.csv"))
            #TODO train and test split and prepare train and test csv artifacts
            train_data, test_data = train_test_split(raw_data, test_size=0.3)
            logging.info("Raw data has been splitted !!")

            #TODO save train_data artifacts 
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            #TODO save test data artifacts
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data Ingestion has been completed !!")
            return ( 
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )
        except Exception as e:
            logging.info("Exception occured while data ingestion")
            raise customexception(e, sys)

#obj=DataIngestion()
#obj.initiate_data_ingestion()
