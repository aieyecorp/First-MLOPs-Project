# adding src to the system path
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
from src.exception import customexception
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            train_csv_path, test_csv_path=data_ingestion.initiate_data_ingestion()
            return train_csv_path, test_csv_path
        except Exception as e:
            raise customexception(e, sys)

    def start_data_transformation(self, train_csv_path, test_csv_path):
        try:
            data_transformation=DataTransformation()
            train_arr, test_arr=data_transformation.initiate_data_transform(train_csv_path, test_csv_path)
            return train_arr, test_arr
        except Exception as e:
            raise customexception(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer=ModelTrainer()
            model_trainer.initiate_model_training(train_arr,test_arr)
        except Exception as e:
            raise customexception(e, sys)

    def start_training(self):
        try:
            train_csv_path, test_csv_path = self.start_data_ingestion()
            train_arr, test_arr=self.start_data_transformation(train_csv_path, test_csv_path)
            self.start_model_training(train_arr, test_arr)
        except Exception as e:
            raise customexception(e,sys)
        
obj=TrainingPipeline()
obj.start_training()
