# adding src to the system path
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from src.logger.custom_logging import logging
from src.exception import customexception
from src.utils.utils import save_object, evaluate_model


class ModelTrainerConfig:
    #TODO define path for saved model path
    trained_model_path=os.path.join("../../artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting dependent and independent variable from train and test araay")
            #TODO split traina_arr and test_arr in dependent and independent feature
            x_train, y_train, x_test, y_test = \
            (train_arr[:,:-1], train_arr[:,-1], test_arr[:, :-1], test_arr[:,-1])
             
            #TODO Load various model to be train
            models = {
                    "LinearRegression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "Elasticnet": ElasticNet(),
                    "XGBoost": XGBRegressor()
                    }
            logging.info(f"training varios models ....")
            model_report=evaluate_model(x_train,y_train, x_test, y_test, models)
            logging.info(f"Model evalution report: {model_report}")
            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f"Best model with higest r2_score: {best_model_name}") 
            best_model=models[best_model_name]
            #TODO save best model 
            save_object(
                    file_path=self.trainer_config.trained_model_path, 
                    obj=best_model
                    )
        except Exception as e:
            logging.info("Exception occured while data ingestion")
            raise customexception(e, sys)

#from src.components.data_transformation import DataTransformation, DataTransformationConfig
#
#obj=DataTransformation()
#train_arr, test_arr=obj.initiate_data_transform("../../artifacts/train.csv", "../../artifacts/test.csv")
#
#obj1=ModelTrainer()
#obj1.initiate_model_training(train_arr, test_arr)
