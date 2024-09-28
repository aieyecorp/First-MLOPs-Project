# adding src to the system path
import os
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
from src.exception import customexception
from src.logger.custom_logging import logging
from src.utils.utils import load_object
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub
dagshub.init(repo_owner='aieyecorp', repo_name='my-first-repo', mlflow=True)

class ModelEvolution:
    def __init__(self):
        pass
    def calculate_metrics(self, y_test, y_pred):
        rmse=np.sqrt(mean_squared_error(y_test, y_pred))
        mae=mean_absolute_error(y_test, y_pred)
        score=r2_score(y_test, y_test)
        return rmse, mae, score
    def initiate_model_evolution(self, test_arr):
        try:
            #TODO split dependent and independent variable from train and tes array
            x_test, y_test=(test_arr[:, :-1], test_arr[:, -1])
            logging.info(f"Model evolution has started ....")
            model_path=os.path.join("../../artifacts", "model.pkl")
            model=load_object(model_path)

            #TODO Use mlflow to register model
            mlflow.set_registry_uri("https://dagshub.com/aieyecorp/my-first-repo.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            #TODO use mlflow to track metrics and parameter
            with mlflow.start_run():
                y_pred=model.predict(x_test)
                #calculate rmse, mae, r2_score
                rmse, mae, r2=self.calculate_metrics(y_test, y_pred)
                #TODO log model metric values 
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                if tracking_url_type_store != "file":
                    print("I AM Here")
                    mlflow.sklearn.log_model(
                            model, "ml_model",
                            )
                else:
                    mlflow.sklearn.log_model(model, "ml_model")

        except Exception as e:
            logging.info("Exception occured while model evolution ...")
            raise customexception(e,sys)

model_evolution=ModelEvolution()
#TODO load "../../artifacts/test.csv"
test_data=pd.read_csv(os.path.join("../../artifacts", "test.csv"))
#TODO perform trandformation on test data
test_feature_dependent=test_data.drop(['id', 'price'],axis=1)
test_feature_independent=test_data['price']
preprocessor=load_object(os.path.join("../../artifacts", "preprocessor.pkl"))
trans_test_feature_dependent=preprocessor.transform(test_feature_dependent)
final_concat_test_data=np.c_[trans_test_feature_dependent, np.array(test_feature_independent)]
model_evolution.initiate_model_evolution(final_concat_test_data)
