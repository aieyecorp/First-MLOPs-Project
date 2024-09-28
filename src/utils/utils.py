import os
import pickle
# adding src to the system path
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
from src.exception import customexception
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)

def evaluate_model(x_train,y_train,x_test,y_test, models):
    try:
        report={}
        for ith_model in range(len(models)):
            model=list(models.values())[ith_model]
            model_name=list(models.keys())[ith_model]
            print(f"Training {model_name}")
            model.fit(x_train, y_train)
            #TODO prediction 
            y_predict=model.predict(x_test)
            ith_model_score=r2_score(y_test, y_predict)
            report[model_name] = ith_model_score

        return report
    except Exception as e:
        raise customexception(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise customexception(e, sys)
