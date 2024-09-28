# adding src to the system path
import sys
sys.path.insert(0, '/media/thirdeye/Data/ai.corp.eye/First-MLOPs-Project/')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger.custom_logging import logging
from src.exception import customexception
from src.utils.utils import save_object

class DataTransformationConfig:
    #TODO define path transformation pickle artifacts
    preprocessor_path=os.path.join("../../artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transfomation_config=DataTransformationConfig()

    def initiate_data_transform(self, train_path, test_path):
        try:
            #TODO Load train and test csv data
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            #TODO get catgorical and numerical coloumns
            target_column_name = 'price'
            train_target_feature_data=train_data[target_column_name]
            train_data=train_data.drop(['id', target_column_name], axis=1)
            test_target_feature_data=test_data[target_column_name]
            test_data=test_data.drop(['id', target_column_name],axis=1)

            catagorical_cols=train_data.select_dtypes(include="object").columns
            numerical_cols=train_data.select_dtypes(exclude="object").columns
            logging.info(f"Catagarical_columns: {catagorical_cols}")
            logging.info(f"Numerical_columns: {numerical_cols}")

            #TODO get all unique values for each catagorical columns
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            #TODO define catagarical pipeline
            catagorical_pipeline=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinalencoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                        ("scaler", StandardScaler())
                        ]
                    )
            #TODO define numerical pipeline
            numerical_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler())
                        ]
                    )

            #TODO perform transformation
            preprocessor=ColumnTransformer(
                    [
                        ("numerical_pipeline", numerical_pipeline, numerical_cols),
                        ("catagorical_pipeline", catagorical_pipeline, catagorical_cols)
                        ]
                    )

            #TODO concat transfomed train and test feature with corresponding target feature and return it
            trans_train_data=preprocessor.fit_transform(train_data)
            trans_test_data=preprocessor.fit_transform(test_data)

            #TODO column stacking
            final_train_data=np.c_[trans_train_data, np.array(train_target_feature_data)]
            final_test_data=np.c_[trans_test_data, np.array(test_target_feature_data)]
            
            #TODO save preprocessor object in artifacts in pickle format
            save_object(
                    file_path=self.transfomation_config.preprocessor_path, 
                    obj=preprocessor
                    )

            return (
                    final_train_data,
                    final_test_data)
                    
        except Exception as e:
            logging.info("Exception occured while data ingestion")
            raise customexception(e, sys)

obj=DataTransformation()
obj.initiate_data_transform("../../artifacts/train.csv", "../../artifacts/test.csv")
