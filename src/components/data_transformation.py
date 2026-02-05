import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ..exceptions import CustomException
from ..logger import logging

from src.utils import save_object

import os

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    '''
        This function si responsible for data trnasformation
        
    '''
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            cat_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipline,numerical_columns),
                    ("cat_piplines",cat_pipline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):
        try:

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading training and testing is completed")

            preprocessor_obj=self.get_data_transformer_object()
            
            target_column="math_score"
            x_train=train_df.drop(target_column,axis=1)
            y_train=train_df[target_column]
            x_test=test_df.drop(target_column,axis=1)
            y_test=test_df[target_column]


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            x_train_scaled=preprocessor_obj.fit_transform(x_train)
            x_test_scaled=preprocessor_obj.transform(x_test)

            '''this np.c_ concates the x_tran_scaled 
            and the y_tran as a last column '''
            train_arr=np.c_[
                x_train_scaled,np.array(y_train)
            ]
            test_arr=np.c_[
                x_test_scaled,np.array(y_test)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocesser_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        