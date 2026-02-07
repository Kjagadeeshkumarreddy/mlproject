'''here we will write all the code 
thar requird to take the data from the 
database like mogodb or cloud'''

import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

import pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in models:
            model=models.get(i)
            param=params.get(i)

            gs=GridSearchCV(model,param,cv=3,n_jobs=-1)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[i]={
                "test_model_score":test_model_score,
                "train_model_score":train_model_score,
                "best_parameters":gs.best_params_
            }
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            model=pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    return model
