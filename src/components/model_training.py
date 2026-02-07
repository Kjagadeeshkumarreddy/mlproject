import sys
import os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exceptions import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts',"model.pkl")

class ModelTrinner:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            x_train=train_array[:,:-1]
            y_train=train_array[:,-1]
            x_test=test_array[:,:-1]
            y_test=test_array[:,-1]
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test
                                         ,models=models,params=params)
            print(model_report)
            best_model=dict()
            max_score=0
            for i in  model_report:
                if(model_report.get(i).get('test_model_score')>max_score):
                    max_score=model_report.get(i).get('test_model_score')
                    best_model={i:model_report.get(i)}

            print(max_score)
            print(best_model)
            best_model_instance=models.get(list(best_model)[0])
            best_model_instance.fit(x_train,y_train)
            y_pred=best_model_instance.predict(x_test)
            best_score=r2_score(y_test,y_pred)
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model_instance
            )
            return best_score
        except Exception as e:
            raise CustomException(e,sys)