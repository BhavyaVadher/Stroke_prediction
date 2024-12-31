import pandas as pd
import os
from src.mlProject import logger
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import ElasticNet
import joblib
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.utils.common import evaluteModel
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self , model_evaluation_config):

        try:

            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            logger.info("train and test data read successfully.")

            train_x = train_data.drop([self.config.target_column], axis=1)
            test_x = test_data.drop([self.config.target_column], axis=1)
            train_y = train_data[[self.config.target_column]]
            test_y = test_data[[self.config.target_column]]

            models = {
                "Logistic Regression" : LogisticRegression(),
                "SVC" : SVC(),
                "Guassian Naive Bayes" : GaussianNB(),
                "RandomForestClassifier" : RandomForestClassifier(),
              
            }

            hyperparameters = {
                "Logistic Regression": {
                    "penalty": ["l2"],
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "class_weight": ['balanced'],
                    "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                },

                "SVC": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "gamma": ["scaled", "auto"],
                    "class_weight": ['balanced']
                },

                "Guassian Naive Bayes": {},

                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["gini", "entropy"],
                    "class_weight": ['balanced']
                }
              
            }

            
            logger.info("evaluate model started")

            model_performance:dict = evaluteModel(train_x , test_x , train_y , test_y , models , hyperparameters , model_evaluation_config)
            
            best_model_score = max(sorted(model_performance.values()))

            best_model_name = list(model_performance.keys())[
                list(model_performance.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No best model found")            

            print( best_model_name , best_model_score)
            logger.info("Best model found with score" , best_model_score)

            joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))
            

        except Exception as e:
            raise Exception(e, sys)



