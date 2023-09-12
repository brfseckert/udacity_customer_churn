from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from udacity_customer_churn.constants import RF_PARAM_GRID, LR_PARAM_GRID

from typing import List, Dict, Any

class RandomForestModel:
    def __init__(self):
        self.name = 'random_forest'
        self.model_type = RandomForestClassifier()
        self.param_grid = RF_PARAM_GRID
        self.fold_number = 5
        self.model = self.assemble_hyperparameters()
        
    def assemble_hyperparameters(self)->  RandomForestClassifier:
         return GridSearchCV(
             estimator=self.model_type, 
             param_grid=self.param_grid,
             cv=self.fold_number
         )

class LogisticRegressionModel:
    def __init__(self):
        self.name = 'logistic_regression'
        self.model = LogisticRegression()
        self.param_grid = LR_PARAM_GRID
        self.fold_number = 5
        self.model = self.assemble_hyperparameters()
        
    def assemble_hyperparameters(self)->  LogisticRegression:
         return GridSearchCV(
             estimator=self.model, 
             param_grid=self.param_grid,
             cv=self.fold_number
         )
