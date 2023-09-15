"""
Defines the constants used throughout the project

Author: Felipe Eckert
Date: 2023-09-15
""""

DATA_FOLDER_PATH = 'data/bank_data.csv'
EDA_FOLDER_PATH='images/eda/'

CATEGORICAL_COLUMNS = [
    'gender',
    'education_level',
    'marital_status',
    'income_category',
    'card_category'        
]

KEEP_COLUMNS = [
    'customer_age', 'dependent_count', 'months_on_book',
    'total_relationship_count', 'months_inactive_12_mon',
    'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
    'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
    'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio',
    'gender_churn', 'education_level_churn', 'marital_status_churn', 
    'income_category_churn', 'card_category_churn', 'churn'
]

SEED = 1234

CV_FOLD_NUMBER = 5

RF_PARAM_GRID = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

LR_PARAM_GRID = { 
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['none', 'l2', 'l1'],
    'C': [100, 10, 1.0, 0.1],
    'max_iter':[100, 1000, 3000]
}
