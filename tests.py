"""
Tests and logs functions and classes used in the project

Author: Felipe Eckert
Date: 2023-09-15
"""

import logging
from main import Pipeline
import pytest
from constants import DATA_FOLDER_PATH, EDA_FOLDER_PATH, CATEGORICAL_COLUMNS, KEEP_COLUMNS
from matplotlib.pyplot import figure, savefig, close, title
from unittest.mock import MagicMock, call, Mock, patch, PropertyMock
from pandas import DataFrame
from models import LogisticRegressionModel

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@pytest.fixture
def pipeline_instance():
	mock_model = MagicMock()
	mock_model.model_name.return_value ='mock_name'
	return Pipeline(mock_model)

@pytest.fixture(scope='function')
def perform_eda():
    def _save_figure(figure_name, expression, folder):
        figure(figsize=(20, 10))
        title(figure_name)
        eval(expression)
        yield savefig(f'{folder}{figure_name}.png')
        close()
    return _save_figure

def test_prepare_raw_dataframe(pipeline_instance):
	"""Test prepare_raw_dataframe method"""
	try:
		data_frame = pipeline_instance.prepare_raw_dataframe(DATA_FOLDER_PATH)
		logging.info("Testing preare_raw_dataframe: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing preare_raw_dataframe: The file wasn't found")
		raise err

	try:
		assert data_frame.shape[0] > 0
		assert data_frame.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing preare_raw_dataframe: The file doesn't appear to have rows and columns")
		raise err

def test_perform_eda(pipeline_instance):
	"""Test perfom_eda method"""
	try:
		pipeline_instance.perform_eda(EDA_FOLDER_PATH)
		assert True
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: ERROR function does yields matplotlib.pyplot methods")
		raise err

def test_perform_feature_engineering(pipeline_instance):
	"""Test perform_feature_engineering method"""
	try:
		data_frame = pipeline_instance.perform_feature_engineering(CATEGORICAL_COLUMNS, KEEP_COLUMNS)
		assert len(data_frame.columns) == len(KEEP_COLUMNS)
		logging.info("Testing perform_feature_engineering: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_feature_engineering: ERROR dataframe does not have keep columns or categorical columns were not modified")
		raise err

def test_split_train_and_test_datasets(pipeline_instance):
	"""Test split_train_and_test_datasets method"""
	try:
		list_dataframes = pipeline_instance.split_train_and_test_datasets()
		assert len(list_dataframes) == 4
		logging.info("Testing split_train_and_test_datasets: SUCCESS")
	except AssertionError as err:
		logging.error("Testing split_train_and_test_datasets: ERROR train and test datasets split was not sucessfull")
		raise err
	
def test_train_model(pipeline_instance):
	"""Test train_model method"""
	try:
		trained_model = pipeline_instance.train_model()
		assert trained_model.cv_results_ is not None
		logging.info("Testing train_model: SUCCESS")
	except AssertionError as err:
		logging.error("Testing train_model: ERROR model(s) were not sucessfully trained")
		raise err
	
def test_get_best_estimator(pipeline_instance):
	"""Test get_best_estimator method"""
	try:
		mock_train_model = pipeline_instance.train_model()
		mock_train_model.best_estimator_ = "MockedBestEstimator"
		best_estimator = pipeline_instance.get_best_estimator()
		assert best_estimator == "MockedBestEstimator"
		logging.info("Testing get_best_estimator: SUCCESS")
	except AssertionError as err:
		logging.error("Testing get_best_estimator: ERROR model(s) were not sucessfully trained")
		raise err
	

def test_predict(pipeline_instance):
	"""Test predict method"""
	try:
		features = DataFrame(data={'customer_age': [1, 2,3], 'credit_limit': [3, 4,3]})
		predictions = DataFrame(data={'predictions': [1, 2, 3]})
		pipeline_instance.predict = MagicMock(return_value=predictions)
		pipeline_instance.predict(model=pipeline_instance.get_best_estimator(), features=features)
		pipeline_instance.predict.assert_called()
		logging.info("Testing predict: SUCCESS")
	except AssertionError as err:
		logging.error("Testing predict: ERROR model(s) were not sucessfully trained")
		raise err
	
def test_save_estimator_evaluation_metric(pipeline_instance):
	"""Test save_model method"""
	try:
		pipeline_instance.save_estimator_evaluation_metrics= MagicMock()
		mock_model = pipeline_instance.get_best_estimator()
		pipeline_instance.save_estimator_evaluation_metrics(mock_model)
		pipeline_instance.save_estimator_evaluation_metrics.assert_called()
		logging.info("Testing save_estimator_evaluation_metrics: SUCCESS")
	except AssertionError as err:
		logging.error("Testing save_estimator_evaluation_metrics: ERROR metrics were not saved")
		raise err
	
@patch('main.joblib.dump')
def test_save_model(mock_dump: MagicMock, pipeline_instance):
	"""Test save_model method"""
	try:
		mock_model = MagicMock()
		mock_name = pipeline_instance.model_name
		pipeline_instance.save_model(mock_model)
		logging.info(f'{mock_model}')
		logging.info(f'{pipeline_instance.model_name}')
		logging.info(f'{mock_dump.call_args_list}')
		mock_dump.assert_called_with(mock_model, f"./models/{str(mock_name)}_model.pkl")
		logging.info("Testing run_pipeline: SUCCESS")
	except AssertionError as err:
		logging.error("Testing save_model: ERROR model(s) were not successfully saved")
		raise err

@patch('main.Pipeline.get_best_estimator', return_value=MagicMock())
@patch('main.Pipeline.save_model')
@patch('main.Pipeline.save_estimator_evaluation_metrics')
def test_run_pipeline(mock_model: MagicMock, mock_save_model: MagicMock, mock_saved_metrics: MagicMock, pipeline_instance):
	"""Test run_pipeline method"""
	try:
		pipeline_instance.run_pipeline()
		mock_model.assert_called_once()
		mock_save_model.assert_called_once()
		mock_saved_metrics.assert_called_once()
		logging.info("Testing run_pipeline: SUCCESS")
	except AssertionError as err:
		logging.error("Testing run_pipeline: pipeline was not sucessfully completed")
		raise err










								
								 

	
	









