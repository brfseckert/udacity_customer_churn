"""
Runs the end-to-end machine learning pipeline to predict customer churn
"""

from typing import List, Type
import joblib

import numpy as np
from pandas import read_csv, DataFrame

from matplotlib.pyplot import figure, savefig, close, title, ylabel, xticks, bar
from seaborn import histplot, heatmap # pylint: disable=W0611

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from udacity_customer_churn.models import RandomForestModel, LogisticRegressionModel
from udacity_customer_churn.constants import DATA_FOLDER_PATH, EDA_FOLDER_PATH, CATEGORICAL_COLUMNS, KEEP_COLUMNS, SEED

class Pipeline:
    """Pipeline object containing the steps of the training machine learning pipeline"""

    def __init__(self, model):
        self.data_path = DATA_FOLDER_PATH
        self.model = model.model
        self.model_name = model.name
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_train_and_test_datasets()

    @staticmethod
    def _save_figure(
            figure_name: str,
            expression: str,
            data_frame: DataFrame,
            folder: str) -> None:
        """
        Saves a figure based on the evaluation of an expression
        Arguments:
            figure_name: name of the figure to be saved
            expression: expression to be evaluated. pyplot object
            df: dataframe against where the expression will be performed
            folder: destination path for the figure
        """
        figure(figsize=(20, 10))
        title(figure_name)
        eval(expression)
        savefig(f'{folder}{figure_name}.png')
        close()

    def _save_classification_report(
            self,
            model,
            response: DataFrame,
            features: DataFrame,
            folder: str,
            name: str) -> None:
        report = classification_report(
            response, self.predict(
                model, features), output_dict=True)
        data_frame = DataFrame(report).transpose()
        data_frame.to_csv(f'{folder}{self.model_name}_{name}.csv')

    @staticmethod
    def prepare_raw_dataframe(path) -> DataFrame:
        """Prepares the raw dataframe for subsequent steps"""
        data_frame = read_csv(path)
        data_frame.columns = [column.lower() for column in data_frame.columns]
        return data_frame.assign(churn=data_frame['attrition_flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1))

    @staticmethod
    def perform_eda(eda_folder: str) -> None:
        """
        Saves figures defined for subsequent exploratory data analysis (eda)
        """

        eda_figures = {
            'customer_churn_histogram': 'data_frame["churn"].hist()',
            'customer_age_histogram': 'data_frame["customer_age"].hist()',
            'martial_status_breakdown': 'data_frame["marital_status"].value_counts("normalize").plot(kind="bar")', # pylint: disable=C0301
            'transaction_count_density': 'histplot(data_frame["total_trans_ct"], stat="density", kde=True)', # pylint: disable=C0301
            'correlation_analysis': 'heatmap(data_frame.corr(), annot=False, cmap="Dark2_r", linewidths = 2)'} # pylint: disable=C0301
        data_frame = Pipeline.prepare_raw_dataframe(DATA_FOLDER_PATH)
        return list(map(lambda figure: Pipeline._save_figure(
            figure[0], figure[1], data_frame, eda_folder), eda_figures.items()))

    def perform_feature_engineering(self,
                                    categorical_columns: List[str],
                                    keep_columns: List[str]) -> DataFrame:
        """
        Transform each categorical column into it's propotion of churn for each category
        Filters only relevant columns for model training
        Arguments:
            categorical_columns: list of categorical columns to be transformed
            keep_columns: list of relevant columns for training the model
        """
        raw_data = Pipeline.prepare_raw_dataframe(DATA_FOLDER_PATH)
        categorical_columns = {f'{column}_churn': raw_data.groupby(
            column)['churn'].transform('mean') for column in categorical_columns}
        raw_data = raw_data.assign(**categorical_columns)
        return raw_data[keep_columns]

    def split_train_and_test_datasets(self) -> List[DataFrame]:
        """Split the data into a training and testing dataframes"""
        data_frame = self.perform_feature_engineering(
            CATEGORICAL_COLUMNS, KEEP_COLUMNS)
        features = data_frame[filter(lambda x: x != 'churn', data_frame.columns)]
        response = data_frame['churn']
        x_train, x_test, y_train, y_test = train_test_split(
            features, response, test_size=0.3, random_state=SEED)

        return [x_train, x_test, y_train, y_test]

    def train_model(self) -> Type:
        """Train model(s) using the training data."""
        return self.model.fit(self.x_train, self.y_train)

    def get_best_estimator(self) -> Type:
        """Gets the best trained model based on hyperparameter tunning."""
        return self.train_model().best_estimator_

    def predict(self, model: Type, features: DataFrame) -> DataFrame:
        """
        Predicts the reponse variable based on the trained model and input data
        Arguments
            x: input data for prediction
        """
        return model.predict(features)

    def save_estimator_evaluation_metrics(
            self, model: Type, evaluation_folder='images/results/') -> None:
        """Saves trained model key evaluation metrics"""
        # ROC curve
        figure(figsize=(15, 8))
        plot_roc_curve(model, self.x_test, self.y_test, alpha=0.8)
        savefig(f'{evaluation_folder}{self.model_name}_roc.png')

        # Classification report results
        self._save_classification_report(
            model,
            self.y_train,
            self.x_train,
            evaluation_folder,
            'train_results')
        self._save_classification_report(
            model,
            self.y_test,
            self.x_test,
            evaluation_folder,
            'test_results')

        # Feature importance
        if self.model_name == 'random_forest':
            figure(figsize=(20, 5))

            # Calculate feature importances
            importances = model.feature_importances_

            # Sort feature importances in descending order
            indices = np.argsort(importances)[::-1]

            # Rearrange feature names so they match the sorted feature
            # importances
            names = [self.x_train.columns[i] for i in indices]

            title("Feature Importance")
            ylabel('Importance')
            bar(range(self.x_train.shape[1]), importances[indices])
            xticks(range(self.x_train.shape[1]), names, rotation=90)
            savefig(
                f'{evaluation_folder}{self.model_name}_feature_importance.png')

    def save_model(self, model: Type) -> None:
        """
        Serialize and saves trained model
        Arguments:
            model: trained model to be serialized and saved
        """
        joblib.dump(model, f'./models/{self.model_name}_model.pkl')

    def run_pipeline(self) -> None:
        """Runs the end-to-end machine learning pipeline"""
        best_model = self.get_best_estimator()
        self.save_model(best_model)
        self.save_estimator_evaluation_metrics(best_model)
