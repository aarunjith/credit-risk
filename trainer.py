from credit_data_actual_values import substitute
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from loguru import logger
import json

import warnings
warnings.filterwarnings('ignore')


class ColumnNotFoundError(Exception):
    pass


class Trainer:
    def __init__(self):
        with open('data/train.data', 'rb') as file:
            self.X_train, self.y_train = pickle.load(file)

        with open('data/test.data', 'rb') as file:
            self.X_test, self.y_test = pickle.load(file)

        with open('models/scaler.model', 'rb') as file:
            self.sc = pickle.load(file)

        self.all_features = self.X_train.columns
        self.continuos_features = ['Duration in month',
                                   'Credit amount',
                                   'Installment rate in percentage of disposable income',
                                   'Age in years',
                                   'Number of existing credits at this bank',
                                   'Number of people being liable to provide maintenance for']

        self.categorical_columns = ['Status of existing checking account',
                                    'Credit history', 'Purpose', 'Savings account/bonds',
                                    'Present employment since',
                                    'Personal status and sex', 'Other debtors / guarantors',
                                    'Present residence since', 'Property',
                                    'Other installment plans', 'Housing',
                                    'Job', 'Telephone',
                                    'foreign worker']

        self.current_model = open(
            'models/latest_version.txt', 'r').readlines()[-1]

        with open(f'models/{self.current_model}', 'rb') as file:
            self.model = pickle.load(file)

    def clean_column_name(self, columns):
        '''Method to clean up the column names'''

        cleaned = []
        for column in columns:
            cleaned.append(column.replace("<", "lt").replace(
                ">", "gt").replace(",", " "))
        return cleaned

    def preprocess(self, data):
        '''Method to preprocess the input data into model compatible dataframe'''

        data_cols = list(data.columns)
        for col in self.continuos_features+self.categorical_columns:
            if col in data_cols:
                continue
            else:
                logger.info(f'Column {col} not found in data')
                raise(ColumnNotFoundError)
        data = data[self.continuos_features+self.categorical_columns]

        # Scale the continuous values
        data[self.continuos_features] = self.sc.transform(
            data[self.continuos_features])

        # One Hot Encode categorical values
        data = pd.get_dummies(data, columns=self.categorical_columns)
        data.columns = self.clean_column_name(data.columns)

        # Filter for columns trained only and add mising categorical columns
        for col in self.all_features:
            if col in data.columns:
                continue
            else:
                if not col in self.continuos_features:
                    data[col] = 0
        data = data[self.all_features]
        return data

    def train(self):
        '''Train the model using prebuilt dataframe'''

        logger.info(f'Shape of the training data: {self.X_train.shape}')

        filename = f'xgboost_v{int(self.current_model.split("_v")[-1][0])+1}.model'
        logger.info('Training the model using saved training data')
        model = XGBClassifier(learning_rate=0.01,
                              n_estimators=1000, max_depth=5)
        model.fit(self.X_train, self.y_train,
                  compute_sample_weight("balanced", self.y_train))

        logger.info(f'Dumping new model {filename} into models/ ')
        with open(f'models/{filename}', 'wb') as file:
            pickle.dump(model, file)
        version = open('latest_version.txt', 'w')
        version.writelines(f'\n{filename}')
        version.close()
        logger.info(f'Training Complete')

        logger.info('Evaluation in progress')
        y_pred = model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred)
        y_class_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
        # Calculating results for various evaluation metric
        y_class_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
        precision = precision_score(self.y_test, y_class_pred, average='micro')
        recall = recall_score(self.y_test, y_class_pred, average='micro')
        f1 = f1_score(self.y_test, y_class_pred)

        logger.info(f"AUC: {auc}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Precision: {precision}")
        logger.info(f"F1-score: {f1}")

        return {"AUC": auc, "recall": recall, "precision": precision, "f1": f1}


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
