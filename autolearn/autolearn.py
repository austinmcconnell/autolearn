# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder

from .encoders import EncodeCategorical


def all_algorithms(func):
    def wrapper(*args, **kwargs):
        with progressbar.ProgressBar(max_value=len(args[0].algorithms)) as pbar:
            for i, algorithm in enumerate(args[0].algorithms):
                kwargs['algorithm'] = algorithm
                func(*args, **kwargs)
                pbar.update(i+1)
    return wrapper


class AutoLearn(object):

    def __init__(self, encode_categoricals=False, onehot=False, impute=False, standardize=False, decompose=False,
                 impute_strategy='mean', missing_values='NaN', target=None, id_col=None, error_metric='rmse',
                 algorithms={'linear', 'ridge', 'lasso', 'bayes', 'bayes_ridge', 'boost', 'forest'}):

        impute_strategy_types = {'mean', 'median', 'most_frequent'}

        assert impute_strategy in impute_strategy_types,\
            'Strategy must be one of the following: {} {} {}'.format('mean', 'median', 'most_frequent')

        self.encode_categoricals = encode_categoricals
        self.onehot = onehot
        self.impute = impute
        self.impute_strategy = impute_strategy
        self.missing_values = missing_values
        self.standardize = standardize
        self.decompose = decompose
        self.target = target
        self.id_col = id_col
        self.error_metric = error_metric
        self.model = {}
        self.algorithms = algorithms
        self.encoder_label = None
        self.imputer = None
        self.encoder_onehot = None
        self.scaler = None
        self.pca = None

        for i, algorithm in enumerate(self.algorithms):
            self.model[algorithm] = {}

    def process_training_data(self, filename):
        training_data = pd.read_csv(filename, sep=',')

        if self.encode_categoricals:
            self.encoder_label = EncodeCategorical()
            self.encoder_label.fit(training_data)
            training_data = self.encoder_label.transform(training_data)

        X = training_data.copy()
        X.drop(self.target, axis=1, inplace=True)
        if self.id_col:
            X.drop(self.id_col, axis=1, inplace=True)

        mask = []
        for column in X.columns:
            if column in X.select_dtypes(include=['object', 'category']).columns:
                mask.append(True)
            else:
                mask.append(False)

        X = X.values
        y = training_data[self.target].values

        if self.impute:
            self.imputer = Imputer(missing_values=self.missing_values,
                                   strategy=self.impute_strategy,
                                   copy=False)
            self.imputer.fit(X)
            X = self.imputer.transform(X)

        if self.onehot:
            self.encoder_onehot = OneHotEncoder(categorical_features=mask,
                                                dtype=np.int,
                                                sparse=False,
                                                handle_unknown='ignore')
            X = self.encoder_onehot.fit_transform(X)

        if self.standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        if self.decompose:
            self.pca = PCA()
            self.pca.fit(X)

        return training_data, X, y

    def process_test_data(self, filename, separator=','):

        test_data = pd.read_csv(filename, sep=separator)

        if self.encode_categoricals:
            test_data = self.encoder_label.transform(test_data)

        X = test_data.copy()
        if self.id_col:
            X.drop(self.id_col, axis=1, inplace=True)
        X = X.values

        if self.impute:
            X = self.imputer.transform(X)

        if self.onehot:
            X = self.encoder_onehot.transform(X)

        return test_data, X

    def train(self, X, y, algorithm):

        if self.standardize:
            X = self.scaler.transform(X)

        if self.decompose:
            X = self.pca.transform(X)

        algs = {'linear': LinearRegression(),
                'logistic': LogisticRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(max_iter=10000),
                'bayes': GaussianNB(),
                'bayes_ridge': BayesianRidge(),
                'boost': GradientBoostingRegressor(),
                'forest': RandomForestRegressor()}

        model = algs[algorithm]

        model.fit(X, y)

        return model

    @all_algorithms
    def train_all(self, X, y, algorithm):
        self.model[algorithm]['model'] = self.train(X, y, algorithm)

    # def tune(self, model, X, y):
    #     print('add this feature')
    #
    # def tune_best(self, model, X, y):
    #     print('add this feature')

    def cross_validate(self, X, y, algorithm):

        skf = StratifiedKFold(n_splits=3)
        scores = []
        # print('X: {}\ty: {}'.format(X.shape, y.shape))
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # print('x_train: {}\tx_test: {}'.format(X_train.shape, X_test.shape))
            # print('y_train: {}\ty_test: {}'.format(y_train.shape, y_test.shape))
            model = self.train(X_train, y_train, algorithm)
            y_test_predictions = self.predict(X_test, model)
            scores.append(self.score(y_test, y_test_predictions))
        return scores

    @all_algorithms
    def cross_validate_all(self, X, y, algorithm):
        # print('Beginning cross validation for: {}'.format(algorithm))
        self.model[algorithm]['cv'] = self.cross_validate(X, y, algorithm)
        # print('Finished cross validation for: {}'.format(algorithm))

    def predict(self, X, model):
        if self.standardize:
            X = self.scaler.transform(X)

        if self.decompose:
            X = self.pca.transform(X)

        y = model.predict(X)

        return y

    @all_algorithms
    def predict_all(self, X, algorithm):
        model = self.model[algorithm]['model']
        self.model[algorithm]['predictions'] = self.predict(X, model)

    @staticmethod
    def visualize(y, y_predictions):

        actuals = y
        predictions = y_predictions
        residuals = actuals - predictions

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.regplot(x=predictions, y=actuals, ax=ax[0], color='#34495e')
        sns.regplot(x=predictions, y=residuals, ax=ax[1], fit_reg=False, color='#34495e')

        ax[0].set_title('Compare Predictions')
        ax[1].set_title('Residuals')

        plt.setp(ax[0].get_xticklabels(), rotation=45)
        plt.setp(ax[1].get_xticklabels(), rotation=45)
        return fig

    @all_algorithms
    def visualize_all(self, y, algorithm):
        y_predictions = self.model[algorithm]['predictions']
        return self.visualize(y, y_predictions)

    @staticmethod
    def root_mean_squared_logarithmic_error(y, y_pred):
        return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y), 2)))

    def score(self, y, y_pred, metric=None):

        if metric is None:
            metric = self.error_metric

        if metric == 'rmsle':
            return self.root_mean_squared_logarithmic_error(y, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y, y_pred)
        else:
            print('No metric defined for {}'.format(metric))
            exit()

    @all_algorithms
    def score_all(self, y, algorithm):
        y_predictions = self.model[algorithm]['predictions']
        self.model[algorithm]['scores'] = self.score(y, y_predictions)
        self.model[algorithm]['variance'] = self.variance(y_predictions)

    @staticmethod
    def variance(y_pred):
        return np.var(y_pred)

    def get_results(self):
        algs = []
        scores = []
        params = []
        variance = []
        cv = []
        for algorithm, item in self.model.items():
            algs.append(algorithm)
            scores.append(item['scores'])
            params.append(item['model'])
            variance.append(item['variance'])
            cv.append(item['cv'])
        data = {'algorithm': algs, 'score': scores, 'parameters': params, 'variance': variance, 'cv': cv}
        results = pd.DataFrame(data=data, index=algs, columns=['score', 'cv', 'variance', 'parameters'])
        results.sort_values('score', ascending=True, inplace=True)
        return results
