import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class EncodeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = None

    def fit(self, data, target=None):
        if self.columns is None:
            self.columns = data.select_dtypes(include=['object', 'category']).columns

        self.encoders = {}
        for column in self.columns:
            self.encoders[column] = LabelEncoder().fit(data[column].dropna())
        return self

    def transform(self, data):
        output = data.copy()
        for column, encoder in self.encoders.items():
            index = data[column].dropna().index
            output.ix[index, [column]] = encoder.transform(data[column].dropna())
        return output

    def inverse_transform(self, data):  # fixme. Not working
        output = data.copy()
        for column, encoder in self.encoders.items():
            print(column, encoder)
            print(encoder.classes_)
            index = data[column].dropna().index
            print(index)
            print(np.asarray(data[column].dropna()))
            output.ix[index, [column]] = encoder.inverse_transform(data[column].dropna())
        return output
