import pytest
import pandas as pd
from autolearn.encoders import *


class TestEncoder:

    def setup_class(self):
        self.data = {
            'number': [1,5,1,2,4,4],
            'fruit': ['apple', 'banana', 'mango', 'apple', 'mango', 'mango'],
            'country': ['Germany', 'Switzerland', 'France', 'Germany', 'Spain', 'Spain']
        }
        self.df = pd.DataFrame(data=self.data)
        self.df['fruit'] = self.df['fruit'].astype('category')

    def test_fit_stored_columns(self):
        encoder_label = EncodeCategorical()
        encoder_label.fit(self.df)

        assert 'number' not in encoder_label.columns.values
        assert 'fruit' in encoder_label.columns.values
        assert 'country' in encoder_label.columns.values

    def test_fit_stored_classes(self):
        encoder_label = EncodeCategorical()
        encoder_label.fit(self.df)

        with pytest.raises(KeyError):
            encoder_label.encoders['number'].classes_
        assert (encoder_label.encoders['fruit'].classes_ == ['apple', 'banana', 'mango']).all
        assert (encoder_label.encoders['country'].classes_ == ['France', 'Germany', 'Spain', 'Switzerland']).all

    def test_fit_specify_columns(self):
        encoder_label = EncodeCategorical(columns=['country'])
        encoder_label.fit(self.df)

        with pytest.raises(KeyError):
            encoder_label.encoders['number'].classes_
        with pytest.raises(KeyError):
            encoder_label.encoders['fruit'].classes_
        assert (encoder_label.encoders['country'].classes_ == ['France', 'Germany', 'Spain', 'Switzerland']).all

    def test_fit_specify_target(self):
        encoder_label = EncodeCategorical()
        encoder_label.fit(self.df, target='fruit')

        with pytest.raises(KeyError):
            encoder_label.encoders['number'].classes_
        with pytest.raises(KeyError):
            encoder_label.encoders['fruit'].classes_
        assert (encoder_label.encoders['country'].classes_ == ['France', 'Germany', 'Spain', 'Switzerland']).all

    # def test_transform(self):
    #
    # def test_inverse_transform(self):
