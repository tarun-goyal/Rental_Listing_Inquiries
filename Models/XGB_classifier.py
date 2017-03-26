import pandas as pd
import numpy as np
import re
from data_cleansing import clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from nltk.corpus import stopwords


# Reading training and test data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
test_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/test.json',
                           convert_dates=['created'])
stop = stopwords.words('english')
feat_to_clean = ['display_address', 'building_id', 'manager_id']


def cleaning_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub('[^\w\s]', ' ', sentence)  # removes punctuations
    sentence = re.sub('\d+', ' ', sentence)  # removes digits
    cleaned = ' '.join([w for w in sentence.split() if w not in stop])
    # removes english stopwords
    cleaned = cleaned.replace('avenue', 'ave')
    cleaned=' '.join([w for w in cleaned.split() if not len(w) <= 2])
    # removes single or double lettered words and digits
    cleaned = cleaned.strip()
    return cleaned

train_rental['display_address'] = train_rental['display_address']\
    .apply(lambda x: cleaning_text(x))
test_rental['display_address'] = test_rental['display_address']\
    .apply(lambda x: cleaning_text(x))

for feature in feat_to_clean:
    label = LabelEncoder()
    label.fit(list(train_rental[feature].values) +
              list(test_rental[feature].values))
    train_rental[feature] = label.transform(list(train_rental[feature].values))
    test_rental[feature] = label.transform(list(test_rental[feature].values))


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix, test_size=0.3)
    return train, validation


class GradientBoostingModel(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values)
                           if ele not in ['interest_level']]

    @staticmethod
    def _build_model(design_matrix, predictors):
        """Building gradient boosting model with default parameters."""
        model = XGBClassifier(
            seed=99, n_estimators=3590, learning_rate=0.05,
            objective='multi:softprob', subsample=0.9, colsample_bytree=0.8,
            max_depth=3, min_child_weight=4, reg_alpha=2)
        model.fit(design_matrix[predictors], design_matrix['interest_level'])
        return model

    def _calculate_log_loss(self, iterations=10):
        """Compute multi-class logarithmic loss using cross-validations"""
        multi_log_loss = []
        for itr in range(iterations):
            training, validation = _train_validation_split(self.design_matrix)
            model = self._build_model(training, self.predictors)
            predicted = model.predict_proba(validation[self.predictors])
            loss = log_loss(validation['interest_level'], predicted)
            print 'log loss: ', loss
            multi_log_loss.append(loss)
        print multi_log_loss
        print np.mean(multi_log_loss)
        return np.mean(multi_log_loss)

    def _make_predictions(self):
        """Make predictions on test data"""
        test_data = clean_design_matrix(test_rental)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        model = self._build_model(self.design_matrix, predictors)
        predictions = model.predict_proba(test_data[predictors])
        return model, predictions

    def submission(self):
        """Submitting solutions"""
        model, predictions = self._make_predictions()
        print model.classes_
        submission = test_rental[['listing_id']]
        submission['high'] = predictions[:, 0]
        submission['medium'] = predictions[:, 2]
        submission['low'] = predictions[:, 1]
        submission.to_csv('../Submissions/XGB_label_encoding_tuned_' + str(
            self._calculate_log_loss()) + '.csv', index=False)

GradientBoostingModel().submission()
