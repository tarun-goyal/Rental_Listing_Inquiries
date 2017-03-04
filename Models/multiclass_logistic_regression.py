import pandas as pd
import numpy as np
from data_cleansing import _clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# Reading training and test data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json')
test_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/test.json')


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix, test_size=0.3)
    return train, validation


class MultiClassLogRegression(object):

    def __init__(self):
        self.design_matrix = _clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values)
                           if ele not in ['interest_level']]

    @staticmethod
    def _build_model(design_matrix, predictors):
        """Building multi-class logistic model with default parameters using
        cross-validation. Predictor: Price"""
        model = LogisticRegressionCV(max_iter=1000, scoring='neg_log_loss')
        model.fit(design_matrix[predictors], design_matrix[['interest_level']]
                  .values.ravel())
        return model

    def _calculate_log_loss(self, iterations=10):
        """Compute multi-class logarithmic loss using cross-validations"""
        multi_log_loss = []
        for itr in range(iterations):
            training, validation = _train_validation_split(self.design_matrix)
            model = LogisticRegression(max_iter=1000)
            model.fit(training[self.predictors], training[['interest_level']]
                      .values.ravel())
            predicted = model.predict_proba(validation[self.predictors])
            loss = log_loss(validation[['interest_level']], predicted)
            print 'log loss: ', loss
            multi_log_loss.append(loss)
        print multi_log_loss
        print np.mean(multi_log_loss)
        return np.mean(multi_log_loss)

    def _make_predictions_using_cv_fit(self):
        """Make predictions on test data"""
        test_data = _clean_design_matrix(test_rental)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        model = self._build_model(self.design_matrix, predictors)
        predictions = model.predict_proba(test_data[predictors])
        return model, predictions

    def submission(self):
        """Submitting solutions"""
        model, predictions = self._make_predictions_using_cv_fit()
        print model.classes_
        submission = test_rental[['listing_id']]
        submission['high'] = predictions[:, 0]
        submission['medium'] = predictions[:, 2]
        submission['low'] = predictions[:, 1]
        self._calculate_log_loss()
        # submission.to_csv('../Submissions/log_reg3_CV_' + str(
        #     self._calculate_log_loss()) + '.csv', index=False)

MultiClassLogRegression().submission()
