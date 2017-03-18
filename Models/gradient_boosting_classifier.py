import pandas as pd
import numpy as np
from data_cleansing import clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier


# Reading training and test data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
test_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/test.json',
                           convert_dates=['created'])


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix, test_size=0.33)
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
        model = GradientBoostingClassifier(
            n_estimators=2500, learning_rate=0.01, criterion='friedman_mse',
            min_samples_split=2, min_samples_leaf=3, subsample=0.5)
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

    def _make_predictions_using_cv_fit(self):
        """Make predictions on test data"""
        test_data = clean_design_matrix(test_rental)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        feature_imp = pd.DataFrame(columns=predictors)
        model = self._build_model(self.design_matrix, predictors)
        feature_imp.loc[0] = model.feature_importances_
        predictions = model.predict_proba(test_data[predictors])
        return model, predictions, feature_imp

    def submission(self):
        """Submitting solutions"""
        model, predictions, importance = self._make_predictions_using_cv_fit()
        print model.classes_
        submission = test_rental[['listing_id']]
        submission['high'] = predictions[:, 0]
        submission['medium'] = predictions[:, 2]
        submission['low'] = predictions[:, 1]
        submission.to_csv('../Submissions/GB2_tuned_' + str(
            self._calculate_log_loss()) + '.csv', index=False)
        importance.to_csv('../Model_results/GB_feature_imp2.csv', index=False)

GradientBoostingModel().submission()
