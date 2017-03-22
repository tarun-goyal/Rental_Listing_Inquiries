from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
from data_cleansing import clean_design_matrix


# Reading data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])


class Model(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values)
                           if ele not in ['interest_level']]

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        """Define model fit function & parameters"""
        regressor = XGBClassifier(seed=99,
            objective='multi:softprob', max_depth=5, min_child_weight=2,
            gamma=0.3, subsample=0.85, colsample_bytree=0.75)
        parameters = {'reg_alpha': [i/100.0 for i in range(195, 210, 5)]}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=5, verbose=4,
                             scoring='neg_log_loss', iid=False)
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['interest_level'])
        print model.best_params_
        print model.best_score_
        cv_results = model.cv_results_
        results = DataFrame.from_dict(cv_results, orient='columns')
        results.to_csv('../Model_results/XGB_GridSearch6.csv',
                       index=False)

Model().grid_search_for_best_estimator()
