from sklearn.linear_model import LogisticRegression
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
        regressor = LogisticRegression(max_iter=5000, penalty='l2',
                                       solver='newton-cg')
        parameters = {'multi_class': ['ovr', 'multinomial']}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=5, verbose=2,
                             scoring='neg_log_loss', iid=False)
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['interest_level'])
        print model.best_params_
        print model.best_score_
        cv_results = model.cv_results_
        results = DataFrame.from_dict(cv_results, orient='columns')
        results.to_csv('../Model_results/LogReg_GridSearch3_results.csv',
                       index=False)

Model().grid_search_for_best_estimator()
