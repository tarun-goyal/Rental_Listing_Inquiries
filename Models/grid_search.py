from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
import re
from data_cleansing import clean_design_matrix
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


# Reading data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
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

for feat in feat_to_clean:
    label = LabelEncoder()
    label.fit(list(train_rental[feat].values))
    train_rental[feat] = label.transform(list(train_rental[feat].values))


class Model(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values)
                           if ele not in ['interest_level']]

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        """Define model fit function & parameters"""
        regressor = XGBClassifier(
            seed=99, n_estimators=232, learning_rate=0.5,
            objective='multi:softprob')
        parameters = {'max_depth': [3],
                      'min_child_weight': [4],
                      'subsample': [0.9],
                      'colsample_bytree': [0.8],
                      'reg_alpha': [1.8, 1.9, 2., 2.1, 2.2]}
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
