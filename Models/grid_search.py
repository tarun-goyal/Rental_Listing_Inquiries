from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
import re
from data_cleansing import clean_design_matrix
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# Reading data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
stop = stopwords.words('english')
feat_to_label_encode = ['street_address', 'building_id', 'manager_id']
feat_to_clean = ['features', 'description']


def cleaning_text(sentence):
    stemmer = PorterStemmer()
    sentence = sentence.lower()
    sentence = re.sub('[^\w\s]', ' ', sentence)  # removes punctuations
    sentence = re.sub('\d+', ' ', sentence)  # removes digits
    cleaned = ' '.join([w for w in sentence.split() if w not in stop])
    # removes english stopwords
    cleaned=' '.join([w for w in cleaned.split() if not len(w) <= 2])
    # removes single or double lettered words and digits
    word_list = [stemmer.stem(word) for word in cleaned.split(" ")]
    cleaned = ' '.join([w for w in word_list])
    cleaned = cleaned.strip()
    return cleaned

train_rental['street_address'] = train_rental['street_address']\
    .apply(lambda x: cleaning_text(x))

for feature in feat_to_label_encode:
    label = LabelEncoder()
    label.fit(list(train_rental[feature].values))
    train_rental[feature] = label.transform(list(train_rental[feature].values))

train_rental['features'] = train_rental["features"].apply(
    lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

for feature in feat_to_clean:
    if feature == 'description':
        train_rental[feature] = train_rental[feature].apply(
            lambda x: cleaning_text(x))
    train_rental = train_rental.reset_index(drop=True)
    vectors = CountVectorizer(max_features=100)
    train_counts = pd.DataFrame(vectors.fit_transform(
        train_rental[feature]).toarray(), columns=vectors.get_feature_names())
    train_rental = train_rental.join(train_counts, rsuffix='_N')


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
            seed=99, n_estimators=239, learning_rate=0.5,
            objective='multi:softprob')
        parameters = {'max_depth': [3],
                      'min_child_weight': [2],
                      'subsample': [0.9],
                      'colsample_bytree': [0.8],
                      'gamma': [0.],
                      'reg_alpha': [3.5, 4., 4.5]}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=3, verbose=4,
                             scoring='neg_log_loss', iid=False)
        model.fit(self.design_matrix[self.predictors].as_matrix(),
                  self.design_matrix['interest_level'].as_matrix())
        print model.best_params_
        print model.best_score_
        cv_results = model.cv_results_
        results = DataFrame.from_dict(cv_results, orient='columns')
        results.to_csv('../Model_results/XGB_GridSearch6.csv',
                       index=False)

Model().grid_search_for_best_estimator()
