import pandas as pd
import numpy as np
import re
from data_cleansing import clean_design_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import PorterStemmer


# Reading training and test data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
test_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/test.json',
                           convert_dates=['created'])
stop = stopwords.words('english')
feat_to_label_encode = ['street_address', 'building_id', 'manager_id']
feat_to_clean = ['features']


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


def cleaning_text2(sentence):
    sentence = sentence.lower()
    rdict = {
        'reduced_fee': 'no_fee'
    }
    robj = re.compile('|'.join(rdict.keys()))
    cleaned = robj.sub(lambda m: rdict[m.group(0)], sentence)
    cleaned = cleaned.strip()
    return cleaned

train_rental['street_address'] = train_rental['street_address']\
    .apply(lambda x: cleaning_text(x))
test_rental['street_address'] = test_rental['street_address']\
    .apply(lambda x: cleaning_text(x))

for feature in feat_to_label_encode:
    label = LabelEncoder()
    label.fit(list(train_rental[feature].values) +
              list(test_rental[feature].values))
    train_rental[feature] = label.transform(list(train_rental[feature].values))
    test_rental[feature] = label.transform(list(test_rental[feature].values))

train_rental['features'] = train_rental["features"].apply(
    lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_rental['features'] = test_rental["features"].apply(
    lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
train_rental['description'] = train_rental['description'].apply(
    lambda x: cleaning_text(x))
test_rental['description'] = test_rental['description'].apply(
    lambda x: cleaning_text(x))

for feature in feat_to_clean:
    train_rental[feature] = train_rental[feature].apply(
        lambda x: cleaning_text2(x))
    test_rental[feature] = test_rental[feature].apply(
        lambda x: cleaning_text2(x))
    train_rental = train_rental.reset_index(drop=True)
    test_rental = test_rental.reset_index(drop=True)
    vectors = CountVectorizer(max_features=200)
    train_counts = pd.DataFrame(vectors.fit_transform(
        train_rental[feature]).toarray(), columns=vectors.get_feature_names())
    test_counts = pd.DataFrame(vectors.transform(
        test_rental[feature]).toarray(), columns=vectors.get_feature_names())
    train_rental = train_rental.join(train_counts, rsuffix='_N')
    test_rental = test_rental.join(test_counts, rsuffix='_N')


def _train_validation_split(design_matrix):
    """Split into training and validation sets"""
    train, validation = train_test_split(design_matrix.as_matrix(),
                                         test_size=0.3)
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
            seed=99, n_estimators=3553, learning_rate=0.05, silent=False,
            objective='multi:softprob', subsample=0.9, colsample_bytree=0.8,
            max_depth=3, min_child_weight=2, gamma=0, reg_alpha=4)
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
        submission.to_csv('../Submissions/XGB_FeatText5_tuned.csv',
                          index=False)
        # str(self._calculate_log_loss()) + '.csv', index=False)

GradientBoostingModel().submission()
