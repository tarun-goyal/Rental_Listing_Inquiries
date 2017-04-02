import pandas as pd
import xgboost as xgb
import re
import matplotlib.pylab as plt
from data_cleansing import clean_design_matrix
from matplotlib.pylab import rcParams
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer


rcParams['figure.figsize'] = 18, 12

# Reading training and test data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
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

for feature in feat_to_label_encode:
    label = LabelEncoder()
    label.fit(list(train_rental[feature].values))
    train_rental[feature] = label.transform(list(train_rental[feature].values))

train_rental['features'] = train_rental["features"].apply(
    lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
train_rental['description'] = train_rental['description'].apply(
    lambda x: cleaning_text(x))

for feature in feat_to_clean:
    train_rental[feature] = train_rental[feature].apply(
        lambda x: cleaning_text2(x))
    train_rental = train_rental.reset_index(drop=True)
    vectors = CountVectorizer(max_features=200)
    train_counts = pd.DataFrame(vectors.fit_transform(
        train_rental[feature]).toarray(), columns=vectors.get_feature_names())
    train_rental = train_rental.join(train_counts, rsuffix='_N')


class XGBoostingModel(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['interest_level']]

    @staticmethod
    def _build_model():
        """Model: Extreme Gradient Boosting model using tuned parameters"""
        model = XGBClassifier(
            seed=99, n_estimators=5000, learning_rate=0.05,
            objective='multi:softprob', subsample=0.9, colsample_bytree=0.8,
            max_depth=3, min_child_weight=2, gamma=0, reg_alpha=4)
        return model

    def _execute_cross_validation(self, model, cv_folds=5,
                                  early_stopping_rounds=50):
        """Get cross validated results based on Log loss"""
        xgb_param = model.get_xgb_params()
        xgb_param['num_class'] = 3
        xgtrain = xgb.DMatrix(self.design_matrix[self.predictors].values,
                              label=self.design_matrix['interest_level']
                              .values)
        cv_results = xgb.cv(xgb_param, xgtrain, nfold=cv_folds,
                            metrics='mlogloss',
                            num_boost_round=model.get_params()['n_estimators'],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=True)
        model.set_params(n_estimators=cv_results.shape[0])

        # Fit the algorithm on the data
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['interest_level'],
                  eval_metric='mlogloss')

        # Print model report:
        print "\nCV Results"
        print cv_results
        # cv_results.to_csv("../Model_results/XGB_tuned_CV9_results.csv")
        feat_imp = pd.Series(model.booster().get_fscore()).sort_values(
            ascending=False)
        feat_imp = feat_imp[:50]
        print feat_imp
        # feat_imp.plot(kind='bar', title='Feature Importance')
        # plt.ylabel('Feature Importance Score')
        # plt.savefig('../Model_results/XGB_Top50_feat_imp2.png')

    def submit_solution(self):
        """Submit the solution file"""
        self._execute_cross_validation(model=self._build_model())
        # predictions = self._make_predictions()
        # submission = test[['Id']]
        # submission['SalePrice'] = predictions['SalePrice']
        # submission.to_csv(
        #     "../Submissions/submission_XGB_CV_tuned2.csv", index=False)

XGBoostingModel().submit_solution()
