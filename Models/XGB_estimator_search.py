import pandas as pd
import xgboost as xgb
import re
import matplotlib.pylab as plt
from data_cleansing import clean_design_matrix
from matplotlib.pylab import rcParams
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


# Reading data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
rcParams['figure.figsize'] = 18, 12
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
            seed=99, n_estimators=10000, learning_rate=0.05,
            objective='multi:softprob', subsample=0.9, colsample_bytree=0.8,
            max_depth=3, min_child_weight=4, reg_alpha=2)
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
        # feat_imp = pd.Series(model.booster().get_fscore()).sort_values(
        #     ascending=False)
        # feat_imp = feat_imp[:50]
        # feat_imp.plot(kind='bar', title='Feature Importance')
        # plt.ylabel('Feature Importance Score')
        # plt.savefig('../Model_results/XGB_Top50_feat9_imp.png')

    def submit_solution(self):
        """Submit the solution file"""
        self._execute_cross_validation(model=self._build_model())
        # predictions = self._make_predictions()
        # submission = test[['Id']]
        # submission['SalePrice'] = predictions['SalePrice']
        # submission.to_csv(
        #     "../Submissions/submission_XGB_CV_tuned2.csv", index=False)

XGBoostingModel().submit_solution()
