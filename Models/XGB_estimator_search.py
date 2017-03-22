import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt
from data_cleansing import clean_design_matrix
from matplotlib.pylab import rcParams
from xgboost.sklearn import XGBClassifier


# Reading data
train_rental = pd.read_json('../../Rental_Listing_Inquiries_Data/train.json',
                            convert_dates=['created'])
rcParams['figure.figsize'] = 18, 12


class XGBoostingModel(object):

    def __init__(self):
        self.design_matrix = clean_design_matrix(train_rental, train=True)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['interest_level']]

    @staticmethod
    def _build_model():
        """Model: Extreme Gradient Boosting model using tuned parameters"""
        model = XGBClassifier(seed=99,
            n_estimators=5000, learning_rate=0.01, objective='multi:softprob',
            max_depth=5, min_child_weight=2, subsample=0.85, gamma=0.3,
            colsample_bytree=0.75, reg_alpha=2.0)
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
