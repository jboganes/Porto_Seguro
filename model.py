import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from vecstack import stacking

# Importing the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Setting up
id_test = test['id'].values
target_train = train['target'].values

# Removing superfluous rows from data
train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

# Dropping the unimportant features
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

# Replacing missing data
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

# Selecting the 'cat' features for one-hot encoding
cat_features = [a for a in train.columns if a.endswith('cat')]
for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train,temp],axis=1).drop([column],axis=1)
    test = pd.concat([test,temp],axis=1).drop([column],axis=1)
    
# Setting up model parameters for three models
params = {}
params['max_depth'] = 3
params['subsample'] = 0.8
params['subsample_freq'] = 10
params['learning_rate'] = 0.06
params['colsample_bytree'] = 0.75
params2 = params
params3 = params

params['n_estimators'] = 500
params2['n_estimators'] = 1000
params3['n_estimators'] = 1500

# Creating the three models for stacking
lgb_model = LGBMClassifier(**params)
lgb_model1 = lgb_model
lgb_model2 = LGBMClassifier(**params2)
lgb_model3 = LGBMClassifier(**params3)

# Creating voting classifier, fitting, and predicting
vclf = VotingClassifier(estimators=[('1', lgb_model1), ('2', lgb_model2), ('3', lgb_model3)], voting='hard')
vclf = vclf.fit(train, target_train)
y_pred = vclf.predict_proba(S_test, raw_scores=True)

# Creating submission to fit Kaggles requirements
submission = pd.DataFrame()
submission['id'] = id_test
submission['target'] = 1-y_pred[:,0]
submission.to_csv('submission.csv', index=False)
print("Done")
