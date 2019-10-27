import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
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

# Dropping the unimportant features which we found in the EDA to be the 'ps_calc' ones
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

# Replace missing values with the mode of the feature
df_train = train
for i, feature in enumerate(list(df_train.drop(['id'], axis=1))):
    if df_train[feature].isnull().sum() > 0:
        df_train[feature].fillna(df_train[feature].mode()[0],inplace=True)

# Selecting the 'cat' features for one-hot encoding
cat_features = [a for a in train.columns if a.endswith('cat')]
for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train,temp],axis=1).drop([column],axis=1)
    test = pd.concat([test,temp],axis=1).drop([column],axis=1)
    
# Setting up three models with optimized parameters
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
lgb_model1 = LGBMClassifier(**params)
lgb_model2 = LGBMClassifier(**params2)
lgb_model3 = LGBMClassifier(**params3)
lgb_model = lgb_model1

'''# Creating a classifier based on the voting result of the three previous ones
vclf = VotingClassifier(estimators=[('1', lgb_model1), ('2', lgb_model2), ('3', lgb_model3)], voting='soft')

# Fitting the new model
vclf = vclf.fit(train, target_train)'''

# This ended up taking too long, ran it for hours to no avail.
# Stacking the three models
print("Stacking") 
models = [lgb_model1, lgb_model2, lgb_model3]
S_train, S_test = stacking(models, train, target_train, test)

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3, objective='multi:softprob')
    
model = model.fit(S_train, y_train)

y_pred = model.predict_proba(S_test)

#print("Fitting")
# Fitting stacker-predicted data on new model
#lgb_model.fit(S_train, target_train)
#print("Predicting")
# Predicting form the test set
#y_pred = lgb_model.predict_proba(S_test, raw_scores=True)

# Creating submission to fit Kaggles requirements
submission = pd.DataFrame()
submission['id'] = id_test
submission['target'] = 1-y_pred[:,0]
submission.to_csv('submission.csv', index=False)
print("Done")
