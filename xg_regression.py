# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 01:23:12 2018

@author: aarti jugdar
"""

def xg_regression(X,Y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xgboost
    from sklearn import cross_validation, tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import explained_variance_score
    data=pd.read_csv()
    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # XGtrain matrix
    xgtrain = xgboost.DMatrix(X_train, label=y_train)
    xgbreg = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
    xgb_params = xgbreg.get_xgb_params()
    cvresult=xgboost.cv(xgb_params,xgtrain,num_boost_round=200, nfold=10, metrics=['rmse'])
    print('Best number of trees = {}'.format(cvresult.shape[0]))
    xgbreg.set_params(n_estimators=cvresult.shape[0])
    print('Fit on the trainingsdata')
    xgbreg.fit(X_train, y_train, eval_metric='rmse')
    
    #Predictions
    pred = xgbreg.predict(X_test, ntree_limit=cvresult.shape[0])
    
    # Evaluate the accuracy
    print(explained_variance_score(pred,y_test))
    