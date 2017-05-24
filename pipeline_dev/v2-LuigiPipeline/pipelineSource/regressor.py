from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from external_helper_luigi import custom_out, forward_selected, auto_grid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
sys.path.insert(0, '../userInput/')
from user_input import *


target_variable = ["price_doc"]
cv_all = 10

def train_xg_boost(dataframe, xgb_param_dict = 'predefined'):
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])
                                                        
    custom_out('Shape of split training dataset = {}'.format(X_train.shape))
    df_columns = X_train.columns
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtrain_all = xgb.DMatrix(dataframe.drop(target_variable,axis=1), dataframe[target_variable[0]])
    dval = xgb.DMatrix(X_test, y_test, feature_names=df_columns)

    if xgb_param_dict =='predefined':
        user_xgb_param()
    else:
        xgb_params = xgb_param_dict
        xgb_params['eval_metric']='rmse'
        xgb_params['eta'] = 0.05

    for param in xgb_params:
        custom_out('XGB Parameter {} = {}'.format(param, xgb_params[param]))
                                                                    
    
    cv_output = xgb.cv(xgb_params, dtrain_all, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)

    nround = cv_output.loc[cv_output['test-rmse-mean'] == min(cv_output['test-rmse-mean'])].index.values[0]

    custom_out('CV XGBoost output nround = {}'.format(nround))
    
    xgb_out = xgb.train(xgb_params, dtrain, num_boost_round=nround, evals=[(dval, 'val')],
                        early_stopping_rounds=20, verbose_eval=20)

    cv_test_pred = xgb_out.predict(xgb.DMatrix(X_test))
    ### test your predictions
    mse = metrics.mean_squared_error(y_test, cv_test_pred)
    r2 = metrics.r2_score(y_test, cv_test_pred)
    ### and print the report
    custom_out('XGBoost Final RMSE = {}\nXGBoost Final R2 = {}'.format(mse**0.5, r2))

    feature_importance = xgb_out.get_fscore()
    df_features = pd.DataFrame({'names':feature_importance.keys(),'values':feature_importance.values()})
    print df_features.sort_values('values', ascending=False)[0:5]
    #custom_out('Feature importance: {}.format()')

    return xgb_out

def train_random_forest(dataframe):
    np.random.seed(0)
    X_train = dataframe.drop(target_variable, axis=1)
    y_train = dataframe[target_variable[0]]
    parameter_grid = user_forest_param()
    custom_out("Training Random Forest")
    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("scaler", StandardScaler()),
                          ("forest", RandomForestRegressor(random_state=0,
                                                           n_estimators=100))])
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameter_grid,
                               cv=cv_all,
                               verbose=2,
                               n_jobs=5,
                               refit=True)
    return auto_grid(grid_search, X_train, y_train, 'RF')

def train_forward_selected(dataframe):
    #I'm assuming we're only being fed numeric valued-columns
    #Leaving cross-validation up to Yabin
    custom_out("Training LinReg Forward Selection")

    linreg_model = forward_selected(dataframe, target_variable[0])
    print linreg_model.summary()
    return linreg_model

def train_model_ridge(dataframe):
    np.random.seed(0)
    X_train = dataframe.drop(target_variable, axis=1)
    y_train = dataframe[target_variable[0]]
    parameter_grid = user_ridge_param()
    custom_out("Training Ridge Regression")
    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("scaler", StandardScaler()),
                          ("ridge", linear_model.Ridge())])
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameter_grid,
                               cv=cv_all,
                               verbose=2,
                               n_jobs=5,
                               refit=True)
    return auto_grid(grid_search, X_train, y_train, 'RLR')


def train_model_xgb_grid(dataframe):
    X_train = dataframe.drop(target_variable, axis=1)
    y_train = dataframe[target_variable[0]]

    print "Training XGB Grid Search"
    pipeline = Pipeline([('xgb',xgb.XGBRegressor())])

    parameter_grid = user_xgbgrid_param()
    
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameter_grid,
                               cv=cv_all,
                               verbose=2,
                               n_jobs=5,
                               refit=True)
    
    grid_search.fit(X_train, y_train)
    custom_out('Best score {}: {}'.format('XGB', grid_search.best_score_))
    custom_out('Best parameters {}: {}'.format('XGB', grid_search.best_params_))
    print '\n All grid results: \n'
    print grid_search.grid_scores_
    xgb_param_dict = grid_search.best_params_

    return train_xg_boost(dataframe, xgb_param_dict)

def train_Huber(dataframe):
    custom_out("Training Huber Regression")
    np.random.seed(0)
    X_train = dataframe.drop(target_variable, axis=1)
    y_train = dataframe[target_variable[0]]

    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("scaler", StandardScaler()),
                          ("huber", linear_model.HuberRegressor())])

    parameter_grid = user_huber_param()

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameter_grid,
                               cv=cv_all,
                               verbose=2,
                               n_jobs=5,
                               refit=True)
    return auto_grid(grid_search, X_train, y_train, 'HLR')

def train_model_xgb_rand(dataframe):
    #Not yet implemented
    X_train = dataframe.drop(target_variable, axis=1)
    y_train = dataframe[target_variable[0]]

    print "Training XGB Grid Search"
    pipeline = Pipeline([('xgb',xgb.XGBRegressor())])

    parameter_grid = dict(xgb__learning_rate=sp_randint(0.05, 0.15),
                          xgb__max_depth=[3, 5],
                          xgb__subsample=[1.0],
                          xgb__colsample_bytree=[0.7],
                          xgb__objective=['reg:linear'],
                          xgb__silent=[True],
                          xgb__gamma=[0.1,0.5,1,2]
                          )
    
    n_iter_search = 20
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                                 n_iter=n_iter_search)
    print random_search.best_score_
    return random_search.best_estimator_

def train_elastic_model(dataframe):
    pass
#    custom_out("Training ElasticNetCV Regression")
#    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
#                                                        dataframe[target_variable[0]])
#
#    pipeline = Pipeline([("imputer", Imputer(strategy="median",
#                                              axis=0)),
#                          ("ridge", linear_model.ElasticNetCV())])
#
#    en_score = cross_val_score(pipeline, X_train, y_train)
#    custom_out("EN Accuracy: {} +/- {}".format(round(en_score.mean(),2), round(2*en_score.std(),2)))
#    return pipeline.fit(X_train, y_train)

if __name__ == '__main__':
    pass
