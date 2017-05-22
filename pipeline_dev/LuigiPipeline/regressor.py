from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from external_helper_luigi import custom_out, forward_selected, cv_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

target_variable = ["price_doc", "id"]

def train_xg_boost(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])
                                                        
    custom_out('Shape of split training dataset = {}'.format(X_train.shape))
    df_columns = X_train.columns
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtrain_all = xgb.DMatrix(dataframe.drop(target_variable,axis=1), dataframe[target_variable[0]])
    dval = xgb.DMatrix(X_test, y_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
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

    return xgb_out

def train_random_forest(dataframe):
    #I'm assuming we're only being fed numeric valued-columns
    #df = dataframe.select_dtypes(include=['float64', 'int'])

    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])
    custom_out("Training Random Forest")
    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("forest", RandomForestRegressor(random_state=0,
                                                           n_estimators=100))])

    rf_score = cross_val_score(pipeline, X_train, y_train)
    custom_out("RF R^2: {} +/- {}".format(round(rf_score.mean(),2), round(2*rf_score.std(),2)))
    #rf = RandomForestRegressor(n_estimators=100, criterion='mse', n_jobs=5,verbose=2)
    #return rf.fit(X_train, y_train)
    return pipeline.fit(X_train, y_train)

def train_forward_selected(dataframe):
    #I'm assuming we're only being fed numeric valued-columns
    #df = dataframe.select_dtypes(include=['float64', 'int'])

    #X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
    #                                                    dataframe[target_variable[0]])
    custom_out("Training LinReg Forward Selection")

    linreg_model = forward_selected(dataframe.drop(target_variable[1], axis=1), target_variable[0])
    print linreg_model.summary()
    return linreg_model

def train_model_ridge(dataframe):
    custom_out("Training Ridge Regression")
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])

    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("ridge", linear_model.Ridge(alpha=0.5))])

    rlr_score = cross_val_score(pipeline, X_train, y_train)
    custom_out("RLR Accuracy: {} +/- {}".format(round(rlr_score.mean(),2), round(2*rlr_score.std(),2)))
    return pipeline.fit(X_train, y_train)


def train_model_xgb_grid(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])

    print "Training XGB Grid Search"
    pipeline = Pipeline([('xgb',xgb.XGBRegressor())])

    parameter_grid = dict(xgb__learning_rate=[0.05, 0.10],
                          xgb__max_depth=[3, 5],
                          xgb__subsample=[1.0],
                          xgb__colsample_bytree=[0.7],
                          xgb__objective=['reg:linear'],
                          xgb__silent=[True]
                          )
    
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameter_grid,
                               cv=5,
                               verbose=2,
                               n_jobs=5,
                               refit=True)
    
    grid_search.fit(X_train, y_train)
    print grid_search.best_score_
    print cv_report(grid_search.cv_results_)
    return grid_search.best_estimator_

def xgb_to_linreg(dataframe):
    #Convert price_doc to price/sq

    #Split dataframe

    #Train XGB

    #Append predicted price/sq to dataframe

    #Convert price_doc (original) to log(price+1)

    #Split dataframe

    #Train linear model

    #Return fit
    return stacked_model


def stacked_pipeline(dataframe):
    pass
    #This doesn't work....can't stack XGB at front of pipeline

    #X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
    #                                                    dataframe[target_variable[0]])
    #custom_out("Training Stacked Pipeline")

    #pipeline = Pipeline([("xgb", xgb.XGBRegressor()),
    #    ("imputer", Imputer(strategy="median",
    #        axis=0)),
    #    ("linreg",linear_model.LinearRegression())])

    #pipeline.fit(X_train, y_train)
    #custom_out('MSE (Stacked):'.format(metrics.mean_squared_error(pipeline.predict(X_test), y_test)))
    #return pipeline

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

def train_Huber(dataframe):
    custom_out("Training Huber Regression")
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe[target_variable[0]])

    pipeline = Pipeline([("imputer", Imputer(strategy="median",
                                              axis=0)),
                          ("huber", linear_model.HuberRegressor())])

    pipeline.fit(X_train, y_train)
    custom_out('MSE (Huber):'.format(metrics.mean_squared_error(pipeline.predict(X_test), y_test)))
    return pipeline

if __name__ == '__main__':
    pass
