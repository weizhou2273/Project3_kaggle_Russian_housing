from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn import metrics
from helper_luigi import custom_out

target_variable = ["price_doc", "id"]

def train_model_ridge(dataframe):
    model_object = {}
    print "Training Ridge Regression"
    print dataframe.columns.tolist()
    ridge = linear_model.Ridge(alpha=0.5)
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe.price_doc)
    model = ridge.fit(X_train, y_train)
    model.fit(X_train, y_train)
    model_object["model"] = model
    model_object["training_features"] = X_train.columns.tolist()
    return model_object


def train_model_with_grid_search(dataframe):
    print "Training Ridge Regression"
    print dataframe.columns
    ridge = linear_model.Ridge()
    params_grid = {
        "alpha": [0.01, 0.05, 0.1, 0.5]
    }
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe.price_doc)
    model = GridSearchCV(ridge, param_grid=params_grid, verbose=2, cv=5, refit=True)
    model.fit(X_train, y_train)
    return model.best_estimator_

def train_xg_boost(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe.price_doc)
                                                        
    custom_out('Shape of split training dataset = {}'.format(X_train.shape))
    df_columns = X_train.columns
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtrain_all = xgb.DMatrix(dataframe.drop(target_variable,axis=1), dataframe.price_doc)
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
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe.price_doc)
    print "Training Random Forest"
    rf = RandomForestRegressor(n_estimators=100, criterion='mse', n_jobs=5,verbose=2)
    rf.fit(X_train, y_train)
    return rf


def test_model_ridge(dataframe, model):
    df = dataframe.drop(target_variable, axis=1)
    return model.predict(df)

def stacked_pipeline(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(target_variable, axis=1),
                                                        dataframe.price_doc)

    print "Training Stacked Models"
    
    estimators = [('xgb',xgb.XGBClassifier())]
    pipeline = make_pipeline(estimators)

    parameter_grid = dict(xgb__learning_rate=[0.05, 0.10, 0.15],
                          xgb__max_depth=[3,5,7],
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
    return grid_search.best_estimator_

if __name__ == '__main__':
    pass
