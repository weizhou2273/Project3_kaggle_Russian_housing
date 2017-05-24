#############  Tree  #################
## Single XGB parameter input ########
def user_xgb_param():
	return  {'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1,
            'seed': 0}

###########  Tree  ###################
#### Grid XGB parameter input ########
def user_xgbgrid_param():
	return dict(xgb__subsample=[0.7],
				xgb__learning_rate=[0.05],
				xgb__max_depth=[5],
				xgb__colsample_bytree=[0.7],
				xgb__objective=['reg:linear'],
				xgb__silent=[1],
				xgb_seed=[0])

# max_depth=3, 
# learning_rate=0.1, 
# n_estimators=100, 
# silent=True, 
# objective='reg:linear', 
# booster='gbtree', 
# n_jobs=1, 
# nthread=None, 
# gamma=0, 
# min_child_weight=1, 
# max_delta_step=0, 
# subsample=1, 
# colsample_bytree=1, 
# colsample_bylevel=1, 
# reg_alpha=0, 
# reg_lambda=1, 
# scale_pos_weight=1, 
# base_score=0.5, 
# random_state=0, 
# seed=None, 
# missing=None, 
# **kwarg

#################  Tree  ###########################
######## Random Forest parameter input #############
def user_forest_param():
	return {'forest__n_estimators': [8, 10, 12],
			'forest__max_depth': [None, 10],
			'forest__min_samples_split': [2, 4],
			'forest_min_samples_leaf': [1, 4]
			}

# All options:
# n_estimators=10, 
# criterion='mse', 
# max_depth=None, 
# min_samples_split=2, 
# min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0, 
# max_features='auto', 
# max_leaf_nodes=None, 
# min_impurity_split=1e-07, 
# bootstrap=True, 
# oob_score=False, 
# n_jobs=1, 
# random_state=None, 
# verbose=0, 
# warm_start=False

################  LR  ##############################
######## Ridge Reg parameter input #################
def user_ridge_param():
	return {'ridge__alpha': [0.01, 0.1, 1]}

################  LR  ##############################
###### Huber Lin Reg parameter input ###############
def user_huber_param():
	return {'huber__epsilon': [1.2, 1.35, 1.5]}

if __name__ == '__main__':
    pass
