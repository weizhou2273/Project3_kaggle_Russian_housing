#############  Tree  #################
## Single XGB parameter input ########
def user_xgb_param():
	return  {'eta': 0.05,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1}

###########  Tree  ###################
#### Grid XGB parameter input ########
def user_xgbgrid_param():
	return dict(xgb__subsample=[1.0],
				xgb__learning_rate=[0.05, 0.10],
				xgb__max_depth=[3, 5],
				xgb__colsample_bytree=[0.7],
				xgb__objective=['reg:linear'],
				xgb__silent=[True])

#################  Tree  ###########################
######## Random Forest parameter input #############
def user_forest_param():
	return {'forest__n_estimators': [5, 10, 15, 20],
			'forest__max_depth': [2, 5, 7, 9]}

################  LR  ##############################
######## Ridge Reg parameter input #################
def user_ridge_param():
	return {'ridge__alpha': [0.01, 0.1, 1]}

################  LR  ##############################
###### Huber Lin Reg parameter input ###############
def user_huber_param():
	return {'huber__epsilon': [1.2, 1.35, 1.5]}