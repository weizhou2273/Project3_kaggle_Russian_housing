import time
import pandas as pd
from regressor import *
import xgboost as xgb
import sys
import numpy as np
from external_helper_luigi import custom_out

def function_mappings():
	return {'xgb':train_xg_boost,
						'rf':train_random_forest,
						'flr':train_forward_selected,
						'rlr':train_model_ridge,
						'xgbgrid':train_model_xgb_grid,
						'sp':stacked_pipeline,
						'en':train_elastic_model,
						'hlr':train_Huber}

def prediction_to_submission(dataframe, predictions, model_string):
    #Post processing from log(price+1) to price
    #Replace with regex search for 'lr' in model_string
    debug = True
    if debug:
    	pass
    else:
    	if (str(model_string) == 'flr') | (str(model_string) == 'rlr'):
    		predictions = np.exp(predictions) - 1
    		custom_out('Transformed back from log(price)')
    	#Post processing for price/sqft to price
    	else:
    		predictions = dataframe['totalsq']*predictions
    return pd.DataFrame({ 'id': dataframe['id'],
    					  'price_doc': predictions})

def model_choice(model_string, dataframe):
	return function_mappings()[model_string](dataframe)

def test_postprocess(model_string, dataframe):
	if (str(model_string) == 'xgb') | (str(model_string) == 'xgbgrid'):
		return xgb.DMatrix(dataframe.drop('id', axis=1))
	else:
		return dataframe.drop('id', axis=1)

if __name__ == '__main__':
    pass