import luigi
import pandas as pd
import os
import pickle
from urllib2 import Request,urlopen,URLError
import time
from regressor import *
from helper_luigi import custom_out

class TrainDataPreProcessing(luigi.Task):

    def output(self):
        return luigi.LocalTarget("/tmp/train_clean_out.csv")

    def run(self):
        custom_out('TrainDataPreProcessing Node initiated')
        train_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train_clean.csv"))
        #####################################################
        ## This space saved for future Train-PreProcessing ##
        #####################################################
        train_df.to_csv(self.output().path, index=False)
        custom_out('TrainDataPreProcessing Node finished')

class TestDataPreProcessing(luigi.Task):

    def run(self):
        custom_out('TestDataPreProcessing Node initiated')
        test_df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_clean.csv"))
        ####################################################
        ## This space saved for future Test-PreProcessing ##
        ####################################################
        test_df.to_csv(self.output().path, index=False)
        custom_out('TestDataPreProcessing Node finished')

    def output(self):
        return luigi.LocalTarget("/tmp/test_clean_out.csv")        

class Train(luigi.Task):

    def requires(self):
        return TrainDataPreProcessing()

    def output(self):
        return luigi.LocalTarget("/tmp/russian_housing_model.pkl")

    def run(self):
        custom_out('Train Node initiated')
        prices_model = train_xg_boost(pd.read_csv(self.input().path))
        custom_out('Successfully trained model')
        with open(self.output().path, 'wb') as f:
            pickle.dump(prices_model, f)
        custom_out('Successfully wrote model to pickle')

class Predict(luigi.Task):

    def requires(self):
        yield Train()
        yield TestDataPreProcessing()

    def output(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return luigi.LocalTarget(os.path.join(os.getcwd(), "logs", "predictions/russian_housing_submission_{}.csv".format(timestr)))

    def run(self):
        custom_out('Predict Node initiated')

        prices_model = pd.read_pickle(Train().output().path)
        test_df = pd.read_csv(TestDataPreProcessing().output().path)
        test_x = xgb.DMatrix(test_df.drop('id', axis=1))
        predictions = prices_model.predict(test_x)

        submission = pd.DataFrame({ 'id': test_df['id'],
                                    'price_doc': predictions})
        submission.to_csv(self.output().path, index=False)

        custom_out('Write of submission to csv successful')


if __name__ == '__main__':
    luigi.run()
    
