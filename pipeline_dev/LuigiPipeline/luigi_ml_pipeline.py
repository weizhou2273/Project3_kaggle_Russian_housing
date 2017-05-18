import luigi
import pandas as pd
import os
import pickle
import time
from regressor import *

class TrainDataPreProcessing(luigi.Task):

    def output(self):
        return luigi.LocalTarget("/tmp/train_clean_out.csv")

    def run(self):
        print '\nTrain Data Pre Proc\n'
        old_path = os.path.join(os.getcwd())
        os.chdir('../../data')
        train_df = pd.read_csv(os.path.join(os.getcwd(), "processed", "train_clean.csv"))
        os.chdir(old_path)
        #####################################################
        ## This space saved for future Train-PreProcessing ##
        #####################################################
        print train_df.columns
        train_df.to_csv(self.output().path, index=False)



class TestDataPreProcessing(luigi.Task):

    def run(self):
        print '\nTest Data Pre Proc\n'
        old_path = os.path.join(os.getcwd())
        os.chdir('../../data')
        test_df = pd.read_csv(os.path.join(os.getcwd(), "processed", "test_clean.csv"))
        os.chdir(old_path)
        ####################################################
        ## This space saved for future Test-PreProcessing ##
        ####################################################
        test_df.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("/tmp/test_clean_out.csv")        

class Train(luigi.Task):

    def requires(self):
        return TrainDataPreProcessing()

    def output(self):
        return luigi.LocalTarget("/tmp/russian_housing_model.pkl")

    def run(self):
        print 'Train Model'
        prices_model = train_xg_boost(pd.read_csv(self.input().path))
        
        with open(self.output().path, 'wb') as f:
            pickle.dump(prices_model, f)

class Predict(luigi.Task):

    def requires(self):
        yield Train()
        yield TestDataPreProcessing()

    def output(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return luigi.LocalTarget(os.path.join(os.getcwd(), "data", "russian_housing_submission_{}.csv".format(timestr)))

    def run(self):
        print 'Predict values'
        prices_model = pd.read_pickle(Train().output().path)
        print prices_model
        test_df = pd.read_csv(TestDataPreProcessing().output().path)
        test_x = xgb.DMatrix(test_df.drop('id', axis=1))
        predictions = prices_model.predict(test_x)

        submission = pd.DataFrame({ 'id': test_df['id'],
                                    'price_doc': predictions})
        submission.to_csv(self.output().path, index=False)
        print 'Write of submission to csv successful'


if __name__ == '__main__':
    luigi.run()
    
