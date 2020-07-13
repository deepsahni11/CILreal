from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import pickle 
import random
import pdb
from sklearn.externals import joblib 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import load
from numpy import save


test_x = np.array(pd.read_excel('Data_metrics_real_final_done.xlsx'))

bootstrapped_samples = 50


for i in range(21):

 # number of bootstrapped samples

    predictions_per_sampling_method = []
    for b in range(bootstrapped_samples):

        joblib_file = 'regressor_model_sampling_method' + str(i+1) + 'bootstrap_sample_' + str(b+1) + '.pkl'
      
        pickle_model = joblib.load(joblib_file)

       
        Ypredict = pickle_model.predict(test_x)


        if(b == 0):
            predictions_per_sampling_method.append(Ypredict)
        else: 
            predictions_per_sampling_method = np.add((predictions_per_sampling_method),Ypredict)
            
            
    np.savetxt("predictions_bootstrap_" + str(b) + ".csv" , predictions_per_sampling_method)