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


training_x = np.load("X_train_datasets_1_bootstrap3.npy", allow_pickle = True) 
# test_x = np.load("X_test_real_datasets_bootstrap.npy", allow_pickle = True)
test_x = np.array(pd.read_excel('Data_metrics_real_final_done.xlsx'))

training_y_recall = np.load("y_train_recall_datasets_1_bootstrap3.npy", allow_pickle = True)
training_y_precision = np.load("y_train_precision_datasets_1_bootstrap3.npy", allow_pickle = True)


predictions_recall = []


for b in range(len(training_x)): # number of bootstrapped samples

    Xtrain = training_x[b]
    ytrainp = training_y_precision[b]
    ytrainr = training_y_recall[b]
    Xtest = test_x

    predictions = []

    
  

  
    c = 1
    for p in [20]: 
        for q in [15]: 

            reg = MLPRegressor(alpha=1e-4,
                               hidden_layer_sizes=(p, q),
                               random_state=1,
                               activation="tanh",
                               batch_size= 64,
                               max_iter=500)
            

            for i in range(21):
                predictions = []
                
                ytrain_p = ytrainp[:,i].reshape(-1,1)
                ytrain_r = ytrainr[:,i].reshape(-1,1)
                ytrain_pr = np.concatenate((ytrain_p,ytrain_r), axis = 1)
        
        
                reg.fit(Xtrain, ytrain_pr)

                pred_y_test = reg.predict(Xtest)
               

                predictions.append(pred_y_test)
                    

                joblib.dump(reg, 'regressor_model_sampling_method' + str(i+1) + 'bootstrap_sample_' + str(b+1) + '.pkl')  
                    

                
                c= c+1



    np.save("recall_predictions_bootstrap_" + str(b) + ".csv" , predictions)
