from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import pickle 
import random
import pdb
import pandas as pd
import math
import numpy as np
from sklearn.externals import joblib 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import load
from numpy import save


test_x = np.array(pd.read_excel('Data_metrics_real_final_done.xlsx', header = None))

bootstrapped_samples = 50
bootstrap = 1

predictions_precision = []
predictions_recall = []


file = open('regressor_model.pkl', 'rb')

for i in range(21):

 # number of bootstrapped samples

    predictions_per_sampling_method = []
    for b in range(bootstrapped_samples):

#         joblib_file = 'regressor_model_sampling_method' + str(i+1) + 'bootstrap_sample_' + str(b+1) + '.pkl'
      
        joblib_file = pickle.load(file)
    
        pickle_model = joblib.load(joblib_file)

       
        Ypredict = pickle_model.predict(test_x)


        if(b == 0):
            predictions_per_sampling_method.append(Ypredict)
        else: 
            predictions_per_sampling_method = np.add(np.array(predictions_per_sampling_method),Ypredict)
            
#    print(" size:" + str(np.array(predictions_per_sampling_method).shape)) 

    precision  = np.squeeze((predictions_per_sampling_method)/bootstrapped_samples)[:,0].reshape(-1,1)
    recall = np.squeeze((predictions_per_sampling_method)/bootstrapped_samples)[:,1].reshape(-1,1)
#    print("precision" , str(precision.shape))

    if(i == 0):
        #print(predictions_precision.shape)
        predictions_precision.append(precision)
        predictions_recall.append(recall)
    elif(i == 1):
        print("i == 1")
        print(np.array(predictions_precision).shape)
        predictions_precision = np.concatenate((np.squeeze(np.array(predictions_precision)).reshape(-1,1),precision), axis = 1)

        predictions_recall = np.concatenate((np.squeeze(np.array(predictions_recall)).reshape(-1,1),recall), axis = 1)


    else:
        print("else one")
        print(np.array(predictions_precision).shape)
        predictions_precision = np.concatenate((np.squeeze(np.array(predictions_precision)),precision) , axis = 1)
        predictions_recall = np.concatenate((np.squeeze(np.array(predictions_recall)), recall ), axis = 1)
        

        
file.close()        
        
        
np.savetxt("precision_predictions_sampling_method.csv" , predictions_precision, delimiter =  ',')
np.savetxt("recall_predictions_sampling_method.csv", predictions_precision, delimiter = ',')


dfp = pd.read_excel('E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CODS-COMAD 2021\\Precision_one_dataset.xlsx', header= None)
dfr = pd.read_excel('E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CODS-COMAD 2021\\Recall_one_dataset.xlsx', header= None)

precision = []
recall = []


for i in range(14):
    for j in range(20):
        if(math.floor(dfr.values[i][j]) == 1):
            recall.append(math.floor(dfr.values[i][j]))
            
        else:
            recall.append(dfr.values[i][j])
        
for i in range(14):
    for j in range(20):
        if(math.floor(dfp.values[i][j]) == 1):
            precision.append(math.floor(dfp.values[i][j]))
        else:
            precision.append(dfp.values[i][j])
        
        
recall_f = recall
precision_f = precision
remove_index = []

l = len(recall_f)
for i in range(280):
    for j in range(280):
        if(recall[i] < recall[j]  and precision[i] < precision[j] ): #and recall[i] != recall[j]  and precision[i] != precision[j]):
            remove_index.append(i)
    #         l = len(recall_f)
            
        
        
        
mylist = list(set(remove_index))

recall_final = []
precision_final = []
for i in range(280):
    if(i not in mylist):
#         if(math.floor(recall[i]) == 1):
#             recall_final.append(math.floor(recall[i]))
#         else:
        recall_final.append(recall[i])
            
#         if(math.floor(precision[i]) == 1):
#             precision_final.append(math.floor(precision[i]))
#         else:
        precision_final.append(precision[i])
#         else:
        
        
# pr
recall_final = np.array(recall_final).reshape(-1,1)
precision_final = np.array(precision_final).reshape(-1,1)
pr = np.concatenate((precision_final, recall_final), axis=1)


# d
d = []
for i in range(280):
    if(i not in mylist):
        if((i+1)%(20) == 0):
            d.append(20)
        else:
            d.append((i+1)%(20))

            
# t           
threshold = np.arange(0.2, 0.851, 0.05).tolist()
t = []
for i in range(280):
    if(i not in mylist):
#         print(math.ceil(((i+1)/20))-1)

        t.append(threshold[math.ceil(((i+1)/20)-1)])


dataset_excel = pd.DataFrame({'Precision': pr[:, 0], 'Recall': pr[:, 1], 'Sampling Methods': (np.array(d)) , 'Thresholds': np.array(t)})
dataset = pd.DataFrame({'Precision': pr[:, 0], 'Recall': pr[:, 1], 'Sampling Methods': (np.array(d)) })
dataset_excel.to_csv('E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CODS-COMAD 2021\\pareto_excel_final.csv')
print(dataset_excel) 
