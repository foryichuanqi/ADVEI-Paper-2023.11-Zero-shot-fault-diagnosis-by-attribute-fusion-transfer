#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 23:10:20 2022

@author: cqu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:23:49 2022

@author: cqu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:21:06 2022

@author: cqu
"""



"""
Created on Tue May 14 22:44:05 2019

@author: Robot Hands (github.com/nvd919)
"""

#[]
import sys


import copy

import pandas as pd
import os
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn.model_selection import train_test_split as tts
from keras import backend as K

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import fetch_species_distributions, load_digits
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.neighbors import KernelDensity





import numpy as np

from sklearn import preprocessing


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import naive_bayes
from scipy.stats import pearsonr

from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers.core import Dense,Activation, Dropout

import keras 
import keras.backend as backend


LEARNING_RATE = 0.1
WEIGHT_DACAY =5e-4
EPOCHS = 1500
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_FOLDER =  last_path+r"\TE"
NUM_FILES = 15
experiment_index=2
pearsonr_value=0.1

num_7=[0,1]
time_step=16


num_components=10

# min_loss=0.2

load_classifier='Ture'



well_trained_list_0=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
well_trained_list_1=[]
well_trained_list_2=[]
well_trained_list_3=[]
well_trained_list_4=[]



well_trained_list_0_4=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19]



def sum_list(items):  
    sum_numbers = 0  
    for x in items:  
        sum_numbers += x  
    return sum_numbers 

def data_to_multi_time_step_list(test_data):
    
    test_data_list=[]
    for i in range(len(test_data)):
        
        test_data_i_list=[]
        
        for j in range(test_data[i].shape[0]-time_step+1):
            # print(test_data[i].shape[0]-time_step)
            
            test_data_i_list.append(test_data[i][j:j+time_step,:].T)
        
        test_data_list.append(np.array(test_data_i_list))
    return test_data_list
        

        



def data_to_data_normalized(x_test):
    test_data_normalized=[]
    for i in range(len(data)):
        test_data_normalized.append(x_test[480*i:480*(i+1)])
    return test_data_normalized

def test_data_to_test_data_normalized(x_test):
    test_data_normalized=[]
    for i in range(len(test_data)):
        test_data_normalized.append(x_test[480*i:480*(i+1)])
    return test_data_normalized
    
def train_data_to_train_data_normalized(x_train):
    test_data_normalized=[]
    for i in range(len(training_data)):
        test_data_normalized.append(x_train[480*i:480*(i+1)])
    return test_data_normalized
def accuracy_for_two_array_non_one_hot(x1,x2):
    
    right_num=0
    for i in range(x1.shape[0]):
        if x1[i]==x2[i]:
            right_num+=1
    
    return right_num/x1.shape[0] 



def test_for_test(test_numpy,y_test_non_one_hot):



    test_y_pred_non_one_hot=[]
    for i in range(test_numpy.shape[0]):
        dist_list=[]
        target_faults_dist_list=[]
        for j in range(15):
            ##print(test_numpy[i].shape)
            ##print(fault_attribute_array_well_trained_list[j].shape)
            ##print(test_numpy[i])
            ##print(fault_attribute_array_well_trained_list[j])                
        #     dist_list.append(np.linalg.norm(test_numpy[i]-fault_attribute_array_well_trained_list[j]))
        # test_y_pred_non_one_hot.append(dist_list.index(min([dist_list[target_faults_list[0]],dist_list[target_faults_list[1]],dist_list[target_faults_list[2]]])))
    # ##print('aaaaaaaaaaaaaaaaa')

            dist_list.append(np.linalg.norm(test_numpy[i]-fault_attribute_array_well_trained_list[j]))
            
        target_faults_dist_list.append(dist_list[target_faults_list[0]])
        target_faults_dist_list.append(dist_list[target_faults_list[1]])
        target_faults_dist_list.append(dist_list[target_faults_list[2]])
        test_y_pred_non_one_hot.append(target_faults_dist_list.index(min(target_faults_dist_list)))
    # ##print('aaaaaaaaaaaaaaaaa')        
    
    
    ##print(test_y_pred_non_one_hot[:480])
    ##print(test_y_pred_non_one_hot[480:960])
    ##print(test_y_pred_non_one_hot[960:1440])
    ##print(test_numpy[:10])
    ##print(test_numpy[970:980])
    ##print(test_numpy[480:490])

    # ##print(y_test_non_one_hot[:10])
    # ##print('xxxxxxxxxxxxxx')
    test_y_pred_non_one_hot_array=np.array(test_y_pred_non_one_hot)
    accuracy_0_480=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[0:480],test_y_pred_non_one_hot_array[0:480])
    accuracy_480_960=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[480:960],test_y_pred_non_one_hot_array[480:960])
    accuracy_960_1440=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[960:1440],test_y_pred_non_one_hot_array[960:1440])
    accuracy=accuracy_for_two_array_non_one_hot(y_test_non_one_hot,test_y_pred_non_one_hot_array)  
    return accuracy, accuracy_0_480, accuracy_480_960, accuracy_960_1440#test_y_pred_non_one_hot   #accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()




def y_train_fault_attribute_array_012_to_y_train_and_test_fault_attribute_array(y_train_fault_attribute_array_012):
    


    fault_attribute_list=["11101001100001000000","11011001100001000000","00000111010000000000","00000001001110000100","00000001000110000100","10100001100001000000","01101001100000000000","111110101000001000000","00000110010000100000","01001000010000100000","00000000001100100100","00000000000110100100","00000000000000011000","00000000001000000111","00000000000010000111"]
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[0])
    fault_attribute_0=fault_attribute_array
    ##print(fault_attribute_0)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[1])
    fault_attribute_1=fault_attribute_array
    ##print(fault_attribute_1)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[2])
    fault_attribute_2=fault_attribute_array
    ##print(fault_attribute_2)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[3])
    fault_attribute_3=fault_attribute_array
    ##print(fault_attribute_3)
    
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[4])
    fault_attribute_4=fault_attribute_array
    ##print(fault_attribute_4)
    
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[5])
    fault_attribute_5=fault_attribute_array
    ##print(fault_attribute_5)
    
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[6])
    fault_attribute_6=fault_attribute_array
    ##print(fault_attribute_6)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[7])
    fault_attribute_7=fault_attribute_array
    ##print(fault_attribute_7)
    
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[8])
    fault_attribute_8=fault_attribute_array
    ##print(fault_attribute_8)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[9])
    fault_attribute_9=fault_attribute_array
    ##print(fault_attribute_9)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[10])
    fault_attribute_10=fault_attribute_array
    ##print(fault_attribute_10)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[11])
    fault_attribute_11=fault_attribute_array
    ##print(fault_attribute_11)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[12])
    fault_attribute_12=fault_attribute_array
    ##print(fault_attribute_12)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[13])
    fault_attribute_13=fault_attribute_array
    ##print(fault_attribute_13)
    
    fault_attribute_array=np.zeros((480-time_step+1,1))
    fault_attribute_array[:,0]=int(y_train_fault_attribute_array_012[14])
    fault_attribute_14=fault_attribute_array
    ##print(fault_attribute_14)
    
    
    y_all_fault_attribute_array=[fault_attribute_0,fault_attribute_1,fault_attribute_2,fault_attribute_3,fault_attribute_4,fault_attribute_5,fault_attribute_6,fault_attribute_7,fault_attribute_8,fault_attribute_9,fault_attribute_10,fault_attribute_11,fault_attribute_12,fault_attribute_13,fault_attribute_14]
    
    y_train_all_fault_attribute_array=[]
    for i in untarget_fault_list:
        y_train_all_fault_attribute_array.append(y_all_fault_attribute_array[i])
            
    y_train_fault_attribute_array=np.concatenate(tuple(y_train_all_fault_attribute_array))
    
  
    y_test_fault_attribute_list=[]
    for i in target_faults_list:
        y_test_fault_attribute_list.append(y_all_fault_attribute_array[i])  


        
    y_test_fault_attribute_array=np.concatenate(tuple(y_test_fault_attribute_list))
    # y_test_fault_attribute_cuda=torch.tensor(y_test_fault_attribute_array,device=DEVICE).float()   
    
    return   y_train_fault_attribute_array.reshape(-1,), y_test_fault_attribute_array.reshape(-1,)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y.astype(int)] 

y_test_non_one_hot= np.concatenate((np.ones(480-time_step+1)*0,np.ones(480-time_step+1)*1,np.ones(480-time_step+1)*2)) 
          
fault_attribute_matrix=["11101001100001000000","11011001100001000000","00000111010000000000","00000001001110000100","00000001000110000100","10100001100001000000","01101001100000000000","111110101000001000000","00000110010000100000","01001000010000100000","00000000001100100100","00000000000110100100","00000000000000011000","00000000001000000111","00000000000010000111"]





fault_attribute_array_well_trained_list=np.zeros((15,len(well_trained_list_0_4)))

# # well_trained_list_0_4 = well_trained_list_0 + well_trained_list_1 + well_trained_list_2 + well_trained_list_3 + well_trained_list_4






for i in range(15):
    for j in range(len(well_trained_list_0_4)):
        # #print(i,j)
        fault_attribute_array_well_trained_list[i,j]=int(fault_attribute_matrix[i][well_trained_list_0_4[j]])                  
            
        # test_mask_logits = logits
        # predict_y = test_mask_logits.max(1)[1]
        # accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()


fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list<0.1]=0.5
fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list>0.5]=0
fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list>0.1]=1






all_fault_attribute_array=np.zeros((15,len(well_trained_list_0)))

for i in range(15):
    for j in range(len(well_trained_list_0)):
        # #print(i,j)
        all_fault_attribute_array[i,j]=int(fault_attribute_matrix[i][well_trained_list_0[j]])                  
            
        # test_mask_logits = logits
        # predict_y = test_mask_logits.max(1)[1]
        # accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()


all_fault_attribute_array[all_fault_attribute_array<0.1]=0.5
all_fault_attribute_array[all_fault_attribute_array>0.5]=0
all_fault_attribute_array[all_fault_attribute_array>0.1]=1 


#PCA_NB_timestep_grid_list 206







#########atr14   68.37  #[0.4292821606254442, 0.12864250177683015, 0.4598436389481166, 0.6062544420753376, 0.4953802416488984, 0.6624022743425728, 0.6574271499644634, 0.8024164889836531, 0.6638237384506042, 0.6645344705046198, 0.6744847192608386, 0.48258706467661694, 0.6368159203980099, 0.7647476901208244, 0.29637526652452023, 0.47547974413646055, 0.6119402985074627, 0.5493958777540867, 0.5167022032693674, 0.6403695806680881, 0.3333333333333333, 0.4449182658137882, 0.3333333333333333, 0.6624022743425728, 0.3333333333333333, 0.6631130063965884, 0.6616915422885572, 0.6638237384506042, 0.7398720682302772, 0.5948827292110874, 0.6552949538024165, 0.6538734896943852, 0.6538734896943852, 0.660270078180526, 0.660270078180526, 0.6645344705046198, 0.6140724946695096, 0.7242359630419332, 0.3290689410092395, 0.7654584221748401, 0.5835110163468372, 0.5067519545131486, 0.681592039800995, 0.3333333333333333, 0.658137882018479, 0.3333333333333333, 0.6560056858564322, 0.6567164179104478, 0.6538734896943852, 0.7704335465529495, 0.6666666666666666, 0.3368869936034115, 0.3312011371712864, 0.4626865671641791, 0.3603411513859275, 0.42217484008528783, 0.6588486140724946, 0.6552949538024165, 0.4292821606254442, 0.4605543710021322, 0.4292821606254442, 0.3333333333333333, 0.31627576403695806, 0.3333333333333333, 0.3333333333333333, 0.33759772565742713, 0.3404406538734897, 0.6609808102345416, 0.5067519545131486, 0.4662402274342573, 0.5970149253731343, 0.48471926083866385, 0.6545842217484008, 0.728500355366027, 0.4150675195451315, 0.4228855721393035, 0.4250177683013504, 0.3333333333333333, 0.23525230987917556, 0.3333333333333333, 0.6567164179104478, 0.3333333333333333, 0.3411513859275053, 0.6176261549395877, 0.6567164179104478, 0.650319829424307, 0.7078891257995735, 0.6368159203980099, 0.550817341862118, 0.39445628997867804, 0.3816631130063966, 0.5074626865671642, 0.3333333333333333, 0.32338308457711445, 0.3333333333333333, 0.5501066098081023, 0.3333333333333333, 0.47334754797441364, 0.5053304904051172, 0.5259417199715707, 0.673773987206823, 0.550817341862118, 0.6076759061833689, 0.5202558635394456, 0.3333333333333333, 0.4683724235963042, 0.3333333333333333, 0.43567874911158494, 0.3333333333333333, 0.47476901208244493, 0.4164889836531628, 0.35678749111584934, 0.7199715707178393, 0.3333333333333333, 0.3333333333333333, 0.417910447761194, 0.3333333333333333, 0.42430703624733473, 0.19971570717839374, 0.3333333333333333, 0.4235963041933191, 0.48116560056858565, 0.46766169154228854, 0.3333333333333333, 0.3333333333333333, 0.6112295664534471, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3390191897654584, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.658137882018479, 0.3333333333333333, 0.658137882018479, 0.6545842217484008]


files = [os.path.join(DATA_FOLDER, "d{:0>2}.dat".format(i)) for i in range(1,16,1)]
# test_files = [os.path.join(DATA_FOLDER, "d{:0>2}_te.dat".format(i)) for i in range(1,16,1)]

data = [pd.read_csv(f, sep='   ',  header=None,index_col = None) for f in files]



target_faults_list_all=[[0, 2, 7],[1,10,11],[3, 5, 8],[4, 6, 13],[9,12,14],
                [0, 4, 9], [1, 3, 11],[2, 6, 13],[5, 7,8],[10,12,14],
                [0, 11,14],[1, 2, 10],[ 3,8,12],[4,5, 13],[6,7,9],
                [0, 4, 13],[1, 9, 14],[2,3,12],[5,10,11], [6,7, 8],
                [0, 4, 6],[1, 11,13],[2, 5, 10],[3, 7, 9],[8,12,14]]  

target_list=[]
accuracy_list=[]


accuracy_sum_list_epoch_index=[]
accuracy_list_epoch_index=[]
    
for epoch_index in [600,600,600]    :
    

    
    for ii in range(len(target_faults_list_all)):
        
        experiment_index=ii 
    
        all_faults_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        faults_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        
        target_faults_list=target_faults_list_all[experiment_index]
        faults_list.remove(target_faults_list[0])
        faults_list.remove(target_faults_list[1])
        faults_list.remove(target_faults_list[2])
        untarget_fault_list=faults_list
        
    
    
    
    
    
    
      
        
        
        y_train_non_one_hot=np.concatenate((np.ones(480-time_step+1)*(untarget_fault_list[0]),np.ones(480-time_step+1)*(untarget_fault_list[1]),np.ones(480-time_step+1)*(untarget_fault_list[2]),np.ones(480-time_step+1)*(untarget_fault_list[3]),np.ones(480-time_step+1)*(untarget_fault_list[4]),np.ones(480-time_step+1)*(untarget_fault_list[5]),np.ones(480-time_step+1)*(untarget_fault_list[6]),np.ones(480-time_step+1)*(untarget_fault_list[7]),np.ones(480-time_step+1)*(untarget_fault_list[8]),np.ones(480-time_step+1)*(untarget_fault_list[9]),np.ones(480-time_step+1)*(untarget_fault_list[10]),np.ones(480-time_step+1)*(untarget_fault_list[11])))
        
        
        
          
        
   
        
        y_test_non_one_hot= np.concatenate((np.ones(480-time_step+1)*0,np.ones(480-time_step+1)*1,np.ones(480-time_step+1)*2))  
         
        
        
        
        all_data=[]
        for i in all_faults_list:
            all_data.append(np.array(data[i]))
        
        
        test_data=[]
        for i in target_faults_list:
            test_data.append(np.array(data[i]))
            
        training_data=[]
        for i in untarget_fault_list:
            
            training_data.append(np.array(data[i]))
            
        x_for_all_array=np.concatenate(tuple(data_to_multi_time_step_list(all_data)))
        x_train_array=np.concatenate(tuple(data_to_multi_time_step_list(training_data)))
        x_test_array=np.concatenate(tuple(data_to_multi_time_step_list(test_data)))  
        print(x_train_array.shape[0])
        print('xxxxxxx')
                
        
        
        
   
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train_array.reshape(x_train_array.shape[0],-1)).reshape(x_train_array.shape[0],x_train_array.shape[1],x_train_array.shape[2])
        x_test=min_max_scaler.transform(x_test_array.reshape(x_test_array.shape[0],-1)).reshape(x_test_array.shape[0],x_test_array.shape[1],x_test_array.shape[2])
        x_for_all=min_max_scaler.transform(x_for_all_array.reshape(x_for_all_array.shape[0],-1)).reshape(x_for_all_array.shape[0],x_for_all_array.shape[1],x_for_all_array.shape[2])
        
        
        
        
        
        
        
 
        
        
        
        
        
    

        
      ######################get   y_train_fault_attribute_array_2
        
        all_fault_attribute_array_no_recti=np.zeros((15,len(well_trained_list_0)))
        
        for i in range(15):
            for j in range(len(well_trained_list_0)):
                # #print(i,j)
                all_fault_attribute_array_no_recti[i,j]=int(fault_attribute_matrix[i][well_trained_list_0[j]])                  
                    
                # test_mask_logits = logits
                # predict_y = test_mask_logits.max(1)[1]
                # accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
        
        
        all_fault_attribute_array_no_recti[all_fault_attribute_array_no_recti<0.1]=0.5
        all_fault_attribute_array_no_recti[all_fault_attribute_array_no_recti>0.5]=0
        all_fault_attribute_array_no_recti[all_fault_attribute_array_no_recti>0.1]=1     
        
    
    
        fault_attribute_list=["11101001100001000000","11011001100001000000","00000111010000000000","00000001001110000100","00000001000110000100","10100001100001000000","01101001100000000000","111110101000001000000","00000110010000100000","01001000010000100000","00000000001100100100","00000000000110100100","00000000000000011000","00000000001000000111","00000000000010000111"]
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[0][well_trained_list_0_4[i]])
        fault_attribute_0=fault_attribute_array
        ##print(fault_attribute_0)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[1][well_trained_list_0_4[i]])
        fault_attribute_1=fault_attribute_array
        ##print(fault_attribute_1)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[2][well_trained_list_0_4[i]])
        fault_attribute_2=fault_attribute_array
        ##print(fault_attribute_2)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[3][well_trained_list_0_4[i]])
        fault_attribute_3=fault_attribute_array
        ##print(fault_attribute_3)
        
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[4][well_trained_list_0_4[i]])
        fault_attribute_4=fault_attribute_array
        ##print(fault_attribute_4)
        
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[5][well_trained_list_0_4[i]])
        fault_attribute_5=fault_attribute_array
        ##print(fault_attribute_5)
        
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[6][well_trained_list_0_4[i]])
        fault_attribute_6=fault_attribute_array
        ##print(fault_attribute_6)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[7][well_trained_list_0_4[i]])
        fault_attribute_7=fault_attribute_array
        ##print(fault_attribute_7)
        
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[8][well_trained_list_0_4[i]])
        fault_attribute_8=fault_attribute_array
        ##print(fault_attribute_8)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[9][well_trained_list_0_4[i]])
        fault_attribute_9=fault_attribute_array
        ##print(fault_attribute_9)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[10][well_trained_list_0_4[i]])
        fault_attribute_10=fault_attribute_array
        ##print(fault_attribute_10)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[11][well_trained_list_0_4[i]])
        fault_attribute_11=fault_attribute_array
        ##print(fault_attribute_11)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[12][well_trained_list_0_4[i]])
        fault_attribute_12=fault_attribute_array
        ##print(fault_attribute_12)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[13][well_trained_list_0_4[i]])
        fault_attribute_13=fault_attribute_array
        ##print(fault_attribute_13)
        
        fault_attribute_array=np.zeros((480-time_step+1,len(well_trained_list_0_4)))
        for i in range(len(well_trained_list_0_4)):
                # ##print(i,j)
                fault_attribute_array[:,i]=int(all_fault_attribute_array_no_recti[14][well_trained_list_0_4[i]])
        fault_attribute_14=fault_attribute_array
        ##print(fault_attribute_14)
        
        
        y_all_fault_attribute_array_no_recti=[fault_attribute_0,fault_attribute_1,fault_attribute_2,fault_attribute_3,fault_attribute_4,fault_attribute_5,fault_attribute_6,fault_attribute_7,fault_attribute_8,fault_attribute_9,fault_attribute_10,fault_attribute_11,fault_attribute_12,fault_attribute_13,fault_attribute_14]
        
        y_train_all_fault_attribute_array_no_recti=[]
        for i in untarget_fault_list:
            y_train_all_fault_attribute_array_no_recti.append(y_all_fault_attribute_array_no_recti[i])
                
        y_train_fault_attribute_array_no_recti=np.concatenate(tuple(y_train_all_fault_attribute_array_no_recti))    
    
        pca = PCA(n_components=num_components)
        pca.fit(x_train[:,:,:].reshape(x_train.shape[0],-1))
        x_train_pca=pca.transform(x_train[:,:,:].reshape(x_train.shape[0],-1))
        x_test_pca=pca.transform(x_test[:,:,:].reshape(x_test.shape[0],-1))    
        # x_test_accuracy=0
        
        prob_test_list=[]
        for i in range(len(well_trained_list_0_4)):
            
            
            num_1=0
            num_set_2=[]
            for num in range(len(fault_attribute_array_well_trained_list[:,i] )):
                if num  in untarget_fault_list and fault_attribute_array_well_trained_list[:,i][num]==1 and num in num_7:
                    num_1+=1
                    num_set_2.append(num)
                    
            if len(num_set_2)==0 :
                # print("aaaaaaaaaaaaa")
    
            # if 1:            
                y_train_fault_attribute_array_012= copy.deepcopy(fault_attribute_array_well_trained_list[:,i])   
                # print(int(y_train_fault_attribute_array_012[0]))
                y_train_fault_attribute_array,y_test_fault_attribute_array=y_train_fault_attribute_array_012_to_y_train_and_test_fault_attribute_array(y_train_fault_attribute_array_012)
    
                NB = naive_bayes.GaussianNB()
                NB.fit(x_train_pca,y_train_fault_attribute_array)
                prob_test =NB.predict_proba(x_test_pca)
                # print(prob_test.shape)
                
                if prob_test.shape[1]==3:               
                    prob_test_aggregated=prob_test[:,1]+prob_test[:,2]
                if prob_test.shape[1]==2:
                    prob_test_aggregated=prob_test[:,1]
                prob_test_list.append(prob_test_aggregated.reshape(-1,1))
                
            else:
                
                
                    
                y_train_fault_attribute_array_012= copy.deepcopy(fault_attribute_array_well_trained_list[:,i])
                y_train_fault_attribute_array_012[num_set_2]=2
                y_train_fault_attribute_array,y_test_fault_attribute_array=y_train_fault_attribute_array_012_to_y_train_and_test_fault_attribute_array(y_train_fault_attribute_array_012)
        
                

        
                    
  
                print(well_trained_list_0_4[i])
                NB = naive_bayes.GaussianNB()
                NB.fit(x_train_pca,y_train_fault_attribute_array)
                
                NB_no_rect = naive_bayes.GaussianNB()
                NB_no_rect.fit(x_train_pca,y_train_fault_attribute_array_no_recti[:,i])
                
        

        
                prob_train =NB.predict_proba(x_train_pca)
        
                prob_test =NB.predict_proba(x_test_pca)  
                
                
                prob_train_no_rect =NB_no_rect.predict_proba(x_train_pca)
        
                
                prob_test_no_rect =NB_no_rect.predict_proba(x_test_pca)
                
                print(prob_train[:,1:].shape)             
                print(prob_train_no_rect[:,1].reshape(-1,1).shape)            
                
                
                x_train_prob=np.concatenate((prob_train[:,1:],prob_train_no_rect[:,1].reshape(-1,1)),axis=1)
        

                y_train_prob=y_train_fault_attribute_array_no_recti[:,i]
                
                x_test_prob=np.concatenate((prob_test[:,1:],prob_test_no_rect[:,1].reshape(-1,1)),axis=1)
                

                Y_train=y_train_prob
        
                
                input_layer=keras.Input(shape=(num_components,))
                input_layer1=keras.Input(shape=(3,))
                
                
        
                layer_1=Dense(8, activation='relu')(input_layer)

                layer_1=(Dense(5, activation='relu'))(layer_1)

                weight=Dense(3, activation='relu')(layer_1)
        
    
                out=keras.layers.dot([weight,input_layer1],axes=1)
                    
    
        

                
        
                
                
                model = keras.Model(inputs=[input_layer,input_layer1], outputs=out)
                
                model.compile(
        loss = 'mean_squared_error',
        optimizer = 'adam', #adam,SGD 
        metrics = ['mean_squared_error']
        )
        
        

        #prepare basic setups
                batch_size = 120  #批次
                nb_class = 2
                nb_epochs = epoch_index
                
 
                print(x_train_prob.shape)
                Training = model.fit([x_train_pca,x_train_prob], Y_train, 
                          epochs=nb_epochs, batch_size = batch_size) 

                prob_test_pred=model.predict([x_test_pca,x_test_prob])
                
                
                
        
                
        
                
                
                
        
        
        
                         
                prob_test_aggregated=prob_test_pred
                
                print(x_test_prob.shape)
                print(prob_test_pred.shape)
                    
                    
                prob_test_list.append(prob_test_aggregated.reshape(-1,1))
            # print(prob_test_list)
            # print("accuracy_train")
            # print(accuracy_train)
            # print("accuracy_test")
            # print(accuracy_test)
            # print("\n")
            # x_test_accuracy+=accuracy_test
        
        # print(prob_test_list.shape)
        # print('xxxxxxxxxxxxxxxxxx')
        prob_test_array=np.concatenate(tuple(prob_test_list),axis=1)
        
        accuracy_test=test_for_test(prob_test_array,y_test_non_one_hot)
        accuracy_list.append(accuracy_test[0])
        target_list.append(target_faults_list)
        print(accuracy_test[0])
        print(ii)
        backend.clear_session()
    print(target_list)
    print(ii)
    print('AFT')
    
    
    print(sum_list(accuracy_list))
    print(pearsonr_value)
    print(time_step)
    print(num_components)
    
    print(accuracy_list)
    
    accuracy_sum_list_epoch_index.append(sum_list(accuracy_list))
    accuracy_list_epoch_index.append(accuracy_list)
    accuracy_list=[]
print(accuracy_list_epoch_index)
print(accuracy_sum_list_epoch_index)



