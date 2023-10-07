
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:52:40 2022

@author: Administrator
"""





import copy

import sys

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
 

last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))

DATA_FOLDER =  last_path+r"\TE"
NUM_FILES = 15
experiment_index=2
pearsonr_value=0


time_step=16


num_components=10



load_classifier='Ture'


well_trained_list_0=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

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


            dist_list.append(np.linalg.norm(test_numpy[i]-fault_attribute_array_well_trained_list[j]))
            
        target_faults_dist_list.append(dist_list[target_faults_list[0]])
        target_faults_dist_list.append(dist_list[target_faults_list[1]])
        target_faults_dist_list.append(dist_list[target_faults_list[2]])
        test_y_pred_non_one_hot.append(target_faults_dist_list.index(min(target_faults_dist_list)))
    # ##print('aaaaaaaaaaaaaaaaa')        
    
    

    test_y_pred_non_one_hot_array=np.array(test_y_pred_non_one_hot)
    accuracy_0_480=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[0:480],test_y_pred_non_one_hot_array[0:480])
    accuracy_480_960=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[480:960],test_y_pred_non_one_hot_array[480:960])
    accuracy_960_1440=accuracy_for_two_array_non_one_hot(y_test_non_one_hot[960:1440],test_y_pred_non_one_hot_array[960:1440])
    accuracy=accuracy_for_two_array_non_one_hot(y_test_non_one_hot,test_y_pred_non_one_hot_array)  
    return accuracy, accuracy_0_480, accuracy_480_960, accuracy_960_1440#test_y_pred_non_one_hot   #accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


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
            



fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list<0.1]=0.5
fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list>0.5]=0
fault_attribute_array_well_trained_list[fault_attribute_array_well_trained_list>0.1]=1






all_fault_attribute_array=np.zeros((15,len(well_trained_list_0)))

for i in range(15):
    for j in range(len(well_trained_list_0)):
        # #print(i,j)
        all_fault_attribute_array[i,j]=int(fault_attribute_matrix[i][well_trained_list_0[j]])                  
            



all_fault_attribute_array[all_fault_attribute_array<0.1]=0.5
all_fault_attribute_array[all_fault_attribute_array>0.5]=0
all_fault_attribute_array[all_fault_attribute_array>0.1]=1 








files = [os.path.join(DATA_FOLDER, "d{:0>2}.dat".format(i)) for i in range(1,16,1)]
# test_files = [os.path.join(DATA_FOLDER, "d{:0>2}_te.dat".format(i)) for i in range(1,16,1)]

data = [pd.read_csv(f, sep='   ',  header=None,index_col = None) for f in files]



target_faults_list_all=[[0, 2, 7],[1,10,11],[3, 5, 8],[4, 6, 13],[9,12,14],
                [0, 4, 9], [1, 3, 11],[2, 6, 13],[5, 7,8],[10,12,14],
                [0, 11,14],[1, 2, 10],[ 3,8,12],[4,5, 13],[6,7,9],
                [0, 4, 13],[1, 9, 14],[2,3,12],[5,10,11], [6,7, 8],
                [0, 4, 6],[1, 11,13],[2, 5, 10],[3, 7, 9],[8,12,14]]  

num_set_2_list=    [[0,1]]

 
# num_set_2_list=[[0,1]]             

accuracy_test_list=[]

print(len(num_set_2_list))
# for num_7 in  num_set_2_list:

for xxx in  range(0,len(num_set_2_list)):
    
    num_7=num_set_2_list[xxx]
    
    
    print(num_set_2_list.index(num_7))
    print(xxx)
    print(len(num_set_2_list))
                   
                    
                    
    target_list=[]
    accuracy_list=[]    
    for ii in range(len(target_faults_list_all)):
        
        print(ii)
        print('target_faults')
        print(num_set_2_list.index(num_7))
        print(xxx)
        print(len(num_set_2_list))
        
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
   
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train_array.reshape(x_train_array.shape[0],-1)).reshape(x_train_array.shape[0],x_train_array.shape[1],x_train_array.shape[2])
        x_test=min_max_scaler.transform(x_test_array.reshape(x_test_array.shape[0],-1)).reshape(x_test_array.shape[0],x_test_array.shape[1],x_test_array.shape[2])
        x_for_all=min_max_scaler.transform(x_for_all_array.reshape(x_for_all_array.shape[0],-1)).reshape(x_for_all_array.shape[0],x_for_all_array.shape[1],x_for_all_array.shape[2])
        np.random.seed(0)
        


        
      ######################get   y_train_fault_attribute_array_2
        
  
        


        pca = PCA(n_components=num_components)
        pca.fit(x_train[:,:,:].reshape(x_train.shape[0],-1))
        x_train_pca=pca.transform(x_train[:,:,:].reshape(x_train.shape[0],-1))
        x_test_pca=pca.transform(x_test[:,:,:].reshape(x_test.shape[0],-1))
    
        
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
    
                
                NB = naive_bayes.GaussianNB()
                NB.fit(x_train_pca,y_train_fault_attribute_array)            
                prob_test =NB.predict_proba(x_test_pca)
                if prob_test.shape[1]==3:                
                    prob_test_aggregated=prob_test[:,1]+prob_test[:,2]
                if prob_test.shape[1]==2:
                    prob_test_aggregated=prob_test[:,1]
                
    
                prob_test_list.append(prob_test_aggregated.reshape(-1,1))
                    

                    
                    
    
        
        prob_test_array=np.concatenate(tuple(prob_test_list),axis=1)    
        accuracy_test=test_for_test(prob_test_array,y_test_non_one_hot)
        accuracy_list.append(accuracy_test[0])

            
    print(accuracy_list)
    print(target_list)

    accuracy_test_list.append(sum_list(accuracy_list))
    
print(accuracy_test_list)
print(max(accuracy_test_list))
print(num_set_2_list[accuracy_test_list.index(max(accuracy_test_list))])


