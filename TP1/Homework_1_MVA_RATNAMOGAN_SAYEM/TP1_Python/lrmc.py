# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:37:59 2017

@Pirashanth Ratnamogan & Othmane Sayem
"""

import numpy as np
from loadimage import read_necessary_images
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

sign = lambda x : -1 if x<0 else 1


def create_space_omega(X):
    '''
    Funciton that returns a matrix with coefficient m_ij= 1 if the coefficient x_ij in  
    the input matrix is different of 0 and 0 else
    Parameter: X : (np.array) X
    '''
    Omega = np.array(X!=0,dtype=int)
    return Omega

def D_Shrinkage(Singular_values,tau):
    '''
    Compute the shrinkage of singular_values with the parameter tau
    Parameter : Singular_values : the singular values
    tau : the paramers
    '''
    return np.array([sign(Singular_values[i])*max(abs(Singular_values[i])-tau,0) \
                          for i in range(Singular_values.shape[0])])

def Shrinkage_Thresholding_Operator(Matrix,tau):
    '''
    Compute the shrinkage of singular_values of the matrix Matrixwith the parameter tau
    Parameter : Matrix : Input matrix
    tau : the paramers
    '''
    u,sigma,v = np.linalg.svd(Matrix);
    sigma_shrink = D_Shrinkage(sigma,tau)
    S = np.zeros(Matrix.shape)
    S[:sigma_shrink.shape[0],:sigma_shrink.shape[0]] = np.diag(sigma_shrink)
    return np.dot(u,np.dot(S,v))
    
def create_random_mask(size,proba_zero):
    '''
    Create a random mask matrix (with only 0 and 1) to compute a "missing values" space
    Parameter: size = size of the mask matrix
    proba_zero: is the probability that a given coefficient get a 0 value (else 1)
    '''
    return np.random.choice([0, 1], size=size, p=[proba_zero,1-proba_zero])
    
#Omega = create_space_omega(X) #pour test seulement 

def lrmc(X,W,tau,beta):
    '''
    Compute the full Low rank matrix completion algorithm
    Parameter: X : matrix to complete
    W: Known mask that give the known positions of the missing values
    tau: shrinkage parameter
    beta: learning parameter
    '''
    Z = np.zeros(X.shape)
    A= np.ones(X.shape)
    i=0
    while((np.linalg.norm(W*X-W*A)>1e-2)& (i<10000)):
        A= Shrinkage_Thresholding_Operator(Z*W,tau)
        print(np.linalg.norm(W*X-W*A))
        Z= Z+ beta*(W*X - W*A)
        i= i + 1 
    return A


def compute_the__image_matrix_completion():
    '''
    Compute the matrix completion taking only the first image
    The full code for the matrix completion is available in the matlab code
    '''
    images_flatten = read_necessary_images(r'C:\Users\Pirashanth\Desktop\Unsupervised Learning\YaleB-Dataset\images',1)
    Mask = create_random_mask(images_flatten[0].shape,0.2)
    Corrupted_data = Mask*images_flatten[0]
    beta = min(2,Corrupted_data.size/sum(Mask))
    Datas = lrmc(Corrupted_data,Mask,10000,beta)
    return Datas

def compute_movies_recommandation_csv():
    '''
    Compute the code for the movie recommandation
    Output a csv with the MSE and the MAE for different values of tau and of test rate values
    '''
    file=open(r'./resultats_genre2.txt',mode='a') #outputfile
    #file.write('tau,beta,taux_mask,MAE,MLE \n')
    file.close()
    data = pd.read_csv(r'/home/ratnamogan/Documents/Unsupervid Learning/TP1/ratings_medium_n4_Horror_Romance_42.csv',delimiter=',',index_col=0)
    
    list_tau = [1,10,100,1000,10000]
    list_proba_0 = [0.2,0.4,0.6]
    
    for cur_tau in list_tau:
        for proba_0_data in list_proba_0:
            
            data_train= pd.DataFrame(data[data.genreId==2]) #Compute the genreId=2
            mask = create_random_mask(data_train.rating.shape,proba_0_data)
            
            Train_matrix = np.zeros((max(data_train.userInd)+1,max(data_train.movieInd)+1))
            Complete_Matrix= np.zeros((max(data_train.userInd)+1,max(data_train.movieInd)+1))
                
            for i,j,rate,i_mask in zip(data_train.userInd,data_train.movieInd,data_train.rating,mask):
                Train_matrix[i,j]= i_mask*rate
                Complete_Matrix[i,j] = rate
            
            Omega = create_space_omega(Train_matrix)
            
            
            cur_beta =2  
            Completed_matrix = lrmc(Train_matrix,Omega,cur_tau,cur_beta)
            
            y_true = []
            y_pred = []
            for i,j,rate,i_mask in zip(data_train.userInd,data_train.movieInd,data_train.rating,mask):
                if (i_mask==0):
                    y_true.append(rate)
                    y_pred.append(Completed_matrix[i,j])
            
            MSE_Score = mean_squared_error(y_true,y_pred)
            MAE_score = mean_absolute_error(y_true,y_pred)
            
            str_to_write = str(cur_tau) +','+ str(cur_beta)+',' +str(proba_0_data)+',' +str(MAE_score) +','+str(MSE_Score) +'\n'
            
            file=open(r'./resultats_genre2.txt',mode='a')
            
            file.write(str_to_write)
            
            file.close()

        