#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:22:17 2017

@author: ratnamogan
"""

import numpy as np
import scipy
from sklearn.cluster import KMeans
from scipy.io import loadmat
from error_evaluation import evaluate_error
from sklearn.decomposition import PCA

epsilon = 1e-1


def ksubspaces(data, n , d, replicates):
    N = data.shape[1]
    D = data.shape[0]
       
    err = []
    for r in range(replicates) :
        
         mu = np.zeros((n,D)) 
         for i in range(n):
            mu[i,:]= data[:,np.random.randint(0,N)]
        
         mu_prev = np.ones(mu.shape)
    
        
        
        ### randomly selecting U 
         U = np.random.randn(n,D,int(d[0]))
         for m in range(n):
            for i in range(int(d[0])):
                U[m,:,i] = U[m,:,i]/np.linalg.norm(U[m,:,i])
        
         U_prev = np.ones((n,D,int(d[0])))
         y = np.zeros((D,N))
         
         print('mu Error : ' , np.linalg.norm(mu-mu_prev) , 'U error', np.linalg.norm(U-U_prev))        
    
         while (np.linalg.norm(mu-mu_prev)>epsilon or np.linalg.norm(U-U_prev)>epsilon ):
        
            U_prev = U
            #U = np.zeros((n,D,int(d[0])))
            mu_prev = mu
            w = np.zeros((n,N))
            
            distance = np.zeros((n,N)) ##used for multiple restart
            
            U2 = np.zeros((n,D,D))
            for i in range(n) : 
                U2[i,:,:] = np.identity(D)-np.dot(U[i,:,:],U[i,:,:].transpose())
                    
            ## Segmentation
            for j in range(N):
                l=[]
                for i in range(n):
                    lii = np.dot(U2[i,:,:],np.reshape(data[:,j]-mu[i,:],(D,1)))
                    l.append(np.linalg.norm(lii))
                    distance[i,j] = np.linalg.norm(lii)
                i = np.argmin(l)
                w[i,j] = 1
            #print(i)
            
            
            U = np.zeros((n,D,int(d[0])))
            # Estimation
            mu = np.dot(w, data.transpose())
            for i in range(n):
                mu[i,:] = mu[i]/sum(w[i,:])
                
                A = np.zeros((D,D))
                
                for j in range(N) : 
                    if (w[i,j]==1):
                        p = np.reshape(data[:,j]-mu[i,:],(D,1))
                        A +=  w[i,j]*np.dot(p,p.transpose()) 
                        
                e, v = np.linalg.eigh(A)
                U[i,:,:] = v[:,-int(d[i,0]):]
                
                
                for j in range(N):
                    if (w[i,j] == 1):
                        y[0:int(d[i,0]),j] = np.dot(U[i,:,:].transpose(),np.reshape(data[:,j]-mu[i,:],(D,1)))[:,0]
        
            print('mu Error : ' , np.linalg.norm(mu-mu_prev) , 'U error', np.linalg.norm(U-U_prev))        

        
         error_cur = np.linalg.norm(w*distance)
         err.append(error_cur)
         
         if (error_cur == min(np.array(err))):
             w_opt = w
             mu_opt = mu
             U_opt =U
             y_opt = y
             
    print (err)
    
    return w_opt, mu_opt, U_opt, y_opt


mat = loadmat('../data/ExtendedYaleB.mat')
data = mat['EYALEB_DATA']
true_label = mat['EYALEB_LABEL']
data = np.array(data, dtype = np.int64)
true_label = true_label[:,:640]

data=data[:,0:640]

data_transform = PCA(n_components=90).fit_transform(data.T)
data_transform= data_transform.T

w_opt, mu_opt, U_opt, y_opt= ksubspaces(data_transform,10,5*np.ones((10,1)),20)
eval_label = (KMeans(n_clusters=10).fit(w_opt.T)).labels_
true_label= true_label-1
true_label= true_label.reshape(-1)
error= evaluate_error(eval_label,true_label)
