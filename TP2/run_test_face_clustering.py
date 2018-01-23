# -*- coding: utf-8 -*-

import numpy as np

#Algorithm and error
from error_evaluation import *
from spectral_clustering import *
from SSC import *
from ksubspaces import *
from scipy.io import loadmat
from sklearn.cluster import KMeans

nb_ind = 2

mat = loadmat('./data/ExtendedYaleB.mat')
data = mat['EYALEB_DATA']
true_label = mat['EYALEB_LABEL']
data = np.array(data, dtype = np.int64)

true_label = true_label[:,:nb_ind*64]
true_label = true_label-1
true_label=true_label.reshape((-1,))
data=data[:,0:nb_ind*64]

############parameters SSC#############
mu2 = 30
tau = 1e-5

##########Evaluate SSC#######
print('Evaluate SSC')
C_m, eval_label = SSC(data,nb_ind,tau,mu2)
error_SSC= evaluate_error(eval_label,true_label)

############parameters K-Subspaces#############
nb_restart = 3
d= 3

##########Evaluate K-Subspaces#######
print('Evaluate K-subspaces')
eval_label_ks= ksubspaces(data,nb_ind,[d]*nb_ind,nb_restart)[0][0]
error_ks = evaluate_error(eval_label_ks,true_label)

##########Evaluate Spectral Clustering#######
knn = 5
sigma = 100


##########Evaluate Spectral Clustering#######
print('Evaluate Spectral Clustering')
W = gaussian_affinity(data, knn,sigma)
print ("Percentage of W filled : {}%".format((W > 0).sum() / len(W)**2 * 100))
eval_label_sc = SC(W, nb_ind)

error_SC=evaluate_error(eval_label_sc,true_label)


print("Error SC=",error_SC)
print("Error SSC=",error_SSC)
print("Error_Kubspaces=",error_ks)



