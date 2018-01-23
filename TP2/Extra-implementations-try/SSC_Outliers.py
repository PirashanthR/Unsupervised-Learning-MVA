# -*- coding: utf-8 -*-
import numpy as np
import scipy
from sklearn.cluster import KMeans

epsilon=1e-1

def soft_thresholding(mat, threshold):
    """
    Soft thresholding operator
    Proximal operator of the L1 norm

    Parameters:
    -----------
    :param mat:     array-like
                    matrix to threshold
    :param threshold: real value
                    value of threshold

    Returns:
    --------
    mat:            array-like
                    thresholded matrix
    """

    mat[np.abs(mat) < threshold] = 0
    mat[mat > threshold] -= threshold
    mat[mat < -threshold] += threshold

    return mat


def compute_sparse_C(data,mu1,mu2,verbose):
    """
    Matrix C with corrupted entries Minimization by ADMM

    Parameters:
    -----------
    data:    array, shape[D, N]
             data matrix, N examples of dimension D
    mu1:     positive real
             penalisation of errors for the sparse representation
    mu2:     postive real
             parameter of the augmented Lagrangian method
    verbose: boolean
             level of verbosity

    Returns:
    --------
    C:       array, shape[N, N]
             robust sparse representation of data
    """
    global epsilon
    C = np.zeros((data.shape[1],data.shape[1]))
    lambda2 = np.zeros((data.shape[1],data.shape[1]))
    Z= np.ones((data.shape[1],data.shape[1]))*float('inf')
    lambda1 = np.zeros(data.shape)
    E = np.zeros(data.shape)
    Z_1 = np.linalg.inv(mu1*np.dot(data.transpose(),data)+ mu2*np.eye(data.shape[1]))
    itern = 0
    
    ## Computation of ADMM algorithm ##
    while ((np.linalg.norm((Z-C))>epsilon)or(np.linalg.norm(data-np.dot(data,Z)-E)>epsilon))and(itern<150):
        itern += 1         

        Z_2 = mu1*np.dot(data.T,data-E+lambda1/mu1) + mu2*(C-lambda2/mu2)
        Z = np.dot(Z_1,Z_2)
        
        C = soft_thresholding(Z+lambda2/mu2,1/mu2)
        np.fill_diagonal(C,0)
                
        E = soft_thresholding(data-np.dot(data,Z)+lambda1/mu1,mu1/mu2)
        lambda1 = lambda1 + mu1*(data-np.dot(data,Z)-E)
        
        lambda2 = lambda2 + mu2*(Z-C)
        
        print(np.linalg.norm((Z-C)),np.linalg.norm(data-np.dot(data,Z)))    
    return C


def SSC(data, n, mu1, mu2, verbose = False):
    """
    Sparse Subspace algorithm

    Inspired from : algorithm 8.5
    Vidal, René, Yi Ma, and S. Shankar Sastry. "Principal Component Analysis." Generalized Principal Component Analysis. Springer New York, 2016

    Parameters:
    -----------
    data:    array, shape[D, N]
             data matrix, N examples of dimension D
    n:       positive integer
             number of subspaces
    mu1:     positive real
             penalisation of errors for the sparse representation
    mu2:     postive real
             parameter of the augmented Lagrangian method
    verbose: boolean
             level of verbosity

    Returns:
    --------
    C:       array, shape[N, N]
             robust sparse representation of data
    groups:  array-like, shape (N,)
             labels (between 0 and n-1) of each point
    """
    # Compute robust sparse representation of data
    C = compute_sparse_C(data, mu1, mu2, verbose = verbose)

    # Compute affinity matrix
    W = np.abs(C) + np.abs(C.transpose())

    # Compute D and L associated to graph
    D = np.diag(W.sum(axis = 1))
    L = D - W

    if verbose:
        print (D)

    # Normalized spectral clustering
    D_root = np.linalg.inv(scipy.linalg.sqrtm(D))
    _, evectors = np.linalg.eigh(np.dot(np.dot(D_root, L), D_root))
    kmeans = KMeans(n_clusters = n, init = 'random').fit(evectors[:, :n])

    return C, kmeans.labels_
