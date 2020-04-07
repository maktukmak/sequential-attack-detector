import numpy as np
from numpy.linalg import inv

def softmax(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    e_x = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    out = tmp - e_x
    prob = np.exp(out)
    
    return prob


def logsumexp(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    lse = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    return lse

def logdet(x):
    
    return np.log(np.linalg.det(x))

def mse_unc(mean_user, sigma_user, mean_item, sigma_item, Rtrain, Otrain, muR_user, muR_item):

    sec_mom = (np.einsum('ijj->ij', sigma_user) + (mean_user.T)**2) @ (np.einsum('ijj->ij', sigma_item) + (mean_item.T)**2).T
    Sigma = sec_mom - ((mean_user.T)**2) @ (mean_item**2)
    nloglik_user = Sigma + (mean_user.T @ mean_item + muR_user[None].T + muR_item)**2 - 2*(mean_user.T @ mean_item  + muR_user[None].T + muR_item)*Rtrain + Rtrain**2
    nloglik_user[np.where(Otrain == 0)] = 0
    mse = np.sum(nloglik_user) / np.sum(Otrain == 1)
    
    return mse
