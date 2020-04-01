# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:59 2018

@author: Mehmet
"""

import numpy as np
#from lightfm.datasets import fetch_stackexchange
import time
from utils import softmax
import os
dirname = os.path.dirname(__file__)

def train_test_split(ratedata, InfoData, option = 'warm'):
    
    """
        Train\Test splitting of interactions
        Argument 'option' choses between 'warm' or 'cold' start scenario
    """

    J = InfoData.J
    I = InfoData.I
    
    ratedata_test = np.empty((0,ratedata.shape[1]))
    ratedata_train = np.empty((0,ratedata.shape[1]))
    ratedata_val = np.empty((0,ratedata.shape[1]))
    
    if option == 'warm':
        for i in range(0, J):
           itemdatarate = len(np.where(ratedata[:,1] == i + 1)[0])
           if itemdatarate < 5:
               ind = np.where(ratedata[:,1] == i+1)[0]
               ratedata_train = np.append(ratedata_train, ratedata[ind, :], axis = 0)
           else:
               itemdatarate_test = int(float(itemdatarate) / 5)
               ind = np.where(ratedata[:,1] == i+1)[0]
               np.random.shuffle(ind)
               ratedata_test = np.append(ratedata_test, ratedata[ind[0:itemdatarate_test], :], axis = 0)
               ratedata_val = np.append(ratedata_val, ratedata[ind[itemdatarate_test:2*itemdatarate_test], :], axis = 0)
               ratedata_train = np.append(ratedata_train, ratedata[ind[2*itemdatarate_test:], :], axis = 0)
           
    elif option == 'cold_item':
        indtest = np.arange(J)
        np.random.shuffle(indtest)
        for i in range(0, J):
            ind = np.where(ratedata[:,1] == indtest[i]+1)[0]
            if i < int(J/5):
                ratedata_test = np.append(ratedata_test, ratedata[ind, :], axis = 0)
            elif i < int(2*J/5):
                ratedata_val = np.append(ratedata_val, ratedata[ind, :], axis = 0)
            else:
                ratedata_train = np.append(ratedata_train, ratedata[ind, :], axis = 0)
                
    elif option == 'cold_user':
        indtest = np.arange(I)
        np.random.shuffle(indtest)
        for i in range(0, I):
            ind = np.where(ratedata[:,0] == indtest[i]+1)[0]
            if i < int(I/5):
                ratedata_test = np.append(ratedata_test, ratedata[ind, :], axis = 0)
            elif i < int(2*I/5):
                ratedata_val = np.append(ratedata_val, ratedata[ind, :], axis = 0)
            else:
                ratedata_train = np.append(ratedata_train, ratedata[ind, :], axis = 0)
    
    return ratedata_train, ratedata_test, ratedata_val

def rate_to_matrix(ratedata, I, J):
    
    R = np.zeros((I,J))
    C = np.zeros((I,J), dtype = np.int8)
    R[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = ratedata[:,2]
    C[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = 1
    
    return R, C



class DataSetInfo:
    def __init__(self):
        self.I = 900
        self.J = 1700
        self.Du = 1
        self.Mu = np.array([ 1, 20])
        self.Di = 1
        self.Mi = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

def mm_data_generation(params, impfeed = 0, K = 20 ): 
    
    """
        Synthetic data geenration based on params
    """
    
    lmd_u = 1
    lmd_v = 1
    cij = 1
    miss_frac = 0.95
    
    Du = params.Du
    I = params.I
    Mu= params.Mu
    Di = params.Di
    J = params.J
    Mi = params.Mi

        
    mean = np.zeros((K))
    cov = (1/lmd_u) * np.identity(K)
    U = np.random.multivariate_normal(mean, cov, I)
    U = U.T
    
    mean = np.zeros((K))
    cov = (1/lmd_v) * np.identity(K)
    V = np.random.multivariate_normal(mean, cov, J)
    V = V.T
    
    W = np.random.multivariate_normal(np.zeros(K), np.identity(K), Du)
    #mu_W = np.random.normal(0, 1, Dx)
    mu_W = np.zeros((Du,1)) 
    Sig_X = 10 * np.eye(Du)
    Xorg = np.zeros((Du,I))
    mean = W @ U + mu_W
    for i in range(0,I):
        Xorg[:,i] = np.random.multivariate_normal(mean[:,i], Sig_X, 1)
    
    
    H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(Mu))
    #mu_H = np.random.normal(0, 1, np.sum(Mu))
    Yorg = np.zeros((np.sum(Mu),I))
    mean = H @ U
    for i in range(0,Mu.shape[0]):
        ind = range(np.sum(Mu[0:i]), np.sum(Mu[0:i+1]))
        prob = softmax(mean[ind, :])
        for j in range(0,I):
            Yorg[ind,j] = np.random.multinomial(1, prob[:,j], size=1)[0, 0:Mu[i]]
            
    FeatureUser = np.append(Xorg, Yorg, axis = 0)
    
    
    A = np.random.multivariate_normal(np.zeros(K), np.identity(K), Di)
    #mu_A = np.random.normal(0, 1, Dz)
    mu_A = np.zeros((Di,1))
    Sig_Z = np.eye(Di)
    Zorg = np.zeros((Di,J))
    mean = A @ V + mu_A
    for j in range(0,J):
        Zorg[:,j] = np.random.multivariate_normal(mean[:,j], Sig_Z, 1)
    
    
    B = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(Mi))
    Porg = np.zeros((np.sum(Mi),J))
    mean = B @ V
    for i in range(0,Mi.shape[0]):
        ind = range(np.sum(Mi[0:i]), np.sum(Mi[0:i+1]))
        prob = softmax(mean[ind, :])
        for j in range(0,J):
            Porg[ind,j] = np.random.multinomial(1, prob[:,j], size=1)[0, 0:Mi[i]]
    
    FeatureItem = np.append(Zorg, Porg, axis = 0)
    
    # Rating Generation
    C = np.random.choice([0, 1], size=(I, J) , p=[miss_frac, 1-miss_frac])
    RateData = np.zeros((C.sum(), 3))
    ind = np.where(C == 1)
    RateData[:,0] = ind[0] + 1
    RateData[:,1] = ind[1] + 1
    mean = (U.T @ V)[ind]
    
    if impfeed == 0:
        for i in range(0, len(RateData)):
            RateData[i,2] = np.random.normal(mean[i], cij, 1)
            
        mu_R = np.mean(RateData[:,2])
        std_R = np.std(RateData[:,2])
        RateData[:,2] = (RateData[:,2] - mu_R) / std_R
        prob = softmax(RateData[:,2][None])[0,:]
        RateData[:,2] = np.round((prob * 4) + 1)
        
        
    else:
        prob = softmax(mean[None])[0,:]
        for i in range(0, len(RateData)):
            #Rate[i,2] = np.random.binomial(1, prob[i])
            if prob[i] > 0.5:
                RateData[i,2] = 1
            
        RateData = np.delete(RateData, np.where(RateData[:,2] == 0), 0)
    
    
    #ratedata_train, ratedata_test = train_test_split(Rate, I, J, option)

    #Rtrain, Ctrain = rate_to_matrix(ratedata_train, I, J)
    #Rtest, Ctest = rate_to_matrix(ratedata_test, I, J)
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    print('Dataset Generated with sparsity level: {:.3f} ({:d} interactions)'.format(1 - (len(RateData) / (C.shape[0] * C.shape[1])), len(RateData)))
    
    return FeatureUser, FeatureItem, RateData, InfoData

def movie100kprep(ImpFeed = 0):

    filename = os.path.join(dirname, "Datasets/ml-100k/u.user")
    usrdatac = np.genfromtxt(filename, delimiter='|', usecols= [0, 1])
    usrdatad = np.genfromtxt(filename, delimiter='|', usecols= [2, 3], dtype = str)
    Ddu = []
    ind = np.unique(usrdatad[:,0])
    Ddu.append(ind.shape[0])
    for i in range(0, ind.shape[0]):
        usrdatad[np.where(usrdatad[:,0] == ind[i] ), 0] = i
    ind = np.unique(usrdatad[:,1])
    Ddu.append(ind.shape[0])
    for i in range(0, ind.shape[0]):
        usrdatad[np.where(usrdatad[:,1] == ind[i] ), 1] = i
    Ddu = np.array(Ddu)
    Mu = Ddu - 1
    usrdatad = usrdatad.astype(int)
    
    I = usrdatac.shape[0]
    usrdatacat = np.zeros((I, sum(Ddu)-Ddu.shape[0]))
    temp = np.zeros((I, Mu[0]+1))
    temp[np.arange(I), usrdatad[:,0]-1] = 1
    usrdatacat[:,0:Mu[0]] = temp[:,0:Mu[0]]
    temp = np.zeros((I, Mu[1]+1))
    temp[np.arange(I), usrdatad[:,1]-1] = 1
    usrdatacat[:,Mu[0]:Mu[0] + Mu[1]] = temp[:,0:Mu[1]]
    
    Xorg = usrdatac[:, 1][None]
    Yorg = usrdatacat.T
    Du = 1

    filename = os.path.join(dirname, "Datasets/ml-100k/u.item")
    itmdatac = np.genfromtxt(filename, delimiter='|', usecols= [1], dtype = str)
    itmdatad = np.genfromtxt(filename, delimiter='|', usecols= list(range(5,24)), dtype = str)
    Mi = np.ones(itmdatad.shape[1]).astype(int)
    Porg = itmdatad.astype(int).T
    
    Di = 1
    
    J = itmdatac.shape[0]
    for i in range(0, J):
        itmdatac[i] = itmdatac[i][-5:-1]
    Zorg = itmdatac.astype(float)[None]
    
    filename = os.path.join(dirname, "Datasets/ml-100k/u.data")
    ratedata = np.genfromtxt(filename, delimiter='\t')
    ratedata = np.delete(ratedata,3, axis=1)
    
    if ImpFeed == 1:
        #ratedata[:, 2] = 1
        ratedata[np.where(ratedata[:,2] < 4),2] = -1
        ratedata[np.where(ratedata[:,2] >= 4),2] = 1
        
        
    # Normalization of Cont. Data
    mu_W = np.mean(Xorg, axis=1)
    Xnorm = Xorg - mu_W[None].T
    std_X = np.std(Xnorm, axis=1)
    Xnorm /= std_X[None].T
    Ynorm = Yorg.copy()
    
    FeatureUser = np.append(Xnorm, Ynorm, axis = 0)
        
    mu_A = np.mean(Zorg, axis=1)
    Znorm = Zorg - mu_A[None].T
    std_Z = np.std(Znorm, axis=1)
    Znorm /= std_Z[None].T
    Pnorm = Porg.copy()
    
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    return FeatureUser, FeatureItem, ratedata, InfoData

def movie1mprep(ImpFeed = 0):
    

    filename = os.path.join(dirname, "Datasets/ml-1m/users.dat")
    usrdatac = np.genfromtxt(filename, delimiter='::', usecols= [2])
    usrdatad = np.genfromtxt(filename, delimiter='::', usecols= [1, 3], dtype = str)
    Ddu = []
    ind = np.unique(usrdatad[:,0])
    Ddu.append(ind.shape[0])
    for i in range(0, ind.shape[0]):
        usrdatad[np.where(usrdatad[:,0] == ind[i] ), 0] = i
    ind = np.unique(usrdatad[:,1])
    Ddu.append(ind.shape[0])
    
    Ddu = np.array(Ddu)
    Mu = Ddu - 1
    usrdatad = usrdatad.astype(int)
    
    I = usrdatac.shape[0]
    usrdatacat = np.zeros((I, sum(Ddu)-Ddu.shape[0]))
    temp = np.zeros((I, Mu[0]+1))
    temp[np.arange(I), usrdatad[:,0]-1] = 1
    usrdatacat[:,0:Mu[0]] = temp[:,0:Mu[0]]
    temp = np.zeros((I, Mu[1]+1))
    temp[np.arange(I), usrdatad[:,1]-1] = 1
    usrdatacat[:,Mu[0]:Mu[0] + Mu[1]] = temp[:,0:Mu[1]]
    Xorg = usrdatac.T[None]
    Yorg = usrdatacat.T
    Mu = np.array(Mu)
    Du = 1
    
    filename = os.path.join(dirname, "Datasets/ml-1m/movies.dat")
    itmdatac = np.genfromtxt(filename, delimiter='::', usecols= [1], dtype = str)
    itmdatad = np.genfromtxt(filename, delimiter='::', usecols= [2], dtype = str)
    itmmap = np.genfromtxt(filename, delimiter='::', usecols= [0], dtype = int)
    J = itmdatac.shape[0]
    for i in range(0, J):
        itmdatac[i] = itmdatac[i][-5:-1]
    Zorg = itmdatac.astype(float)[None]
    
    Genres = []
    for i in range(0, len(itmdatad)):
        Genres = np.append(Genres, itmdatad[i].split("|"))
    ind = np.unique(Genres)
    Ddi = (ind.shape[0])
    Mi = np.ones(Ddi).astype(int)
    itmdatacat = np.zeros((Ddi, J))
    for i in range(0, J):
        for j in range(0, len(itmdatad[i].split("|"))):
            itmdatacat[np.where(ind == itmdatad[i].split("|")[j])[0], i] = 1
    Porg = itmdatacat.astype(int)
    Di = 1
    
    filename = os.path.join(dirname, "Datasets/ml-1m/ratings.dat")
    ratedata = np.genfromtxt(filename, delimiter='::')
    ratedata = ratedata[:,0:3]
    for i in range(0, len(ratedata)):
        ratedata[i, 1] = np.where(itmmap == ratedata[i,1])[0]+1
    if ImpFeed == 1:
        ratedata[:, 2] = 1
    
    # Normalization of Cont. Data
    mu_W = np.mean(Xorg, axis=1)
    Xnorm = Xorg - mu_W[None].T
    std_X = np.std(Xnorm, axis=1)
    Xnorm /= std_X[None].T
    Ynorm = Yorg.copy()
        
    mu_A = np.mean(Zorg, axis=1)
    Znorm = Zorg - mu_A[None].T
    std_Z = np.std(Znorm, axis=1)
    Znorm /= std_Z[None].T
    Pnorm = Porg.copy()
        
    FeatureUser = np.append(Xnorm, Ynorm, axis = 0)
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    return FeatureUser, FeatureItem, ratedata, InfoData

def movie10mprep(ImpFeed = 0):
    

    filename = os.path.join(dirname, "Datasets/ml-10M100K/movies.dat")
    itmdatac = np.genfromtxt(filename, delimiter='::', usecols= [1], encoding='utf-8', dtype = str)
    itmdatad = np.genfromtxt(filename, delimiter='::', usecols= [2], encoding='utf-8', dtype = str)
    itmmap = np.genfromtxt(filename, delimiter='::', usecols= [0], encoding='utf-8', dtype = int)
    J = itmdatac.shape[0]
    for i in range(0, J):
        itmdatac[i] = itmdatac[i][-5:-1]
    Zorg = itmdatac.astype(float)[None]
    
    Genres = []
    for i in range(0, len(itmdatad)):
        Genres = np.append(Genres, itmdatad[i].split("|"))
    ind = np.unique(Genres)
    Ddi = (ind.shape[0])
    Mi = np.ones(Ddi).astype(int)
    itmdatacat = np.zeros((Ddi, J))
    for i in range(0, J):
        for j in range(0, len(itmdatad[i].split("|"))):
            itmdatacat[np.where(ind == itmdatad[i].split("|")[j])[0], i] = 1
    Porg = itmdatacat.astype(int)
    Di = 1
    
    filename = os.path.join(dirname, "Datasets/ml-10M100K/ratings.dat")
    ratedata = np.genfromtxt(filename, delimiter='::')
    ratedata = ratedata[:, 0:3]
    for i in range(0, len(ratedata)):
        ratedata[i, 1] = np.where(itmmap == ratedata[i,1])[0]+1
    if ImpFeed == 1:
        ratedata[:, 2] = 1
        
    
    mu_A = np.mean(Zorg, axis=1)
    Znorm = Zorg - mu_A[None].T
    std_Z = np.std(Znorm, axis=1)
    Znorm /= std_Z[None].T
    Pnorm = Porg.copy()
    
    I = 71567
    Xnorm = np.zeros((1,I))
    Ynorm = np.zeros((1,I))
    Mu = np.array([1])
    Du = 0
    
    FeatureUser = np.append(Xnorm, Ynorm, axis = 0)
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    return FeatureUser, FeatureItem, ratedata, InfoData

def decathlonprep():
    
    filename = os.path.join(dirname, "Datasets/decathlon/user_products_09062018_1536261381.csv")
    usrmap = np.genfromtxt(filename, delimiter=',', skip_header = 1, usecols= [0], encoding='utf-8', dtype = int)
    itmmap = np.genfromtxt(filename, delimiter=',', skip_header = 1, usecols= [6], encoding='utf-8', dtype = int)
    ratedata = np.append(usrmap[None] + 1, itmmap[None] + 1, axis = 0).T
    ratedata = np.append(ratedata, np.ones((ratedata.shape[0],1)), axis = 1)
    
    R, C = rate_to_matrix(ratedata, int(max(usrmap)) + 1, max(itmmap)+1)
    usr_filt = np.where(np.sum(R, axis = 1) < 10)[0]
    R_filt = np.delete(R, usr_filt, axis = 0)
    item_filt = np.where(np.sum(R_filt, axis = 0) == 0)[0]
    R_filt = np.delete(R_filt, item_filt, axis = 1)
    
    ratedata = np.append(np.where(R_filt == 1)[0][None].T + 1, np.where(R_filt == 1)[1][None].T + 1, axis = 1)
    ratedata = np.append(ratedata, np.ones((ratedata.shape[0],1)), axis = 1)
    
    I = R_filt.shape[0]
    J = R_filt.shape[1]
    
    itmdatad = np.genfromtxt(filename, delimiter=',', skip_header = 1, usecols= [6, 7], encoding='utf-8', dtype = int)
    itmdatadsrt = itmdatad[itmdatad[:,0].argsort()]
    itmdpt = np.unique(itmdatadsrt, axis=0)
    
    Mi = max(itmdpt[:, 1])
    temp = np.zeros((max(itmmap)+1, Mi+1))
    temp[np.arange(max(itmmap)+1), itmdpt[:,1]] = 1
    itmdatacat = temp[:,0:Mi]
    
    Pnorm = itmdatacat.astype(int).T
    Pnorm = np.delete(Pnorm, item_filt, axis = 1)
    Xnorm = np.zeros((1,I))
    Ynorm = np.zeros((1,I))
    Znorm = np.zeros((1,J))
    Du = 0
    Mu = np.array([1])
    Di = 1
    
    FeatureUser = np.append(Xnorm, Ynorm, axis = 0)
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = np.array([Mi])
    InfoData.Mu = Mu
    
    print('Decathlon dataset loaded with sparsity level: {:.3f} ({:d} interactions)'.format(1 - (len(ratedata) / (I * J)), len(ratedata)))
    
    
    return FeatureUser, FeatureItem, ratedata, InfoData

def bookprep(ImpFeed):
    
    filename = os.path.join(dirname, "Datasets/book/Book_Rating.csv")
    ratedata = np.genfromtxt(filename, delimiter=',')
    
    filename = os.path.join(dirname, "Datasets/book/Book_UserFeature.csv")
    userattr = np.genfromtxt(filename, delimiter=',')
        
    ratedata[:, [0,1]] = ratedata[:, [0,1]] + 1
    ratedata[:, 2] = ratedata[:, 2] / 2
    
    userattr_pr = np.zeros((userattr.shape[0], int(np.max(userattr[:,2]))+1))
    userattr_pr[:,0] = (userattr[:,1] - np.mean(userattr[:,1])) / np.std(userattr[:,1])
    tmp = np.zeros((userattr.shape[0], int(np.max(userattr[:,2]))+1))
    tmp[np.arange(len(userattr)), userattr[:,2].astype(int)] = 1
    userattr_pr[:,1:] = tmp[:,:-1]
    
    FeatureUser = userattr_pr.T
    I = len(userattr_pr)
    J = int(np.max(ratedata[:, 1]))
    
    Di = 1
    Mi = np.array([1])
    Du = 1
    Mu = np.array([int(np.max(userattr[:,2]))])
    
    Znorm = np.zeros((1,J))
    Pnorm = np.zeros((1,J))
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    return FeatureUser, FeatureItem, ratedata, InfoData

def lastfmprep(ImpFeed):
    
    filename = os.path.join(dirname, "Datasets/lastfm/LastFM_Rating.csv")
    ratedata = np.genfromtxt(filename, delimiter=',')
    
    filename = os.path.join(dirname, "Datasets/lastfm/LastFM_UserFeature.csv")
    userattr = np.genfromtxt(filename, delimiter=',')
        
    ratedata[:, [0,1]] = ratedata[:, [0,1]] + 1
    
    Di = 1
    Mi = np.array([1])
    Du = 1
    Mu = np.array([1, int(np.max(userattr[:,3]))])
    
    userattr_pr = np.zeros((userattr.shape[0], int(np.max(userattr[:,3]))+2))
    userattr_pr[:,0] = (userattr[:,1] - np.mean(userattr[:,1])) / np.std(userattr[:,1])
    
    tmp = np.zeros((userattr.shape[0], 2))
    tmp[np.arange(len(userattr)), userattr[:,2].astype(int)] = 1
    userattr_pr[:,1] = tmp[:,0]
    
    tmp = np.zeros((userattr.shape[0], int(np.max(userattr[:,3]))+1))
    tmp[np.arange(len(userattr)), userattr[:,3].astype(int)] = 1
    userattr_pr[:,2:] = tmp[:,:-1]
    
    FeatureUser = userattr_pr.T
    I = len(userattr_pr)
    J = int(np.max(ratedata[:, 1]))
       
    Znorm = np.zeros((1,J))
    Pnorm = np.zeros((1,J))
    FeatureItem = np.append(Znorm, Pnorm, axis = 0)
    
    
    InfoData = DataSetInfo()
    InfoData.I = I
    InfoData.J = J
    InfoData.Di = Di
    InfoData.Du = Du
    InfoData.Mi = Mi
    InfoData.Mu = Mu
    
    return FeatureUser, FeatureItem, ratedata, InfoData
    