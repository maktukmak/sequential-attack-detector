# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:59 2018

@author: Mehmet
"""

import numpy as np
#from lightfm.datasets import fetch_stackexchange
import time
from utils import softmax
import matplotlib.pyplot as plt
import os
dirname = os.path.dirname(__file__)

class DatasetPrep(object):
    def __init__(self):
        
        self.I = 0
        self.J = 0
        self.Di = 0
        self.Du = 0
        self.Mi = 0
        self.Mu = 0
        
        self.FeatureUser = []
        self.FeatureItem = []
        self.ratedata = []
        
    def dataset_stat(self): 
        
        R, O = rate_to_matrix(self.ratedata, self.I, self.J)

        print("Rating Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(self.ratedata[:, 2].mean(),
                                                                                  self.ratedata[:, 2].std(),
                                                                                  self.ratedata[:, 2].min(),
                                                                                  self.ratedata[:, 2].max()))
        
        
        
        print("User Filler Mean: {}, Std: {}, Min: {}, Max: {}".format(int(O.sum(axis = 1).mean()),
                                                                     int(O.sum(axis = 1).std()),
                                                                     int(O.sum(axis = 1).min()),
                                                                     int(O.sum(axis = 1).max())))
        
        
        
        print("Item Filler Mean: {}, Std: {}, Min: {}, Max: {}".format(int(O.sum(axis = 0).mean()),
                                                                     int(O.sum(axis = 0).std()),
                                                                     int(O.sum(axis = 0).min()),
                                                                     int(O.sum(axis = 0).max())))
        
        plt.hist(O.sum(axis = 1), bins=100)
        plt.xlabel("Histogram of user filler numbers")
        plt.show()
        plt.hist(O.sum(axis = 0), bins=100)
        plt.xlabel("Histogram of item filler numbers")
        plt.show()
        
        
        print("User real-valued features Mean: {}, Std: {}, Min: {}, Max: {}".format(self.FeatureUser[0:self.Du].mean(axis = 1),
                                                                                                     self.FeatureUser[0:self.Du].std(axis = 1),
                                                                                                     self.FeatureUser[0:self.Du].min(axis = 1),
                                                                                                     self.FeatureUser[0:self.Du].max(axis = 1)))
        counts = self.FeatureUser[self.Du:].sum(axis = 1).astype(int)
        print("User cat-valued feature counts:")               
        ind = 0                                                                         
        for i in self.Mu:
            print(counts[ind:ind + i], self.I - counts[ind:ind + i].sum())
            ind = ind + i
            
        
        print("Item real-valued features Mean: {}, Std: {}, Min: {}, Max: {}".format(self.FeatureItem[0:self.Di].mean(axis = 1),
                                                                                                     self.FeatureItem[0:self.Di].std(axis = 1),
                                                                                                     self.FeatureItem[0:self.Di].min(axis = 1),
                                                                                                     self.FeatureItem[0:self.Di].max(axis = 1)))
        counts = self.FeatureItem[self.Di:].sum(axis = 1).astype(int)
        print("Item cat-valued feature counts:")               
        ind = 0                                                                         
        for i in self.Mi:
            print(counts[ind:ind + i], self.I - counts[ind:ind + i].sum())
            ind = ind + i
        
        
    def movie1mload(self):
        
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
            
        
        self.FeatureUser = np.append(Xorg, Yorg, axis = 0)
        self.FeatureItem = np.append(Zorg, Porg, axis = 0)
        self.ratedata = ratedata
        
        self.I = I
        self.J = J
        self.Di = Di
        self.Du = Du
        self.Mi = Mi
        self.Mu = Mu
        
        
               

def split_users(dataset, SplitSize, random, offset):
    
    '''
    Split the users into two sets and adjust rating and feature vectors accordingly
    '''
    
    ind_gen_user = np.arange(0, dataset.I)
    if random == 1:
        np.random.shuffle(ind_gen_user)
    else:
        ind_gen_user = ind_gen_user + offset
        
    R, O = rate_to_matrix(dataset.ratedata, dataset.I, dataset.J)
    R_split = R[ind_gen_user[0:SplitSize], :]
    O_split = O[ind_gen_user[0:SplitSize], :]
    R_rest = np.delete(R, ind_gen_user[0:SplitSize], axis = 0)
    O_rest = np.delete(O, ind_gen_user[0:SplitSize], axis = 0)
    
    dataset.ratedata = np.append(np.where(O_rest > 0)[0][None].T + 1, np.where(O_rest > 0)[1][None].T + 1, axis = 1)
    dataset.ratedata = np.append(dataset.ratedata, R_rest[np.where(O_rest > 0)][None].T, axis = 1)
    
    ratedata_split = np.append(np.where(O_split > 0)[0][None].T + 1, np.where(O_split > 0)[1][None].T + 1, axis = 1)
    ratedata_split = np.append(ratedata_split, R_split[np.where(O_split > 0)][None].T, axis = 1)
    
    FeatureUserSplit = dataset.FeatureUser[:, ind_gen_user[0:SplitSize]]
    dataset.FeatureUser = np.delete(dataset.FeatureUser, ind_gen_user[0:SplitSize], axis = 1)
    
    dataset.I = dataset.I - SplitSize
    
    dataset_split = DatasetPrep()
    dataset_split.I = SplitSize
    dataset_split.J = dataset.J
    dataset_split.Di = dataset.Di
    dataset_split.Du = dataset.Du
    dataset_split.Mi = dataset.Mi
    dataset_split.Mu = dataset.Mu
    
    dataset_split.FeatureUser = FeatureUserSplit
    dataset_split.FeatureItem = dataset.FeatureItem
    dataset_split.ratedata = ratedata_split
    
    
    return dataset, dataset_split



        

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


