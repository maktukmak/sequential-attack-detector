# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:55:01 2019

@author: Mehmet
"""

import numpy as np
from numpy.linalg import inv
from DatasetPrep import rate_to_matrix

class eval_cls:
    def __init__(self):
        self.Test_MSE = 0
        self.Train_MSE = 0
        self.Val_MSE = 0
        self.Test_recall = 0
        self.Train_recall = 0
        self.Test_auc = 0
        self.Train_auc = 0
        self.ild = 0
        self.coverage = 0

        
def auc_eval(Rpred, Rtest, Ctrain):
    auc_vec = np.zeros(Rpred.shape[0])
    for u in range(0, Rpred.shape[0]):
        auc = 0
        usr_test = np.where(Rtest[u, :] > 0)[0]
        if len(usr_test) > 0:
            usr_train = np.where(Ctrain[u, :] == 1)[0]
            usr_set = np.union1d(usr_train, usr_test)
            e_set = np.arange(0, Rpred.shape[1])
            e_set = np.delete(e_set, usr_set)
            for i in usr_test:
                auc = auc + len(np.where(Rpred[u, i] > Rpred[u, e_set])[0])
#                for j in e_set:
#                    if Rpred[u, i] > Rpred[u, j]:
#                        auc = auc + 1
            auc_vec[u] = auc / (len(e_set) * len(usr_test))
        else:
            auc_vec[u] = np.nan
    Test_auc = np.nanmean(auc_vec)
    return Test_auc

def auc_stat(Rpred, Rtest):
    auc_vec = np.zeros(Rpred.shape)
    O_auc = np.zeros(Rpred.shape)
    for u in range(0, Rpred.shape[0]):
        usr_test = np.where(Rtest[u, :] > 0)[0]
        if len(usr_test) > 0:
            e_set = np.where(Rtest[u, :] < 0)[0]
            if len(e_set) > 0:
                for i in usr_test:
                    auc_vec[u, i] = len(np.where(Rpred[u, i] > Rpred[u, e_set])[0]) / (len(e_set))
                    O_auc[u, i] = 1
    return auc_vec, O_auc

def ild_eval(Rpred, Rtest, Ctest, at_K, Mean_vec, Sigma_vec):
    ild_sum = 0
    for i in range(0, Rpred.shape[0]):
        pred_like = np.flip(np.argsort(Rpred[i,:]), axis = 0)[0:at_K]
        dist_sum = 0
        for j in range(0, at_K-1):
            for k in range(j+1, at_K):
                dist = np.trace(inv(Sigma_vec[pred_like[k]]) @ Sigma_vec[pred_like[j]]) + ((Mean_vec[:,pred_like[k]] - Mean_vec[:,pred_like[j]])[None] @ Sigma_vec[pred_like[k]] @ (Mean_vec[:,pred_like[k]] - Mean_vec[:,pred_like[j]])[None].T)[0][0] - Mean_vec[:,pred_like[j]].shape[0] + np.log(np.linalg.det(Sigma_vec[pred_like[k]]) / np.linalg.det(Sigma_vec[pred_like[j]]) )
                dist_sum = dist_sum + dist
        ild_u = dist_sum / (at_K * (at_K-1))
        ild_sum = ild_sum + ild_u
    ild = ild_sum / Rpred.shape[0]
    return ild
        
    

def recall_eval(Rpred, Rtest, Ctest, at_K):
    Test_recall = 0
    cnt = 0
    recall_vec = np.zeros(Rpred.shape[0])
    coverage = []
    for i in range(0, Rpred.shape[0]):
        usr_like = np.where(Rtest[i, :] > 0)[0]
        if len(usr_like) > 0:
            ind = np.where(Ctest[i, :] == 1)[0]
            #pred_like = np.flip(np.argsort(Rpred[i,:]), axis = 0)[0:at_K]
            pred_like = ind[np.flip(np.argsort(Rpred[i,ind]), axis = 0)][0:at_K]
            #recall = len(np.intersect1d(pred_like, usr_like)) / len(usr_like)
            recall = len(np.intersect1d(pred_like, usr_like)) / min(at_K, len(usr_like))
            Test_recall = Test_recall + recall
            cnt = cnt + 1
            recall_vec[i] = recall
            coverage = np.union1d(coverage, pred_like)
        else:
            recall_vec[i] = np.nan
    Test_recall = Test_recall / cnt
    coverage = len(coverage) / Rpred.shape[1]
    return Test_recall, recall_vec, coverage

def eval_res(model, ratedata_train, ratedata_test, ratedata_val, at_K):
    
    Rpred = model.latentparams_u.U_mean.T @ model.latentparams_v.U_mean
    
    model_eval_res = eval_cls()
    
    Rtrain, Ctrain = rate_to_matrix(ratedata_train, model.infodata.I, model.infodata.J)
    Rtest, Ctest = rate_to_matrix(ratedata_test, model.infodata.I, model.infodata.J)
    Rval, Cval = rate_to_matrix(ratedata_val, model.infodata.I, model.infodata.J)
    
    model_eval_res.Test_recall, _, model_eval_res.coverage = recall_eval(Rpred, Rtest, Ctest, at_K)
    model_eval_res.Test_auc = auc_eval(Rpred, Rtest, Ctrain)
    
    #Rpred[np.where(Ctest == 0)] = 0

    model_eval_res.Test_MSE = (np.mean(np.square(Rpred - Rtest)[np.where(Ctest == 1)]))
    model_eval_res.Train_MSE = (np.mean(np.square(Rpred - Rtrain)[np.where(Ctrain == 1)]))
    model_eval_res.Val_MSE = (np.mean(np.square(Rpred - Rval)[np.where(Cval == 1)]))
    if model.ImpFeed == 1:
        
        #model_eval_res.Train_recall,_ = recall_eval(Rpred, Rtrain, Ctrain, at_K)
        model_eval_res.Test_auc = auc_eval(Rpred, Rtest, Ctrain)
        #model_eval_res.Train_auc = auc_eval(Rpred, Rtrain, Rtest)
        #model_eval_res.ild = ild_eval(Rpred, Rtest, Ctest, at_K, Mean_vec, Sigma_vec)
    
    return model_eval_res
    