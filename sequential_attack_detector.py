
from MF_MSI import MF_MSI
from DatasetPrep import rate_to_matrix
from DatasetPrep import split_users
import numpy as np

class sequential_attack_detector(object):
    
    def __init__(self, dataset):
        
        self.NLL_alpha_item = 0
        self.NLL_alpha_user = 0
                    
        self.stat_vec_user = []
        self.stat_vec_item = []
        
        
    def compute_baseline(self, dataset, HoldSize, target_itemno, K = 10, lmd_u = 1, lmd_v = 1, lvm_epochno = 10):
        
        self.NLL_alpha_item = []
        self.NLL_alpha_user = []
        for fold in range(0, 1):
            dataset_train, dataset_val = split_users(dataset,
                                                     HoldSize,
                                                     random = 0,
                                                     offset = fold * HoldSize)
            
            self.model = MF_MSI(dataset_train,
                                K = K,
                                lmd_u = lmd_u,
                                lmd_v = lmd_v)
        
            self.model.fit(dataset_train = dataset_train,
                           dataset_val = dataset_train,
                           dataset_test = dataset_train,
                           epochno = lvm_epochno)
        
            # Compute NLL statistic
            nLogLik_val_users, _ = self.model.loglik_analytic(dataset_val)
            
            _, Oval = rate_to_matrix(dataset_val.ratedata, dataset_val.I, dataset_val.J)
            NLL_vec_item = nLogLik_val_users[:, target_itemno - 1]
            NLL_vec_item = NLL_vec_item[np.where(Oval[:, target_itemno-1] == 1)[0]]
            NLL_vec_user =  np.sum(nLogLik_val_users, axis = 1) / np.sum(Oval, axis = 1)
                
            ind=np.argsort(NLL_vec_item)
            NLL_vec_sort = NLL_vec_item[ind]
            self.NLL_alpha_item.append(NLL_vec_sort[int(len(NLL_vec_sort) * 0.95)])
            
            ind=np.argsort(NLL_vec_user)
            NLL_vec_sort = NLL_vec_user[ind]
            self.NLL_alpha_user.append(NLL_vec_sort[int(len(NLL_vec_sort) * 0.95)])
         
        
    def compute_test_score(self, dataset, target_itemno):
        
        # Compute NLL statistic test users
        _, O = rate_to_matrix(dataset.ratedata, dataset.I, dataset.J)
        nLogLik_users_new, _ = self.model.loglik_analytic(dataset)
        NLL_vec_item = nLogLik_users_new[:, target_itemno - 1]
        NLL_vec_user =  np.sum(nLogLik_users_new, axis = 1) / np.sum(O, axis = 1)
        self.stat_vec_item = NLL_vec_item - np.mean(self.NLL_alpha_item)
        self.stat_vec_user = NLL_vec_user - np.mean(self.NLL_alpha_user)
        
        g_t = 0  
        self.g_t_vec = np.zeros(len(self.stat_vec_user))
        for i in range(0, len(self.stat_vec_user)):
            g_t = max(0, g_t + self.stat_vec_user[i])
            self.g_t_vec[i] = g_t
        

        
        
        
        