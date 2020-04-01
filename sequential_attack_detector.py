
from MF_MSI import MF_MSI
from DatasetPrep import rate_to_matrix
from utils_attack import split_users
import numpy as np
import copy

class sequential_attack_detector(object):
    
    def __init__(self, InfoData):
        
        self.NLL_alpha_item = 0
        self.NLL_alpha_user = 0
                    
        self.stat_vec_user = []
        self.stat_vec_item = []
        
        self.InfoData = copy.deepcopy(InfoData)
        
    def compute_baseline(self, FeatureUser, FeatureItem, ratedata, HoldSize, target_itemno, K = 10, lmd_u = 1, lmd_v = 1, lvm_epochno = 10):
        
        # Rating normalization
        self.mean_r = np.mean(ratedata[:,2])
        self.std_r = np.std(ratedata[:,2])
        ratedata_norm = np.copy(ratedata)
        ratedata_norm[:, 2] = (ratedata[:,2] - self.mean_r) / self.std_r
        
        
        
        ratedata_train, FeatureUserTrain, ratedata_val, FeatureUserVal = split_users(ratedata_norm,
                                                                                           self.InfoData.I,
                                                                                           self.InfoData.J,
                                                                                           FeatureUser,
                                                                                           HoldSize,
                                                                                           random = 0,
                                                                                           offset = 0)
        
        self.InfoData.I = self.InfoData.I - HoldSize
        
        self.model = MF_MSI(self.InfoData,
                            K = K,
                            lmd_u = lmd_u,
                            lmd_v = lmd_v)
        
        self.model.fit(FeatureUserTrain,
                      FeatureItem,
                      ratedata_train,
                      ratedata_train,
                      ratedata_train,
                      epochno = lvm_epochno)
        
        
        Rval, Oval = rate_to_matrix(ratedata_val, HoldSize, self.InfoData.J)
        mean_val_user, sigma_val_user = self.model.infer_user_posterior(FeatureUserVal, Rval, Oval)
        
        # Compute NLL statistic
        nLogLik_val_users, _ = self.loglik_analytic(Rval, Oval, mean_val_user, sigma_val_user)
        
        NLL_vec_item = nLogLik_val_users[:, target_itemno - 1]
        NLL_vec_item = NLL_vec_item[np.where(Oval[:, target_itemno-1] == 1)[0]]
        NLL_vec_user =  np.sum(nLogLik_val_users, axis = 1) / np.sum(Oval, axis = 1)
            
        ind=np.argsort(NLL_vec_item)
        NLL_vec_sort = NLL_vec_item[ind]
        self.NLL_alpha_item = NLL_vec_sort[int(len(NLL_vec_sort) * 0.95)]
        
        ind=np.argsort(NLL_vec_user)
        NLL_vec_sort = NLL_vec_user[ind]
        self.NLL_alpha_user =  NLL_vec_sort[int(len(NLL_vec_sort) * 0.95)]
        
    def compute_test_score(self, FeatureUserTest, R, O, target_itemno):
        
        R = (R - self.mean_r) / self.std_r
        
        mean_test_user, sigma_test_user = self.model.infer_user_posterior(FeatureUserTest, R, O)

        # Compute NLL statistic test users
        nLogLik_users_new, _ = self.loglik_analytic(R, O, mean_test_user, sigma_test_user)
        NLL_vec_item = nLogLik_users_new[:, target_itemno - 1]
        NLL_vec_user =  np.sum(nLogLik_users_new, axis = 1) / np.sum(O, axis = 1)
        self.stat_vec_item = NLL_vec_item - self.NLL_alpha_item
        self.stat_vec_user = NLL_vec_user - self.NLL_alpha_user
        
        g_t = 0  
        self.g_t_vec = np.zeros(len(self.stat_vec_user))
        for i in range(0, len(self.stat_vec_user)):
            g_t = max(0, g_t + self.stat_vec_user[i])
            self.g_t_vec[i] = g_t
        
    def loglik_analytic(self, R, O, mean_new_user, sigma_new_user):
    
        mean_item = self.model.latentparams_v.U_mean
        sigma_item = self.model.latentparams_v.Sigma_u
        
        sec_mom = (np.einsum('ijj->ij', sigma_new_user) + (mean_new_user.T)**2) @ (np.einsum('ijj->ij', sigma_item) + (mean_item.T)**2).T
        Sigma = sec_mom - ((mean_new_user.T)**2) @ (mean_item**2)
        
        nloglik_user = Sigma + (mean_new_user.T @ mean_item)**2 - 2*(mean_new_user.T @ mean_item)*R + R**2
        nloglik_user[np.where(O == 0)] = 0
        
        MSE = np.mean(((mean_new_user.T @ mean_item) - R)[np.where(O == 1)]**2)
                
        return nloglik_user, MSE
        
        
        
        