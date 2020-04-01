import numpy as np
from numpy.linalg import inv
from utils import softmax
from utils import logsumexp
from utils import logdet
from eval_model import eval_res
from DatasetPrep import rate_to_matrix

class MF_MSI(object):
    
    def __init__(self, infodata, K = 10, lmd_u = 1, lmd_v = 1):
        
        self.infodata = infodata
        self.ImpFeed_a = 0
        self.ImpFeed_b = 1
        self.logeval = 0
        
        self.modelparams_u = self.model_params(infodata.Du, infodata.Mu, K, lmd_u, infodata.I)
        self.modelparams_v = self.model_params(infodata.Di, infodata.Mi, K, lmd_v, infodata.J)
        self.latentparams_u = self.latent_params(self.modelparams_u)
        self.latentparams_v = self.latent_params(self.modelparams_v)
    
    class model_params:
        def __init__(self, D, M, K, lmd, I):
            
            self.D = D
            self.M = M
            self.I = I
            self.K = K
            
            # Initialize global params
            self.W = np.random.multivariate_normal(np.zeros(K), np.identity(K), D)
            self.H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(M))
            self.muR = np.zeros(I)
            self.Sigma_x = np.identity(D)
            self.Prec_x = np.identity(D)
            
            # Fixed model priors
            self.U_mean_prior = np.random.normal(0, 1, K)
            self.Prec_u_prior = lmd * np.identity(K)
            self.c = 1
            self.nu0 = -(self.K+1)
            self.lmd = lmd
            self.alpha = 0.00
            self.alpha_mu = 0
            self.param_a = 3
            self.param_b = 0.5
            self.s0 = 0 * np.eye(self.K)
            
            # Enable/disable data sources
            self.Xon = 1
            self.Yon = 1
            self.Ron = 1
            
            
            self.LogLikOn = 0
            self.LogLik = 0
            self.LogRating = 0
            
            # Fixed model params
            self.F_u = []
            for i in range(0,M.shape[0]):
                self.F_u.append(1/2 * (np.identity(M[i]) - (1/(M[i]+1)) * np.ones((M[i],1)) * np.ones((M[i],1)).T))
        
    class latent_params:
        def __init__(self, modelparams):
            
            # Initialize local params and sufficient statistics
            self.SS_SecMoment = 0
            self.SS_Mean = 0
            self.U_mean = np.tile(modelparams.U_mean_prior[:,None], [1,modelparams.I])
            self.Psi_u = modelparams.H @ self.U_mean
            self.U_SecMoment = np.tile(modelparams.Prec_u_prior + modelparams.U_mean_prior[:,None].T @ modelparams.U_mean_prior[:,None], [modelparams.I, 1, 1])
            self.Psd_X = np.zeros((np.sum(modelparams.M) + modelparams.D, modelparams.I))
            self.Sigma_u = np.zeros((modelparams.I, modelparams.K, modelparams.K))

            
    def e_step(self, modelparams, latentparams, X, Y, R, O, MeanVecCouple, SecMomCouple, muR):
          
        
        M = modelparams.M
        I = modelparams.I
        D = modelparams.D
        K = modelparams.K
        
        # Fetch model parameters
        c = modelparams.c
        H = modelparams.H
        W = modelparams.W
        Prec_x = modelparams.Prec_x
        Sigma_x = modelparams.Sigma_x
        U_mean_prior = modelparams.U_mean_prior
        F_u = modelparams.F_u
        
        
        Xon = modelparams.Xon
        Yon = modelparams.Yon
        Ron = modelparams.Ron
        
        
        # Infer posterior covariance and precision
        Sigma_u = np.zeros((I, K, K))
        InfPrec = modelparams.Prec_u_prior.copy()
        if Xon == 1:
            InfPrec = InfPrec + (W.T @ Prec_x @ W)
        if Yon == 1:
            InfCat = 0
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                InfCat = InfCat +  H[ind,:].T @ F_u[i] @ H[ind]
            InfPrec = InfPrec + InfCat 
        Sigma_u = inv(Ron * np.tensordot(c * O, SecMomCouple, axes = 1) + InfPrec)
        
        # Infer posterior mean
        if Yon == 1:
            iter_psi = 5 # Iterations of bound convergence for categorical info
        else:
            iter_psi = 1 # No bound iteration in the absence of categorical info
        
        for iterPsi in range(0,iter_psi):
            G_u = np.zeros((np.sum(M), I))
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                Psi_u_d = softmax(latentparams.Psi_u[ind, :])
                G_u[ind,:] = F_u[i] @ latentparams.Psi_u[ind, :] - Psi_u_d[0:-1, :]

            SS_SecMoment = np.zeros((K, K))
            InfSum = np.zeros((K, I))
            InfSum = InfSum + (modelparams.Prec_u_prior @ U_mean_prior)[None].T
            if Xon == 1:
                InfSum = InfSum + (W.T @ (X / np.diag(Sigma_x)[None].T))
            if Yon == 1:
                InfSum = InfSum + (H.T @ (Y + G_u))
           
            U_mean = np.einsum('ijk,ik->ij', Sigma_u, (InfSum + Ron * c * (MeanVecCouple @ np.multiply(R - muR, O).T)).T).T
            U_SecMoment = Sigma_u + np.einsum('ijk,ikl->ijl', U_mean[None].T, np.reshape(U_mean.T, (I, 1, K)))
            SS_SecMoment = np.sum(U_SecMoment, axis = 0)
        
            Psi_u_old = latentparams.Psi_u.copy()
            Psi_u = H @ U_mean
            latentparams.Psi_u = Psi_u
            conv = np.sum((Psi_u_old - Psi_u)**2) / (Psi_u.shape[0] * Psi_u.shape[1])
            #print(conv)
            if conv < 1e-5:
                #print("Converged")
                break;
        
        
        # Fuse multimodal observations
        Psd_Cov = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X

        Psd_Cov[0:D,0:D] =  Sigma_x
        Psd_Prec[0:D,0:D] = Prec_x
        for i in range(0,M.shape[0]):
            ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
            Psi_u_d = softmax(Psi_u[ind, :])
            G_u[ind,:] = F_u[i] @ Psi_u[ind, :] - Psi_u_d[0:-1, :]
            ind_tilde = range(np.sum(M[0:i]) + D, np.sum(M[0:i+1]) + D)
            Y_tilde = inv(F_u[i]) @ (Y[ind, :] + G_u[ind,:]) 
            Psd_X[ind_tilde, :] = Y_tilde
            Psd_Cov[np.ix_((ind_tilde),(ind_tilde))] = inv(F_u[i])
            Psd_Prec[np.ix_((ind_tilde),(ind_tilde))] = F_u[i]
            
        latentparams.SS_Mean = Psd_X @ U_mean.T
        latentparams.Psi_u = Psi_u
        latentparams.SS_SecMoment =  SS_SecMoment
        latentparams.U_SecMoment = U_SecMoment
        latentparams.U_mean = U_mean
        latentparams.Psd_X = Psd_X
        latentparams.Sigma_u = Sigma_u

    
    def m_step(self, modelparams, latentparams, X, Y):
        
            D = modelparams.D
            
            YY = np.sum(X * X, axis = 1)
            
            # Estimate global model parameters
            U_mean_prior = np.mean(latentparams.U_mean,axis = 1)
            W = latentparams.SS_Mean[0:D, :] @ (inv(latentparams.SS_SecMoment) + modelparams.alpha * np.eye(modelparams.K))
            Sigma_x = np.diag((2*modelparams.param_b + YY - 2 * np.diag(W @ latentparams.SS_Mean[0:D, :].T) + np.diag(W @ latentparams.SS_SecMoment @ W.T) + np.sum((W ** 2) * modelparams.alpha , axis = 1)) / (modelparams.I + 2*(modelparams.param_a+1)))
            Prec_x = np.diag(1/np.diag(Sigma_x))
            H = latentparams.SS_Mean[D:, :] @ (inv(latentparams.SS_SecMoment))
            
            modelparams.U_mean_prior = U_mean_prior
            modelparams.Prec_x = Prec_x
            modelparams.Sigma_x = Sigma_x
            modelparams.W = W
            modelparams.H = H
            
    def loglik(self, modelparams, latentparams, X, Y, R, O, MeanVecCouple):
        
        M = modelparams.M
        I = modelparams.I
        D = modelparams.D
        K = modelparams.K
        
        W = modelparams.W
        H = modelparams.H
        U_mean = latentparams.U_mean
        Sigma_u = latentparams.Sigma_u
        U_mean_prior = modelparams.U_mean_prior
        c = modelparams.c
        Psi_u = latentparams.Psi_u
        F_u = modelparams.F_u
        
        Xon = modelparams.Xon
        Yon = modelparams.Yon
        
        LogMult = 0
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X
        Psd_Prec[0:D,0:D] = modelparams.Prec_x
        for i in range(0,M.shape[0]):
            ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
            Psi_u_d = softmax(Psi_u[ind, :])
            G_u = F_u[i] @ Psi_u[ind, :] - Psi_u_d[0:-1, :]
            ind_tilde = range(np.sum(M[0:i]) + D, np.sum(M[0:i+1]) + D)
            Y_tilde = inv(F_u[i]) @ (Y[ind, :] + G_u) 
            Psd_X[ind_tilde, :] = Y_tilde
            Psd_Prec[np.ix_((ind_tilde),(ind_tilde))] = F_u[i]
            LogMult_c = 0.5 * np.sum(Psi_u[ind, :] * (F_u[i] @ Psi_u[ind, :]), axis = 0) - np.sum(Psi_u_d[0:-1, :] * Psi_u[ind, :], axis = 0) + logsumexp(Psi_u[ind, :])
            LogMult = LogMult + 0.5 * np.log(2*np.pi) * M[i] + 0.5 * logdet(inv(F_u[i])) + 0.5 * np.sum(Y_tilde * (F_u[i] @ Y_tilde), axis = 0) - LogMult_c
        
        LogMult = Yon * np.sum(LogMult)
        
        if Xon == 1 or Yon == 1:
            if Xon == 1 and Yon == 1:
                Psd_Beta = np.append(W,H,axis=0)
            elif Xon == 1:
                Psd_Beta = W.copy()
            elif Yon == 1:
                Psd_Beta = H.copy()
            Psd_Mean =  Psd_Beta @ U_mean
            LogLink =  np.sum(0.5 * (logdet(Psd_Prec) - (np.sum(M) + D) * np.log(2*np.pi)) - 0.5 * np.sum((Psd_X - Psd_Mean) * (Psd_Prec @ (Psd_X - Psd_Mean)), axis = 0))
            for i in range(0, I):
                LogLink = LogLink - 0.5 * np.trace( Psd_Prec @ Psd_Beta @ Sigma_u[i] @ Psd_Beta.T )
        else:
            LogLink = 0
    
        Entropy = 0
        LogLatent = np.sum(0.5 * (logdet(modelparams.Prec_u_prior) - (K) * np.log(2*np.pi)) - 0.5 * np.sum((U_mean_prior[None].T - U_mean) * (latentparams.Prec_u_prior @ (U_mean_prior[None].T - U_mean)), axis = 0))
        for i in range(0, I):
            LogLatent = LogLatent - 0.5 * np.trace( modelparams.Prec_u_prior @ Sigma_u[i])
            Entropy =Entropy + 0.5 * (np.log(2*np.pi) * K + logdet(Sigma_u[i]))
        
        
        LogPrior = -(modelparams.param_a + 1) * np.sum(np.log(np.diag(modelparams.Sigma_x))) - modelparams.param_b * np.sum(np.diag(modelparams.Prec_x))
        LogPrior = Xon * LogPrior + 0.5 * (modelparams.nu0 + K + 1) * logdet(modelparams.Prec_u_prior) - 0.5 * np.trace(modelparams.s0 @ latentparams.Prec_u_prior)
        
        LogLik = (LogLink + LogMult + LogLatent + Entropy + LogPrior) / I
        
        LogRating = (0.5 * np.sum( O * (np.log(c) - c * ((U_mean.T @ MeanVecCouple - R)**2) - np.log(2*np.pi)))) / np.sum(O)
    
        return LogLik, LogRating
            
    def fit(self, FeatureUser, FeatureItem, ratedata_train, ratedata_test, ratedata_val, epochno = 10):
        
        Rtrain, Otrain = rate_to_matrix(ratedata_train, self.infodata.I, self.infodata.J)
        Rtest, Otest = rate_to_matrix(ratedata_test, self.infodata.I, self.infodata.J)
        Rval, Oval = rate_to_matrix(ratedata_val, self.infodata.I, self.infodata.J)
        
        X = FeatureUser[0:self.infodata.Du, :]
        Y = FeatureUser[self.infodata.Du:, :]
        Z = FeatureItem[0:self.infodata.Di, :]
        P = FeatureItem[self.infodata.Di:, :]
        
        for epoch in range(0,epochno):
            
            self.e_step(self.modelparams_u,
                        self.latentparams_u, 
                        X, Y, Rtrain, Otrain,  
                        self.latentparams_v.U_mean,
                        self.latentparams_v.U_SecMoment, 
                        self.modelparams_v.muR)
        
            self.e_step(self.modelparams_v,
                        self.latentparams_v,
                        Z, P, Rtrain.T, Otrain.T,
                        self.latentparams_u.U_mean,
                        self.latentparams_u.U_SecMoment,
                        self.modelparams_u.muR)

            self.m_step(self.modelparams_u,
                        self.latentparams_u,
                        X, Y)
            self.m_step(self.modelparams_v,
                        self.latentparams_v,
                        Z, P)
            
            c_pred = (2*self.modelparams_u.param_b + np.sum(Rtrain**2  - (self.latentparams_u.U_mean.T @ self.latentparams_v.U_mean) * Rtrain)) / (np.sum(Otrain) + 2*(self.modelparams_u.param_a+1))
            self.modelparams_u.c = c_pred
            self.modelparams_v.c = c_pred
            
            LogLik = 0
            if self.modelparams_u.LogLikOn == 1:
                LogLik_u,_ = self.loglik(self.modelparams_u, self.latentparams_u, X, Y, Rtrain, Otrain, self.latentparams_v.U_mean)
                LogLik_v, LogRating = self.loglik(self.modelparams_v, self.latentparams_v, Z, P, Rtrain.T, Otrain.T, self.latentparams_u.U_mean)
                LogLik = LogLik_u + LogLik_v + LogRating

            if self.logeval == 1:
                model_eval_res = eval_res(self, ratedata_train, ratedata_test, ratedata_val, 10)
                #print("LH = {:.2f}, LH residual = {:.2f},  MSETr = {:.3f}, Recall = {:.3f}, AUC = {:.3f}, MSETe = {:.3f}, MSEVal = {:.3f}".format(LogLik, LogLik - tmp_log,  model_eval_res.Train_MSE, model_eval_res.Test_recall,model_eval_res.Test_auc, model_eval_res.Test_MSE, model_eval_res.Val_MSE))
                print("MSETr = {:.3f}, MSEVal = {:.3f}, MSETe = {:.3f}, Recall = {:.3f}, AUC = {:.3f},".format(model_eval_res.Train_MSE, model_eval_res.Val_MSE, model_eval_res.Test_MSE, model_eval_res.Test_recall,model_eval_res.Test_auc))
                #print("Rating Err = {:.3f}, Side Cont Err= {:.3f}, Side Disc Err= {:.3f}, MSEVal= {:.3f}, MSETest= {:.3f}".format( model_eval_res.Train_MSE, mseX, crossY, model_eval_res.Val_MSE, model_eval_res.Test_MSE))
                #print("LH = {:.2f}, LH residual = {:.2f},  MSE = {:.3f}, Recall = {:.3f}, AUC = {:.3f}, ILD = {:.3f}".format(LogLik, LogLik - tmp_log,  model_eval_res.Train_MSE, model_eval_res.Train_recall, model_eval_res.Test_auc, model_eval_res.Test_recall))
            
    def infer_user_posterior(self, FeatureUser, R, O):
        
        X = FeatureUser[0:self.infodata.Du, :]
        Y = FeatureUser[self.infodata.Du:, :]
        self.modelparams_u.I = R.shape[0]
        
        latentparams_new = self.latent_params(self.modelparams_u)
        self.e_step(self.modelparams_u,
                    latentparams_new,
                    X, Y, R, np.zeros(O.shape),
                    self.latentparams_v.U_mean,
                    self.latentparams_v.U_SecMoment,
                    0)
        
        mean = latentparams_new.U_mean
        cov = latentparams_new.Sigma_u
        
        return mean, cov
        