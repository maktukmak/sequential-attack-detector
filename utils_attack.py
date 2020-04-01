import numpy as np
from DatasetPrep import rate_to_matrix
import os
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import interp
dirname = os.path.dirname(__file__)


def attack_gen(InfoData, FeatureUser, ratedata, target_itemno = 1, user_start_id = 1, no_of_newratings = 10, attack_size = 20, attack_type = 0, pop_frac = 1.0, rand_rating = 0, prob = [0,0,0,0,1]):

    glob_mean = np.mean(ratedata[:, 2])
    glob_std = np.std(ratedata[:, 2])
    item_mean = np.zeros(InfoData.J)
    item_std = np.zeros(InfoData.J)

    R, O = rate_to_matrix(ratedata, InfoData.I, InfoData.J)
    glob_mean = np.ma.array(R, mask = 1-O).mean()
    glob_std = np.ma.array(R, mask = 1-O).std()
    
    item_mean = np.array(np.ma.array(R, mask = 1-O).mean(axis = 0))
    item_std = np.array(np.ma.array(R, mask = 1-O).std(axis = 0))
    itemdatarate = np.sum(O, axis = 0)
    
    attack_type_int = np.copy(attack_type)
    
    ratedata_attack = np.empty((0, 3))
    FeatureUserAttack = np.empty((int(InfoData.Du + np.sum(InfoData.Mu)), 0))
    for i in range(0, attack_size):
        ratedata_new = np.zeros((no_of_newratings, 3))
        ratedata_new[:, 0] = user_start_id + i
        ratedata_new[0, 1] = target_itemno
        
        if attack_type == 0:    # Mixed push
            attack_set = [2, 3, 5]
            attack_type_int = random.choice(attack_set)
        
        if attack_type == 1:    # Mixed nuke
            attack_set = [6, 8]
            attack_type_int = random.choice(attack_set)
            
        if attack_type == 9:    # Mixed obfuscation
            attack_set = [4, 7]
            attack_type_int = random.choice(attack_set)
        
        if attack_type_int == 2:    # Random Attack
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.where(rated_items == target_itemno-1))
            np.random.shuffle(rated_items)
            ratedata_new[1:, 1] = rated_items[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = np.maximum(np.minimum(np.round(np.random.normal(glob_mean, glob_std, no_of_newratings-1)), 5), 0)
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == 3:  # Average Attack
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.where(rated_items == target_itemno-1))
            np.random.shuffle(rated_items)
            ratedata_new[1:, 1] = rated_items[0:no_of_newratings-1] + 1
            for j in range(1, len(ratedata_new)):
                mean = item_mean[int(ratedata_new[j, 1]) - 1]
                std = item_std[int(ratedata_new[j, 1]) - 1]
                ratedata_new[j, 2] = np.maximum(np.minimum(np.round(np.random.normal(mean, std, 1)), 5), 0)
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == 4:  # AOP attack %20
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            pop_item = pop_list[0: int(InfoData.J * pop_frac)]
            pop_item = np.delete(pop_item, np.where(pop_item == target_itemno-1))
            np.random.shuffle(pop_item)
            ratedata_new[1:, 1] = pop_item[0:no_of_newratings-1] + 1
            for j in range(1, len(ratedata_new)):
                mean = item_mean[int(ratedata_new[j, 1]) - 1]
                std = item_std[int(ratedata_new[j, 1]) - 1]
                ratedata_new[j, 2] = np.maximum(np.minimum(np.round(np.random.normal(mean, std, 1)), 5), 0)
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == 5:  # Bandwagon attack
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            band_item_no = 5
            band_item = pop_list[np.argsort(item_mean[pop_list[0:25]])[-band_item_no:]]
            band_item = np.delete(band_item, np.argwhere(band_item == target_itemno - 1))
            band_item_no = len(band_item)

            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.append(np.where(rated_items == target_itemno-1)[0], np.argwhere(np.in1d(rated_items, band_item)==True)))
            np.random.shuffle(rated_items)
            ratedata_new[band_item_no+1:, 1] = rated_items[0:no_of_newratings-band_item_no-1] + 1
            ratedata_new[band_item_no+1:, 2] = np.maximum(np.minimum(np.round(np.random.normal(glob_mean, glob_std, no_of_newratings-band_item_no-1)), 5), 0)
            ratedata_new[1:band_item_no+1, 1] = band_item + 1
            ratedata_new[1:band_item_no+1, 2] = 5
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
        
        elif attack_type_int == 6:  # Reverse Bandwagon attack
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            band_item_no = 5
            band_item = pop_list[np.argsort(item_mean[pop_list[0:25]])[0:band_item_no]]
            band_item = np.delete(band_item, np.argwhere(band_item == target_itemno - 1))
            band_item_no = len(band_item)
            
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.append(np.where(rated_items == target_itemno-1)[0], np.argwhere(np.in1d(rated_items, band_item)==True)))
            np.random.shuffle(rated_items)
            ratedata_new[band_item_no+1:, 1] = rated_items[0:no_of_newratings-band_item_no-1] + 1
            ratedata_new[band_item_no+1:, 2] = np.maximum(np.minimum(np.round(np.random.normal(glob_mean, glob_std, no_of_newratings-band_item_no-1)), 5), 0)
            ratedata_new[1:band_item_no+1, 1] = band_item + 1
            if rand_rating == 1:
                ratedata_new[1:band_item_no+1, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[1:band_item_no+1, 2] = 1
                ratedata_new[0, 2] = 1
            
        elif attack_type_int == 7:  # Popular attack
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            pop_item = pop_list[0: int(InfoData.J * pop_frac)]
            pop_item = np.delete(pop_item, np.where(pop_item == target_itemno-1))
            np.random.shuffle(pop_item)
            ratedata_new[1:, 1] = pop_item[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = 1
            ratedata_new[np.where(item_mean[(ratedata_new[1:, 1].astype(int))-1] < glob_mean)[0] + 1, 2] = 2
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == 8:    # Love Hate Attack
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.where(rated_items == target_itemno-1))
            np.random.shuffle(rated_items)
            ratedata_new[1:, 1] = rated_items[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = 5
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 1
        
        X = FeatureUser[0:InfoData.Du, np.random.randint(0, InfoData.I)][None].T
        Y = np.empty((0,1))
        ind = InfoData.Du
        for m in InfoData.Mu:
            Y = np.append(Y, FeatureUser[ind:ind+m, np.random.randint(0, InfoData.I)][None].T, axis = 0)
            ind = ind + m
        
        ratedata_attack = np.append(ratedata_attack, ratedata_new, axis = 0)
        FeatureUserAttack = np.append(FeatureUserAttack, np.append(X, Y)[None].T, axis = 1)
   
    return ratedata_attack, FeatureUserAttack

def statistic_attack(ratedata_train, Rseq, Oseq, GeniuneSize, MixSize, target_itemno):
    ratings_train = ratedata_train[np.where(ratedata_train[:, 1] ==  target_itemno)[0], 2]
    train_dist = (np.histogram(ratings_train, bins = 5, range = (1,5))[0] + 1) / (len(ratings_train) + 5)
    train_mean = np.mean(ratings_train)
    print("Train attack ratings:", len(ratings_train))
    print("Train attack dist:", np.array_str(train_dist, precision =2))
    print("Train attack mean:", (train_mean))
    
    
    ind = np.where(Oseq[:, target_itemno-1] == 1)[0]
    ratings_pre = np.delete(Rseq[ind, target_itemno-1], np.where(ind >= GeniuneSize - MixSize))
    pre_dist = (np.histogram(ratings_pre, bins = 5, range = (1,5))[0] + 1) / (len(ratings_pre) + 5)
    pre_mean = np.mean(ratings_pre)
    print("Pre attack ratings:", len(ratings_pre))
    print("Pre attack dist:", np.array_str(pre_dist, precision =2))
    print("Pre attack mean:", pre_mean)
    
    
    ind = np.where(Oseq[:, target_itemno-1] == 1)[0]
    ratings_post = np.delete(Rseq[ind, target_itemno-1], np.where(ind < GeniuneSize - MixSize))
    post_dist = (np.histogram(ratings_post, bins = 5, range = (1,5))[0] + 1) / (len(ratings_post) + 5)
    post_mean = np.mean(ratings_post)
    print("Post attack ratings:", len(ratings_post))
    print("Post attack dist:", np.array_str(post_dist, precision =2))
    print("Post attack mean:", post_mean)
    


def split_users(ratedata, I, J, FeatureUser, SplitSize, random, offset):
    
    '''
    Split the users into two sets and adjust rating and feeature vectors accordingly
    '''
    
    ind_gen_user = np.arange(0, I)
    if random == 1:
        np.random.shuffle(ind_gen_user)
    else:
        ind_gen_user = ind_gen_user + offset
        
    R, O = rate_to_matrix(ratedata, I, J)
    R_split = R[ind_gen_user[0:SplitSize], :]
    O_split = O[ind_gen_user[0:SplitSize], :]
    R_rest = np.delete(R, ind_gen_user[0:SplitSize], axis = 0)
    O_rest = np.delete(O, ind_gen_user[0:SplitSize], axis = 0)
    
    ratedata_rest = np.append(np.where(O_rest > 0)[0][None].T + 1, np.where(O_rest > 0)[1][None].T + 1, axis = 1)
    ratedata_rest = np.append(ratedata_rest, R_rest[np.where(O_rest > 0)][None].T, axis = 1)
    
    ratedata_split = np.append(np.where(O_split > 0)[0][None].T + 1, np.where(O_split > 0)[1][None].T + 1, axis = 1)
    ratedata_split = np.append(ratedata_split, R_split[np.where(O_split > 0)][None].T, axis = 1)
    
    FeatureUserSplit = FeatureUser[:, ind_gen_user[0:SplitSize]]
    FeatureUserRest = np.delete(FeatureUser, ind_gen_user[0:SplitSize], axis = 1)
    
    return ratedata_rest, FeatureUserRest, ratedata_split, FeatureUserSplit
    
def generate_mix_attack(ratedata_attack, ratedata_geniune, FeatureUserAttack, FeatureUserHold, J, GeniuneSize, AttackSize, MixSize):
    
    R_at,O_at = rate_to_matrix(ratedata_attack, AttackSize, J)
    R_gen,O_gen = rate_to_matrix(ratedata_geniune, GeniuneSize, J)
    
    mix_ind = np.arange(0, GeniuneSize)
    np.random.shuffle(mix_ind)
    mix_ind = mix_ind[0:MixSize]
    
    mix_ind2 = np.arange(0, AttackSize)
    np.random.shuffle(mix_ind2)
    mix_ind2 = mix_ind2[0:MixSize]
    
    new_ind = np.arange(0, AttackSize + GeniuneSize)
    new_ind[mix_ind2 + GeniuneSize] = mix_ind
    new_ind = np.append(new_ind, mix_ind2 + GeniuneSize)
    new_ind = np.delete(new_ind, mix_ind)
    
    np.random.shuffle(new_ind[0:GeniuneSize-MixSize])
    
    feature_new = np.append(FeatureUserHold, FeatureUserAttack, axis = 1)
    feature_new = feature_new[:, new_ind]
    
    R = np.append(R_gen, R_at, axis = 0)
    R = R[new_ind, :]
    O = np.append(O_gen, O_at, axis = 0)
    O = O[new_ind, :]
    
    ground_truth_vec = np.append(np.zeros(GeniuneSize-MixSize), np.ones(AttackSize+MixSize))
    ground_truth_vec[mix_ind2 + GeniuneSize-MixSize] = 0
    
    return R, O, ground_truth_vec, feature_new
    

def plot_roc(ground_truth_vec, stat_vec):
    
    base_fpr = np.linspace(0, 1, 101)
    fpr, tpr, _ = roc_curve(ground_truth_vec, stat_vec)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(base_fpr, np.array(tpr), color = 'g', linestyle = '-', label = 'Proposed-User')
    ax.set_xlim([0.0, 0.25])
    ax.set_ylim([0.0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc = 4)
    
def plot_gt(g_t_vec):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(g_t_vec)), g_t_vec)
    ax.set_ylabel('Decision Statistic')
    ax.set_xlabel('User Index')
    ax.set_ylim([0.0, 100])

