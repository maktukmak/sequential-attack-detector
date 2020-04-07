import numpy as np
from DatasetPrep import rate_to_matrix, DatasetPrep
import copy
import os
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import interp
dirname = os.path.dirname(__file__)


def attack_gen(dataset, target_itemno = 1, user_start_id = 1, no_of_newratings = 10, attack_size = 20, attack_type = 0, pop_frac = 1.0, rand_rating = 0, prob = [0,0,0,0,1]):

    glob_mean = np.mean(dataset.ratedata[:, 2])
    glob_std = np.std(dataset.ratedata[:, 2])

    R, O = rate_to_matrix(dataset.ratedata, dataset.I, dataset.J)
    
    item_mean = np.array(np.ma.array(R, mask = 1-O).mean(axis = 0))
    item_std = np.array(np.ma.array(R, mask = 1-O).std(axis = 0))
    itemdatarate = np.sum(O, axis = 0)
    
    attack_type_int = np.copy(attack_type)
    
    ratedata_attack = np.empty((0, 3))
    FeatureUserAttack = np.empty((int(dataset.Du + np.sum(dataset.Mu)), 0))
    for i in range(0, attack_size):
        ratedata_new = np.zeros((no_of_newratings, 3))
        ratedata_new[:, 0] = user_start_id + i
        ratedata_new[0, 1] = target_itemno
        
        if attack_type == "mix_push":    # Mixed push
            attack_set = ["random", "average", "band"]
            attack_type_int = random.choice(attack_set)
        
        if attack_type == "mix_nuke":    # Mixed nuke
            attack_set = ["reverse_band", "love_hate"]
            attack_type_int = random.choice(attack_set)
            
        if attack_type == "mix_obf":    # Mixed obfuscation
            attack_set = ["aop", "popular"]
            attack_type_int = random.choice(attack_set)
        
        if attack_type_int == "random":    # Random Attack
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.where(rated_items == target_itemno-1))
            np.random.shuffle(rated_items)
            ratedata_new[1:, 1] = rated_items[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = np.maximum(np.minimum(np.round(np.random.normal(glob_mean, glob_std, no_of_newratings-1)), 5), 0)
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == "average":  # Average Attack
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
            
        elif attack_type_int == "aop":  # AOP attack %20
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            pop_item = pop_list[0: int(dataset.J * pop_frac)]
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
            
        elif attack_type_int == "band":  # Bandwagon attack
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
        
        elif attack_type_int == "reverse_band":  # Reverse Bandwagon attack
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
            
        elif attack_type_int == "popular":  # Popular attack
            pop_list = np.flip(np.argsort(itemdatarate), axis = 0)
            pop_item = pop_list[0: int(dataset.J * pop_frac)]
            pop_item = np.delete(pop_item, np.where(pop_item == target_itemno-1))
            np.random.shuffle(pop_item)
            ratedata_new[1:, 1] = pop_item[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = 1
            ratedata_new[np.where(item_mean[(ratedata_new[1:, 1].astype(int))-1] < glob_mean)[0] + 1, 2] = 2
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 5
            
        elif attack_type_int == "love_hate":    # Love Hate Attack
            rated_items = np.where(itemdatarate > 1)[0]
            rated_items = np.delete(rated_items, np.where(rated_items == target_itemno-1))
            np.random.shuffle(rated_items)
            ratedata_new[1:, 1] = rated_items[0:no_of_newratings-1] + 1
            ratedata_new[1:, 2] = 5
            if rand_rating == 1:
                ratedata_new[0, 2] = np.argmax(np.random.multinomial(1, prob)) + 1
            else:
                ratedata_new[0, 2] = 1
        
        X = dataset.FeatureUser[0:dataset.Du, np.random.randint(0, dataset.I)][None].T
        Y = np.empty((0,1))
        ind = dataset.Du
        for m in dataset.Mu:
            Y = np.append(Y, dataset.FeatureUser[ind:ind+m, np.random.randint(0, dataset.I)][None].T, axis = 0)
            ind = ind + m
        
        ratedata_attack = np.append(ratedata_attack, ratedata_new, axis = 0)
        FeatureUserAttack = np.append(FeatureUserAttack, np.append(X, Y)[None].T, axis = 1)
        
    dataset_attack = copy.deepcopy(dataset)
    dataset_attack.I = attack_size
    dataset_attack.FeatureUser = FeatureUserAttack
    dataset_attack.ratedata = ratedata_attack
   
    return dataset_attack

def statistic_attack(dataset_train, dataset_seq, GeniuneSize, MixSize, target_itemno):
    ratings_train = dataset_train.ratedata[np.where(dataset_train.ratedata[:, 1] ==  target_itemno)[0], 2]
    train_dist = (np.histogram(ratings_train, bins = 5, range = (1,5))[0] + 1) / (len(ratings_train) + 5)
    train_mean = np.mean(ratings_train)
    print("Target item no:", target_itemno)
    print("Train ratings:", len(ratings_train))
    print("Train dist:", np.array_str(train_dist, precision =2))
    print("Train mean:", (train_mean))
    
    
    Rseq, Oseq = rate_to_matrix(dataset_seq.ratedata, dataset_seq.I, dataset_seq.J)
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
    

def generate_target_item(ratedata_train, InfoData):
    
    itemdatarate = np.sum(rate_to_matrix(ratedata_train, InfoData.I, InfoData.J)[1], axis = 0)
    rated_items = np.where(itemdatarate > 10)[0]
    np.random.shuffle(rated_items)
    target_itemno = rated_items[0] + 1
    
    return target_itemno


def auc_eval(ground_truth_vec, stat_vec):
    auc = 0
    for i in np.where(ground_truth_vec == 1)[0]:
        auc = auc + len((np.where(stat_vec[i] > stat_vec[np.where(ground_truth_vec == 0)[0]]))[0])
    auc = auc / (len(np.where(ground_truth_vec == 1)[0]) * len(np.where(ground_truth_vec == 1)[0]))
    return auc
    

    
def generate_mix_attack(dataset_attack, dataset_test_sub, MixSize):
    
    GeniuneSize = dataset_test_sub.I
    AttackSize = dataset_attack.I
    J = dataset_test_sub.J
    
    R_at,O_at = rate_to_matrix(dataset_attack.ratedata, AttackSize, J)
    R_gen,O_gen = rate_to_matrix(dataset_test_sub.ratedata, GeniuneSize, J)
    
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
    
    feature_new = np.append(dataset_test_sub.FeatureUser, dataset_attack.FeatureUser, axis = 1)
    feature_new = feature_new[:, new_ind]
    
    R = np.append(R_gen, R_at, axis = 0)
    R = R[new_ind, :]
    O = np.append(O_gen, O_at, axis = 0)
    O = O[new_ind, :]
    
    ground_truth_vec = np.append(np.zeros(GeniuneSize-MixSize), np.ones(AttackSize+MixSize))
    ground_truth_vec[mix_ind2 + GeniuneSize-MixSize] = 0
    
    
    ratedata = np.append(np.where(O > 0)[0][None].T + 1, np.where(O > 0)[1][None].T + 1, axis = 1)
    ratedata = np.append(ratedata, R[np.where(O > 0)][None].T, axis = 1)
    
    
    dataset_mix = copy.deepcopy(dataset_attack)
    dataset_mix.I = GeniuneSize + AttackSize
    dataset_mix.FeatureUser = feature_new
    dataset_mix.ratedata = ratedata
    
    return dataset_mix, ground_truth_vec
    

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

