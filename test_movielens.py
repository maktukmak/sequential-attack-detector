
from DatasetPrep import DatasetPrep
from DatasetPrep import split_users, rate_to_matrix
from utils_attack import attack_gen
from utils_attack import plot_roc, plot_gt
from utils_attack import generate_mix_attack
from utils_attack import statistic_attack
from sequential_attack_detector import sequential_attack_detector
import matplotlib.pyplot as plt

test_cond = {
        "attack_type"           : "mix_push",
        "attack_size"           : 0.05,
        "filler_size"           : 0.04,
        "popularity"            : 0.2,
        
        "expno"                 : 1,
        
        "canidate_interval_high": 2.5,
        "canidate_interval_low" : 2.0,
        "canidate_rating"       : 500,
        "target_itemno"         : 1,
        
        "rating_prob_push"      : [0, 0, 0, 0, 1],
        "rating_prob_nuke"      : [1, 0, 0, 0, 0],
        
        "Test_ratio"            : 0.1, # of the users
        "TestSub_ratio"         : 0.5, # of the test users
        "Mix_ratio"             : 0.1, # of the test sub users  
        "Val_ratio"             : 0.1  # of the train users
        
        }
            

dataset_inp = DatasetPrep()
dataset_inp.movie1mload()
dataset_inp.dataset_stat()


TestSize = int(dataset_inp.I * test_cond['Test_ratio'] )
TrainSize = dataset_inp.I - TestSize
ValSize = int(TrainSize * test_cond['Val_ratio'])
TrainSize = TrainSize - ValSize

TestSubSize = int(TestSize * test_cond['TestSub_ratio'] )
HoldSize = TestSize - TestSubSize

MixSize = int(TestSubSize * test_cond['Mix_ratio'])
GeniuneSize = TestSubSize - MixSize
AttackSize = int(dataset_inp.I * test_cond['attack_size'])


# Print test conditions
print('Train-Test-Val Split:')
print('Training Users: {}, Validation Users: {}, Test Users: {}'.format(TrainSize, ValSize, TestSize))
print('Sequential Attack:')
print('Geniune: {}, Mix: {}, Attack: {}'.format(GeniuneSize, MixSize, AttackSize))




# Train-Test split
dataset_train, dataset_test = split_users(dataset_inp,
                                          SplitSize = TestSize,
                                          random = 1,
                                          offset = 0)
        
# Compute baseline statistics from training data
detector = sequential_attack_detector(dataset_train)

detector.compute_baseline(dataset_train,
                          ValSize,
                          test_cond["target_itemno"],
                          K = 40, lmd_u = 10, lmd_v = 1,
                          lvm_epochno = 10)


# Generate mixed sequential atttack
dataset_attack = attack_gen(dataset_train,
                            target_itemno = test_cond["target_itemno"],
                            user_start_id = 1,
                            no_of_newratings =  max(8, int(dataset_train.J * test_cond['filler_size'])),
                            attack_size = AttackSize,
                            attack_type = test_cond["attack_type"],
                            pop_frac = test_cond['popularity'],
                            rand_rating = 1,
                            prob = test_cond['rating_prob_push'])

_, dataset_test_sub = split_users( dataset_test,
                                      SplitSize = TestSubSize,
                                      random = 1,
                                      offset = 0)
        
dataset_seq, ground_truth_vec = generate_mix_attack(dataset_attack,
                                                    dataset_test_sub,
                                                    MixSize)


statistic_attack(dataset_train,
                 dataset_seq,
                 GeniuneSize = GeniuneSize,
                 MixSize = MixSize,
                 target_itemno = test_cond["target_itemno"])

# Test the sequence 
detector.compute_test_score(dataset_seq,
                            test_cond["target_itemno"])



# Plot the results
plot_roc(ground_truth_vec, detector.stat_vec_user)
plot_gt(detector.g_t_vec)






