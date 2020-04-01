
from DatasetPrep import movie1mprep
from utils_attack import split_users
from utils_attack import attack_gen
from utils_attack import plot_roc, plot_gt
from utils_attack import generate_mix_attack
from utils_attack import statistic_attack
from sequential_attack_detector import sequential_attack_detector

test_cond = {
        "attack_type"           : "mix_push",
        "attack_size"           : 0.05,
        "filler_size"           : 0.04,
        "popularity"            : 1.0,
        
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
            

FeatureUser, FeatureItem, ratedata, InfoData = movie1mprep(ImpFeed = 0)


# Print 
TestSize = int(InfoData.I * test_cond['Test_ratio'] )
TrainSize = InfoData.I - TestSize
ValSize = int(TrainSize * test_cond['Val_ratio'])
TrainSize = TrainSize - ValSize

TestSubSize = int(TestSize * test_cond['TestSub_ratio'] )
HoldSize = TestSize - TestSubSize

MixSize = int(TestSubSize * test_cond['Mix_ratio'])
GeniuneSize = TestSubSize - MixSize
AttackSize = int(InfoData.I * test_cond['attack_size'])

print('Train-Test-Val Split:')
print('Training Users: {}, Validation Users: {}, Test Users: {}'.format(TrainSize, ValSize, TestSize))
print('Sequential Attack:')
print('Geniune: {}, Mix: {}, Attack: {}'.format(GeniuneSize, MixSize, AttackSize))


# Train-Test split
ratedata_train, FeatureUserTrain, ratedata_test, FeatureUserTest = split_users(ratedata = ratedata,
                                                                               I = InfoData.I,
                                                                               J = InfoData.J,
                                                                               FeatureUser = FeatureUser,
                                                                               SplitSize = TestSize,
                                                                               random = 1,
                                                                               offset = 0)
        

InfoData.I = InfoData.I - TestSize


# Compute baseline statistics from training data
detector = sequential_attack_detector(InfoData)

detector.compute_baseline(FeatureUserTrain,
                          FeatureItem,
                          ratedata_train,
                          ValSize,
                          test_cond["target_itemno"],
                          K = 40, lmd_u = 10, lmd_v = 1,
                          lvm_epochno = 10)


# Generate mixed sequential atttack
ratedata_attack, FeatureUserAttack = attack_gen(InfoData,
                                                FeatureUserTrain,
                                                ratedata_train,
                                                target_itemno = test_cond["target_itemno"],
                                                user_start_id = 1,
                                                no_of_newratings =  max(8, int(InfoData.J * test_cond['filler_size'])),
                                                attack_size = AttackSize,
                                                attack_type = 1,
                                                pop_frac = test_cond['popularity'],
                                                rand_rating = 1,
                                                prob = test_cond['rating_prob_push'])

_, _, ratedata_test_sub, FeatureUserTestSub = split_users(ratedata = ratedata_test,
                                                          I = TestSize,
                                                          J = InfoData.J,
                                                          FeatureUser = FeatureUserTest,
                                                          SplitSize = TestSubSize,
                                                          random = 1,
                                                          offset = 0)
        
Rseq, Oseq, ground_truth_vec, FeatureUserSeq = generate_mix_attack(ratedata_attack,
                                                          ratedata_test_sub,
                                                          FeatureUserAttack,
                                                          FeatureUserTestSub,
                                                          InfoData.J, 
                                                          TestSubSize,
                                                          AttackSize,
                                                          MixSize)

statistic_attack(ratedata_train = ratedata_train,
                 Rseq = Rseq,
                 Oseq = Oseq,
                 GeniuneSize = GeniuneSize,
                 MixSize = MixSize,
                 target_itemno = test_cond["target_itemno"])

# Test the sequence 
detector.compute_test_score(FeatureUserSeq,
                            Rseq, Oseq,
                            test_cond["target_itemno"])



# Plot the results
plot_roc(ground_truth_vec, detector.stat_vec_user)
plot_gt(detector.g_t_vec)






