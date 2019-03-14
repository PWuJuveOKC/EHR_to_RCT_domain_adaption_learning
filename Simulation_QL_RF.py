import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from Qlearn_RF import QLearn
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
from sklearn.preprocessing import scale


strat = 'benefit'
ran_p = 3
obs_p = 3
unknown = 'known'
whole = 'sub'
ran_size = 500

if whole == 'sub':
    crit_point1 = -0.4
    crit_point2 = 0.4
else:
    crit_point1 = -9999
    crit_point2 = 9999

if unknown == 'unknown':
    unobs_latent = obs_p - 1
elif unknown == 'known':
    unobs_latent = obs_p


def phi_eta(Frame_X):
    phi = 0.5 * (Frame_X[:, 0] + Frame_X[:, 1] > 0) - \
          (Frame_X[:, 0] + Frame_X[:, 1] <= 0) * (0.5 + 0.5 * (Frame_X[:, 1] <= - 0.5)) + \
           (Frame_X[:, 2] ** 2 - Frame_X[:, 1] ** 2)
    eta = Frame_X[:, 0] - 0.5 * Frame_X[:, 1]

    return phi, eta

# Generate data
def generate_data(sample_size, p, seed=0, obs=True, prob=0.5):
    r = np.random.RandomState(seed)
    Val_X = r.standard_normal((sample_size, p))
    try:
        if obs:
            z = 1 + 2 * Val_X[:, 0] + Val_X[:, 1]
            pr = 1 / (1 + np.exp(-z))
        elif not obs:
            pr = np.repeat(prob, sample_size)
        else:
            pr = None
            raise ValueError
    except ValueError:
        print("Invalid setting!")

    r2 = np.random.RandomState(seed + 1)
    Val_A = r2.binomial(1, pr) * 2 - 1


    r3 = np.random.RandomState(seed + 2)
    Val_N = r3.standard_normal((sample_size)) / 2

    phi, eta = phi_eta(Val_X)

    eta = Val_X[:, 0] - 0.5 * Val_X[:, 1]
    Val_Q = eta + phi * Val_A + Val_N
    Val_A_True = np.sign(phi )
    Potential = eta + phi * Val_A_True + Val_N

    return Val_Q, Potential, Val_X, Val_N, Val_A, Val_A_True



# Learn observational data
def learn_obs(X,Q,A,seed=0):

    model_obs = QLearn(propensity='obs')
    model_obs.fit(X, Q, A)
    pred_obs_Q = model_obs.predict(X)
    lm = LogisticRegression()
    lm.fit(X,A)
    prob = 1 - lm.predict_proba(X)[:, 0]

    # Predict optimal outcome
    X_optimal = X[np.where(pred_obs_Q==A)]
    A_optimal = A[np.where(pred_obs_Q==A)]
    Q_optimal = Q[np.where(pred_obs_Q==A)]
    PSwt = 1 / prob[np.where(pred_obs_Q==A)]

    model_out = ensemble.RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=seed)
    model_out.fit(X_optimal, Q_optimal, sample_weight=PSwt)

    # Predict nonoptimal outcome
    X_nonoptimal = X[np.where(pred_obs_Q != A)]
    A_nonoptimal = A[np.where(pred_obs_Q != A)]
    Q_nonoptimal = Q[np.where(pred_obs_Q != A)]
    PSwt_non = 1 / prob[np.where(pred_obs_Q != A)]

    model_out2 = ensemble.RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=seed)
    model_out2.fit(X_nonoptimal, Q_nonoptimal, sample_weight=PSwt_non)


    # Predict optimal TRT
    model_trt = ensemble.RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=seed)
    model_trt.fit(X_optimal, A_optimal, sample_weight=PSwt)

    return model_out, model_out2, model_trt


# Predict Obs-related scores in RCT data
def RCT_scores(RCT_X, model_out, model_out2, model_trt):

    RCT_popt_out1 = model_out.predict(RCT_X)
    RCT_popt_out2 = model_out2.predict(RCT_X)
    RCT_benefit = RCT_popt_out1 - RCT_popt_out2
    RCT_prob = 1 -  model_trt.predict_proba(RCT_X)[:, 0]

    return RCT_benefit, RCT_prob


########## Simulation ###########
import time
start_time = time.time()
num_cores = multiprocessing.cpu_count()
sim_Q_TEST, sim_Potential_TEST, sim_X_TEST, sim_N_TEST, sim_A_TEST, sim_A_True_TEST = generate_data(10000, ran_p, seed=9999, obs=False)

def Learning(t):
    sim_Q_obs, sim_Potential_obs, sim_X_obs, sim_N_obs, sim_A_obs, sim_A_True_obs = generate_data(sample_size, obs_p, seed=t)
    sim_Q_ran, sim_Potential_ran, sim_X_ran, sim_N_ran, sim_A_ran, sim_A_True_ran = generate_data(10000, ran_p, seed=t, obs=False)

    ## inclusion criterion
    np.random.seed(t)
    subgroup = np.where((sim_X_ran[:, 0] >= crit_point1) & (sim_X_ran[:, 0] <= crit_point2))
    ran_index = np.random.choice(len(subgroup[0]), ran_size)
    sim_Q_ran = sim_Q_ran[subgroup][ran_index]
    sim_Potential_ran = sim_Potential_ran[subgroup][ran_index]
    sim_A_ran = sim_A_ran[subgroup][ran_index]
    sim_X_ran = sim_X_ran[subgroup][ran_index]

    ####### strategy 1:
    My_RCT_X1 = sim_X_ran[:, 0:(unobs_latent)]
    TEST_RCT_X1 = sim_X_TEST[:, 0:(unobs_latent)]

    model_QL_s1 = QLearn(propensity=0.5)
    model_QL_s1.fit(My_RCT_X1, sim_Q_ran, sim_A_ran)

    TEST_Pred_QL_s1 = model_QL_s1.predict(TEST_RCT_X1)
    phi_TEST1, eta_TEST1 = phi_eta(sim_X_TEST)

    TEST_Potential_QL_s1 = np.mean(eta_TEST1 + phi_TEST1 * TEST_Pred_QL_s1 + sim_N_TEST)
    Ben1 = 2 * np.mean(phi_TEST1 * TEST_Pred_QL_s1)

    ####### strategy 2: Add features directly
    My_model_out1, My_model_out2, My_model_trt1 = learn_obs(sim_X_obs[:, 0:(unobs_latent)], sim_Q_obs, sim_A_obs, seed=t * t)
    My_RCT_out1, My_RCT_prob1 = RCT_scores(sim_X_ran[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1)
    My_RCT_scores1 = np.column_stack((scale(My_RCT_prob1),scale(My_RCT_out1)))
    My_RCT_X2 = np.append(sim_X_ran[:, 0:(unobs_latent)], My_RCT_scores1, axis=1)

    TEST_RCT_out1, TEST_RCT_prob1 = RCT_scores(sim_X_TEST[:, 0:(unobs_latent)],  My_model_out1, My_model_out2, My_model_trt1)
    TEST_RCT_scores1 = np.column_stack((scale(TEST_RCT_prob1), scale(TEST_RCT_out1)))
    TEST_RCT_X2 = np.append(sim_X_TEST[:, 0:(unobs_latent)], TEST_RCT_scores1, axis=1)

    model_QL_s2 = QLearn( propensity=0.5)
    model_QL_s2.fit(My_RCT_X2, sim_Q_ran, sim_A_ran)
    TEST_Pred_QL_s2 = model_QL_s2.predict(TEST_RCT_X2)
    #print(np.mean(TEST_Pred_QL_s3==sim_A_True_TEST))
    TEST_Potential_QL_s2 = np.mean(eta_TEST1 + phi_TEST1 * TEST_Pred_QL_s2 + sim_N_TEST)
    Ben2 =  2 * np.mean(phi_TEST1 * TEST_Pred_QL_s2)

    ####### Strategy 3: Stratification
    My_model_out1, My_model_out2, My_model_trt1 = learn_obs(sim_X_obs[:, 0:(unobs_latent)], sim_Q_obs, sim_A_obs,seed=t * t)
    My_RCT_benefit, My_RCT_prob1 = RCT_scores(sim_X_ran[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1)
    np.random.seed(t)
    My_RCT_prob1 = My_RCT_prob1 + np.random.uniform(-1e-10, 1e-10, size=ran_size)
    if strat == 'prob':
        My_RCT_scores1 = np.column_stack((scale(My_RCT_benefit), scale(My_RCT_prob1)))
    elif strat == 'benefit':
        My_RCT_scores1 = np.column_stack((scale(My_RCT_prob1), scale(My_RCT_benefit)))

    cutoff = 0
    My_RCT_X3 = np.append(sim_X_ran[:, 0:(unobs_latent)], My_RCT_scores1, axis=1)

    My_RCT_X3_hi = My_RCT_X3[np.where(My_RCT_scores1[:, 1] >= cutoff)]
    sim_Q_ran_hi = sim_Q_ran[np.where(My_RCT_scores1[:, 1] >= cutoff)]
    sim_A_ran_hi = sim_A_ran[np.where(My_RCT_scores1[:, 1] >= cutoff)]

    My_RCT_X3_lo = My_RCT_X3[np.where(My_RCT_scores1[:, 1] < cutoff)]
    sim_Q_ran_lo = sim_Q_ran[np.where(My_RCT_scores1[:, 1] < cutoff)]
    sim_A_ran_lo = sim_A_ran[np.where(My_RCT_scores1[:, 1] < cutoff)]

    My_RCT_X3_hi = My_RCT_X3_hi[:, :-1]
    My_RCT_X3_lo = My_RCT_X3_lo[:, :-1]

    TEST_RCT_benefit, TEST_RCT_prob1 = RCT_scores(sim_X_TEST[:, 0:(unobs_latent)], My_model_out1, My_model_out2, My_model_trt1)
    np.random.seed(t * t)
    TEST_RCT_prob1 = TEST_RCT_prob1 + np.random.uniform(-1e-10, 1e-10, size=10000)
    if strat == 'prob':
        TEST_RCT_scores1 = np.column_stack((scale(TEST_RCT_benefit), scale(TEST_RCT_prob1)))
    elif strat == 'benefit':
        TEST_RCT_scores1 = np.column_stack((scale(TEST_RCT_prob1), scale(TEST_RCT_benefit)))
    TEST_RCT_X3 = np.append(sim_X_TEST[:, 0:(unobs_latent)], TEST_RCT_scores1, axis=1)

    TEST_RCT_X3_hi = TEST_RCT_X3[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]
    sim_N_TEST_hi = sim_N_TEST[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]
    sim_X_TEST_hi = sim_X_TEST[np.where(TEST_RCT_scores1[:, 1] >= cutoff)]

    TEST_RCT_X3_lo = TEST_RCT_X3[np.where(TEST_RCT_scores1[:, 1] < cutoff)]
    sim_N_TEST_lo = sim_N_TEST[np.where(TEST_RCT_scores1[:, 1] < cutoff)]
    sim_X_TEST_lo = sim_X_TEST[np.where(TEST_RCT_scores1[:, 1] < cutoff)]

    TEST_RCT_X3_hi = TEST_RCT_X3_hi[:, :-1]
    TEST_RCT_X3_lo = TEST_RCT_X3_lo[:, :-1]

    ### HIGH GROUP
    model_QL_s3_hi = QLearn( propensity=0.5)
    model_QL_s3_hi.fit(My_RCT_X3_hi, sim_Q_ran_hi, sim_A_ran_hi)
    TEST_Pred_QL_s3_hi = model_QL_s3_hi.predict(TEST_RCT_X3_hi)
    phi_TEST3_hi, eta_TEST3_hi = phi_eta(sim_X_TEST_hi)

    TEST_Potential_QL_s3_hi = np.mean(eta_TEST3_hi + phi_TEST3_hi * TEST_Pred_QL_s3_hi + sim_N_TEST_hi)
    TEST_Potential_QL_s3_hi = TEST_Potential_QL_s3_hi * len(TEST_RCT_X3_hi) / 10000
    Ben3_hi = 2 * np.mean(phi_TEST3_hi * TEST_Pred_QL_s3_hi) * len(TEST_RCT_X3_hi) / 10000

    ### LOW GROUP
    model_QL_s3_lo = QLearn( propensity=0.5)
    model_QL_s3_lo.fit(My_RCT_X3_lo, sim_Q_ran_lo, sim_A_ran_lo)
    TEST_Pred_QL_s3_lo = model_QL_s3_lo.predict(TEST_RCT_X3_lo)
    phi_TEST3_lo, eta_TEST3_lo = phi_eta(sim_X_TEST_lo)

    TEST_Potential_QL_s3_lo = np.mean(eta_TEST3_lo + phi_TEST3_lo * TEST_Pred_QL_s3_lo + sim_N_TEST_lo)
    TEST_Potential_QL_s3_lo = TEST_Potential_QL_s3_lo * len(TEST_RCT_X3_lo) / 10000
    Ben3_lo = 2 * np.mean(phi_TEST3_lo * TEST_Pred_QL_s3_lo) * len(TEST_RCT_X3_lo) / 10000

    TEST_Potential_QL_s3 = TEST_Potential_QL_s3_hi + TEST_Potential_QL_s3_lo
    Ben3 = Ben3_hi + Ben3_lo


    print('iteration_time: ', t, ' results: ', np.mean(sim_Potential_ran)), \
    print( "strategy 1: " , TEST_Potential_QL_s1,  Counter(TEST_Pred_QL_s1), Ben1),
    print("strategy 2: " , TEST_Potential_QL_s2,  Counter(TEST_Pred_QL_s2), Ben2),
    print("strategy 3: " , TEST_Potential_QL_s3,  Counter(TEST_Pred_QL_s3_hi), Counter(TEST_Pred_QL_s3_lo), Ben3 ),
    print("----------------------------------------------------------------------------------")

    return TEST_Potential_QL_s1, TEST_Potential_QL_s2,TEST_Potential_QL_s3, Ben1, Ben2, Ben3


results = {}
iters = 100
sim_size = [1000]
for sample_size in sim_size:
    results[str(sample_size)] = Parallel(n_jobs= (num_cores-1))(delayed(Learning)(t) for t in range(iters))

cost = time.time() - start_time
print('Time Consumed: ', cost)


print(np.median(np.array(results[str(sample_size)])[:,0]),np.mean(np.array(results[str(sample_size)])[:,0]), np.max(np.array(results[str(sample_size)])[:,0]), np.std(np.array(results[str(sample_size)])[:,0])),
print(np.median(np.array(results[str(sample_size)])[:,1]),np.mean(np.array(results[str(sample_size)])[:,1]), np.max(np.array(results[str(sample_size)])[:,1]), np.std(np.array(results[str(sample_size)])[:,1]))
print(np.median(np.array(results[str(sample_size)])[:,2]),np.mean(np.array(results[str(sample_size)])[:,2]), np.max(np.array(results[str(sample_size)])[:,2]), np.std(np.array(results[str(sample_size)])[:,2]))

print ('------------------------------------------------------')
print(np.median(np.array(results[str(sample_size)])[:,3]),np.mean(np.array(results[str(sample_size)])[:,3]), np.max(np.array(results[str(sample_size)])[:,3]), np.std(np.array(results[str(sample_size)])[:,3])),
print(np.median(np.array(results[str(sample_size)])[:,4]),np.mean(np.array(results[str(sample_size)])[:,4]), np.max(np.array(results[str(sample_size)])[:,4]), np.std(np.array(results[str(sample_size)])[:,4]))
print(np.median(np.array(results[str(sample_size)])[:,5]),np.mean(np.array(results[str(sample_size)])[:,5]), np.max(np.array(results[str(sample_size)])[:,5]), np.std(np.array(results[str(sample_size)])[:,5]))

res = pd.DataFrame(results[str(sample_size)],columns=['s1', 's2', 's3', 's1_ben', 's2_ben', 's3_ben'])
#res.to_csv('Output_s1/Sim_Q_s1_' + whole+ '_' + unknown +'_' + str(ran_size) + '_' +  str(sim_size[0]) + '_' + strat+ '_rf.csv', index=None)
