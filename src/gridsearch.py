import sys
sys.path.append('../src')

import importlib
import dyn_accept_loop as dal
importlib.reload(dal)
import reject_inference as rinf
importlib.reload(rinf)
import helper as h
importlib.reload(h)

import matplotlib.pyplot as plt

import lightgbm as lgbm

import numpy as np

import lightgbm as lgbm

import pickle

import autoencoder as aenc
import var_autoencoder as vaenc
import importlib
importlib.reload(aenc)
importlib.reload(vaenc)

import timeit



n_years = 20
model = lgbm.LGBMClassifier(random_state=1234) # DecisionTreeClassifier(min_samples_leaf=40) LogisticRegression(max_iter=400)
datasetname = "gmsc.csv"

## ------------------------------------------------------------
## Get a Baseline Bias Measue without any reject inference
## ------------------------------------------------------------

sim = dal.Simulate_acceptance_loop(datasetname, model, 0.1, 0.1, n_years, enc_features=False, rej_inf=None)
results_generator = sim.run()

metrics = None
for iteration in results_generator:
    #print(f'Itteration: {iteration[0]}) Accepted: {iteration[1].count(True)} | Denied: {iteration[1].count(False)} - New train set size: {iteration[2]}')
    metrics = iteration[3]

last_n_years = 5
baseline_bias = h.measure_bias(metrics["oracle"]["holdout"]['roc_auc'], metrics["model"]["holdout"]['roc_auc'], last_n_years)
baseline_roc_auc = sum(metrics["model"]["holdout"]['roc_auc'][-last_n_years:]) / last_n_years
print(f'Baseline Sampling Bias: {round(baseline_bias, 5)}\nBaseline ROC-AUC: {round(baseline_roc_auc, 5)}')


## -----------------------------------------------------------------------
## Loop through Autoencoder training and testing for a parameter
## -----------------------------------------------------------------------

weight_test = np.array(range(0, 11, 1)) / 10
epochs_test = [1] + list(range(1, 7, 1))
shape_test = np.array(range(4, 40, 3))
BATCH_SIZE = 2000
#EPOCHS = 10
LR = 1e-3

sampling_bias = dict()
sampling_bias_flat = []
roc_auc = dict()
roc_auc_flat = []

fig, axes = plt.subplots(nrows=len(weight_test), ncols=len(shape_test), figsize=(40, 20))

for weight in weight_test:
    sampling_bias[weight] = dict()
    roc_auc[weight] = dict()
    for layer in shape_test:

        start = timeit.default_timer()
        
        # Train Autoencoder
        LOSSFUNCWEIGHTS = [weight, 1 - weight, 0.0]  #[MMSE, KLDiv, MMD]

        dataset = aenc.CreditscoringDataset(datasetname)      # load and prepare Dataset to Tensor
        data_loader = aenc.DataLoader(                       # create Dataloader for batching
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        shape = [dataset.x.shape[1], 45, layer, 45, dataset.x.shape[1]]  # define shape of Autoencoder PARAM = 25
        net3 = aenc.Autoencoder(shape)
        #print(net)
        net3.to("cpu")

        sampling_bias[weight][layer] = []
        roc_auc[weight][layer] = []

        prev_ep = 0
        for EPOCHS in epochs_test:

            train_loss, train_loss_mmse, train_loss_mmd, train_loss_kld = aenc.train(net3, data_loader, 2**EPOCHS, LR, LOSSFUNCWEIGHTS, verbose = False)

            # Simulate on encoded Data to measure sampling bias

            sim = dal.Simulate_acceptance_loop(datasetname, model, 0.1, 0.1, n_years, enc_features=True, encoder=net3)
            results_generator = sim.run()

            metrics7 = None
            for iteration in results_generator:
                metrics7 = iteration[3]


            sampling_bias[weight][layer].append(h.measure_bias(metrics["oracle"]["holdout"]['roc_auc'], metrics7["model"]["holdout"]['roc_auc'], last_n_years))
            roc_auc[weight][layer].append(sum(metrics7["model"]["holdout"]['roc_auc'][-last_n_years:]) / last_n_years)

            sampling_bias_flat.append(h.measure_bias(metrics["oracle"]["holdout"]['roc_auc'], metrics7["model"]["holdout"]['roc_auc'], last_n_years))
            roc_auc_flat.append(sum(metrics7["model"]["holdout"]['roc_auc'][-last_n_years:]) / last_n_years)

            print(f'PARAM: W({weight}), L({layer}), E({prev_ep + 2**EPOCHS}) | Sampling Bias: {round(sampling_bias[weight][layer][-1], 5)} // ROC-AUC: {round(roc_auc[weight][layer][-1], 5)}')

            prev_ep += 2**EPOCHS
        
        stop = timeit.default_timer()

        print(f'Saving current results... - Last iteration took {stop - start} sec.')
        pickle.dump(roc_auc, open('roc-auc_grid-result.p', 'wb'))
        pickle.dump(roc_auc_flat, open('roc-auc-flat_grid-result.p', 'wb'))

        pickle.dump(sampling_bias, open('sampling-bias-result.p', 'wb'))
        pickle.dump(sampling_bias_flat, open('sampling-bias-flat-result.p', 'wb'))

print("I am done ya selame!")
