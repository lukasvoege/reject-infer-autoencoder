from math import floor
from numpy.lib import tracemalloc_domain
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score
from sklearn.base import clone
import sklearn.preprocessing as pre
import torch

import helper as h
import autoencoder as aenc


class Simulate_acceptance_loop():

    def __init__(self, dataset_name: str, model, model_fit_split: float, holdout_test_split: float, n_loops: int, enc_features: bool = False, encoder: aenc.Autoencoder = None, rej_inf = None, incl_rejected: bool = False):
        self.n_loops = n_loops
        self.rej_inf = rej_inf

        # custom Transformer, that standard scales and then encodes the data with Autoencoder
        def std_enco(x):
            x_np = x.values.reshape(-1, x.shape[1]).astype('float32')
            x_trans = pre.StandardScaler().fit_transform(x_np)
            with torch.no_grad():
                x_encoded = encoder.encode(torch.from_numpy(x_trans)).numpy()
            return x_encoded

        # Either normal StandardScaler or StandardScaler and Encoder
        if enc_features:
            self.transformer  =  pre.FunctionTransformer(std_enco)
        else:
            self.transformer  = pre.StandardScaler()

        # load dataset
        complete_data = pd.read_csv(f'../prepared_data/{dataset_name}', sep=',')
        complete_data['BAD'] = np.where(complete_data['BAD'] == 'BAD', 1, 0).astype(np.int64)

        obj_cols = complete_data.select_dtypes('object').columns
        complete_data[obj_cols] = complete_data[obj_cols].astype('category')

        # For the sake of simplicity when dealing with neural nets later, let's just make everything categorical continous
        for col in obj_cols:
            woe_calc = h.IV_Calc(complete_data, feature=col, target='BAD')
            woe = woe_calc.full_summary()['WOE_adj'].to_dict()
            complete_data[col] = complete_data[col].map(woe)
            complete_data[col] = complete_data[col].astype('float64')

        complete_X = complete_data.iloc[:, complete_data.columns != 'BAD']
        complete_y = complete_data['BAD']

        # get part of the data for inital model fitting
        X_remaining, X_model_fit, y_remaining, y_model_fit = train_test_split(complete_X, complete_y, test_size=model_fit_split, stratify=complete_y, random_state=123)

        # reserve holdout data for model evaluation
        X_simulation, X_holdout, y_simulation, y_holdout = train_test_split(X_remaining, y_remaining, test_size=holdout_test_split, stratify=y_remaining, random_state=123)

        self.holdout_test_X = X_holdout
        self.holdout_test_y = y_holdout

        # put simulation data back together
        self.simulation_data = pd.concat([X_simulation, y_simulation], axis=1)

        # initial fit of model and oracle model on remaining data
        self.model = model
        self.oracle = clone(model)

        self.transformer.fit(X_model_fit)
        X_model_fit_trans = self.transformer.transform(X_model_fit)
        self.model.fit(X_model_fit_trans, y_model_fit)

        self.oracle.fit(X_model_fit_trans, y_model_fit)

        # store all available train data
        self.all_train_X = X_model_fit
        self.all_train_y = y_model_fit

        self.oracle_all_train_X = X_model_fit
        self.oracle_all_train_y = y_model_fit

        self.info = f'''
        Data SPLIT ({dataset_name}):\n
        Total rows: {complete_X.shape[0]}, Total columns: {complete_X.shape[1]}\n
        Used for initial model training:{X_model_fit.shape[0]}\t({round(X_model_fit.shape[0]/complete_X.shape[0], 4)})\n
        Used for model evaluation:\t{X_holdout.shape[0]}\t({round(X_holdout.shape[0]/complete_X.shape[0], 4)})\n
        Remaining used for simulation:\t{X_simulation.shape[0]}\t({round(X_simulation.shape[0]/complete_X.shape[0], 4)})
        '''
        #print(self.info)


    def run(self, accpet_fraction = 0.3):

        self.simulation_data = self.simulation_data.sample(frac=1, random_state=123)
        self.data_splits = np.array_split(self.simulation_data, self.n_loops)

        metrics = {"model": {"rolling": {"roc_auc": [], "accuracy": [], "f1": [], "precision": []},
                    "holdout": {"roc_auc": [], "accuracy": [], "f1": [], "precision": []}},
                "oracle": {"rolling": {"roc_auc": [], "accuracy": [], "f1": [], "precision": []},
                    "holdout": {"roc_auc": [], "accuracy": [], "f1": [], "precision": []}}}
        
        for year in range(self.n_loops):
            # 1. get the new data for that year
            new_data = self.data_splits[year].reset_index(drop=True)
            X = new_data.iloc[:, new_data.columns != 'BAD']
            y = new_data['BAD']
            
            # 2. predict data for this year with model and oracle
            predicted_proba = self.model.predict_proba(self.transformer.transform(X))[:, 1]
            predicted_proba_oracle = self.oracle.predict_proba(self.transformer.transform(X))[:, 1]

            threshold = sorted(predicted_proba)[floor(len(predicted_proba)*accpet_fraction)] # accept top n% of aplicants
            if threshold == 0.0: threshold = 0.0001 # needed to allow for pure Tree Classifier Leafs

            predicted_abs = np.where(predicted_proba < threshold, 0, 1)
            predicted_abs_oracle = np.where(predicted_proba_oracle < 0.5, 0, 1)

            # 3. add accepted data points to all available training data
            accepted = [True if x == 0 else False for x in predicted_abs]
            
            self.all_train_X = pd.concat([self.all_train_X, X[accepted]], ignore_index=True)
            self.all_train_y = pd.concat([self.all_train_y, y[accepted]], ignore_index=True)

            # 3.2 add same number of points (but random points - so no acceptance bias) to oracle model
            random.shuffle(accepted)
            self.oracle_all_train_X = pd.concat([self.oracle_all_train_X, X[accepted]], ignore_index=True)
            self.oracle_all_train_y = pd.concat([self.oracle_all_train_y, y[accepted]], ignore_index=True)

            # verbose
            # print(f'Itteration: {year}) Accepted: {accepted.count(True)} | Denied: {accepted.count(False)} - New train set size: {self.all_train_X.shape}')
            predicted_abs = np.where(predicted_proba < 0.5, 0, 1)

            # 4.1 save rolling_metrics for data of that year
            metrics["model"]["rolling"]['roc_auc'].append(roc_auc_score(y, predicted_proba))
            metrics["model"]["rolling"]['accuracy'].append(accuracy_score(y, predicted_abs))
            metrics["model"]["rolling"]['f1'].append(f1_score(y, predicted_abs))
            metrics["model"]["rolling"]['precision'].append(precision_score(y, predicted_abs))

            metrics["oracle"]["rolling"]['roc_auc'].append(roc_auc_score(y, predicted_proba_oracle))
            metrics["oracle"]["rolling"]['accuracy'].append(accuracy_score(y, predicted_abs_oracle))
            metrics["oracle"]["rolling"]['f1'].append(f1_score(y, predicted_abs_oracle))
            metrics["oracle"]["rolling"]['precision'].append(precision_score(y, predicted_abs_oracle))

            # 4.2 save metrics on evaluation hold out
            predicted_proba = self.model.predict_proba(self.transformer.transform(self.holdout_test_X))[:, 1]
            predicted_abs = np.where(predicted_proba < 0.5, 0, 1)

            predicted_proba_oracle = self.oracle.predict_proba(self.transformer.transform(self.holdout_test_X))[:, 1]
            predicted_abs_oracle = np.where(predicted_proba_oracle < 0.5, 0, 1)

            metrics["model"]["holdout"]['roc_auc'].append(roc_auc_score(self.holdout_test_y, predicted_proba))
            metrics["model"]["holdout"]['accuracy'].append(accuracy_score(self.holdout_test_y, predicted_abs))
            metrics["model"]["holdout"]['f1'].append(f1_score(self.holdout_test_y, predicted_abs))
            metrics["model"]["holdout"]['precision'].append(precision_score(self.holdout_test_y, predicted_abs))

            metrics["oracle"]["holdout"]['roc_auc'].append(roc_auc_score(self.holdout_test_y, predicted_proba_oracle))
            metrics["oracle"]["holdout"]['accuracy'].append(accuracy_score(self.holdout_test_y, predicted_abs_oracle))
            metrics["oracle"]["holdout"]['f1'].append(f1_score(self.holdout_test_y, predicted_abs_oracle))
            metrics["oracle"]["holdout"]['precision'].append(precision_score(self.holdout_test_y, predicted_abs_oracle))

            # 5. train model on all available data to improve
            self.transformer.fit(self.all_train_X)
            all_train_X_trans = self.transformer.transform(self.all_train_X)

            self.transformer.fit(self.oracle_all_train_X)
            oracle_all_train_X_trans = self.transformer.transform(self.oracle_all_train_X)
            
            if(self.rej_inf is None):
                self.model.fit(all_train_X_trans, self.all_train_y)
            else:
                self.rej_inf.fit(self.model,all_train_X_trans, self.all_train_y)
                
            self.oracle.fit(oracle_all_train_X_trans, self.oracle_all_train_y)

            yield year, accepted, all_train_X_trans.shape, metrics

        return metrics
