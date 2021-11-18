import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score

import lightgbm as lgbm
import matplotlib.pyplot as plt

class Simulate_acceptance_loop():

    def __init__(self, dataset_name, model, model_fit_split, n_loops):
        self.n_loops = n_loops

        # load dataset
        complete_data = pd.read_csv(f'C:/Projects/Information-Systems-Seminar/prepared_data/{dataset_name}.csv', sep=',')
        complete_data['BAD'] = np.where(complete_data['BAD'] == 'BAD', 1, 0).astype(np.int64)

        obj_cols = complete_data.select_dtypes('object').columns
        complete_data[obj_cols] = complete_data[obj_cols].astype('category')

        complete_X = complete_data.iloc[:, complete_data.columns != 'BAD']
        complete_y = complete_data['BAD']

        # get part of the data for inital model fitting
        X_simulation, X_model_fit, y_simulation, y_model_fit = train_test_split(complete_X, complete_y, test_size=model_fit_split, stratify=complete_y, random_state=123)

        # put simulation data back together
        self.simulation_data = pd.concat([X_simulation, y_simulation], axis=1)

        # initial fit of model on remaining data
        self.model = model
        self.model.fit(X_model_fit, y_model_fit)
        
        # store all available train data
        self.all_train_X = X_model_fit
        self.all_train_y = y_model_fit


    def run(self):

        self.data_splits = np.array_split(self.simulation_data, self.n_loops)

        metrics = {"roc_auc": [], "accuracy": [], "f1": [], "precision": []}
        
        for year in range(self.n_loops):
            # 1. get the new data for that year
            new_data = self.data_splits[year].reset_index(drop=True)
            X = new_data.iloc[:, new_data.columns != 'BAD']
            y = new_data['BAD']
            
            # 2. predict data for this year
            predicted_proba = self.model.predict_proba(X)[:, 1]
            predicted_abs = np.where(predicted_proba < 0.5, 0, 1)

            # 3. save metrics for that year
            metrics['roc_auc'].append(roc_auc_score(y, predicted_proba))
            metrics['accuracy'].append(accuracy_score(y, predicted_abs))
            metrics['f1'].append(f1_score(y, predicted_abs))
            metrics['precision'].append(precision_score(y, predicted_abs))

            s = [True if x == 0 else False for x in predicted_abs]

            # 4. add accepted data points to all available training data
            self.all_train_X = pd.concat([self.all_train_X, X[s]], ignore_index=True)
            self.all_train_y = pd.concat([self.all_train_y, y[s]], ignore_index=True)

            print(self.all_train_X.shape)


            # 5. train model on all available data to improve
            self.model.fit(self.all_train_X, self.all_train_y)


        return metrics


sim = Simulate_acceptance_loop("pakdd", lgbm.LGBMClassifier(), 0.0005, 10)
results = sim.run()


x = range(1, 11)

plt.plot(x, results['roc_auc'], label = 'roc_auc')
plt.plot(x, results['precision'], label = 'precision')
plt.plot(x, results['f1'], label = 'f1')
plt.plot(x, results['accuracy'], label = 'accuracy')
plt.legend()
plt.show()


        