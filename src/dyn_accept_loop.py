import pandas as pd
import numpy as np

import lightgbm as lgbm

class simulate_acceptance_loop():

    def __init__(self, dataset_name, model, n_loops): ### define split for intial model fitting
        # load dataset
        self.prepared_df = pd.read_csv(f'../prepared_data/{dataset_name}.csv', sep=',')
        self.prepared_df['BAD'] = np.where(self.prepared_df['BAD'] == 'BAD', 1, 0).astype(np.int64)

        self.model = model
        self.n_loops = n_loops

    def split_dataset(self):
        self.data_splits = np.array_split(self.prepared_df, self.n_loops)

    def run(self):
        pass


        