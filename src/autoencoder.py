from tkinter import E
from turtle import shape
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import mmd_loss as mmd

import helper as h


# Custom Dataset class, that loads the standardized and prepared data from function load_data_to_tensor()
class CreditscoringDataset(Dataset):
    def __init__(self, dataset_name):
        self.x, self.y = load_data_to_tensor(dataset_name)
        
    def __getitem__(self,index):      
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, shape):
        super(Autoencoder, self).__init__()

        if shape[0] != shape[-1]:
            print('Warning! First and last layer of encoder do not have the same size.')

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        # Build encoder part
        for i in range(shape.index(min(shape))):
            self.enc.append(nn.Linear(in_features = shape[i], out_features = shape[i + 1]))
        
        # Build decoder part
        for i in range(shape.index(min(shape)), len(shape) - 1):
            self.dec.append(nn.Linear(in_features = shape[i], out_features = shape[i + 1]))

    def forward(self, x):
        x = self.decode(self.encode(x))
        return x

    def encode(self, x):
        for e in self.enc:
            x = torch.tanh(e(x))
        return x
        
    def decode(self, x):
        for d in self.dec:
            x = torch.tanh(d(x))
        return x


# train any net
def train(net, trainloader, epochs, learningrate):
    criterion = nn.MSELoss()
    criterion2 = nn.KLDivLoss(log_target=True, reduction="batchmean")
    criterion3 = mmd.MMD_loss()
    optimizer = optim.Adam(net.parameters(), lr=learningrate)

    train_loss = []
    train_loss_mmse = []
    train_loss_mmd = []
    train_loss_kld = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_mmse = 0.0
        running_loss_mmd = 0.0
        running_loss_kld = 0.0
        for data in trainloader:
            data_x, data_y = data
            optimizer.zero_grad()
            outputs = net(data_x)
            encoded = net.encode(data_x)

            # split encoded data into good and bad subsets
            good = [True if x == 0 else False for x in data_y]
            enc_good = encoded[good]
            enc_bad = encoded[[not value for value in good]]
            #print(f'Enc_good shape: {enc_good.shape} Enc_bad shape: {enc_bad.shape}')

            # build MultiNorm Distributions from subsets and create log_probs of enc_good for both distributions to compare with KLDIVLOSS
            MN_dist_good = dis.multivariate_normal.MultivariateNormal(torch.mean(enc_good, dim=0), torch.corrcoef(torch.transpose(enc_good, 0, 1)))
            MN_dist_bad = dis.multivariate_normal.MultivariateNormal(torch.mean(enc_bad, dim=0), torch.corrcoef(torch.transpose(enc_bad, 0, 1)))

            sample = MN_dist_good.sample((1000,))

            enc_good = enc_good[:min([len(enc_good),len(enc_bad)])]
            enc_bad = enc_bad[:min([len(enc_good),len(enc_bad)])]

            KLDivLoss = criterion2(MN_dist_bad.log_prob(sample), MN_dist_good.log_prob(sample)) * 1000000
            MMSELoss = criterion(outputs, data_x)
            MMDLoss = criterion3(enc_good,enc_bad) * 10

            loss = 0.2 * MMSELoss + 0.8 * KLDivLoss + 0.0 * MMDLoss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_mmse += MMSELoss.item()
            running_loss_mmd += MMDLoss.item()
            running_loss_kld += KLDivLoss.item()

        loss = running_loss / len(trainloader)
        MMSELoss = running_loss_mmse / len(trainloader)
        MMDLoss = running_loss_mmd / len(trainloader)
        KLDivLoss = running_loss_kld / len(trainloader)
        train_loss.append(loss)
        train_loss_mmse.append(MMSELoss)
        train_loss_mmd.append(MMDLoss)
        train_loss_kld.append(KLDivLoss)
        
        print('Epoch {} of {}, Train Loss: {:.4f} (MMSE: {:.4f} | MMD: {:.4f} | KLD: {:.4f})'.format(epoch+1, epochs, loss, MMSELoss, MMDLoss, KLDivLoss))

    return train_loss, train_loss_mmse, train_loss_mmd, train_loss_kld


# load data from csv file, woe encode categorical features (TO-DO), standardize values, make tensor with shape [n_rows, n_features]
def load_data_to_tensor(dataset_name):
    complete_data = pd.read_csv(f'../prepared_data/{dataset_name}', sep=',')
    complete_data['BAD'] = np.where(complete_data['BAD'] == 'BAD', 1, 0).astype(np.int64)

    #print(complete_data.shape)

    complete_data = pd.concat([complete_data[complete_data['BAD'] == 0].sample(complete_data['BAD'].value_counts()[1]), complete_data[complete_data['BAD'] == 1]])

    #print(complete_data.shape)

    obj_cols = complete_data.select_dtypes('object').columns
    complete_data[obj_cols] = complete_data[obj_cols].astype('category')

    # For the sake of simplicity when dealing with neural nets later, let's just make everything categorical continous
    for col in obj_cols:
        woe_calc = h.IV_Calc(complete_data, feature=col, target='BAD')
        woe = woe_calc.full_summary()['WOE_adj'].to_dict()
        complete_data[col] = complete_data[col].map(woe)
        complete_data[col] = complete_data[col].astype('float64')

    ###### TRAIN ENCODER ON SUBSET OF THE DATA ########################

    #complete_data = complete_data[complete_data['BAD'] == 0]    # Only on BAD (1) or GOOD (0)
    #print(f'Shape of Autoencoder training data: {complete_data.shape}')
    ###################################################################

    complete_X = complete_data.iloc[:, complete_data.columns != 'BAD']
    complete_y = complete_data['BAD']

    x_np = complete_X.values.reshape(-1, complete_X.shape[1]).astype('float32')
    y_np = complete_y.values.reshape(-1, 1).astype('float32')

    # stadardize values
    standardizer = StandardScaler()
    x_stand = standardizer.fit_transform(x_np)

    # we actually dont need train or test splits
    return torch.from_numpy(x_stand), torch.from_numpy(y_np)