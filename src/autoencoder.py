import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



# Custom Dataset class, that loads the standardized and prepared data from function load_data_to_tensor()
class CreditscoringDataset(Dataset):
    def __init__(self, dataset_name):
        self.x = load_data_to_tensor(dataset_name)
        
    def __getitem__(self,index):      
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


# Autoencoder
class Autoencoder(nn.Module):       ## parametrisieren!
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
            x = nn.functional.relu(e(x))
        return x
        
    def decode(self, x):
        for d in self.dec:
            x = nn.functional.relu(d(x))
        return x


# train any net
def train(net, trainloader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=1e-3)

    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, epochs, loss))

    return train_loss


# load data from csv file, woe encode categorical features (TO-DO), standardize values, make tensor with shape [n_rows, n_features]
def load_data_to_tensor(dataset_name):
    complete_data = pd.read_csv(f'../prepared_data/{dataset_name}', sep=',')
    complete_data['BAD'] = np.where(complete_data['BAD'] == 'BAD', 1, 0).astype(np.int64)

    obj_cols = complete_data.select_dtypes('object').columns
    complete_data[obj_cols] = complete_data[obj_cols].astype('category') ## woe an der stelle

    complete_X = complete_data.iloc[:, complete_data.columns != 'BAD']

    x_np = complete_X.values.reshape(-1, complete_X.shape[1]).astype('float32')

    # stadardize values
    standardizer = StandardScaler()
    x_stand = standardizer.fit_transform(x_np)

    # we actually dont need train or test splits and also no y, just x
    return torch.from_numpy(x_stand)