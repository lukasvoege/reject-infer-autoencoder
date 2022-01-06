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
    def __init__(self, features):
        super(Autoencoder, self).__init__()

        self.enc1 = nn.Linear(in_features = features, out_features=15)
        self.enc2 = nn.Linear(in_features = 15, out_features=6)

        self.dec2 = nn.Linear(in_features = 6, out_features = 15)
        self.dec1 = nn.Linear(in_features = 15, out_features = features)

    def forward(self, x):
        x = self.decode(self.encode(x))
        return x

    def encode(self, x):
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        return x
        
    def decode(self, x):
        x = nn.functional.relu(self.dec2(x))
        x = nn.functional.relu(self.dec1(x))
        return x



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



######################################################################################################################
######################################################################################################################


BATCH_SIZE = 1000

dataset = CreditscoringDataset("gmsc.csv")      # load and prepare Dataset to Tensor
data_loader = DataLoader(                       # create Dataloader for batching
    dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)


net = Autoencoder(dataset.x.shape[1])
print(net)
net.to("cpu")

train_loss = train(net, data_loader, 50)

plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()