import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from topology import *


class BayesNetDataset(Dataset):
    def __init__(self, dataset_name, is_train=True):
        if is_train:
            self.x = np.load('./data/{}_x_train.npy'.format(dataset_name))
            self.y = np.load('./data/{}_y_train.npy'.format(dataset_name))
            self.d = np.load('./data/{}_d_train.npy'.format(dataset_name))
        else:
            self.x = np.load('./data/{}_x_test.npy'.format(dataset_name))
            self.y = np.load('./data/{}_y_test.npy'.format(dataset_name))
            self.d = np.load('./data/{}_d_test.npy'.format(dataset_name))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (torch.from_numpy(self.x[idx,:]).float(), torch.from_numpy(self.y[idx,:]).float(), torch.from_numpy(self.d[idx,:]).float())

def get_bayesnet_dataloaders(dataset_name, length, batch_size):
    # Create dataloaders
    train_loader = DataLoader(BayesNetDataset(dataset_name, is_train=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(BayesNetDataset(dataset_name, is_train=False), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_network(dataset_name):
    with open('./data/{}_network.pkl'.format(dataset_name), 'rb') as f:
        network = pickle.load(f)
    return network

def get_bayesnet_model(dataset_name):
    with open('./data/{}_model.pkl'.format(dataset_name), 'rb') as f:
        bayesmodel = pickle.load(f)
    with open('./data/{}_layout.pkl'.format(dataset_name), 'rb') as f:
        layout = pickle.load(f)
    with open('./data/{}_encoders.pkl'.format(dataset_name), 'rb') as f:
        encoders = pickle.load(f)
    nfeatures = np.load('./data/{}_nfeatures.npy'.format(dataset_name))
    nclass = np.load('./data/{}_nclass.npy'.format(dataset_name))
    ndomain = np.load('./data/{}_ndomain.npy'.format(dataset_name))
    return bayesmodel, layout, encoders, nfeatures, nclass, ndomain