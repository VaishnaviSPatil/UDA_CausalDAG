import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
        
    def forward(self, x):
        return x.view(self.shape)

    
class Generator(nn.Module):
    def __init__(self, nfeatures, ndomain=ndomain):
        super(Generator, self).__init__()
        self.latent_dim = len(nfeatures)
        self.input_dim = self.latent_dim + nfeatures.sum() + ndomain
        self.hidden_dim = 64
        self.output_dim = nfeatures.sum()
        self.length = len(nfeatures)
        self.feature_idxs = np.split(np.arange(nfeatures.sum()), np.cumsum(nfeatures)[:-1])
        model = [nn.Linear(self.latent_dim, self.hidden_dim, bias=True), nn.BatchNorm1d(self.hidden_dim), nn.LeakyReLU(0.01)]
        model += [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.BatchNorm1d(self.hidden_dim), nn.LeakyReLU(0.01)] * 2
        model += [nn.Linear(self.hidden_dim, self.output_dim, bias=True)]
        self.nn = nn.Sequential(*model)

    def forward(self, input_data, tau):
        x = self.nn(input_data)
        x = torch.cat([F.gumbel_softmax(x[:,self.feature_idxs[node]], tau=tau, hard=True) for node in range(self.length)], 1)
        return x
    
    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
    

class Discriminator(nn.Module):
    def __init__(self, local_nfeatures, ndomain):
        super(Discriminator, self).__init__()
        self.input_dim = local_nfeatures
        self.hidden_dim = 16
        self.output_dim = ndomain
        model = [nn.Linear(self.input_dim, self.hidden_dim, bias=True), nn.LeakyReLU(0.01)]
        model += [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.LeakyReLU(0.01)] * 1
        model += [nn.Linear(self.hidden_dim, self.output_dim, bias=True)]
        self.nn = nn.Sequential(*model)

    def forward(self, input_data):
        x = self.nn(input_data)
        x = F.gumbel_softmax(x, tau=tau, hard=True)
        return x
    
    
class Classifier(nn.Module):
    def __init__(self, nfeatures, nclass):
        super(Classifier, self).__init__()
        self.input_dim = nfeatures.sum()
        self.hidden_dim = 64
        self.output_dim = nclass
        model = [nn.Linear(self.input_dim, self.hidden_dim, bias=True), nn.LeakyReLU(0.01)]
        model += [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.LeakyReLU(0.01)] * 2
        model += [nn.Linear(self.hidden_dim, self.output_dim, bias=True)]
        self.nn = nn.Sequential(*model)

    def forward(self, input_data):
        x = self.nn(input_data).squeeze()
        x = F.gumbel_softmax(x, tau=tau, hard=True)
        return x