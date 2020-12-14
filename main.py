import numpy as np
import pickle
import torch
from torch.autograd import Variable
from torch.nn import ModuleList
import matplotlib.pyplot as plt
from IPython import display
import time
import pandas as pd
import networkx as nx

from dataloaders import *
from models import Generator, Discriminator
from train import Trainer
from evaluation import *
from visualization import *
from topology import *
    

def main(mode, dataset_name, expid, epochs, lr, batch_size, ntrials, struc_learn_method):
    network_dict = {}
    result_dict = {}
    losses_dict = {}
    bayesmodel_dict = {}

    # Get network
    network = get_network(dataset_name)
    length = len(network)
    bayesmodel, layout, encoders, nfeatures, nclass = get_bayesnet_model(dataset_name)

    # Obtain models
    generator = Generator(nfeatures=nfeatures, ndomain=ndomain)
    discriminator = ModuleList([Discriminator(local_nfeatures=nfeatures[local].sum(), ndomain=ndomain) for local in network])
    classifier = Classifier(nfeatures=nfeatures, nclass=nclass)

    # Obtain dataloader
    data_loader, test_data_loader = get_bayesnet_dataloaders(dataset_name=dataset_name, length=length, batch_size=batch_size)

    # Initialize optimizers
    G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(.9, .99))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(.9, .99))
    G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=1)

    # Set up trainer
    expname = '{}'.format(dataset_name.upper())
    trainer = Trainer(mode, network, bayesmodel, layout, encoder, nfeatures, expname, generator, discriminator, G_optimizer, D_optimizer, G_scheduler)

    # Train model
    expdesc = "{}-e{}lr{}bs{}pe{}nt{}-{}".format(mode, epochs, int(10000*lr), batch_size, pretrain_epochs, ntrials, expid)
    generator_state_dict = trainer.train(data_loader, test_data_loader, expdesc, epochs=epochs)

    # Get result
    display.clear_output(wait=True)
    #generator.load_state_dict(generator_state_dict)
    generator.eval()
    result_dict = {}
    ess, dss, rsss, bmps, spas = [], [], [], [], []
    for trial in range(ntrials):
        iter_test_data = iter(test_data_loader)
        sampled_data = [next(iter_test_data).data.numpy() for i in range(16)]
        sampled_data = np.concatenate(sampled_data, axis=0)
        fixed_latents = Variable(generator.sample_latent(sampled_data.shape[0]))
        generated = generator(fixed_latents.cuda(), 0.1).detach().cpu().data.numpy()
        ess.append(energy_statistics(sampled_data, generated))
        dss.append(discriminative_score(sampled_data, generated, diagnose=False))
        rsss.append(relative_structure_score(sampled_data, generated, encoder, bayesmodel))
        bmps.append(structure_prediction(generated, encoder, bayesmodel, method=struc_learn_method, score_type='bic'))
        spas.append(structure_prediction_accuracy(bayesmodel, bmps[-1]))
        print("Evaluating... Progress {:.2f}%".format((trial+1)/ntrials*100), end='\r')
    ess, dss, rsss, spas = np.array(ess), np.array(dss), np.array(rsss), np.array(spas)
    result_dict['energy_statistics'] = (ess.mean(), ess.std()/np.sqrt(ntrials))
    result_dict['discriminative_score'] = (dss.mean(), dss.std()/np.sqrt(ntrials))
    result_dict['relative_structure_score'] = (rsss.mean(), rsss.std()/np.sqrt(ntrials))
    bayesmodel_dict = bmps[0]
    result_dict['structure_prediction_accuracy'] = (spas.mean(), spas.std()/np.sqrt(ntrials))
    losses_dict = {}
    losses_dict['wasserstein_loss'] = trainer.losses['D']
    losses_dict['energy_statistics'] = trainer.losses['energy_statistics']
    losses_dict['relative_structure_score'] = trainer.losses['relative_structure_score']
    losses_dict['structure_prediction_iteration'] = trainer.losses['structure_prediction_iteration']
    losses_dict['structure_prediction_accuracy'] = trainer.losses['structure_prediction_accuracy']
    losses_dict['bayesmodel_predicted'] = trainer.losses['bayesmodel_predicted']
    network_dict = network