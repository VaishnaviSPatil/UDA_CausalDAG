import numpy as np
import multiprocessing
from MulticoreTSNE import MulticoreTSNE as TSNE
from torch_two_sample.statistics_diff import EnergyStatistic
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch


from models import Detector
from visualization import scatter_plot


def compute_tsne(samples):
    return TSNE(n_jobs=16).fit_transform(samples)


def tsne_visualization(real, fake, ndisplay=500, save_path=None):
    # Validation
    assert len(real.shape) == 3 and len(fake.shape) == 3 and real.shape == fake.shape and real.shape[0] >= ndisplay, 'Invalid real and fake samples.'
    # Computing
    idx = np.random.choice(real.shape[0], ndisplay, replace=False)
    real = real[idx]
    fake = fake[idx]
    real = real.reshape(real.shape[0], -1)
    fake = fake.reshape(fake.shape[0], -1)
    pool = multiprocessing.Pool(processes=1)
    async_result  = pool.apply_async(compute_tsne, (np.concatenate((real, fake), axis=0), ))
    real_ebd, fake_ebd = np.split(async_result.get(), 2)
    # Visualization
    scatter_plot(real_ebd, fake_ebd, save_path)


def energy_statistics(real, fake):
    # Computing
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    es_test = EnergyStatistic(real.shape[0], fake.shape[0])
    return float(es_test(torch.from_numpy(real), torch.from_numpy(fake)))


def discriminative_score(real, fake, nfolds=3, nepoches=100, batch_size=128, lr=0.001, diagnose=False):
    # Pre-processing
    nbatches = (real.shape[0]//nfolds*(nfolds-1))//batch_size
    aucs = []
    count = 0
    # Training
    for fold, split in enumerate(KFold(n_splits=nfolds).split(real)):
        train_index, test_index = split
        X_train = Variable(torch.from_numpy(np.concatenate((real[train_index], fake[train_index]), axis=0).astype(np.float32))).cuda()
        y_train = Variable(torch.from_numpy(np.concatenate((np.zeros(len(train_index)), np.ones(len(train_index))), axis=0).astype(np.float32))).cuda()
        X_test = Variable(torch.from_numpy(np.concatenate((real[test_index], fake[test_index]), axis=0).astype(np.float32)), requires_grad=False).cuda()
        y_test = np.concatenate((np.zeros(len(test_index)), np.ones(len(test_index))), axis=0).astype(np.float32)
        detector = Detector(length=real.shape[1]).cuda()
        opt = optim.Adam(detector.parameters(), lr=lr, betas=(0.9, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        for epoch in range(nepoches):
            idx = torch.randperm(X_train.size(0))
            X_train = X_train[idx]
            y_train = y_train[idx]
            for batch in range(nbatches):
                X = X_train[batch*batch_size:(batch+1)*batch_size]
                y = y_train[batch*batch_size:(batch+1)*batch_size]
                opt.zero_grad()
                yh = detector(X)
                loss = criterion(yh, y)
                loss.backward()
                opt.step()        
                losses.append(float(loss.data.cpu().numpy()))
        if diagnose:
            plt.plot(losses)
            plt.show()
        # Evaluation
        yh_tests = []
        for batch in range(X_test.size(0)//batch_size):
            yh_tests.append(detector(X_test[batch*batch_size:(batch+1)*batch_size, :]).cpu().detach().numpy())
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:(X_test.size(0)//batch_size)*batch_size], np.concatenate(yh_tests, axis=0), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc = 0.5 + np.abs(auc-0.5)
        aucs.append(auc)
        torch.cuda.empty_cache()
        count += 1
    return float(np.mean(np.array(aucs)))


def relative_structure_score(real, fake, encoder, bayesmodel_true, score_type='bic'):
    Score = {'bic': BicScore, 'k2': K2Score, 'bdeu': BDeuScore}[score_type]
    real, fake = encoder.inverse_transform(real), encoder.inverse_transform(fake)
    nodes = list(bayesmodel_true.nodes())
    idx_to_node = dict(list(zip(range(len(nodes)), nodes)))
    real, fake = pd.DataFrame(real).rename(columns=idx_to_node), pd.DataFrame(fake).rename(columns=idx_to_node)
    return Score(fake).score(bayesmodel_true) - Score(real).score(bayesmodel_true)
    
    
def structure_prediction(samples, encoder, bayesmodel_true, method='hc', score_type='bic'):
    Score = {'bic': BicScore, 'k2': K2Score, 'bdeu': BDeuScore}[score_type]
    samples = encoder.inverse_transform(samples)
    nodes = list(bayesmodel_true.nodes())
    idx_to_node = dict(list(zip(range(len(nodes)), nodes)))
    samples = pd.DataFrame(samples).rename(columns=idx_to_node)
    if method == 'ex':
        bayesmodel_predicted = ExhaustiveSearch(samples, scoring_method=Score(samples)).estimate()
    else:
        bayesmodel_predicted = HillClimbSearch(samples, scoring_method=Score(samples)).estimate(start_dag=bayesmodel_true.copy(), show_progress=False)
    return bayesmodel_predicted


def structure_prediction_accuracy(bayesmodel_true, bayesmodel_predicted):
    #ntotal = len(bayesmodel_true.nodes()) * (len(bayesmodel_true.nodes()) -1) // 2
    nerror = len(set(bayesmodel_true.edges()).symmetric_difference(set(bayesmodel_predicted.edges())))
    return nerror


def bayesmodel_predicted_majority_vote(list_of_bayesmodel_predicted):
    ntrails = len(list_of_bayesmodel_predicted)
    all_edges = []
    for bayesmodel_predicted in list_of_bayesmodel_predicted:
        all_edges.extend(bayesmodel_predicted.edges())
    edges = []
    for edge in set(all_edges):
        if all_edges.count(edge) > ntrails/2:
            edges.append(edge)
    DAG = nx.DiGraph()
    DAG.add_nodes_from(list(bayesmodel_predicted.nodes()))
    DAG.add_edges_from(edges)
    return DAG