import time
import numpy as np


def modified_network(network, dist):
    np.random.seed(int(time.time()))
    old_network = network
    network = network.copy()
    length = len(network)
    for _ in range(dist):
        while True:
            i = np.random.choice(length)
            j = np.random.choice(length)
            if i!=j and j in network[i]:
                tmp = network[i].copy()
                tmp.remove(j)
                network[i] = tmp
                break
        while True:
            i = np.random.choice(length)
            j = np.random.choice(length)
            if i != j and j not in network[i] and j not in old_network[i]:
                tmp = network[i].copy()
                tmp.append(j)
                network[i] = tmp
                break
    return network


def random_network(network):
    np.random.seed(int(time.time()))
    length = len(network)
    p = 2 * (sum([len(local) for local in network]) - length) / length / (length -1)
    idx = np.random.permutation(length)
    network = []
    for i in range(length):
        local = [idx[i]]
        for j in range(i+1, length):
            if np.random.uniform() < p:
                local.append(idx[j])
        network.append(local)
    return network