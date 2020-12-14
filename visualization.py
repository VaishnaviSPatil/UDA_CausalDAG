import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx


def scatter_plot(real_ebd, fake_ebd, save_path=None):
    '''Draw the scatter plot to compare between the real and fake distributions.
    
    :param (ndarray, float64) read_ebd: 2-dim embedding of the real samples.
    :param (ndarray, float64) fake_ebd: 2-dim embedding of the fake samples.
    :param (str) save_path: path to save the figure.
    '''
    # Validation
    assert len(real_ebd.shape) == 2 and len(fake_ebd.shape) == 2 and real_ebd.shape[0] == fake_ebd.shape[0] and real_ebd.shape[1] == 2 and fake_ebd.shape[1] == 2, 'Invalid real_ebd and fake_ebd.'
    assert save_path is None or isinstance(save_path, str), 'Invalid save_path.'
    # Drawing
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout(pad=0)
    plt.scatter(real_ebd[:,0], real_ebd[:,1], label='Real', alpha=1)
    plt.scatter(fake_ebd[:,0], fake_ebd[:,1], label='Fake', alpha=1)
    plt.legend(loc='upper left', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.get_offset_text().set_size(16)
    if save_path is not None:
        plt.savefig(save_path)
        
        
def bayesnet_plot(bayesmodel, name, ax, pos):
    nx.draw_networkx_nodes(bayesmodel, ax=ax, pos=pos, node_size=6000, node_color='g', alpha=0.6)
    nx.draw_networkx_labels(bayesmodel, ax=ax, pos=pos, font_size=16, font_weight='bold')
    nx.draw_networkx_edges(bayesmodel, ax=ax, pos=pos, width=4, edge_color='k', style='solid', alpha=0.6, arrowsize=40, min_source_margin=45, min_target_margin=45)
    
def bayesnet_compare_plot(bayesmodel_true, bayesmodel_predicted, ax, pos):
    nx.draw_networkx_nodes(bayesmodel_true, ax=ax, pos=pos, node_size=800, node_color='g', alpha=0.6)
    nx.draw_networkx_labels(bayesmodel_true, ax=ax, pos=pos, font_size=16, font_weight='bold')
    nx.draw_networkx_edges(nx.intersection(bayesmodel_true, bayesmodel_predicted), ax=ax, pos=pos, width=4,  edge_color='k', style='solid', alpha=0.6, arrowsize=25)
    nx.draw_networkx_edges(nx.difference(bayesmodel_true, bayesmodel_predicted), ax=ax, pos=pos, width=4,  edge_color='r', style='dashed', alpha=0.6, arrowsize=25)
    nx.draw_networkx_edges(nx.difference(bayesmodel_predicted, bayesmodel_true), ax=ax, pos=pos, width=4, edge_color='b', style='dashed', alpha=0.6, arrowsize=25)