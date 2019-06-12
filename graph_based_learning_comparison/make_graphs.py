# Local imports
# ML imports
from sklearn import neighbors
from torchvision.datasets import FashionMNIST, MNIST
from umap_mod import UMAP

# General imports
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('obj/'):
    os.makedirs('obj/')

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def make_graphs(dataset, dataset_name, graphs, n_neighbors, metric='euclidean', save=False, outdir='graphs'):
    """Makes the specified graphs from the dataset.
    Args:
        dataset (Dataset): a torch.utils.data.Dataset object with -1's as labels for unlabeled points
        graphs (list): a list of strings representing the desired graphs ['umap', 'knn', 'tpwrls']
    Returns:
        (dict): a dictionary containing the graphs as matrices
        (dict): a dictionary containing the graph creation times
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    graph_dict = dict({'umap' : None, 'knn' : None, 'tpwrls' : None})
    times = dict({'umap' : None, 'knn' : None, 'tpwrls' : None})
    for graph in graphs:
        
        if graph == 'umap':
            # Right now knn from umap is returning a totally different graph from k_neighbors, so this is commented out for now
#             if 'knn' in graphs:
#                 graph_dict['umap'], graph_dict['knn'], times['umap'], times['knn'] = get_umap_graph(dataset, metric=metric, n_neighbors=n_neighbors, knn=True)
#                 np.save(os.path.join(outdir, f'{dataset_name}_umap.npy'), graph_dict['umap'].todense())
#                 np.save(os.path.join(outdir, f'{dataset_name}_knn_from_umap.npy'), graph_dict['knn'].todense())
#             else:
            graph_dict['umap'], times['umap'] = get_umap_graph(dataset, metric=metric, n_neighbors=n_neighbors, knn=False)
            np.save(os.path.join(outdir, f'{dataset_name}_umap.npy'), graph_dict['umap'].todense())
        elif graph == 'knn':
#             if 'umap' in graphs:
#                 pass
#             else:
            graph_dict['knn'], times['knn'] = get_knn_graph(dataset, metric=metric, n_neighbors=n_neighbors, include_self=False)
            np.save(os.path.join(outdir, f'{dataset_name}_knn_kneighbors.npy'), graph_dict['knn'].todense())
        else:
            assert graph == 'tpwrls'
            graph_dict['tpwrls'], times['tpwrls'] = get_tpwrls_graph(dataset, metric=metric, n_neighbors=n_neighbors)
            np.save(os.path.join(outdir, f'{dataset_name}_tpwrls.npy'), graph_dict['tpwrls'].todense())
            
    save_obj(times, 'graph_creation_times')

    return graph_dict, times
            
def get_knn_graph(dataset, metric, n_neighbors, include_self=False):
    
    num_items = len(dataset)
    data = dataset.data
#     print(data[0, :15, :15])
    # Reshape the data to be compatible with kneighbors_graph
    data_reshaped = data.view(num_items, -1)
    
    start_time = time.time()
    adj_matrix = neighbors.kneighbors_graph(
        data_reshaped, 
        n_neighbors, 
        mode='connectivity',
        metric=metric,
        metric_params=None, 
        include_self=include_self, 
        n_jobs=None
        )
    
#     plt.imshow(adj_matrix[:100, :100].todense())
    return adj_matrix, time.time() - start_time

def get_umap_graph(dataset, metric, n_neighbors, knn=False):
    
    num_items = len(dataset)
    data = dataset.data
    data_labels = dataset.labels
    data_reshaped = data.view(num_items, -1)
    
    if knn:
        umap = UMAP(metric=metric, n_neighbors=n_neighbors)
        umap_graph, knn_graph, umap_time, knn_time = umap.get_graphs(data_reshaped, data_labels, knn=knn)
        
        return umap_graph, knn_graph, umap_time, knn_time

    else:
        umap = UMAP(metric=metric, n_neighbors=n_neighbors)
        umap_graph, _, umap_time, _ = umap.get_graphs(data_reshaped, data_labels, knn=knn)
        return umap_graph, umap_time
    
def get_tpwrls_graph(dataset, metric, n_neighbors, knn=False):
    return None, None