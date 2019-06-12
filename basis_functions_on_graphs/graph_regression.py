import sklearn
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import time
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from numpy.linalg import eig
from umap import UMAP
from numpy import linalg as la
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)

def gen_clusters():
    cluster_1 = np.array([                          
                         [ -0.96465601,   7.90195605,   7.91667933],
#                          [  5.92186477,  17.20470878,   3.33419128],
                         [  5.61553006,   3.86016425,   4.01646273],
#                          [  6.06924842,  14.45108138,  -0.90114994],
#                          [  5.31662539,  14.02265292,  -2.44546148],
#                          [  1.61602595,  15.25780482,  -4.77270065],
                         [  6.15892401,   2.73288038,  -0.59218982],
                         [  2.25183243,   6.05179238,  -4.64676804],
#                          [  2.96607724,  10.74708493,  -4.39909157],
                         [  4.91677882,   2.42326792,  -2.96340473],
#                          [ -5.81975595,  16.27190175,  -8.62135923],
                         [ -6.56370511,   7.13848534,   5.72215369],
#                          [  4.9037716,   18.70935975,  -2.97833649],
#                          [ -6.82298779,  18.74307128,   5.47567277],
#                          [  6.05882398,  17.87794456,   2.94164237],
#                          [  4.43224617,  20.61525508,   5.6610451 ],
#                          [  4.06592216,  13.17889506,  -3.7586063 ],
                         ])
    cluster_2 = np.array([
                         [ 11.38920717,  14.92140348,   6.43324725],
                         [  2.1079105,   20.71412712,  13.82611173],
                         [ 12.60543362,   19.19465939,   1.10891728],
                         [  5.80171168,   18.06536926,  -9.95611463],
#                          [  5.44297002,  10.66865137,  12.60452393],
#                          [  2.49781065,   2.13921709, -10.93851125],
                         [  1.36262826,  17.20606583, -11.03462404],
#                          [  5.78503884,   6.17280578,  12.42006589],
                         [  7.90516102,  16.82515341,  -8.67248813],
#                          [ 11.60912289,   6.34596448,  -3.8887993 ],
                         [ 10.81257788,  14.72056388,  -5.43290428],
                         [ 11.08920938,  11.75993012,  -4.95555382],
#                          [ 11.96864371,   3.16542611,  -2.94330539],
                         [  5.37231179,  13.67154943,  12.64107991],
                         ])
    
    return cluster_1, cluster_2

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def graph_regression(dataset='swiss_roll', n_samples=1000, n_eig_vecs=10, graph='knn', n_neighbors=10):
    """Conducts regression on a graph manifold using sums of basis functions on
    the graph
    """
    if dataset == 'swiss_roll':
        data, _ = datasets.make_swiss_roll(n_samples=n_samples)
    elif dataset == 'sparse_line':
        data = sparse_line(n_samples=n_samples)

#     num_labeled_pts = 10
        
#     # Select two points to be centers of clusters
#     cluster_point_1 = data[0, :]
#     cluster_point_2 = data[1, :]
    
#     # Generate a labeled cluster around each point
#     cluster_1 = np.random.multivariate_normal(cluster_point_1, .1 * np.eye(3), num_labeled_pts//2)
#     cluster_2 = np.random.multivariate_normal(cluster_point_2, .1 * np.eye(3), num_labeled_pts//2)
    
    cluster_1, cluster_2 = gen_clusters()
    len_cluster_1 = len(cluster_1)
    len_cluster_2 = len(cluster_2)
    num_labeled_pts = len_cluster_1 + len_cluster_2
    
    # Append new clusters to data
    data = np.vstack((data, cluster_1, cluster_2))
    labels = np.hstack((.5*np.ones(n_samples), np.ones(len_cluster_1), np.zeros(len_cluster_2)))

    # Regress label to Euclidean coordinates
    A = np.vstack((cluster_1, cluster_2))
    b = labels[-num_labeled_pts:]
    x_hat = la.solve(A.T @ A, A.T @ b)
    pred_labels = data @ x_hat
    
    print("Euclidean Regression")
    print(labels[-num_labeled_pts:])
    print(pred_labels[-num_labeled_pts:])
#     print_full(np.hstack((data, pred_labels.reshape(-1, 1))))
#     print(f"Min Predicted: {min(pred_labels)}, Max Predicted: {max(pred_labels)}")
    
    # Plot original labels
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=labels,
               vmin=min(pred_labels),
               vmax=max(pred_labels),
               s=20, 
               cmap='jet',
               edgecolor='k')
    plt.title("Original Labels")
    
    # Plot predicted labels
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=pred_labels,
               s=20, 
               cmap='jet',
               edgecolor='k')
    plt.title("Regression to Euclidean Coordinates")
    
    # Regress label to coordinates in terms of graph-basis functions
    
    # Make the graph
    if graph == 'knn':
        adj_matrix = sklearn.neighbors.kneighbors_graph(
            data, 
            n_neighbors, 
            mode='connectivity',
            metric='minkowski', 
            p=2, 
            metric_params=None, 
            include_self=False, 
            n_jobs=None
        )
    else:
        umap = UMAP(n_components=2, n_neighbors=n_neighbors)
        umap.fit_transform(data)
        adj_matrix = umap.graph_

    # Symmetrize the adjacency matrix
    adj_matrix = np.maximum( adj_matrix.todense(), adj_matrix.todense().T )

    laplace = laplacian(adj_matrix, normed=True)

    # eigsh gets eigenvectors and values from the graph laplacian
    eig_vals, eig_vecs = eigsh(laplace, k=n_eig_vecs, which='SM')
    
    # Regress label to graph eigenfunction coordinates
    A = eig_vecs[-num_labeled_pts:, :]
    b = labels[-num_labeled_pts:]
    x_hat = la.solve(A.T @ A, A.T @ b)
    pred_labels = eig_vecs @ x_hat
    
#     print("Graph Regression")
#     print(labels[-num_labeled_pts:])
#     print(pred_labels[-num_labeled_pts:])
    
    # Plot original labels
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=labels,
               vmin=min(pred_labels),
               vmax=max(pred_labels),
               s=20, 
               cmap='jet',
               edgecolor='k')
    plt.title("Original Labels")
    
    # Plot predicted labels
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               c=pred_labels,
               s=20, 
               cmap='jet',
               edgecolor='k')
    plt.title("Regression to Graph Basis Function Coordinates")
    
    plt.show()
