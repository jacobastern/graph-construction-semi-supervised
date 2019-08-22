from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
from sklearn import neighbors

import numpy as np

class SpectralClassifier():
    """Takes in an affinity matrix. Obtains the Laplacian
    of that graph and the eigenvectors of the Laplacian. The eigenvectors are
    basis functions on the graph and the entry eig_vecs[pt, vec] gives the
    coordinate of 'pt' in terms of the basis function 'vec'. Thus 
    eig_vecs[pt, :] gives the representation of a point in terms of basis
    functions on the graph."""
    def __init__(self, data):
        # This is used in the embed_new_pts() method for finding the graph coordinates of new points
        # This is really just equivalent to a KNN classifier in Euclidean space
        data_reshaped = data.view(data.shape[0], -1)
        self.ngbrs = neighbors.NearestNeighbors(n_neighbors=3, metric='minkowski', p=2).fit(data_reshaped)
    
    def fit(self, 
             affinity_matrix, 
             labels, 
             n_labeled,
             n_eig_vecs=10,
             classifier_type='knn', # ['knn', 'svc']
             n_neighbors=5          # Only necessary if classifier_type == 'knn'
            ):
        
        
        if issparse(affinity_matrix):
            affinity_matrix = affinity_matrix.todense()
            
        # Symmetrize the adjacency matrix
        self.affinity_matrix = np.maximum(affinity_matrix, affinity_matrix.T)
        
        laplace = laplacian(self.affinity_matrix, normed=True)

        # eigsh gets eigenvectors and values from the graph laplacian. We get an extra 
        # eigenvector because the first vector corresponds to eigenvalue 0
        _, eig_vecs = eigsh(laplace, k=n_eig_vecs + 1, which='SM', maxiter=5000)
        # Don't consider the eigenvector corresponding to eigenvalue 0
        self.embeddings = eig_vecs[:, 1:]
    
        embeddings_labeled = self.embeddings[:n_labeled, :]
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        self.classifier.fit(embeddings_labeled, labels[:n_labeled])
        
    def predict(self, x):
        # Predict
        x_embedded = self.embed_new_pts(x)
        y_pred = self.classifier.predict(x_embedded)
        return y_pred
    
    def embed_new_pts(self, x):
        """Use this method if you want to make an inductive classifier instead of a transductive
        classifier (You will also need to write a new 'predict' function.) Finds the graph coordinates of an unseen point"""
        data_reshaped = x.view(x.shape[0], -1)
        distances, indices = self.ngbrs.kneighbors(data_reshaped)
        # Find the graph coordinates of the new point's nearest neighbors
        nearest_neighbors = self.embeddings[indices, :]
        # Let the new point be the average of its neighbors
        return np.mean(nearest_neighbors, axis=1)