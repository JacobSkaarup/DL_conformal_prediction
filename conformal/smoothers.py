from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class BaseSmoother(BaseEstimator):
    """Interface: Learns patterns in Calibration, Smooths scores in Test."""
    def fit(self, X, y, scores):
        return self._fit_internal(X, y, scores)
    
    def predict_smooth(self, X):
        # Should return the 'signal' (e.g. mean score of neighbors/cluster)
        return self._predict_internal(X)

    def _fit_internal(self, X, scores):
        raise NotImplementedError("Child class must implement _fit_internal")
    
    def _predict_internal(self, X):
        raise NotImplementedError("Child class must implement _predict_internal")
    
class ClusterSmoother(BaseSmoother):
    """
    For K-Means, GMMs, Decision Trees.
    Treats the input model as a discrete grouper.
    """
    def __init__(self, clustering_model):
        self.clustering_model = clustering_model
        self.means_ = {}
        self.global_mean_ = 0

    def _fit_internal(self, X, y, scores):
        # 1. Fit the clustering model (Pipeline, KMeans, Tree, etc.)
        if hasattr(self.clustering_model, "fit_predict"):
            clusters = self.clustering_model.fit_predict(X)
        else:
            self.clustering_model.fit(X, y)
            if hasattr(self.clustering_model, "apply"):
                clusters = self.clustering_model.apply(X)
            else:
                clusters = self.clustering_model.predict(X)
        
        # 2. Learn Means
        self.global_mean_ = np.mean(scores)
        for c in np.unique(clusters):
            self.means_[c] = np.mean(scores[clusters == c])
            
        return self

    def _predict_internal(self, X):
        # 1. Get Cluster IDs for Test Data
        if hasattr(self.clustering_model, "apply"):
            clusters = self.clustering_model.apply(X)
        else:
            clusters = self.clustering_model.predict(X)
            
        # 2. Map IDs to Learned Means
        # Use a vectorized map or list comprehension for speed
        # Fallback to global_mean_ if a test cluster was never seen in calibration
        signal = np.array([self.means_.get(c, self.global_mean_) for c in clusters])
        return signal

class KNNSmoother(BaseSmoother):
    """
    For KNN. 
    Instead of discrete clusters, we use the average score of 
    the k-nearest neighbors in the Calibration set.
    """
    def __init__(self, knn_regressor_):
        self.knn_regressor_ = knn_regressor_

    def _fit_internal(self, X, y, scores):
        self.knn_regressor_.fit(X, scores)
        return self

    def _predict_internal(self, X):
        return self.knn_regressor_.predict(X)