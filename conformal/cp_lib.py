from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. PyTorch Adapter
# ==========================================

class TorchAdapter(BaseEstimator, ClassifierMixin):
    """
    Adapts a trained PyTorch nn.Module to conform to the Scikit-Learn classifier interface.
    """
    def __init__(self, model, classes, device='cpu'):
        self.model = model
        self.classes = classes
        self.classes_ = np.array(classes)
        self.device = device
        
        # Ensure the model is on the correct device
        self.model.to(self.device)

    def fit(self, X, y=None):
        """
        For compatibility with the Scikit-Learn interface.
        """
        return self

    def predict_logits(self, X):
        """Returns raw logits from the model."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(X_tensor)
            return logits.cpu().numpy()

    def predict_proba(self, X):
        """Returns softmax probabilities."""
        logits_np = self.predict_logits(X)
        
        # Convert logits back to tensor to apply softmax efficiently
        logits_tensor = torch.from_numpy(logits_np)
        probabilities = nn.functional.softmax(logits_tensor, dim=1)
        return probabilities.numpy()

    def predict(self, X):
        """Returns the predicted class label."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    


# ==========================================
# 2. BASE CONFORMAL CLASS
# ==========================================
class BaseConformalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, head_model, alpha=0.1, allow_zero_sets=True, rand=True):
        self.head_model = head_model   # Final fully connected layer model
        self.alpha = alpha
        self.allow_zero_sets = allow_zero_sets
        self.rand = rand
        self.q_hat = None
        self.classes_ = None

    def _get_scores(self, probs, y=None):
        raise NotImplementedError

    def fit(self, X_embeddings, y):
        # 1. Get Probabilities from the Head Model
        cal_probs = self.head_model.predict_proba(X_embeddings)
        self.classes_ = self.head_model.classes_
        
        # 2. Calculate Scores
        scores = self._get_scores(cal_probs, y)
        
        # 3. Compute Quantile
        n = len(y)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, min(q_level, 1.0), method='higher')
        return self

    def predict(self, X_embeddings):
        check_is_fitted(self, "q_hat")
        test_probs = self.head_model.predict_proba(X_embeddings)
        
        # Get Matrix Scores
        scores_matrix = self._get_scores(test_probs, y=None)
        
        # Threshold
        mask = scores_matrix <= self.q_hat
        self.mask = mask

        # Finalize (if Zero Sets are not allowed)
        if not self.allow_zero_sets:
            empty_rows = np.where(mask.sum(axis=1) == 0)[0]
            if len(empty_rows) > 0:
                top_class = test_probs[empty_rows].argmax(axis=1)
                mask[empty_rows, top_class] = True
                
        return mask
    
    def convert_to_sets(self, labels=None):
        """
        Converts a boolean mask into a list of lists containing the actual labels.
        
        Parameters:
        -----------
        labels : list or np.array, optional
            A list of class names (e.g., ['cat', 'dog', 'fish']).
            If None, returns the numerical indices (e.g., [0, 1]).
            
        Returns:
        --------
        list of lists
            e.g. [['cat', 'dog'], ['fish'], []]
        """
        # Convert to indices
        prediction_indices = [np.where(row)[0] for row in self.mask]
        
        # Map to indices if no labels provided
        if labels is None:
            return [idxs.tolist() for idxs in prediction_indices]
        
        # Map indices to labels
        labels = np.array(labels) 
        return [labels[idxs].tolist() for idxs in prediction_indices]
    
    def compute_metrics(self, y_true):
        """
        Computes key metrics for conformal prediction sets.
        Parameters:
        -----------
        y_true : np.array of shape (n_samples,)
            True class labels.
            
        Returns:
        --------
        dict
            returns a dictionary with keys:
            - "Empirical coverage": Fraction of times true label is in set
            - "Efficiency": Average size of prediction sets
            - "Singleton rate": Fraction of sets with size 1
            - "Singleton Hit ratio": Coverage when set size is 1
            - "Class Conditional Coverage": Coverage per class
            - "Max Class Conditional Deviation": Max absolute deviation from nominal coverage
        """
        n = len(y_true)
        
        # 1. Empirical coverage (Fraction of times true label is in set)
        covered = self.mask[np.arange(n), y_true]
        coverage = np.mean(covered)
        
        # 2. Efficiency (Average Set Size)
        set_sizes = np.sum(self.mask, axis=1)
        avg_size = np.mean(set_sizes)
        
        # 3. Singleton rate (Fraction of sets with size 1)
        is_singleton = (set_sizes == 1)
        singleton_rate = np.mean(is_singleton)
        
        # 4. Singleton Hit ratio (Coverage with set size == 1)
        if np.sum(is_singleton) > 0:
            singleton_hit = np.mean(covered[is_singleton])
        else:
            singleton_hit = 0.0

        # 5. Class Conditional Coverage
        cond_cov = np.zeros(len(np.unique(y_true)))

        for label in np.unique(y_true):
            idx = y_true == label
            cond_cov[label] = self.mask[idx, label].mean()

        # 6. Maximum absolute deviation from nominal coverage
        max_abs_dev = np.abs(cond_cov - (1 - self.alpha)).max()

        return {
            "Empirical coverage": coverage,
            "Efficiency": avg_size,
            "Singleton rate": singleton_rate,
            "Singleton Hit ratio": singleton_hit,
            "Class Conditional Coverage": cond_cov,
            "Max Class Conditional Deviation": max_abs_dev
        }

# ==========================================
# 3. CONFORMAL METHODS (TPS, APS, RAPS, DAPS)
# ==========================================

class TPS(BaseConformalClassifier):
    def _get_scores(self, probs, y=None):
        n = len(probs)
        # Simple TPS Score: 1 - P(y_true)
        if y is None:
            scores = 1 - probs
        else:
            scores = 1 - probs[np.arange(n), y]
        return scores


class RAPS(BaseConformalClassifier):
    """
    Regularized Adaptive Prediction Sets. 
    APS is just RAPS with k_reg=0 and lam_reg=0.
    """
    def __init__(self, head_model, alpha=0.1, lam_reg=0.01, k_reg=2, allow_zero_sets=True, rand=True):
        super().__init__(head_model, alpha, allow_zero_sets, rand)
        self.lam_reg = lam_reg
        self.k_reg = k_reg

    def _get_scores(self, probs, y=None):
        n = len(probs)
        pi = probs.argsort(1)[:, ::-1]
        srt = np.take_along_axis(probs, pi, axis=1)
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        
        if y is not None: # Calibration
            cal_true_loc = np.where(pi == y[:, None])[1]
            # Score = CumSum - U*Prob
            scores = (srt + reg_vec).cumsum(axis=1)[np.arange(n), cal_true_loc] - u * (srt + reg_vec)[np.arange(n), cal_true_loc]
            return scores
        else: # Prediction
            score_matrix_sorted = (srt + reg_vec).cumsum(axis=1) - (u[:, None] * (srt + reg_vec))
            score_matrix_original = np.zeros_like(score_matrix_sorted)
            np.put_along_axis(score_matrix_original, pi, score_matrix_sorted, axis=1)
            return score_matrix_original


class DAPS(RAPS): # Inherit reuse _get_scores logic from RAPS
    def __init__(self, head_model, smoother, alpha=0.1, lam_reg=0.01, k_reg=2, beta=0.2, allow_zero_sets=True, rand=True):
        super().__init__(head_model, alpha, lam_reg, k_reg, allow_zero_sets, rand)
        self.smoother = smoother
        self.beta = beta

    def fit(self, X_embeddings, y):
        # 1. Standard Fit (calculates q_hat based on pure RAPS)        
        cal_probs = self.head_model.predict_proba(X_embeddings)
        self.classes_ = self.head_model.classes_
        
        # Get Raw RAPS scores
        raw_scores = self._get_scores(cal_probs, y)

        # 2. Fit Smoother on desired features
        self.smoother.fit(X_embeddings, y, raw_scores)
        
        # 3. Diffuse using signal
        signal = self.smoother.predict_smooth(X_embeddings)
        diff_scores = (1 - self.beta) * raw_scores + self.beta * signal

        # 4. Compute quantile
        n = len(y)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(diff_scores, min(q_level, 1.0), method='higher')
        return self

    def predict(self, X_embeddings):
        check_is_fitted(self, "q_hat")
        test_probs = self.head_model.predict_proba(X_embeddings)
        
        # 1. Get Base Matrix Scores
        matrix_scores = self._get_scores(test_probs, y=None)
        
        # 2. Get Signal
        signal = self.smoother.predict_smooth(X_embeddings)
        
        # 3. Diffuse Scores
        diffused_matrix = (1 - self.beta) * matrix_scores + self.beta * signal[:, None]
        
        # 4. Threshold & Finalize
        mask = diffused_matrix <= self.q_hat
        self.mask = mask
        if not self.allow_zero_sets:
            empty_rows = np.where(mask.sum(axis=1) == 0)[0]
            if len(empty_rows) > 0:
                top_class = test_probs[empty_rows].argmax(axis=1)
                mask[empty_rows, top_class] = True
        return mask

# ==========================================
# 4. FUNCTIONS
# ==========================================

def efficiency_scorer(y_true, y_pred_sets, target_alpha=0.1):
    """
    Custom scoring function for Conformal Prediction.
    Goal: Minimize Average Set Size with heavy penalty if Coverage is violated.
    """
    # 1. Calculate Empirical Coverage
    n = len(y_true)
    covered = y_pred_sets[np.arange(n), y_true]
    coverage = np.mean(covered)
    
    # 2. Calculate Average Set Size
    set_sizes = np.sum(y_pred_sets, axis=1)
    avg_size = np.mean(set_sizes)
    
    # 3. The Penalty Logic
    required_coverage = (1 - target_alpha)
    
    if coverage < required_coverage:
        # Penalty for not meeting coverage that scales
        return -1000 + (coverage * 100) 
    else:
        return -avg_size

