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
    Adapts a trained PyTorch nn.Module (like your FFNN head) 
    to conform to the Scikit-Learn classifier interface.
    """
    def __init__(self, model, classes, device='cpu'):
        # model: The PyTorch nn.Module (e.g., your FFNN head)
        self.model = model
        # classes: Required for Scikit-Learn compatibility (e.g., [0, 1, ..., 9])
        self.classes = classes
        self.classes_ = np.array(classes)
        self.device = device
        
        # Ensure the model is on the correct device
        self.model.to(self.device)

    def fit(self, X, y=None):
        """
        Since we assume the PyTorch model is already trained, 
        this method does nothing but satisfy the Scikit-Learn interface.
        """
        # Note: If you wanted to *fine-tune* the head here, you would add that logic.
        return self

    def predict_logits(self, X):
        """Returns raw logits (pre-softmax) from the PyTorch model."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            # Forward pass
            logits = self.model(X_tensor)
            
            # Return NumPy array
            return logits.cpu().numpy()

    def predict_proba(self, X):
        """Returns softmax probabilities."""
        logits_np = self.predict_logits(X)
        
        # Convert logits back to tensor to apply softmax efficiently
        logits_tensor = torch.from_numpy(logits_np)
        probabilities = nn.functional.softmax(logits_tensor, dim=1)
        
        # Return NumPy array
        return probabilities.numpy()

    def predict(self, X):
        """Returns the predicted class label (not strictly needed by CP, but good practice)."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    


# ==========================================
# 2. BASE CONFORMAL CLASS
# ==========================================
class BaseConformalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, head_model, alpha=0.1, allow_zero_sets=True, rand=True):
        self.head_model = head_model  # This is the Linear Layer (Classifier)
        self.alpha = alpha
        self.allow_zero_sets = allow_zero_sets
        self.rand = rand
        self.q_hat = None
        self.classes_ = None

    def _get_scores(self, probs, y=None):
        raise NotImplementedError

    def fit(self, X_embeddings, y):
        # 1. Get Probabilities from the Head
        # X_embeddings is (N, 512) or similar
        cal_probs = self.head_model.predict_proba(X_embeddings)
        self.classes_ = self.head_model.classes_
        
        # 2. Calculate Scores
        scores = self._get_scores(cal_probs, y)
        self.scores = scores
        
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
        self.scores_matrix = scores_matrix
        
        # Threshold
        mask = scores_matrix <= self.q_hat
        self.mask = mask

        # Finalize (Zero Sets)
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
        mask : np.array of shape (n_samples, n_classes)
            The boolean output from .predict()
        labels : list or np.array, optional
            A list of class names (e.g., ['cat', 'dog', 'fish']).
            If None, returns the numerical indices (e.g., [0, 1]).
            
        Returns:
        --------
        list of lists
            e.g. [['cat', 'dog'], ['fish'], []]
        """
        # Convert to indices (list of arrays)
        prediction_indices = [np.where(row)[0] for row in self.mask]
        
        # Map to indices if no labels provided
        if labels is None:
            return [idxs.tolist() for idxs in prediction_indices]
        
        # Map indices to labels
        labels = np.array(labels) # Ensure it's an array for easy indexing
        return [labels[idxs].tolist() for idxs in prediction_indices]

# ==========================================
# 3. CONFORMAL METHODS (TPS, APS, RAPS, DAPS)
# ==========================================

class TPS(BaseConformalClassifier):
    def _get_scores(self, probs, y=None):
        n = len(probs)
        # Score = 1 - true_class_prob
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
    def __init__(self, head_model, alpha=0.1, lam_reg=0.01, k_reg=2, allow_zero_sets=False, rand=True):
        super().__init__(head_model, alpha, allow_zero_sets, rand)
        self.lam_reg = lam_reg
        self.k_reg = k_reg

    def _get_scores(self, probs, y=None):
        n = len(probs)
        pi = probs.argsort(1)[:, ::-1]
        srt = np.take_along_axis(probs, pi, axis=1)
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        
        if y is not None: # Calibration (Vector)
            cal_true_loc = np.where(pi == y[:, None])[1]
            # Score = CumSum - U*Prob
            scores = (srt + reg_vec).cumsum(axis=1)[np.arange(n), cal_true_loc] - u * (srt + reg_vec)[np.arange(n), cal_true_loc]
            return scores
        else: # Prediction (Matrix)
            # Calculate score for every class
            score_matrix_sorted = (srt + reg_vec).cumsum(axis=1) - (u[:, None] * (srt + reg_vec))
            # Unsort
            score_matrix_original = np.zeros_like(score_matrix_sorted)
            np.put_along_axis(score_matrix_original, pi, score_matrix_sorted, axis=1)
            return score_matrix_original


class DAPS(RAPS): # Inherit from RAPS to reuse _get_scores logic!
    def __init__(self, head_model, smoother, alpha=0.1, lam_reg=0.01, k_reg=2, beta=0.2, allow_zero_sets=False, rand=True):
        # We pass head_model to parent
        super().__init__(head_model, alpha, lam_reg, k_reg, allow_zero_sets, rand)
        self.smoother = smoother
        self.beta = beta

    def fit(self, X_embeddings, y):
        # 1. Standard Fit (calculates q_hat based on pure RAPS)
        # We override this because we need to intercept the scores before quantile calc
        
        cal_probs = self.head_model.predict_proba(X_embeddings)
        self.classes_ = self.head_model.classes_
        
        # Get Raw RAPS scores
        raw_scores = self._get_scores(cal_probs, y)

        # 2. Train Smoother on EMBEDDINGS
        # This is where PCA + KMeans happens
        self.smoother.fit(X_embeddings, y, raw_scores)
        
        # 3. Diffuse
        signal = self.smoother.predict_smooth(X_embeddings)
        diff_scores = (1 - self.beta) * raw_scores + self.beta * signal

        # 4. Quantile
        n = len(y)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(diff_scores, min(q_level, 1.0), method='higher')
        return self

    def predict(self, X_embeddings):
        check_is_fitted(self, "q_hat")
        test_probs = self.head_model.predict_proba(X_embeddings)
        
        # 1. Get Base Matrix (from RAPS parent)
        matrix_scores = self._get_scores(test_probs, y=None)
        
        # 2. Get Signal
        signal = self.smoother.predict_smooth(X_embeddings)
        
        # 3. Diffuse
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
# 4. METRICS FUNCTION
# ==========================================

def ConformalMetrics(prediction_mask, y_true):
        """
        Computes key metrics for conformal prediction sets.
        Parameters:
        -----------
        prediction_mask : np.array of shape (n_samples, n_classes)
            Boolean mask indicating predicted sets.
            y_true : np.array of shape (n_samples,)
            True class labels.
            
        Returns:
        --------
        dict
            returns a dictionary with keys:
            - "Coverage": Fraction of times true label is in set
            - "Avg Set Size": Average size of prediction sets
            - "Singleton Rate": Fraction of sets with size 1
            - "Singleton Hit": Coverage when set size is 1
        """
        n = len(y_true)
        
        # 1. Coverage (Fraction of times true label is in set)
        covered = prediction_mask[np.arange(n), y_true]
        coverage = np.mean(covered)
        
        # 2. Average Set Size
        set_sizes = np.sum(prediction_mask, axis=1)
        avg_size = np.mean(set_sizes)
        
        # 3. Singleton Rate (Fraction of sets with size 1)
        is_singleton = (set_sizes == 1)
        singleton_rate = np.mean(is_singleton)
        
        # 4. Singleton Hit Rate (Coverage with set size == 1)
        if np.sum(is_singleton) > 0:
            singleton_hit = np.mean(covered[is_singleton])
        else:
            singleton_hit = 0.0
            
        return {
            "Coverage": coverage,
            "Avg Set Size": avg_size,
            "Singleton Rate": singleton_rate,
            "Singleton Hit": singleton_hit
        }

# ==========================================
# 5. EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # --- A. Data Generation ---
    print("Generating Data...")
    X, y = datasets.make_classification(n_samples=6000, n_features=20, n_classes=10, n_informative=15, random_state=42)

    # --- B. Robust Splitting (60/20/10/10) ---
    # 1. Split Train (60%) vs Rest (40%)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42)
    
    # 2. Split Rest into Val (20% total -> 50% of rest) vs Cal/Test
    X_val, X_rest2, y_val, y_rest2 = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
    
    # 3. Split Remainder into Cal (10% total -> 50% of remainder) vs Test (10% total)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest2, y_rest2, test_size=0.5, random_state=42)

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Cal: {len(y_cal)}, Test: {len(y_test)}")

    # --- C. Model Training ---
    print("Training MLP...")
    model = MLPClassifier(hidden_layer_sizes=(50,5), activation='relu', solver='adam', max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # Get Probabilities
    cal_probs = model.predict_proba(X_cal)
    test_probs = model.predict_proba(X_test)

    # --- D. Conformal Prediction ---
    print("Fitting Conformal Classifiers...")
    alpha = 0.1
    tps = TPS(head_model=model, alpha=alpha, allow_zero_sets=False, rand=True)
    tps.fit(X_cal, y_cal)
    mask_tps = tps.predict(X_val)
    sets_tps = tps.convert_to_sets(mask_tps)
    tps_metrics = ConformalMetrics(mask_tps, y_val)
    print("TPS Metrics:", tps_metrics)


    raps = RAPS(head_model=model, alpha=alpha, lam_reg=0.01, k_reg=2, allow_zero_sets=False, rand=True)
    raps.fit(X_cal, y_cal)
    mask_raps = raps.predict(X_val)
    sets_raps = raps.convert_to_sets(mask_raps)
    raps_metrics = ConformalMetrics(mask_raps, y_val)
    print("RAPS Metrics:", raps_metrics)

    # Print first 5 prediction sets for inspection
    print("\nFirst 5 Prediction Sets:")
    print("First 5 true labels:", y_val[:5])
    print("First 5 TPS Prediction Sets:", sets_tps[:5])
    print("First 5 RAPS Prediction Sets:", sets_raps[:5])