import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin

# ==========================================
# 1. BASE CONFORMAL CLASS
# ==========================================
class BaseConformal(BaseEstimator, ClassifierMixin):
    """
    Parent class handling common logic for all Conformal Predictors.
    """
    def __init__(self, alpha=0.1, allow_zero_sets=False, rand=True):
        self.alpha = alpha
        self.allow_zero_sets = allow_zero_sets
        self.rand = rand
        self.q_hat = None
        self.n_classes = None
        
    def _get_quantile(self, scores):
        """Calculates the (1-alpha) quantile based on calibration scores."""
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        # Clip q_level to 1.0 to avoid floating point errors exceeding max
        q_level = min(q_level, 1.0)
        return np.quantile(scores, q_level, method='higher')

    def _finalize_sets(self, mask, original_probs, sort_indices=None):
        """
        1. Un-sorts the boolean mask if necessary.
        2. Enforces non-empty sets if allow_zero_sets is False.
        """
        # If data was sorted (APS/RAPS/DAPS), map mask back to original class indices
        if sort_indices is not None:
            unsorted_mask = np.zeros_like(mask, dtype=bool)
            np.put_along_axis(unsorted_mask, sort_indices, mask, axis=1)
            mask = unsorted_mask

        # Force non-empty sets if required
        if not self.allow_zero_sets:
            empty_rows = np.where(mask.sum(axis=1) == 0)[0]
            if len(empty_rows) > 0:
                # Add the class with the highest probability
                top_class = original_probs[empty_rows].argmax(axis=1)
                mask[empty_rows, top_class] = True
                
        return mask
    
    def convert_to_sets(self, mask, labels=None):
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
        # using a list comprehension is usually faster than looping numpy arrays for variable-length outputs
        prediction_indices = [np.where(row)[0] for row in mask]
        
        if labels is None:
            return [idxs.tolist() for idxs in prediction_indices]
        
        # Map indices to labels
        labels = np.array(labels) # Ensure it's an array for easy indexing
        return [labels[idxs].tolist() for idxs in prediction_indices]

# ==========================================
# 2. CONFORMAL METHODS (TPS, APS, RAPS, DAPS)
# ==========================================

class TPS(BaseConformal):
    def fit(self, cal_probs, y_cal):
        n = len(y_cal)
        # Score = 1 - true_class_prob
        cal_scores = 1 - cal_probs[np.arange(n), y_cal]
        self.q_hat = self._get_quantile(cal_scores)
        return self

    def predict(self, test_probs):
        # Include class if prob >= 1 - Q
        mask = test_probs >= (1 - self.q_hat)
        return self._finalize_sets(mask, test_probs, sort_indices=None)


class RAPS(BaseConformal):
    """
    Regularized Adaptive Prediction Sets. 
    Note: APS is just RAPS with k_reg=0 and lam_reg=0.
    """
    def __init__(self, alpha=0.1, k_reg=2, lam_reg=0.3, allow_zero_sets=False, rand=True):
        super().__init__(alpha, allow_zero_sets, rand)
        self.k_reg = k_reg
        self.lam_reg = lam_reg

    def _compute_reg_scores(self, probs):
        # Sort probabilities descending
        pi = probs.argsort(1)[:, ::-1]
        srt = np.take_along_axis(probs, pi, axis=1)
        
        # Add regularization penalty
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        srt_reg = srt + reg_vec
        
        return pi, srt_reg

    def fit(self, cal_probs, y_cal):
        n = len(y_cal)
        pi, srt_reg = self._compute_reg_scores(cal_probs)
        
        # Get score of the true class location
        # map y_cal to the sorted indices
        cal_true_loc = np.where(pi == y_cal[:, None])[1]
        
        # Randomized Cumulative Sum
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        # Score = Cumulative_Sum_at_True - (Random * Current_Val_at_True)
        scores = srt_reg.cumsum(axis=1)[np.arange(n), cal_true_loc] - u * srt_reg[np.arange(n), cal_true_loc]
        
        self.q_hat = self._get_quantile(scores)
        return self

    def predict(self, test_probs):
        n = len(test_probs)
        pi, srt_reg = self._compute_reg_scores(test_probs)
        
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        
        # Include if (CumSum - U*Current) <= Q
        mask = (srt_reg.cumsum(axis=1) - u[:, None] * srt_reg) <= self.q_hat
        
        return self._finalize_sets(mask, test_probs, sort_indices=pi)


class DAPS(BaseConformal):
    """
    Diffused Adaptive Prediction Sets.
    Fix: Learns cluster means on Calibration data, applies them to Test data.
    """
    def __init__(self, alpha=0.1, lam_reg=0.3, k_reg=2, beta=0.1, allow_zero_sets=False, rand=True):
        super().__init__(alpha, allow_zero_sets, rand)
        self.lam_reg = lam_reg
        self.k_reg = k_reg
        self.beta = beta
        self.cluster_means_ = {}
        self.global_mean_ = 0

    def _compute_reg_scores(self, probs):
        # Reusing RAPS logic for base scores
        pi = probs.argsort(1)[:, ::-1]
        srt = np.take_along_axis(probs, pi, axis=1)
        reg_vec = np.array(self.k_reg*[0,] + (probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]
        return pi, srt + reg_vec

    def fit(self, cal_probs, y_cal, cal_clusters):
        # 1. Get Base Scores
        n = len(y_cal)
        pi, srt_reg = self._compute_reg_scores(cal_probs)
        cal_true_loc = np.where(pi == y_cal[:, None])[1]
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        
        # Raw scores before diffusion
        raw_scores = srt_reg.cumsum(axis=1)[np.arange(n), cal_true_loc] - u * srt_reg[np.arange(n), cal_true_loc]
        
        # 2. Learn Cluster Means (The Fix)
        self.global_mean_ = np.mean(raw_scores)
        diff_scores = raw_scores.copy()
        
        for cluster in np.unique(cal_clusters):
            idx = (cal_clusters == cluster)
            mean_score = np.mean(raw_scores[idx])
            self.cluster_means_[cluster] = mean_score
            # Diffuse calibration scores
            diff_scores[idx] = (1 - self.beta) * raw_scores[idx] + self.beta * mean_score

        self.q_hat = self._get_quantile(diff_scores)
        return self

    def predict(self, test_probs, test_clusters):
        # 1. Get Base Test Scores (Vectorized)
        n = len(test_probs)
        pi, srt_reg = self._compute_reg_scores(test_probs)
        
        # We need the full grid of scores to check thresholds
        # This part is slightly more expensive than RAPS because we diffuse the whole matrix
        # Construct base scores for every class position
        base_scores_grid = srt_reg
        
        # 2. Diffuse Test Scores using CALIBRATION means
        diffused_grid = base_scores_grid.copy()
        
        for cluster in np.unique(test_clusters):
            idx = np.where(test_clusters == cluster)[0]
            # Safety: use global mean if cluster was not seen in calibration
            c_mean = self.cluster_means_.get(cluster, self.global_mean_)
            
            # Note: In DAPS paper, diffusion is usually on the score vector. 
            # We approximate this by diffusing the base regularization values.
            diffused_grid[idx] = (1 - self.beta) * base_scores_grid[idx] + self.beta * c_mean

        # 3. Generate Sets
        u = np.random.uniform(0, 1, size=n) if self.rand else np.zeros(n)
        
        # Check threshold
        # Note: DAPS usually checks cumulative sums of diffused values
        mask = (diffused_grid.cumsum(axis=1) - u[:, None] * diffused_grid) <= self.q_hat
        
        return self._finalize_sets(mask, test_probs, sort_indices=pi)

# ==========================================
# 3. METRICS CLASS
# ==========================================

class ConformalMetrics:
    def __init__(self, prediction_mask, y_true):
        self.mask = prediction_mask
        self.y_true = y_true
        
    def get_metrics(self):
        n = len(self.y_true)
        
        # 1. Coverage (Fraction of times true label is in set)
        covered = self.mask[np.arange(n), self.y_true]
        coverage = np.mean(covered)
        
        # 2. Average Set Size
        set_sizes = np.sum(self.mask, axis=1)
        avg_size = np.mean(set_sizes)
        
        # 3. Singleton Rate (Fraction of sets with size 1)
        is_singleton = (set_sizes == 1)
        singleton_rate = np.mean(is_singleton)
        
        # 4. Singleton Hit Rate (Coverage conditional on set size == 1)
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
# 4. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    # --- A. Data Generation ---
    print("Generating Data...")
    X, y = datasets.make_classification(n_samples=10000, n_features=20, n_classes=10, n_informative=15, random_state=42)

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

    # --- D. Clustering (For DAPS) ---
    print("Running PCA + KMeans for DAPS...")
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    
    cluster_model = KMeans(n_clusters=10, random_state=42)
    cluster_model.fit(X_train_pca)

    cal_clusters = cluster_model.predict(pca.transform(X_cal))
    test_clusters = cluster_model.predict(pca.transform(X_test))

    # --- E. Run Conformal Prediction ---
    
    print("\n--- RESULTS (Target Alpha = 0.1) ---")
    
    # 1. TPS
    tps = TPS(alpha=0.1, allow_zero_sets=False)
    tps.fit(cal_probs, y_cal)
    tps_sets = tps.predict(test_probs)
    print("\nTPS Metrics:", ConformalMetrics(tps_sets, y_test).get_metrics())

    # 2. APS (RAPS with lambda=0)
    aps = RAPS(alpha=0.1, k_reg=0, lam_reg=0.0, allow_zero_sets=False)
    aps.fit(cal_probs, y_cal)
    aps_sets = aps.predict(test_probs)
    print("APS Metrics:", ConformalMetrics(aps_sets, y_test).get_metrics())

    # 3. RAPS
    raps = RAPS(alpha=0.1, k_reg=2, lam_reg=0.01, allow_zero_sets=False) # Small lambda for toy data
    raps.fit(cal_probs, y_cal)
    raps_sets = raps.predict(test_probs)
    print("RAPS Metrics:", ConformalMetrics(raps_sets, y_test).get_metrics())

    # 4. DAPS
    daps = DAPS(alpha=0.1, k_reg=2, lam_reg=0.01, beta=0.2, allow_zero_sets=False)
    daps.fit(cal_probs, y_cal, cal_clusters)
    daps_sets = daps.predict(test_probs, test_clusters)
    print("DAPS Metrics:", ConformalMetrics(daps_sets, y_test).get_metrics())