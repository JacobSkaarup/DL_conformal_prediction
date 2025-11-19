import numpy as np

class conformal_metrics():
    def __init__(self, prediction_sets, test_probs, y_test, labels):
        self.prediction_sets = prediction_sets
        self.test_probs = test_probs
        self.y_test = y_test
        self.labels = labels
        self._compute_metrics()
        self._find_preds()
        
    def _compute_metrics(self):
        self._compute_empirical_coverage()
        self._compute_singleton_rate()
        self._compute_singleton_hit_ratio()
        

    def _compute_empirical_coverage(self):
        self.empirical_coverage =  self.prediction_sets[np.arange(len(self.test_probs)), self.y_test].mean()
    
    def _compute_singleton_rate(self):
        mask = np.sum(self.prediction_sets, axis=1) == 1
        if len(mask) == 0:
            self.singleton_rate = None
        else:
            self.singleton_rate = np.mean(mask)
    
    def _compute_singleton_hit_ratio(self):
        mask = np.sum(self.prediction_sets, axis=1) == 1
        if np.sum(mask) == 0:
            self.singleton_hit_ratio = None
        else:
            self.singleton_hit_ratio = np.mean(self.prediction_sets[np.arange(len(self.test_probs)), self.y_test][mask])
        
    def _find_preds(self):
        preds = {}
        for i in range(len(self.test_probs)):
            preds[i] = [self.labels[j] for j in np.where(self.prediction_sets[i])[0]]
        self.preds = preds
    

class TPS_conformal(conformal_metrics):
    def __init__(self, cal_probs, y_cal, test_probs, y_test, labels, alpha=0.1, disallow_zero_sets=True):
        self.cal_probs = cal_probs
        self.y_cal = y_cal
        self.test_probs = test_probs
        self.labels = labels
        self.alpha = alpha
        self.disallow_zero_sets = disallow_zero_sets
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        # Get metrics and predictions
        super().__init__(self.prediction_sets, test_probs, y_test, labels)

        
    def calibrate(self):
        # Determine quantile level
        n = len(self.y_cal)
        q_level = np.ceil((n+1)*(1-self.alpha))/n

        # Calibration scores
        cal_scores = 1-self.cal_probs[np.arange(n), self.y_cal]
        q_hat = np.quantile(cal_scores, q_level, method='higher')

        # Compute prediction sets and empirical coverage
        prediction_sets = self.test_probs >= (1-q_hat)

        # Ensure no empty prediction sets by adding the most probable class
        if self.disallow_zero_sets:
            empty_sets = np.where(np.sum(prediction_sets, axis=1) == 0)[0]
            for i in empty_sets:
                max_index = np.argmax(self.test_probs[i])
                prediction_sets[i, max_index] = True
        
        return cal_scores, q_hat, prediction_sets
    

class APS_conformal(conformal_metrics):
    def __init__(self, cal_probs, y_cal, test_probs, y_test, labels, alpha=0.1, rand=True):
        self.cal_probs = cal_probs
        self.y_cal = y_cal
        self.test_probs = test_probs
        self.labels = labels
        self.alpha = alpha
        self.rand = rand
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        # Get metrics and predictions
        super().__init__(self.prediction_sets, test_probs, y_test, labels)
        
        
    def calibrate(self):
        n = len(self.y_cal)
        q_level = np.ceil((n+1)*(1-self.alpha))/n
        cal_range = np.arange(n)


        # Compute calibration scores
        true_prob = self.cal_probs[cal_range, self.y_cal]                                               # true class probabilities
        cal_pi = self.cal_probs.argsort(1)[:,::-1]                                                      # Indices that would sort probs descending
        cal_srt = np.take_along_axis(self.cal_probs, cal_pi, axis=1).cumsum(axis=1)                     # Sorted cumulative sums
        cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[cal_range, self.y_cal] # Get scores for true class

        # Break ties randomly
        u = np.random.uniform(0, 1, size = n) if self.rand else np.zeros(n)
        cal_scores = cal_scores - u * true_prob   # Subtract random fraction of true class prob to break ties


        # Get the score quantile
        q_hat = np.quantile(cal_scores, q_level, method='higher')

        # Get prediction sets for test data
        test_pi = self.test_probs.argsort(1)[:,::-1]                                     # Indices that would sort test probs descending
        test_srt = np.take_along_axis(self.test_probs, test_pi, axis=1).cumsum(axis=1)   # Sorted cumulative sums

        # Define prediction sets
        mask = test_srt < q_hat                                                         # classes strictly below qhat
        first_cross = (test_srt >= q_hat).argmax(axis=1)                                # index of first crossing
        mask[np.arange(len(mask)), first_cross] = True                                  # ensure inclusion of crossing class
        prediction_sets = np.take_along_axis(mask, test_pi.argsort(axis=1), axis=1)     # Get prediction sets


        return cal_scores, q_hat, prediction_sets
        

class RAPS_conformal(conformal_metrics):
    def __init__(self, cal_probs, y_cal, test_probs, y_test, labels, alpha=0.1, k_reg=2, lambda_reg=0.3, disallow_zero_sets=True, rand=True):
        self.cal_probs = cal_probs
        self.y_cal = y_cal
        self.test_probs = test_probs
        self.y_test = y_test
        self.labels = labels
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.disallow_zero_sets = disallow_zero_sets
        self.rand = rand
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        super().__init__(self.prediction_sets, self.test_probs, self.y_test, self.labels)

        
    def calibrate(self):
        n_cal = len(self.y_cal)
        cal_range = np.arange(n_cal)
        q_level = np.ceil((n_cal+1)*(1-self.alpha))/n_cal
        u_cal = np.random.uniform(0, 1, size = n_cal) if self.rand else np.zeros(n_cal)

        # Regularization vector
        reg_vec = np.array(self.k_reg*[0,] + (self.cal_probs.shape[1]-self.k_reg)*[self.lambda_reg,])[None,:]

        # Compute calibration scores
        cal_pi = self.cal_probs.argsort(1)[:,::-1]; 
        cal_srt = np.take_along_axis(self.cal_probs,cal_pi,axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_true = np.where(cal_pi == self.y_cal[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[cal_range, cal_true] - u_cal * cal_srt_reg[cal_range, cal_true]

        # Get the score quantile
        q_hat = np.quantile(cal_scores, q_level, method='higher')
        
        # Get prediction sets for test data
        n_test = self.test_probs.shape[0]
        u_test = np.random.uniform(0, 1, size = n_test) if self.rand else np.zeros(n_test)

        val_pi = self.test_probs.argsort(1)[:,::-1]
        val_srt = np.take_along_axis(self.test_probs,val_pi,axis=1)
        val_srt_reg = val_srt + reg_vec
        mask = (val_srt_reg.cumsum(axis=1) - u_test[:,None] * val_srt_reg) <= q_hat if self.rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= q_hat
        if self.disallow_zero_sets:
            mask[:,0] = True
        
        prediction_sets = np.take_along_axis(mask, val_pi.argsort(axis=1), axis=1)

        return cal_scores, q_hat, prediction_sets
    
class DAPS_conformal(conformal_metrics):
    def __init__(self, cal_probs, y_cal, cal_clusters, test_probs, y_test, test_clusters, labels, alpha=0.1, lam_reg=0.3, k_reg=2, beta = 0.1, disallow_zero_sets=True, rand=True):
        self.cal_probs = cal_probs
        self.y_cal = y_cal
        self.cal_clusters = cal_clusters
        self.test_probs = test_probs
        self.y_test = y_test
        self.test_clusters = test_clusters
        self.labels = labels
        self.alpha = alpha
        self.lam_reg = lam_reg
        self.k_reg = k_reg
        self.beta = beta
        self.rand = rand
        self.disallow_zero_sets = disallow_zero_sets
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        super().__init__(self.prediction_sets, self.test_probs, self.y_test, self.labels)

        
    def calibrate(self):
        # Determine quantile level
        n_cal = len(self.y_cal)
        cal_range = np.arange(n_cal)
        q_level = np.ceil((n_cal+1)*(1-self.alpha))/n_cal
        u_cal = np.random.uniform(0, 1, size = n_cal) if self.rand else np.zeros(n_cal)

        # Regularization vector
        reg_vec = np.array(self.k_reg*[0,] + (self.cal_probs.shape[1]-self.k_reg)*[self.lam_reg,])[None,:]

        # Compute calibration scores
        cal_pi = self.cal_probs.argsort(1)[:,::-1]; 
        cal_srt = np.take_along_axis(self.cal_probs,cal_pi,axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_true = np.where(cal_pi == self.y_cal[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[cal_range, cal_true] - u_cal * cal_srt_reg[cal_range, cal_true]

        # Diffuse scores with neighboring cluster scores
        diff_scores = cal_scores.copy()
        cluster_means = {}

        for cluster in np.unique(self.cal_clusters):
            cluster_means[cluster] = np.mean(cal_scores[self.cal_clusters == cluster], axis=0)                   # compute cluster mean score
            indices = np.where(self.cal_clusters == cluster)[0]                                                  # get indices of points in cluster
            diff_scores[indices] = (1 - self.beta) * cal_scores[indices] + self.beta * cluster_means[cluster]    # diffuse scores

        # Get the score quantile
        q_hat = np.quantile(diff_scores, q_level, method='higher')
        
            
        # Get base prediction scores for test data
        n_test = self.test_probs.shape[0]
        u_test = np.random.uniform(0, 1, size = n_test) if self.rand else np.zeros(n_test)

        test_pi = self.test_probs.argsort(1)[:,::-1]
        test_srt = np.take_along_axis(self.test_probs, test_pi, axis=1)
        test_scores_base = test_srt + reg_vec

        # Diffuse test scores based on cluster assignments
        test_scores = test_scores_base.copy()

        for cluster in np.unique(self.test_clusters):
            indices = np.where(self.test_clusters == cluster)[0]
            test_scores[indices] = (1 - self.beta) * test_scores_base[indices] + self.beta * cluster_means[cluster]


        # Define prediction sets
        mask = (test_scores.cumsum(axis=1) - u_test[:, None] * test_scores) <= q_hat if self.rand else test_scores.cumsum(axis=1) - test_scores <= q_hat
        if self.disallow_zero_sets: 
            mask[:, 0] = True

        prediction_sets = np.take_along_axis(mask, test_pi.argsort(axis=1), axis=1)
        
        return cal_scores, q_hat, prediction_sets