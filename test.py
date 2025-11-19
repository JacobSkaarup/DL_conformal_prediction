import numpy as np

class conformal_metrics():
    def __init__(self, prediction_sets, test_probs, y_test):
        self.prediction_sets = prediction_sets
        self.test_probs = test_probs
        self.y_test = y_test
        
    def compute_metrics(self):
        empirical_coverage = self._compute_empirical_coverage()
        singleton_rate = self._compute_singleton_rate()
        singleton_hit_ratio = self._compute_singleton_hit_ratio()
        
        return empirical_coverage, singleton_rate, singleton_hit_ratio

    def _compute_empirical_coverage(self):
        return self.prediction_sets[np.arange(len(self.test_probs)), self.y_test].mean()
    
    def _compute_singleton_rate(self):
        return np.mean(np.sum(self.prediction_sets, axis=1) == 1)
    
    def _compute_singleton_hit_ratio(self):
        return np.mean(self.prediction_sets[np.arange(len(self.test_probs)), self.y_test][np.sum(self.prediction_sets, axis=1) == 1])
    

class TPS_conformal(conformal_metrics):
    def __init__(self, calib_probs, y_calib, test_probs, y_test, alpha=0.1):
        self.calib_probs = calib_probs
        self.y_calib = y_calib
        self.test_probs = test_probs
        self.alpha = alpha
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        super().__init__(self.prediction_sets, test_probs, y_test)
        self.empirical_coverage, self.singleton_rate, self.singleton_hit_ratio = self.compute_metrics()

        
    def calibrate(self):
        n = len(self.y_calib)
        q_level = np.ceil((n+1)*(1-self.alpha))/n

        cal_scores = 1 - self.calib_probs[np.arange(n), self.y_calib]
        q_hat = np.quantile(cal_scores, q_level, method='higher')

        prediction_sets = self.test_probs >= (1 - q_hat)
        
        return cal_scores, q_hat, prediction_sets
    

    class APS_conformal(conformal_metrics):
        def __init__(self, calib_probs, y_calib, test_probs, y_test, alpha=0.1):
            self.calib_probs = calib_probs
            self.y_calib = y_calib
            self.test_probs = test_probs
            self.alpha = alpha
            self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

            super().__init__(self.prediction_sets, test_probs, y_test)
            self.empirical_coverage, self.singleton_rate, self.singleton_hit_ratio = self.compute_metrics()

            
        def calibrate(self):
            n = len(self.y_calib)
            q_level = np.ceil((n+1)*(1-self.alpha))/n

            true_prob = self.calib_probs[np.arange(n), self.y_calib]                                         # true class probabilities
            cal_pi = self.calib_probs.argsort(1)[:,::-1]                                                     # Indices that would sort probs descending
            cal_srt = np.take_along_axis(self.calib_probs, cal_pi, axis=1).cumsum(axis=1)                    # Sorted cumulative sums
            cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), self.y_calib] # Get scores for true class

            # Break ties randomly
            u = np.random.uniform(0,1,size=n)
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
            prediction_sets = np.take_along_axis(mask, test_pi.argsort(axis=1), axis=1)     # Get prediction sets, all values with cumsum less than q_hat



            return cal_scores, q_hat, prediction_sets
        

class RAPS_conformal(conformal_metrics):
    def __init__(self, calib_probs, y_calib, test_probs, y_test, alpha=0.1, k_reg=5, lambda_reg=0.01, disallow_zero_sets=True, rand=True):
        self.calib_probs = calib_probs
        self.y_calib = y_calib
        self.test_probs = test_probs
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.disallow_zero_sets = disallow_zero_sets
        self.rand = rand
        self.cal_scores, self.q_hat, self.prediction_sets = self.calibrate()

        super().__init__(self.prediction_sets, test_probs, y_test)
        self.empirical_coverage, self.singleton_rate, self.singleton_hit_ratio = self.compute_metrics()

        
    def calibrate(self):
        n = len(self.y_calib)
        q_level = np.ceil((n+1)*(1-self.alpha))/n

        # Regularization vector
        reg_vec = np.array(self.k_reg*[0,] + (self.calib_probs.shape[1]-self.k_reg)*[self.lambda_reg,])[None,:]


        # Get scores
        cal_pi = self.calib_probs.argsort(1)[:,::-1]; 
        cal_srt = np.take_along_axis(self.calib_probs,cal_pi,axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == self.y_calib[:,None])[1]
        cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]

        # Get the score quantile
        q_hat = np.quantile(cal_scores, q_level, method='higher')
        
        # Define prediction sets
        n_val = self.test_probs.shape[0]
        val_pi = self.test_probs.argsort(1)[:,::-1]
        val_srt = np.take_along_axis(self.test_probs,val_pi,axis=1)
        val_srt_reg = val_srt + reg_vec
        indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= q_hat if self.rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= q_hat
        if self.disallow_zero_sets: indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)

        return cal_scores, q_hat, prediction_sets