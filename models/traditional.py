import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import statsmodels.api as sm
from .base import BaseCausalEstimator


class LinearRegressionEstimator(BaseCausalEstimator):
    def __init__(self, add_intercept=True, use_statsmodels=True):
        super().__init__(name="Linear Regression")
        self.add_intercept = add_intercept
        self.use_statsmodels = use_statsmodels
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        design_matrix = np.column_stack([T, X_scaled])
        
        if self.add_intercept:
            design_matrix = sm.add_constant(design_matrix)
        
        if self.use_statsmodels:
            self.model = sm.OLS(Y, design_matrix).fit()
            self.ate = self.model.params[1]  # coefficient of T
            self.metadata['p_value'] = self.model.pvalues[1]
            self.metadata['r_squared'] = self.model.rsquared
        else:
            self.model = LinearRegression().fit(design_matrix, Y)
            self.ate = self.model.coef_[1]
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def estimate_ate_confidence_interval(self, alpha=0.05) -> Tuple[float, float]:
        if self.use_statsmodels:
            conf_int = self.model.conf_int(alpha=alpha)
            return conf_int[1] 
        return (None, None)
    
    def get_summary(self) -> Dict:
        summary = super().get_summary()
        if self.use_statsmodels:
            summary['p_value'] = self.metadata.get('p_value')
            summary['r_squared'] = self.metadata.get('r_squared')
        return summary


class IPTWEstimator(BaseCausalEstimator):
    def __init__(self, propensity_model=None, stabilize=True, clip_weights=True):
        super().__init__(name="IPTW")
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.stabilize = stabilize
        self.clip_weights = clip_weights
        self.propensity_scores = None
        self.weights = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        self.propensity_model.fit(X, T)
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        if self.clip_weights:
            self.propensity_scores = np.clip(self.propensity_scores, 0.05, 0.95)
        
        self.weights = np.zeros(len(T))
        self.weights[T == 1] = 1 / self.propensity_scores[T == 1]
        self.weights[T == 0] = 1 / (1 - self.propensity_scores[T == 0])
        
        if self.stabilize:
            p_treat = np.mean(T)
            self.weights[T == 1] *= p_treat
            self.weights[T == 0] *= (1 - p_treat)
        
        self.ate = np.average(Y[T == 1], weights=self.weights[T == 1]) - \
                   np.average(Y[T == 0], weights=self.weights[T == 0])
        
        self.is_fitted = True
        
        self.metadata['mean_weight'] = np.mean(self.weights)
        self.metadata['max_weight'] = np.max(self.weights)
        self.metadata['min_weight'] = np.min(self.weights)
        self.metadata['effective_sample_size'] = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def estimate_att(self) -> float:
        weights_att = np.ones(len(self.weights))
        weights_att[T == 0] = self.propensity_scores[T == 0] / (1 - self.propensity_scores[T == 0])
        
        if self.stabilize:
            weights_att[T == 0] *= (1 - p_treat) / p_treat
            
        att = np.mean(Y[T == 1]) - np.average(Y[T == 0], weights=weights_att[T == 0])
        return att


class StratificationEstimator(BaseCausalEstimator):
    def __init__(self, n_strata=5):
        super().__init__(name="Stratification")
        self.n_strata = n_strata
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        
        
        strata = pd.qcut(ps, self.n_strata, labels=False)
        
        
        ate_strata = []
        weights = []
        
        for s in range(self.n_strata):
            mask = strata == s
            if np.sum(mask) > 0:
                y_t = Y[(mask) & (T == 1)]
                y_c = Y[(mask) & (T == 0)]
                
                if len(y_t) > 0 and len(y_c) > 0:
                    ate_s = np.mean(y_t) - np.mean(y_c)
                    ate_strata.append(ate_s)
                    weights.append(np.sum(mask))
        
        
        self.ate = np.average(ate_strata, weights=weights)
        
        self.metadata['strata_ates'] = ate_strata
        self.metadata['strata_weights'] = weights
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate