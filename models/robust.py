import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.base import clone
from typing import Optional, Tuple, Dict
from .base import BaseCausalEstimator


class DoublyRobustEstimator(BaseCausalEstimator):
    def __init__(self, propensity_model=None, outcome_model=None, n_folds=5):
        super().__init__(name="Doubly Robust")
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.outcome_model = outcome_model or RandomForestRegressor(n_estimators=100)
        self.n_folds = n_folds
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        ps_pred = np.zeros(len(X))
        y1_pred = np.zeros(len(X))
        y0_pred = np.zeros(len(X))
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            ps_clf = clone(self.propensity_model)
            ps_clf.fit(X_train, T_train)
            ps_pred[test_idx] = ps_clf.predict_proba(X_test)[:, 1]
            
            treat_model = clone(self.outcome_model)
            treat_mask = T_train == 1
            if np.sum(treat_mask) > 0:
                treat_model.fit(X_train[treat_mask], Y_train[treat_mask])
                y1_pred[test_idx] = treat_model.predict(X_test)
            else:
                y1_pred[test_idx] = np.mean(Y_train)
            
            control_model = clone(self.outcome_model)
            control_mask = T_train == 0
            if np.sum(control_mask) > 0:
                control_model.fit(X_train[control_mask], Y_train[control_mask])
                y0_pred[test_idx] = control_model.predict(X_test)
            else:
                y0_pred[test_idx] = np.mean(Y_train)
        
        ps_pred = np.clip(ps_pred, 0.05, 0.95)
        
        dr_values = y1_pred - y0_pred + \
                    T * (Y - y1_pred) / ps_pred - \
                    (1 - T) * (Y - y0_pred) / (1 - ps_pred)
        
        self.ate = np.mean(dr_values)
        self.dr_scores = dr_values
        
        self.propensity_scores = ps_pred
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def estimate_ate_confidence_interval(self, alpha=0.05) -> Tuple[float, float]:
        n_bootstrap = 1000
        bootstrap_ates = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(self.dr_scores), 
                                     len(self.dr_scores), 
                                     replace=True)
            bootstrap_ates.append(np.mean(self.dr_scores[indices]))
        
        ci_lower = np.percentile(bootstrap_ates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ates, 100 * (1 - alpha / 2))
        
        return (ci_lower, ci_upper)


class AIPWEstimator(BaseCausalEstimator):
    def __init__(self, propensity_model=None, outcome_model=None, n_folds=5):
        super().__init__(name="AIPW")
        self.dr_estimator = DoublyRobustEstimator(
            propensity_model, outcome_model, n_folds
        )
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        self.dr_estimator.fit(X, T, Y)
        self.ate = self.dr_estimator.estimate_ate()
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate