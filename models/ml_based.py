import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from typing import Optional, Dict, Tuple, List
from .base import BaseCausalEstimator, MetaLearner


class CausalForest(BaseCausalEstimator):
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt', random_state=42):
        super().__init__(name="Causal Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.forest = None
        self.ites = None
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        ps_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.05, 0.95)
        
        transformed_y = Y * (T - ps) / (ps * (1 - ps))
        
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.forest.fit(X, transformed_y)
        
        self.ites = self.forest.predict(X)
        
        self.ate = np.mean(self.ites)
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def get_ite(self) -> Optional[np.ndarray]:
        return self.ites
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.forest:
            importance = self.forest.feature_importances_
            return pd.DataFrame({
                'feature': self.forest.feature_names_in_,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return None


class SLearner(MetaLearner):
    def __init__(self, base_learner=None):
        name = "S-Learner"
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        super().__init__(base_learner, name)
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        X_with_t = X.copy()
        X_with_t['treatment'] = T
        
        
        self.models['s_learner'] = clone(self.base_learner)
        self.models['s_learner'].fit(X_with_t, Y)
        
        
        X_t1 = X.copy()
        X_t1['treatment'] = 1
        X_t0 = X.copy()
        X_t0['treatment'] = 0
        
        y1_pred = self.models['s_learner'].predict(X_t1)
        y0_pred = self.models['s_learner'].predict(X_t0)
        
        self.ites = y1_pred - y0_pred
        self.ate = np.mean(self.ites)
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def get_ite(self) -> Optional[np.ndarray]:
        return self.ites


class TLearner(MetaLearner):
    def __init__(self, base_learner=None):
        name = "T-Learner"
        if base_learner is None:
            base_learner = RandomForestRegressor(n_estimators=100, max_depth=5)
        super().__init__(base_learner, name)
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        X_treat = X[T == 1]
        Y_treat = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
       
        self.models['treat_model'] = clone(self.base_learner)
        self.models['treat_model'].fit(X_treat, Y_treat)
        
        self.models['control_model'] = clone(self.base_learner)
        self.models['control_model'].fit(X_control, Y_control)
        
       
        y1_pred = self.models['treat_model'].predict(X)
        y0_pred = self.models['control_model'].predict(X)
        
        self.ites = y1_pred - y0_pred
        self.ate = np.mean(self.ites)
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def get_ite(self) -> Optional[np.ndarray]:
        return self.ites


class XLearner(MetaLearner):
    def __init__(self, base_learner=None, propensity_model=None):
        name = "X-Learner"
        if base_learner is None:
            base_learner = RandomForestRegressor(n_estimators=100, max_depth=5)
        if propensity_model is None:
            propensity_model = RandomForestClassifier(n_estimators=100)
        super().__init__(base_learner, name)
        self.propensity_model = propensity_model
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        
        self.propensity_model.fit(X, T)
        ps = self.propensity_model.predict_proba(X)[:, 1]
        
       
        X_treat = X[T == 1]
        Y_treat = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        
        mu_treat = clone(self.base_learner)
        mu_treat.fit(X_treat, Y_treat)
        
        mu_control = clone(self.base_learner)
        mu_control.fit(X_control, Y_control)
        
        
        D_treat = Y[T == 1] - mu_control.predict(X_treat)
        D_control = mu_treat.predict(X_control) - Y[T == 0]
        
        tau_treat = clone(self.base_learner)
        tau_treat.fit(X_treat, D_treat)
        
        tau_control = clone(self.base_learner)
        tau_control.fit(X_control, D_control)
        
        self.ites = ps * tau_treat.predict(X) + (1 - ps) * tau_control.predict(X)
        self.ate = np.mean(self.ites)
        
        self.models['tau_treat'] = tau_treat
        self.models['tau_control'] = tau_control
        self.models['mu_treat'] = mu_treat
        self.models['mu_control'] = mu_control
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def get_ite(self) -> Optional[np.ndarray]:
        return self.ites


class TARNet(BaseCausalEstimator):
    def __init__(self, n_layers=2, hidden_units=50, learning_rate=0.01, 
                 n_epochs=100, batch_size=32, random_state=42):
        super().__init__(name="TARNet")
        
        self.representation_net = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,
            random_state=random_state
        )
        self.treat_head = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=random_state
        )
        self.control_head = RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=random_state
        )
        
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        self.representation_net.fit(X, Y)
        Phi = self.representation_net.predict(X).reshape(-1, 1)
        
        self.treat_head.fit(Phi[T == 1], Y[T == 1])
        self.control_head.fit(Phi[T == 0], Y[T == 0])
        
        Phi_all = self.representation_net.predict(X).reshape(-1, 1)
        y1_pred = self.treat_head.predict(Phi_all)
        y0_pred = self.control_head.predict(Phi_all)
        
        self.ites = y1_pred - y0_pred
        self.ate = np.mean(self.ites)
        
        self.is_fitted = True
        return self
    
    def estimate_ate(self) -> float:
        return self.ate
    
    def get_ite(self) -> Optional[np.ndarray]:
        return self.ites