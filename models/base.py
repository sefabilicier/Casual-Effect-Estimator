from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Dict, Any, Tuple, Optional, List
import joblib


class BaseCausalEstimator(ABC):
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.estimates = {}
        self.metadata = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray):
        pass
    
    @abstractmethod
    def estimate_ate(self) -> float:
        pass
    
    def estimate_att(self) -> Optional[float]:
        return None
    
    def estimate_ate_confidence_interval(self, alpha=0.05) -> Tuple[float, float]:
        return (None, None)
    
    def get_ite(self) -> Optional[np.ndarray]:
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'estimator': self.name,
            'ate': self.estimate_ate(),
            'att': self.estimate_att(),
            'fitted': self.is_fitted,
            **self.metadata
        }
    
    def save(self, path: str):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


class MetaLearner(BaseCausalEstimator):
    
    def __init__(self, base_learner, name=None):
        super().__init__(name)
        self.base_learner = base_learner
        self.models = {}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[name] = np.abs(model.coef_).flatten()
        
        if importance_dict:
            return pd.DataFrame(importance_dict)
        return None