import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, Union
import streamlit as st


class CausalDataGenerator:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.true_effects = {}
        
    def generate(self) -> Tuple[pd.DataFrame, Dict]:

        n = self.config.n_samples
        p = self.config.n_features
        
        X_c = self._generate_confounders()
        
        Z = self._generate_instruments()
        
        X = self._generate_covariates(X_c, Z)
        
        T, propensity_scores = self._generate_treatment(X, X_c, Z)
        
        Y0, Y1, ite = self._generate_outcomes(X, X_c, T)
        
        Y = Y0.copy()
        Y[T == 1] = Y1[T == 1]
        
        Y += self.rng.normal(0, self.config.noise_level, n)
        
        df = self._create_dataframe(X, X_c, Z, T, Y, propensity_scores)
        
        true_params = {
            'ate': np.mean(Y1 - Y0),
            'att': np.mean((Y1 - Y0)[T == 1]),
            'atc': np.mean((Y1 - Y0)[T == 0]),
            'ite': ite,
            'propensity_scores': propensity_scores,
            'confounders': X_c.columns.tolist(),
            'instruments': Z.columns.tolist() if Z is not None else []
        }
        
        self.true_effects = true_params
        
        return df, true_params
    
    def _generate_confounders(self) -> pd.DataFrame:
        n = self.config.n_samples
        n_c = self.config.n_confounders
        
        X_c = pd.DataFrame()
        for i in range(n_c):
            if self.rng.rand() > 0.5:
                X_c[f'conf_c{i}'] = self.rng.normal(0, 1, n)
            else:
                X_c[f'conf_b{i}'] = self.rng.binomial(1, 0.5, n)
                
        return X_c
    
    def _generate_instruments(self) -> Optional[pd.DataFrame]:
        n_i = self.config.n_instruments
        if n_i == 0:
            return None
            
        n = self.config.n_samples
        Z = pd.DataFrame()
        
        for i in range(n_i):
            Z[f'instrument_{i}'] = self.rng.normal(0, 1, n)
            
        return Z
    
    def _generate_covariates(self, X_c: pd.DataFrame, Z: Optional[pd.DataFrame]) -> pd.DataFrame:
        X = X_c.copy()

        n_pure = self.config.n_features - self.config.n_confounders
        for i in range(n_pure):
            X[f'pure_{i}'] = self.rng.normal(0, 1, self.config.n_samples)
            
        if Z is not None:
            X = pd.concat([X, Z], axis=1)
            
        return X
    
    def _generate_treatment(self, X: pd.DataFrame, X_c: pd.DataFrame, 
                           Z: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        n = self.config.n_samples
        strength = self.config.confounding_strength
        
        lin_pred = np.zeros(n)
        for col in X_c.columns:
            lin_pred += strength * X_c[col].values * self.rng.randn()
        
        if Z is not None:
            for col in Z.columns:
                lin_pred += 0.5 * Z[col].values * self.rng.randn()
        
        lin_pred = (lin_pred - np.mean(lin_pred)) / np.std(lin_pred)
        
        if self.config.treatment_assign == 'logistic':
            prob_t = 1 / (1 + np.exp(-lin_pred))
            prob_t = prob_t * (self.config.treatment_propensity * 2)
            prob_t = np.clip(prob_t, 0.1, 0.9)
            
        elif self.config.treatment_assign == 'threshold':
            prob_t = np.ones(n) * self.config.treatment_propensity
            prob_t[lin_pred > 0] = 0.8
            prob_t[lin_pred <= 0] = 0.2
            
        else: 
            prob_t = 1 / (1 + np.exp(-lin_pred - 0.5 * lin_pred**2))
            prob_t = np.clip(prob_t, 0.1, 0.9)
        
        T = self.rng.binomial(1, prob_t)
        
        return T, prob_t
    
    def _generate_outcomes(self, X: pd.DataFrame, X_c: pd.DataFrame, 
                          T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.n_samples
        ate = self.config.treatment_effect
        
        Y0 = np.zeros(n)
        for col in X_c.columns:
            Y0 += 0.5 * X_c[col].values * self.rng.randn()
        
        if self.config.effect_heterogeneity:
            het_var = X_c.iloc[:, 0].values if len(X_c.columns) > 0 else np.ones(n)
            het_var = (het_var - np.mean(het_var)) / np.std(het_var)
            ite = ate + het_var
        else:
            ite = np.ones(n) * ate
        
        if self.config.nonlinearity == 'moderate':
            ite += 0.2 * X_c.iloc[:, 0].values ** 2 if len(X_c.columns) > 0 else 0
        elif self.config.nonlinearity == 'high':
            ite += 0.5 * np.sin(X_c.iloc[:, 0].values * 2) if len(X_c.columns) > 0 else 0
        
        Y1 = Y0 + ite
        
        return Y0, Y1, ite
    
    def _create_dataframe(self, X: pd.DataFrame, X_c: pd.DataFrame, 
                         Z: Optional[pd.DataFrame], T: np.ndarray, 
                         Y: np.ndarray, ps: np.ndarray) -> pd.DataFrame:
        df = X.copy()
        df['treatment'] = T
        df['outcome'] = Y
        df['propensity_score'] = ps
        
        for col in X_c.columns:
            df[f'{col}_is_conf'] = 1
            
        return df
    
    def generate_multiple_datasets(self, n_datasets: int = 10) -> Dict:
        datasets = {}
        for i in range(n_datasets):
            self.config.random_seed += 1
            self.rng = np.random.RandomState(self.config.random_seed)
            df, true_params = self.generate()
            datasets[f'dataset_{i}'] = {
                'data': df,
                'true_params': true_params
            }
        return datasets