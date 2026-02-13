import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Optional
import warnings


class CausalDiagnostics:
    @staticmethod
    def check_overlap(propensity_scores: np.ndarray, T: np.ndarray) -> Dict:
        ps_treated = propensity_scores[T == 1]
        ps_control = propensity_scores[T == 0]
        
        overlap_stats = {
            'min_treated': np.min(ps_treated),
            'max_treated': np.max(ps_treated),
            'min_control': np.min(ps_control),
            'max_control': np.max(ps_control),
            'overlap_range': max(min(ps_treated), min(ps_control)),
            'violation_ratio': np.sum(propensity_scores > 0.95) / len(propensity_scores)
        }
        
        
        ks_stat, p_value = stats.ks_2samp(ps_treated, ps_control)
        overlap_stats['ks_statistic'] = ks_stat
        overlap_stats['ks_pvalue'] = p_value
        
        return overlap_stats
    
    @staticmethod
    def check_covariate_balance(X: pd.DataFrame, T: np.ndarray, 
                              weights: Optional[np.ndarray] = None) -> pd.DataFrame:
        results = []
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
                
            mean_t = np.mean(X[col][T == 1])
            mean_c = np.mean(X[col][T == 0])
            var_t = np.var(X[col][T == 1])
            var_c = np.var(X[col][T == 0])
            pooled_sd = np.sqrt((var_t + var_c) / 2)
            
            smd_before = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
            
            if weights is not None:
                w_t = weights[T == 1] / np.sum(weights[T == 1])
                w_c = weights[T == 0] / np.sum(weights[T == 0])
                
                weighted_mean_t = np.sum(X[col][T == 1] * w_t)
                weighted_mean_c = np.sum(X[col][T == 0] * w_c)
                
                weighted_var_t = np.sum(w_t * (X[col][T == 1] - weighted_mean_t)**2)
                weighted_var_c = np.sum(w_c * (X[col][T == 0] - weighted_mean_c)**2)
                weighted_pooled_sd = np.sqrt((weighted_var_t + weighted_var_c) / 2)
                
                smd_after = (weighted_mean_t - weighted_mean_c) / weighted_pooled_sd if weighted_pooled_sd > 0 else 0
                balance_improvement = (np.abs(smd_before) - np.abs(smd_after)) / np.abs(smd_before) if smd_before != 0 else 0
            else:
                smd_after = np.nan
                balance_improvement = np.nan
            
            results.append({
                'covariate': col,
                'mean_treated': mean_t,
                'mean_control': mean_c,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'balance_improvement': balance_improvement
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def check_propensity_calibration(propensity_scores: np.ndarray, 
                                   T: np.ndarray,
                                   n_bins: int = 10) -> Dict:
        try:
            fraction_positive, mean_predicted = calibration_curve(
                T, propensity_scores, n_bins=n_bins
            )
            
            brier_score = np.mean((propensity_scores - T) ** 2)
            
            expected_pos = np.sum(propensity_scores)
            expected_neg = np.sum(1 - propensity_scores)
            observed_pos = np.sum(T)
            observed_neg = len(T) - observed_pos
            
            hl_stat = ((observed_pos - expected_pos)**2 / expected_pos + 
                      (observed_neg - expected_neg)**2 / expected_neg)
            
            return {
                'fraction_positive': fraction_positive.tolist(),
                'mean_predicted': mean_predicted.tolist(),
                'brier_score': brier_score,
                'hl_statistic': hl_stat,
                'is_calibrated': brier_score < 0.25
            }
        except:
            return {
                'brier_score': np.nan,
                'is_calibrated': False
            }
    
    @staticmethod
    def placebo_test(X: pd.DataFrame, T: np.ndarray, Y: np.ndarray,
                    n_permutations: int = 100) -> Dict:
        original_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
        
        placebo_ates = []
        np.random.seed(42)
        
        for _ in range(n_permutations):
            T_placebo = np.random.permutation(T)
            ate_placebo = np.mean(Y[T_placebo == 1]) - np.mean(Y[T_placebo == 0])
            placebo_ates.append(ate_placebo)
        
        p_value = np.mean(np.abs(placebo_ates) >= np.abs(original_ate))
        
        return {
            'original_ate': original_ate,
            'mean_placebo': np.mean(placebo_ates),
            'std_placebo': np.std(placebo_ates),
            'p_value': p_value,
            'placebo_ates': placebo_ates
        }