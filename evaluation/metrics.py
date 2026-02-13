import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import scipy.stats as stats


class CausalMetrics:    
    @staticmethod
    def ate_error(estimated_ate: float, true_ate: float) -> Dict:
        try:
            bias = estimated_ate - true_ate
            relative_bias = bias / true_ate if true_ate != 0 else np.nan
            abs_error = np.abs(bias)
            sq_error = bias ** 2
        except:
            bias = np.nan
            relative_bias = np.nan
            abs_error = np.nan
            sq_error = np.nan
        
        return {
            'ate_bias': bias,
            'ate_relative_bias': relative_bias,
            'ate_absolute_error': abs_error,
            'ate_squared_error': sq_error
        }
    
    @staticmethod
    def ite_metrics(estimated_ites: np.ndarray, true_ites: np.ndarray) -> Dict:
        return {
            'ite_mse': mean_squared_error(true_ites, estimated_ites),
            'ite_rmse': np.sqrt(mean_squared_error(true_ites, estimated_ites)),
            'ite_mae': mean_absolute_error(true_ites, estimated_ites),
            'ite_correlation': np.corrcoef(true_ites, estimated_ites)[0, 1],
            'pearson_r': stats.pearsonr(true_ites, estimated_ites)[0],
            'spearman_r': stats.spearmanr(true_ites, estimated_ites)[0]
        }
    
    @staticmethod
    def policy_value(estimated_ites: np.ndarray, true_ites: np.ndarray, 
                    treatment_cost: float = 0.0) -> Dict:
        optimal_treat = true_ites > treatment_cost
        optimal_value = np.mean(true_ites[optimal_treat]) * np.sum(optimal_treat)
        
        estimated_treat = estimated_ites > treatment_cost
        estimated_value = np.mean(true_ites[estimated_treat]) * np.sum(estimated_treat)
        
        random_treat = np.random.binomial(1, 0.5, len(true_ites))
        random_value = np.mean(true_ites[random_treat == 1]) * np.sum(random_treat == 1)
        
        return {
            'optimal_value': optimal_value,
            'estimated_policy_value': estimated_value,
            'random_policy_value': random_value,
            'value_ratio': estimated_value / optimal_value if optimal_value > 0 else np.nan,
            'treat_agreement': np.mean(optimal_treat == estimated_treat)
        }
    
    @staticmethod
    def coverage(estimated_ci_lower: float, estimated_ci_upper: float, 
                true_ate: float) -> bool:
        return (estimated_ci_lower <= true_ate <= estimated_ci_upper)
    
    @staticmethod
    def summarize_all(estimates: Dict, true_ate: float, 
                     true_ites: Optional[np.ndarray] = None) -> pd.DataFrame:
        results = []
        
        for estimator_name, result in estimates.items():
            row = {
                'estimator': estimator_name,
                'ate': result.get('ate', np.nan),
                'bias': result.get('ate', np.nan) - true_ate,
                'abs_error': np.abs(result.get('ate', np.nan) - true_ate)
            }
            
            if 'ci_lower' in result and 'ci_upper' in result:
                row['ci_width'] = result['ci_upper'] - result['ci_lower']
                row['covers_true'] = CausalMetrics.coverage(
                    result['ci_lower'], result['ci_upper'], true_ate
                )
            
            if 'ites' in result and true_ites is not None:
                ite_metrics = CausalMetrics.ite_metrics(
                    result['ites'], true_ites
                )
                row['ite_rmse'] = ite_metrics['ite_rmse']
                row['ite_correlation'] = ite_metrics['ite_correlation']
                
                policy_metrics = CausalMetrics.policy_value(
                    result['ites'], true_ites
                )
                row['policy_value_ratio'] = policy_metrics['value_ratio']
            
            results.append(row)
        
        return pd.DataFrame(results).sort_values('abs_error')