import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Callable, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
from data.generator import CausalDataGenerator
from evaluation.metrics import CausalMetrics


class CausalBenchmark:
    
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self.results = {}
        
    def run_benchmark(self, estimators: Dict[str, Callable], 
                     n_repetitions: int = 10,
                     sample_sizes: List[int] = None) -> pd.DataFrame:
        if sample_sizes is None:
            sample_sizes = [500, 1000, 2000, 5000, 10000]
        
        all_results = []
        
        for n in tqdm(sample_sizes, desc="Sample sizes"):
            self.data_config.n_samples = n
            
            for rep in tqdm(range(n_repetitions), desc=f"Repetitions (n={n})", leave=False):
                
                generator = CausalDataGenerator(self.data_config)
                df, true_params = generator.generate()
                
                X = df.drop(['treatment', 'outcome', 'propensity_score'], axis=1)
                T = df['treatment'].values
                Y = df['outcome'].values
                
                for name, estimator_class in estimators.items():
                    try:
                        start_time = time.time()
                        
                        estimator = estimator_class()
                        estimator.fit(X, T, Y)
                        
                        fit_time = time.time() - start_time
                        
                        ate = estimator.estimate_ate()
                        ites = estimator.get_ite() if hasattr(estimator, 'get_ite') else None
                        
                        metrics = CausalMetrics.ate_error(ate, true_params['ate'])
                        metrics['fit_time'] = fit_time
                        metrics['sample_size'] = n
                        metrics['repetition'] = rep
                        metrics['estimator'] = name
                        
                        if ites is not None:
                            ite_metrics = CausalMetrics.ite_metrics(
                                ites, true_params['ite']
                            )
                            metrics.update(ite_metrics)
                        
                        all_results.append(metrics)
                        
                    except Exception as e:
                        print(f"Error with {name} on repetition {rep}: {e}")
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        if self.results.empty:
            return pd.DataFrame()
        
        summary = self.results.groupby(['estimator', 'sample_size']).agg({
            'ate_bias': ['mean', 'std'],
            'ate_absolute_error': ['mean', 'std'],
            'ite_rmse': ['mean', 'std'] if 'ite_rmse' in self.results else None,
            'fit_time': ['mean', 'std']
        }).round(4)
        
        return summary
    
    def get_best_estimator(self, metric: str = 'ate_absolute_error') -> str:
        if self.results.empty:
            return None
        
        avg_performance = self.results.groupby('estimator')[metric].mean()
        return avg_performance.idxmin()