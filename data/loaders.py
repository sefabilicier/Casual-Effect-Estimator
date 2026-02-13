import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from typing import Dict, Tuple, Optional
import streamlit as st


class RealDataLoader:
    @staticmethod
    def load_lalonde() -> Tuple[pd.DataFrame, Dict]:
        try:
            lalonde = fetch_openml(data_id=413, as_frame=True)
            df = lalonde.frame
            
            df['treatment'] = df['treat'].astype(int)
            
            df['outcome'] = df['re78']
            
            features = ['age', 'education', 'black', 'hispanic', 
                       'married', 'nodegree', 're74', 're75']
            
            X = df[features]
            
            true_params = {
                'known_ate': 1.794,
                'description': 'NSW job training program evaluation'
            }
            
            return df, true_params
            
        except Exception as e:
            st.error(f"Failed to load Lalonde dataset: {e}")
            return None, None
    
    @staticmethod
    def load_twins() -> Tuple[pd.DataFrame, Dict]:
        try:
            twins = fetch_openml(data_id=44129, as_frame=True)
            df = twins.frame
            
            df['treatment'] = df['dbirwt'] < df['birwt']
            
            df['outcome'] = df['mort']
            
            true_params = {
                'description': 'Twins birth weight and mortality',
                'n_pairs': len(df) // 2
            }
            
            return df, true_params
            
        except Exception as e:
            st.error(f"Failed to load Twins dataset: {e}")
            return None, None
    
    @staticmethod
    def load_criteo() -> Tuple[pd.DataFrame, Dict]:
        st.warning("""
        Criteo dataset is large (13GB). 
        Using sample for demonstration.
        """)
        
        n = 10000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, n),
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
        })
        
        df['outcome'] = (df['feature_1'] + df['feature_2'] + 
                        2 * df['treatment'] + np.random.randn(n))
        
        true_params = {
            'description': 'Criteo uplift prediction (synthetic sample)',
            'true_ate': 2.0
        }
        
        return df, true_params