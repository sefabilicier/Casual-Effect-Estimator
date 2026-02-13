from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class DataConfig:
    n_samples: int = 2000
    n_features: int = 5
    n_confounders: int = 3
    n_instruments: int = 1
    treatment_effect: float = 2.0
    effect_heterogeneity: bool = True
    confounding_strength: float = 1.0
    nonlinearity: str = 'none'  # 'none', 'moderate', 'high'
    noise_level: float = 0.5
    random_seed: int = 42
    
    treatment_assign: str = 'logistic'
    treatment_propensity: float = 0.5
    
    test_size: float = 0.3
    val_size: float = 0.2


@dataclass
class ModelConfig:
    run_linear_regression: bool = True
    run_logistic_propensity: bool = True
    run_iptw: bool = True
    run_stratification: bool = True
    
    run_causal_forest: bool = True
    run_bart: bool = True
    run_tarnet: bool = True
    run_xlearner: bool = True
    
    run_doubly_robust: bool = True
    run_aipw: bool = True
    
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    
    n_folds: int = 5
    random_state: int = 42


@dataclass
class UIConfig:
    """Configuration for UI elements"""
    page_title: str = "Causal Effect Estimator"
    layout: str = "wide"
    
    color_primary: str = "#1f77b4"
    color_secondary: str = "#ff7f0e"
    color_success: str = "#2ca02c"
    color_danger: str = "#d62728"
    
    plot_height: int = 500
    plot_width: int = 800