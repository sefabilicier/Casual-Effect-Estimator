import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import streamlit as st


class CausalVisualizer:    
    @staticmethod
    def plot_propensity_distribution(propensity_scores: np.ndarray, 
                                    T: np.ndarray) -> go.Figure:
        """
        Plot distribution of propensity scores by treatment group
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=propensity_scores[T == 1],
            name='Treated',
            opacity=0.75,
            nbinsx=30,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Histogram(
            x=propensity_scores[T == 0],
            name='Control',
            opacity=0.75,
            nbinsx=30,
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title='Propensity Score Distribution by Treatment Group',
            xaxis_title='Propensity Score',
            yaxis_title='Count',
            barmode='overlay',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_covariate_balance(balance_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        # Sort by absolute SMD before
        balance_df = balance_df.sort_values('smd_before', key=abs, ascending=True)
        
        fig.add_trace(go.Scatter(
            x=balance_df['smd_before'],
            y=balance_df['covariate'],
            name='Before Weighting',
            mode='markers',
            marker=dict(symbol='circle', size=10, color='#d62728'),
        ))
        
        fig.add_trace(go.Scatter(
            x=balance_df['smd_after'],
            y=balance_df['covariate'],
            name='After Weighting',
            mode='markers',
            marker=dict(symbol='diamond', size=10, color='#2ca02c'),
        ))
        
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=-0.1, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Covariate Balance: Standardized Mean Differences',
            xaxis_title='Standardized Mean Difference',
            yaxis_title='Covariate',
            height=600,
            template='plotly_white',
            legend=dict(x=0.8, y=0.1)
        )
        
        return fig
    
    @staticmethod
    def plot_cate_distribution(estimates_dict: Dict[str, np.ndarray]) -> go.Figure:
        fig = go.Figure()
        
        for name, ites in estimates_dict.items():
            fig.add_trace(go.Violin(
                y=ites,
                name=name,
                box_visible=True,
                meanline_visible=True,
                opacity=0.7
            ))
        
        fig.update_layout(
            title='Individual Treatment Effect Distributions',
            yaxis_title='Treatment Effect',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_ate_comparison(results_dict: Dict, true_ate: float) -> go.Figure:
        fig = go.Figure()
        
        fig.add_vline(x=true_ate, line_dash="dash", line_color="red",
                    annotation_text="True ATE", annotation_position="top right")
        
        estimators = list(results_dict.keys())
        ates = [results_dict[name].get('ate', np.nan) for name in estimators]
        ci_lowers = [results_dict[name].get('ci_lower', None) for name in estimators]
        ci_uppers = [results_dict[name].get('ci_upper', None) for name in estimators]
        
        fig.add_trace(go.Scatter(
            x=ates,
            y=estimators,
            mode='markers+text',
            marker=dict(size=12, color='#1f77b4'),
            text=[f"{ate:.3f}" for ate in ates],
            textposition="top center",
            name='ATE Estimate',
            showlegend=False
        ))
        
        for i, (estimator, ci_lower, ci_upper) in enumerate(zip(estimators, ci_lowers, ci_uppers)):
            if ci_lower is not None and ci_upper is not None:
                fig.add_trace(go.Scatter(
                    x=[ci_lower, ci_upper],
                    y=[estimator, estimator],
                    mode='lines',
                    line=dict(width=2, color='rgba(31, 119, 180, 0.5)'),
                    showlegend=False if i > 0 else True,
                    name='95% CI'
                ))
        
        fig.update_layout(
            title='ATE Estimates Comparison',
            xaxis_title='Average Treatment Effect',
            yaxis_title='Estimator',
            height=400,
            template='plotly_white',
            showlegend=True
        )
        
        return fig

    @staticmethod
    def plot_overlap_assessment(propensity_scores: np.ndarray,
                               T: np.ndarray) -> go.Figure:
        fig = sp.make_subplots(rows=2, cols=1,
                              subplot_titles=('Propensity Score Overlap',
                                            'Cumulative Distribution'),
                              shared_xaxes=True)
        
        fig.add_trace(
            go.Histogram(x=propensity_scores[T == 1], name='Treated',
                        opacity=0.7, nbinsx=40, marker_color='#1f77b4'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=propensity_scores[T == 0], name='Control',
                        opacity=0.7, nbinsx=40, marker_color='#ff7f0e'),
            row=1, col=1
        )
        
        for group, color, name in [(T == 1, '#1f77b4', 'Treated'),
                                  (T == 0, '#ff7f0e', 'Control')]:
            sorted_ps = np.sort(propensity_scores[group])
            ecdf = np.arange(1, len(sorted_ps) + 1) / len(sorted_ps)
            
            fig.add_trace(
                go.Scatter(x=sorted_ps, y=ecdf, mode='lines',
                          name=name, line=dict(color=color)),
                row=2, col=1
            )
        
        fig.update_layout(height=700, template='plotly_white',
                         showlegend=True)
        fig.update_xaxes(title_text="Propensity Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
        
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(results: Dict) -> go.Figure:
        fig = go.Figure()
        
        gamma_values = results.get('gamma_values', [])
        ate_values = results.get('ate_values', [])
        ci_lower = results.get('ci_lower', [])
        ci_upper = results.get('ci_upper', [])
        
        fig.add_trace(go.Scatter(
            x=gamma_values,
            y=ate_values,
            mode='lines+markers',
            name='Estimated ATE',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=gamma_values + gamma_values[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='Sensitivity Analysis for Unobserved Confounding',
            xaxis_title='Sensitivity Parameter (Γ)',
            yaxis_title='ATE',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_learning_curves(results: Dict) -> go.Figure:
        fig = go.Figure()
        
        for method, method_results in results.items():
            sample_sizes = method_results.get('sample_sizes', [])
            errors = method_results.get('errors', [])
            
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=errors,
                mode='lines+markers',
                name=method
            ))
        
        fig.update_layout(
            title='Learning Curves: Error vs Sample Size',
            xaxis_title='Sample Size',
            yaxis_title='Absolute Error',
            height=500,
            template='plotly_white',
            xaxis_type='log',
            yaxis_type='log'
        )
        
        return fig