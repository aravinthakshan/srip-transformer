"""
Hierarchical Features Module
============================
Wrapper for CatBoost-based hierarchical feature extraction.
Provides μ, σ features from upstream variables based on the
hierarchical Bayesian model.

No gradients - CPU only. This is a feature extraction module,
not a trainable component.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Optional, Union
from scipy.stats import norm

# Import CatBoost if available
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. HierarchicalFeatures will not work.")


class HierarchicalFeatures:
    """
    Wrapper for CatBoost-based hierarchical feature extraction.
    
    Uses pre-trained CatBoost models to extract μ (mean) and σ (std)
    features from upstream variables. The predictions are conditioned
    on rainfall (P=0 vs P>0).
    
    This module has NO learnable parameters and runs on CPU only.
    It is used to inject physics-based features into the neural network.
    
    Args:
        model_path: Path to saved CatBoost models (dict with 4 models)
        quantiles: List of quantiles to compute (default: [0.5] for median)
    """
    
    def __init__(
        self,
        model_path: str = None,
        quantiles: list = None
    ):
        self.quantiles = quantiles or [0.5]
        self.models = {
            'mean_P_eq_0': None,
            'mean_P_gt_0': None,
            'variance_P_eq_0': None,
            'variance_P_gt_0': None
        }
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
            
    def load(self, model_path: str):
        """Load pre-trained CatBoost models."""
        with open(model_path, 'rb') as f:
            saved_models = pickle.load(f)
            
        for key in self.models:
            if key in saved_models:
                self.models[key] = saved_models[key]
                
    def save(self, model_path: str):
        """Save trained CatBoost models."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)
            
    def is_fitted(self) -> bool:
        """Check if models are fitted."""
        return (self.models['mean_P_eq_0'] is not None and 
                self.models['mean_P_gt_0'] is not None)
                
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feeder_col: str,
        rainfall_col: str,
        covariate_cols: list,
        cb_params: dict = None
    ):
        """
        Train CatBoost models on training data.
        
        Args:
            df: Training DataFrame
            target_col: Target streamflow column
            feeder_col: Upstream streamflow column (lagged)
            rainfall_col: Rainfall column (lagged)
            covariate_cols: List of other covariate columns (lagged)
            cb_params: CatBoost hyperparameters
        """
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed. Cannot fit hierarchical models.")
            
        cb_params = cb_params or {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'verbose': 0,
            'random_seed': 42
        }
        
        # Split by rainfall condition
        df_p_eq_0 = df[df[rainfall_col] == 0].copy()
        df_p_gt_0 = df[df[rainfall_col] > 0].copy()
        
        # Train mean models
        print("Training Mean Model for P = 0 case...")
        if not df_p_eq_0.empty:
            X_p_eq_0 = df_p_eq_0[[feeder_col] + covariate_cols]
            y_p_eq_0 = df_p_eq_0[target_col]
            
            self.models['mean_P_eq_0'] = CatBoostRegressor(**cb_params)
            self.models['mean_P_eq_0'].fit(X_p_eq_0, y_p_eq_0)
            
            # Residuals for variance model
            residuals_p_eq_0 = y_p_eq_0 - self.models['mean_P_eq_0'].predict(X_p_eq_0)
            
            # Train variance model
            if len(residuals_p_eq_0) > 1:
                print("Training Variance Model for P = 0 case...")
                log_sq_residuals = np.log(residuals_p_eq_0**2 + 1e-6)
                self.models['variance_P_eq_0'] = CatBoostRegressor(**cb_params)
                self.models['variance_P_eq_0'].fit(X_p_eq_0, log_sq_residuals)
                
        print("Training Mean Model for P > 0 case...")
        if not df_p_gt_0.empty:
            X_p_gt_0 = df_p_gt_0[[feeder_col, rainfall_col] + covariate_cols]
            y_p_gt_0 = df_p_gt_0[target_col]
            
            self.models['mean_P_gt_0'] = CatBoostRegressor(**cb_params)
            self.models['mean_P_gt_0'].fit(X_p_gt_0, y_p_gt_0)
            
            # Residuals for variance model
            residuals_p_gt_0 = y_p_gt_0 - self.models['mean_P_gt_0'].predict(X_p_gt_0)
            
            # Train variance model
            if len(residuals_p_gt_0) > 1:
                print("Training Variance Model for P > 0 case...")
                log_sq_residuals = np.log(residuals_p_gt_0**2 + 1e-6)
                X_var = df_p_gt_0[[feeder_col, rainfall_col] + covariate_cols]
                self.models['variance_P_gt_0'] = CatBoostRegressor(**cb_params)
                self.models['variance_P_gt_0'].fit(X_var, log_sq_residuals)
                
        print("Hierarchical model training complete.")
        
    def transform(
        self,
        feeder_values: np.ndarray,
        rainfall_values: np.ndarray,
        covariate_values: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract μ and σ features for given inputs.
        
        Args:
            feeder_values: Upstream streamflow values [N]
            rainfall_values: Rainfall values [N]
            covariate_values: Other covariates [N, num_covariates] (optional)
            
        Returns:
            mu: Mean predictions [N]
            sigma: Standard deviation predictions [N]
        """
        if not self.is_fitted():
            raise ValueError("Hierarchical models not fitted yet")
            
        n_samples = len(feeder_values)
        mu = np.zeros(n_samples)
        sigma = np.zeros(n_samples)
        
        for i in range(n_samples):
            is_rain = rainfall_values[i] > 0
            
            if is_rain:
                mean_model = self.models['mean_P_gt_0']
                var_model = self.models['variance_P_gt_0']
                features = [feeder_values[i], rainfall_values[i]]
            else:
                mean_model = self.models['mean_P_eq_0']
                var_model = self.models['variance_P_eq_0']
                features = [feeder_values[i]]
                
            if covariate_values is not None:
                features.extend(covariate_values[i].tolist())
                
            features = np.array(features).reshape(1, -1)
            
            if mean_model is not None:
                mu[i] = max(0.1, mean_model.predict(features)[0])
                
                if var_model is not None:
                    log_sigma_sq = var_model.predict(features)[0]
                    sigma[i] = max(0.01, np.sqrt(np.exp(log_sigma_sq)))
                else:
                    sigma[i] = mu[i] * 0.1  # Default: 10% of mean
            else:
                mu[i] = np.nan
                sigma[i] = np.nan
                
        return mu, sigma
    
    def get_quantile_predictions(
        self,
        feeder_values: np.ndarray,
        rainfall_values: np.ndarray,
        covariate_values: np.ndarray = None,
        quantiles: list = None
    ) -> dict:
        """
        Get quantile predictions assuming log-normal distribution.
        
        Args:
            feeder_values: Upstream streamflow values
            rainfall_values: Rainfall values
            covariate_values: Other covariates (optional)
            quantiles: List of quantiles (default: [0.1, 0.5, 0.9])
            
        Returns:
            Dict with quantile keys and prediction arrays
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]
        mu, sigma = self.transform(feeder_values, rainfall_values, covariate_values)
        
        predictions = {}
        for q in quantiles:
            # Convert to log-normal parameters
            sigma_lnQ_sq = np.log(1 + (sigma / np.maximum(mu, 0.01))**2)
            sigma_lnQ = np.sqrt(sigma_lnQ_sq)
            mu_lnQ = np.log(np.maximum(mu, 0.01)) - 0.5 * sigma_lnQ_sq
            
            # Compute quantile
            z_score = norm.ppf(q)
            log_pred = mu_lnQ + sigma_lnQ * z_score
            predictions[f'q{int(q*100)}'] = np.exp(log_pred)
            
        return predictions
    
    def get_feature_vector(
        self,
        feeder_values: np.ndarray,
        rainfall_values: np.ndarray,
        covariate_values: np.ndarray = None,
        include_quantiles: bool = True
    ) -> np.ndarray:
        """
        Get feature vector for injection into neural network.
        
        Args:
            feeder_values: Upstream streamflow
            rainfall_values: Rainfall
            covariate_values: Other covariates
            include_quantiles: If True, include quantile predictions
            
        Returns:
            Feature array [N, num_features]
        """
        mu, sigma = self.transform(feeder_values, rainfall_values, covariate_values)
        
        if include_quantiles:
            quantile_preds = self.get_quantile_predictions(
                feeder_values, rainfall_values, covariate_values,
                quantiles=[0.1, 0.5, 0.9]
            )
            features = np.column_stack([
                mu, sigma,
                quantile_preds['q10'],
                quantile_preds['q50'],
                quantile_preds['q90']
            ])
        else:
            features = np.column_stack([mu, sigma])
            
        return features
