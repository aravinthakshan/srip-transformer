"""
Rating Curve Fitter Module
==========================
Physics-based module for converting water level to streamflow
using a pre-fitted rating curve.

Zero learnable parameters - this is a deterministic transformation.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Callable
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Rating curve functional forms
def quadratic(wse, a, b, c):
    """Quadratic: Q = a*H^2 + b*H + c"""
    return a * (wse ** 2) + b * wse + c


def exponential_offset(wse, a, b):
    """Exponential: Q = a * exp(b * H)"""
    return a * np.exp(b * wse)


def shifted_quadratic(wse, a, b):
    """Shifted quadratic: Q = a * (H - b)^2"""
    return a * (wse - b) ** 2


def shifted_cubic(wse, a, b):
    """Shifted cubic: Q = a * (H - b)^3"""
    return a * (wse - b) ** 3


def power_law(wse, a, b, c):
    """Power law: Q = a * (H - c)^b"""
    return a * np.power(np.maximum(wse - c, 0.001), b)


class RatingCurveFitter:
    """
    Rating curve module for water level to streamflow conversion.
    
    Fits various functional forms to observed (H, Q) pairs and uses
    the best-fit curve for prediction. This is a physics-based module
    with zero trainable parameters.
    
    The rating curve relationship Q = f(H) is a fundamental hydrological
    relationship that should be exploited when available.
    
    Args:
        curve_type: Optional specific curve type to use
                   ('quadratic', 'exponential', 'shifted_quadratic', 
                    'shifted_cubic', 'power_law', or 'auto')
    """
    
    CURVE_FUNCTIONS = {
        "quadratic": (quadratic, [1, 1, 1]),
        "exponential": (exponential_offset, [1e-5, 0.01]),
        "shifted_quadratic": (shifted_quadratic, [1, 260]),
        "shifted_cubic": (shifted_cubic, [1, 260]),
        "power_law": (power_law, [1, 2, 250])
    }
    
    def __init__(self, curve_type: str = 'auto'):
        self.curve_type = curve_type
        self.best_model = None
        self.fitted = False
        
    def fit(
        self,
        water_level: np.ndarray,
        streamflow: np.ndarray,
        verbose: bool = True
    ) -> bool:
        """
        Fit rating curve to observed data.
        
        Tries multiple functional forms and selects the one with
        lowest RMSE.
        
        Args:
            water_level: Observed water levels [N]
            streamflow: Observed streamflow values [N]
            verbose: Print fitting results
            
        Returns:
            True if fitting succeeded, False otherwise
        """
        # Filter out invalid values
        valid_mask = ~(np.isnan(water_level) | np.isnan(streamflow) | 
                       (streamflow <= 0) | (water_level <= 0))
        water_level = water_level[valid_mask]
        streamflow = streamflow[valid_mask]
        
        if len(water_level) < 10:
            print("Warning: Not enough valid data points for rating curve fitting")
            return False
            
        results = []
        
        if self.curve_type == 'auto':
            curves_to_try = self.CURVE_FUNCTIONS.items()
        else:
            if self.curve_type not in self.CURVE_FUNCTIONS:
                raise ValueError(f"Unknown curve type: {self.curve_type}")
            curves_to_try = [(self.curve_type, self.CURVE_FUNCTIONS[self.curve_type])]
            
        for name, (func, init_params) in curves_to_try:
            try:
                params, _ = curve_fit(
                    func, water_level, streamflow,
                    p0=init_params, maxfev=10000
                )
                q_pred = func(water_level, *params)
                
                # Compute metrics
                rmse = np.sqrt(mean_squared_error(streamflow, q_pred))
                mae = mean_absolute_error(streamflow, q_pred)
                r2 = r2_score(streamflow, q_pred)
                
                results.append({
                    "name": name,
                    "function": func,
                    "params": params,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                })
                
                if verbose:
                    print(f"  {name}: RMSE={rmse:.3f}, R²={r2:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  {name}: Failed to fit - {e}")
                    
        if not results:
            print("Warning: No rating curve could be fitted")
            return False
            
        # Select best by RMSE
        results.sort(key=lambda x: x["RMSE"])
        self.best_model = results[0]
        self.fitted = True
        
        if verbose:
            print(f"\nBest Rating Curve: {self.best_model['name']}")
            print(f"  RMSE: {self.best_model['RMSE']:.3f}")
            print(f"  R²: {self.best_model['R2']:.4f}")
            print(f"  Params: {self.best_model['params']}")
            
        return True
    
    def predict(self, water_level: np.ndarray) -> np.ndarray:
        """
        Convert water level to streamflow using fitted curve.
        
        Args:
            water_level: Water level values [N]
            
        Returns:
            streamflow: Predicted streamflow values [N]
        """
        if not self.fitted or self.best_model is None:
            raise ValueError("Rating curve not fitted yet. Call fit() first.")
            
        try:
            predictions = self.best_model["function"](
                water_level, *self.best_model["params"]
            )
            # Ensure non-negative
            return np.maximum(predictions, 0.0)
        except Exception as e:
            print(f"Warning: Rating curve prediction failed: {e}")
            return np.full_like(water_level, np.nan)
            
    def forward(self, water_level_pred: np.ndarray) -> np.ndarray:
        """
        Alias for predict() - for consistency with neural network interface.
        
        Args:
            water_level_pred: Predicted water levels [N]
            
        Returns:
            streamflow_estimate: Estimated streamflow [N]
        """
        return self.predict(water_level_pred)
    
    def save(self, filepath: str):
        """Save fitted rating curve to file."""
        with open(filepath, "wb") as f:
            pickle.dump({
                'best_model': self.best_model,
                'curve_type': self.curve_type,
                'fitted': self.fitted
            }, f)
            
    def load(self, filepath: str):
        """Load fitted rating curve from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        # Handle old format (just best_model dict)
        if isinstance(data, dict) and 'name' in data:
            self.best_model = data
            self.fitted = True
        else:
            self.best_model = data.get('best_model')
            self.curve_type = data.get('curve_type', 'auto')
            self.fitted = data.get('fitted', self.best_model is not None)
            
        # Reconstruct function reference
        if self.best_model and 'name' in self.best_model:
            func_name = self.best_model['name']
            if func_name in self.CURVE_FUNCTIONS:
                self.best_model['function'] = self.CURVE_FUNCTIONS[func_name][0]
                
    def get_info(self) -> dict:
        """Get rating curve information for logging."""
        if not self.fitted:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "name": self.best_model['name'],
            "params": self.best_model['params'].tolist() if hasattr(self.best_model['params'], 'tolist') else list(self.best_model['params']),
            "RMSE": self.best_model['RMSE'],
            "R2": self.best_model['R2']
        }
