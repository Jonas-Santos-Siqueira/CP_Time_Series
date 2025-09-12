from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

class BasePIModel:
    def fit(self, X, y):
        raise NotImplementedError
    def get_prediction(self, X_new, y_true: Optional[np.ndarray] = None):
        raise NotImplementedError
    def predict_interval(self, X_new, y_true: Optional[np.ndarray] = None):
        res = self.get_prediction(X_new, y_true=y_true)
        return res.lower_bounds, res.upper_bounds

@dataclass
class BasePIResults:
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    point_predictions: Optional[np.ndarray]
    alpha: float
    y_true: Optional[np.ndarray] = None
    def summary(self) -> str:
        import numpy as np
        n = len(self.lower_bounds)
        width = np.mean(self.upper_bounds - self.lower_bounds) if n else float("nan")
        s = [f"Results: n={n}, alpha={self.alpha:.3f}, avg_width={width:.4f}"]
        if self.y_true is not None and len(self.y_true) == n:
            y = np.asarray(self.y_true).reshape(-1)
            inside = (y >= self.lower_bounds) & (y <= self.upper_bounds)
            coverage = float(np.mean(inside))
            s.append(f"empirical_coverage={coverage:.3f}")
        return " | ".join(s)
