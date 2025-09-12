from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional, Literal, List
try:
    from sklearn.base import clone
except Exception:
    from copy import deepcopy as clone

from .base import BasePIModel, BasePIResults
from .bootstrap import moving_block_bootstrap_indices

Aggregation = Literal["mean", "median"]
CenterStrategy = Literal["phi", "loo_quantile"]
BootstrapMode = Literal["iid", "block"]

def _bootstrap_indices_iid(n: int, size: int, B: int, rng: np.random.Generator) -> List[np.ndarray]:
    return [rng.integers(0, n, size=size) for _ in range(B)]

class EnbPIResults(BasePIResults):
    def plot_interval(self, ax=None, show=True):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        x = np.arange(len(self.lower_bounds))
        ax.fill_between(x, self.lower_bounds, self.upper_bounds, alpha=0.2, label="EnbPI interval")
        if self.point_predictions is not None:
            ax.plot(x, self.point_predictions, label="Ensemble prediction")
        if self.y_true is not None:
            ax.plot(x, self.y_true, linestyle="--", label="y_true")
        ax.legend()
        ax.set_title(f"EnbPI intervals (1-alpha={1-self.alpha:.2f})")
        ax.set_xlabel("t"); ax.set_ylabel("y")
        if show: plt.show()
        return ax

class EnbPIModel(BasePIModel):
    def __init__(
        self,
        base_model,
        B: int = 30,
        alpha: float = 0.1,
        aggregation: Aggregation = "mean",
        center: CenterStrategy = "loo_quantile",
        batch_size: int = 1,
        window_size: Optional[int] = None,
        random_state: Optional[int] = None,
        bootstrap: BootstrapMode = "iid",
        block_length: Optional[int] = None,
    ):
        self.base_model = base_model
        self.B = int(B); self.alpha = float(alpha)
        self.aggregation = aggregation; self.center = center
        self.batch_size = int(batch_size); self.window_size = window_size
        self.random_state = random_state
        self.bootstrap = bootstrap; self.block_length = block_length
        self.models_ = []; self.in_boot_sample_ = None
        self.resid_hist_ = None; self._train_size = None

    def _aggregate(self, arr: np.ndarray, axis: int) -> np.ndarray:
        if self.aggregation == "mean": return np.mean(arr, axis=axis)
        if self.aggregation == "median": return np.median(arr, axis=axis)
        raise ValueError("aggregation must be 'mean' or 'median'")

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).reshape(-1)
        n = X.shape[0]
        if self.window_size is None: self.window_size = n
        self._train_size = n
        rng = np.random.default_rng(self.random_state)
        if self.bootstrap == "iid":
            boot_idx = _bootstrap_indices_iid(n, n, self.B, rng)
        elif self.bootstrap == "block":
            if not self.block_length: raise ValueError("For bootstrap='block', provide a valid block_length.")
            boot_idx = moving_block_bootstrap_indices(n, self.block_length, self.B, rng)
        else:
            raise ValueError("bootstrap must be 'iid' or 'block'")
        self.models_ = []; self.in_boot_sample_ = np.zeros((self.B, n), dtype=bool)
        preds_train = np.empty((self.B, n), dtype=float)
        for b in range(self.B):
            model_b = clone(self.base_model); idx = boot_idx[b]
            self.in_boot_sample_[b, idx] = True
            if hasattr(model_b, "fit_with_indices") and callable(getattr(model_b, "fit_with_indices")):
                model_b.fit_with_indices(idx)
            else:
                model_b.fit(X[idx], y[idx])
            self.models_.append(model_b)
            preds_train[b] = model_b.predict(X)
        count_incl = self.in_boot_sample_.sum(axis=0)
        sum_all = preds_train.sum(axis=0)
        sum_incl = (preds_train * self.in_boot_sample_).sum(axis=0)
        denom = (self.B - count_incl).astype(float); denom[denom == 0.0] = np.nan
        if self.aggregation == "mean":
            pred_lo = (sum_all - sum_incl) / denom
        else:
            pred_lo = np.empty(n, dtype=float)
            for i in range(n):
                mask = ~self.in_boot_sample_[:, i]
                vals = preds_train[mask, i]
                pred_lo[i] = np.median(vals) if vals.size else np.nan
        resid = np.abs(y - pred_lo)
        from collections import deque
        self.resid_hist_ = deque(resid.tolist()[-self.window_size:], maxlen=self.window_size)
        return self

    def _center_phi(self, preds_test: np.ndarray) -> np.ndarray:
        return self._aggregate(preds_test, axis=0)

    def _center_loo_quantile(self, preds_test: np.ndarray) -> np.ndarray:
        B, T1 = preds_test.shape; n = self._train_size
        sum_all = preds_test.sum(axis=0)
        count_incl = self.in_boot_sample_.sum(axis=0)
        M_incl_sums = self.in_boot_sample_.T @ preds_test
        denom = (self.B - count_incl).astype(float); denom[denom == 0.0] = np.nan
        if self.aggregation == "mean":
            num = (sum_all[None, :] - M_incl_sums)
            f_looi = num / denom[:, None]
        else:
            f_looi = np.empty((n, T1), dtype=float)
            for i in range(n):
                mask = ~self.in_boot_sample_[:, i]
                vals = preds_test[mask, :]
                f_looi[i, :] = np.median(vals, axis=0) if vals.size else np.nan
        q = 1.0 - self.alpha
        return np.nanquantile(f_looi, q, axis=0, method="higher")

    def _current_width(self, resid_array: np.ndarray) -> float:
        return float(np.quantile(resid_array, 1.0 - self.alpha, method="higher"))

    def get_prediction(self, X_new, y_true: Optional[np.ndarray] = None) -> EnbPIResults:
        if not self.models_: raise RuntimeError("Model is not fitted. Call .fit(X, y) first.")
        X_new = np.asarray(X_new); T1 = X_new.shape[0]
        preds_test = np.vstack([m.predict(X_new) for m in self.models_])
        centers = self._center_phi(preds_test) if self.center=="phi" else self._center_loo_quantile(preds_test)
        lowers = np.empty(T1, dtype=float); uppers = np.empty(T1, dtype=float)
        if y_true is None:
            width = self._current_width(np.array(self.resid_hist_))
            lowers[:] = centers - width; uppers[:] = centers + width
            return EnbPIResults(lowers, uppers, centers, self.alpha, None)
        y_true = np.asarray(y_true).reshape(-1)
        if y_true.shape[0] != T1: raise ValueError("y_true must have the same length as X_new when provided.")
        resid_hist = self.resid_hist_.copy()
        for start in range(0, T1, self.batch_size):
            end = min(start + self.batch_size, T1)
            width = self._current_width(np.array(resid_hist))
            lowers[start:end] = centers[start:end] - width
            uppers[start:end] = centers[start:end] + width
            new_res = np.abs(y_true[start:end] - centers[start:end])
            for r in new_res: resid_hist.append(r)
        self.resid_hist_ = resid_hist
        return EnbPIResults(lowers, uppers, centers, self.alpha, y_true)
