"""
Created on Thu Dec 18 08:28:42 2025.

-*- coding: utf-8 -*-
@author: MLechner

Functions for obtaining weights from OLS and ridge regression. Used in bias
adjustment and treatment versions.

"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

from mcf.mcf_cuda import require_torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch import dtype as TorchDType
else:
    Tensor = Any
    TorchDType = Any

type ArrayLike = NDArray[Any] | None
type TensorLike = Tensor | None


def demean_col(x_in: ArrayLike) -> ArrayLike:
    """Substract row mean from columns of numpy array."""
    return None if x_in is None else (x_in - np.mean(x_in, axis=0))


def standardize_col(x_in: ArrayLike, zero_tol: float = 1e-10) -> ArrayLike:
    """Standardize covariates."""
    if x_in is None:
        return None

    x_out = x_in - np.mean(x_in, axis=0)

    std_x = np.std(x_in, axis=0)
    valid = std_x > zero_tol
    if np.all(valid):
        return x_out / std_x

    if np.all(std_x < zero_tol):
        return x_out

    return x_out[:, valid] / std_x[valid]


def project_to_simplex(weight_in: NDArray[np.floating],
                       sum_out: float | np.floating,
                       dtype: DTypeLike = np.float64,
                       ) -> NDArray[np.floating]:
    """
    Euclidean projection of weight_in onto the simplex {a >= 0, sum(a) = sum_out}.
    
    O(n log n) via sorting (Duchi et al., 2008).
    """
    weight_in = np.asarray(weight_in, dtype=dtype)
    if sum_out <= 0:  # Do not do anything
        return weight_in

    u = np.sort(weight_in)[::-1]
    cssv = np.cumsum(u) - sum_out
    rho = np.nonzero(u - cssv / (np.arange(weight_in.size) + 1) > 0)[0]
    if rho.size == 0:
        # all entries project to zero except we must hit the sum z; fall back
        theta = (u.sum() - sum_out) / weight_in.size
    else:
        rho = rho[-1]
        theta = cssv[rho] / (rho + 1.0)
    weight_out = weight_in - theta
    weight_out[weight_out < 0.0] = 0.0

    return weight_out


def ridge_equivalent_weights_for_mean(x: NDArray[Any],
                                      w: NDArray[np.floating],
                                      lam: float, *,
                                      y: NDArray[np.floating] | None = None,
                                      x_eval: ArrayLike | None = None,
                                      int_dtype: DTypeLike = np.float64,
                                      out_dtype: DTypeLike = np.float64,
                                      weights_eval: ArrayLike | None = None,
                                      # Do not penalize first k elements;
                                      # 0: penalize all
                                      not_penalize_first_k: int = 1,
                                      return_residuals: bool = True,
                                      ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """Compute ridge 'equivalent weights' for the mean across rows of x_eval.

    Returns shape (n, 1). If x_eval is None, averages over the training x.
    """
    proj = RidgeProjector(x, w, lam=lam, dtype=int_dtype,
                          not_penalize_first_k=not_penalize_first_k,
                          )
    if x_eval is None:
        x_eval = np.asarray(x, dtype=int_dtype)

    if isinstance(x_eval, list):
        weight_list = []
        resid_list = []
        for x_ev in x_eval:
            weight, resid = proj.weights_matrix_for_mean(x_ev,
                                                         y=y if return_residuals else None,
                                                         avg_w=weights_eval,
                                                         int_dtype=int_dtype,
                                                         out_dtype=out_dtype,
                                                         )
            weight_list.append(weight)
            resid_list.append(resid)

        return weight_list, resid_list

    return proj.weights_matrix_for_mean(x_eval,
                                        y=y if return_residuals else None,
                                        avg_w=weights_eval,
                                        int_dtype=int_dtype,
                                        out_dtype=out_dtype,
                                        )


class RidgeProjector:
    """
    Equivalent-weights projector for weighted Ridge.

    x must already include the intercept column.

    For any x_new, yhat = w_eq(x_new) @ y, where
    w_eq(x_new) = W X (X' W X + lam * P)^(-1) x_new,
    with P = diag([0,1,1,...]) by default (intercept unpenalized).
    """

    def __init__(self,
                 x: NDArray[Any],
                 w: NDArray[Any],
                 lam: float,
                 dtype: DTypeLike = np.float64, *,
                 not_penalize_first_k: int = 0,
                 ):
        x64 = np.ascontiguousarray(np.asarray(x, dtype=dtype))
        w64 = np.asarray(w, dtype=dtype)
        self.x = x64
        self.w = w64
        self.lam = float(lam)
        self.not_penalize_first_k = int(not_penalize_first_k)

        # gram + ridge
        wx = x64 * w64.reshape(-1, 1)
        g = x64.T @ wx                  # (p, p)

        p = x64.shape[1]

        pen_diag = np.ones(p, dtype=dtype)
        if self.not_penalize_first_k > 1:
            pen_diag[:not_penalize_first_k] = 0.0

        a = g + self.lam * np.diag(pen_diag)

        try:
            self._l = np.linalg.cholesky(a)   # a = l l^T
            self._use_chol = True
        except np.linalg.LinAlgError:         # fall back if ill-conditioned
            self._use_chol = False
            self._a_pinv = np.linalg.pinv(a)

    def _solve_a(self,
                 b: NDArray[np.floating],
                 dtype: DTypeLike = np.float64,
                 ) -> NDArray[np.floating]:
        b = np.asarray(b, dtype=dtype)
        if self._use_chol:
            y = np.linalg.solve(self._l, b)
            return np.linalg.solve(self._l.T, y)

        return self._a_pinv @ b

    def weights_matrix_for_mean(self,
                                x_new: NDArray[Any], *,
                                y: NDArray[Any] | None = None,
                                avg_w: ArrayLike | None = None,
                                int_dtype: DTypeLike = np.float64,
                                out_dtype: DTypeLike = np.float64,
                                ) -> NDArray[Any]:
        """
        Compute equivalent weights (n, 1); if y is provided also return residuals (n,).
         
        Weights that produce the average (or weighted avg) prediction over rows of x_new under
        ridge.
        """
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            xbar = xn
        else:
            if avg_w is None:
                xbar = xn.mean(axis=0)
            else:
                a = np.asarray(avg_w, dtype=int_dtype)
                a = a / a.sum()
                xbar = a @ xn

        u = self._solve_a(xbar, dtype=int_dtype)   # (p,)
        z = self.x @ u                              # (n,)

        # return (self.w * z.reshape(-1, 1)).astype(out_dtype, copy=False)
        w_eq = (self.w * z.reshape(-1, 1)).astype(out_dtype, copy=False)

        if y is None:
            return w_eq, None

        y_vec = np.asarray(y, dtype=int_dtype).reshape(-1)      # (n,)
        r_rhs = self.x.T @ (self.w.reshape(-1) * y_vec)                     # (p,)
        beta = self._solve_a(r_rhs, dtype=int_dtype)            # (p,)
        resid = (y_vec - (self.x @ beta)).astype(out_dtype, copy=False)  # (n,)

        return w_eq, resid


# The following procedures are only needed for finding MSE-minimizing penalty
# terms efficiently.
def _ridge_solve_beta(g: NDArray[np.floating],
                      r: NDArray[np.floating],
                      lam: float,
                      pen_diag: NDArray[np.floating],
                      ) -> NDArray[np.floating]:
    # Solve (g + lam*diag(pen_diag)) beta = r
    a = g + lam * np.diag(pen_diag)
    try:
        l = np.linalg.cholesky(a)
        y = np.linalg.solve(l, r)
        return np.linalg.solve(l.T, y)

    except np.linalg.LinAlgError:
        try:
            ginv = np.linalg.pinv(a)
        except np.linalg.LinAlgError:
            ginv = np.eye(len(a))

        return ginv @ r


def ridge_cv_penalty(x: ArrayLike,
                     y: ArrayLike,
                     w: ArrayLike, *,
                     lambdas: ArrayLike | None = None,
                     k: int = 5,
                     shuffle: bool = True,
                     seed: int = 42,
                     not_penalize_first_k: int = 0,
                     dtype: DTypeLike = np.float64,
                     ) -> tuple[float, dict[str, Any]]:
    """
    Choose ridge penalty by k-fold CV minimizing weighted MSE on validation folds.

    Returns (best_lambda, info) where info has:
      - 'lambdas': tested lambdas
      - 'wmse': mean weighted MSE per lambda (shape (L,))
      - 'wmse_per_fold': (L, k)
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=dtype))
    y = np.asarray(y, dtype=dtype).reshape(-1)
    w = np.asarray(w, dtype=dtype).reshape(-1)
    n, p = x.shape

    if lambdas is None:
        lambdas = np.logspace(-6, 3, 20, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)

    # penalty diag (assume first column is intercept)
    pen_diag = np.ones(p, dtype=dtype)
    if not_penalize_first_k > 0:
        pen_diag[:not_penalize_first_k] = 0.0

    # build folds
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, k)

    # precompute per-fold Gram and rhs on TRAIN, and cache VAL pieces
    g_list, r_list, x_val_list, y_val_list, w_val_list = [], [], [], [], []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        xt = x[train_idx]
        yt = y[train_idx]
        wt = w[train_idx]
        wx = xt * wt.reshape(-1, 1)
        g = xt.T @ wx                          # X'WX
        r = xt.T @ (wt * yt)                   # X'Wy

        g_list.append(g)
        r_list.append(r)
        x_val_list.append(x[val_idx])
        y_val_list.append(y[val_idx])
        w_val_list.append(w[val_idx])

    ls = lambdas.size
    wmse = np.empty(ls, dtype=float)
    wmse_per_fold = np.empty((ls, k), dtype=float)

    # evaluate each lambda
    for li, lam in enumerate(lambdas):
        se_sum = 0.0
        w_sum = 0.0
        for i in range(k):
            beta = _ridge_solve_beta(g_list[i], r_list[i], lam, pen_diag)
            y_hat = x_val_list[i] @ beta
            resid = y_val_list[i] - y_hat
            se = float(np.sum(w_val_list[i] * resid * resid))
            wv = float(np.sum(w_val_list[i]))
            wmse_per_fold[li, i] = se / max(wv, 1e-300)
            se_sum += se
            w_sum += wv
        wmse[li] = se_sum / max(w_sum, 1e-300)

    # pick the lambda with smallest mean WMSE (tie → smallest lambda)
    best_idx = int(np.argmin(wmse))
    best_lambda = float(lambdas[best_idx])

    info = {'lambdas': lambdas,
            'wmse': wmse,
            'wmse_per_fold': wmse_per_fold,
            'best_idx': best_idx,
            }
    return best_lambda, info


def wls_equivalent_weights_for_mean(x: NDArray[Any],
                                    w: NDArray[np.floating], *,
                                    x_eval: ArrayLike | list = None,
                                    y: NDArray[Any] | None = None,
                                    int_dtype: DTypeLike = np.float64,
                                    out_dtype: DTypeLike = np.float64,
                                    weights_eval: ArrayLike = None,
                                    return_residuals: bool = True,
                                    ) -> NDArray[np.floating]:
    """Compute new weights for mean."""
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_eval is None:
        x_eval = np.asarray(x, dtype=int_dtype)

    if isinstance(x_eval, list):
        weight_list = []
        resid_list = []
        for x_ev in x_eval:
            weight, resid = proj.weights_matrix_for_mean(x_ev,
                                                         y=y if return_residuals else None,
                                                         avg_w=weights_eval,
                                                         int_dtype=int_dtype, out_dtype=out_dtype,
                                                         )
            weight_list.append(weight)
            resid_list.append(resid)

        return weight_list, resid_list

    return proj.weights_matrix_for_mean(x_eval,
                                        y=y if return_residuals else None,
                                        avg_w=weights_eval,
                                        int_dtype=int_dtype, out_dtype=out_dtype,
                                        )


class WLSProjector:
    """Equivalent-weights projector for WLS with sampling weights.
    
    x must already include the intercept column.

    For any x_new, yhat = w_eq(x_new) @ y, where
    w_eq(x_new) = W X (X' W X)^{-1} x_new.
    """

    def __init__(self,
                 x: NDArray[Any],
                 w: NDArray[Any],
                 dtype: DTypeLike = np.float64,
                 ):
        # accept float32 input, work in float64 internally
        x64 = np.ascontiguousarray(np.asarray(x, dtype=dtype))
        w64 = np.asarray(w, dtype=dtype)
        self.x = x64
        self.w = w64

        wx = x64 * w64.reshape(-1, 1)
        g = x64.T @ wx  # (p, p)
        try:
            self._l = np.linalg.cholesky(g)   # g = l l^T
            self._use_chol = True
        except np.linalg.LinAlgError:   # If not of full rank, use pseudoinverse
            self._use_chol = False
            self._g_pinv = np.linalg.pinv(g)

    def _solve_g(self,
                 b: NDArray[np.floating],
                 dtype: DTypeLike = np.float64,
                 ) -> NDArray[np.floating]:
        b = np.asarray(b, dtype=dtype)
        if self._use_chol:
            y = np.linalg.solve(self._l, b)
            return np.linalg.solve(self._l.T, y)

        return self._g_pinv @ b


    def weights_matrix_for_mean(self,
                                x_new: NDArray[Any], *,
                                y: NDArray[Any] | None = None,
                                avg_w: ArrayLike = None,
                                int_dtype: DTypeLike = np.float64,
                                out_dtype: DTypeLike = np.float64,
                                ) -> NDArray[Any]:
        """Equivalent weights (n, 1); if y is provided also return residuals (n,).
        
        If avg_w is given, it uses a weighted average over rows (avg_w >= 0, sums to 1).
        """
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            xbar = xn
        else:
            if avg_w is None:
                xbar = xn.mean(axis=0)
            else:
                a = np.asarray(avg_w, dtype=int_dtype)
                a = a / a.sum()                      # normalize
                xbar = a @ xn                        # weighted mean row
        # u = self._solve_g(xbar, dtype=int_dtype)     # (p,)
        # z = self.x @ u                              # (n,)
        # return (self.w * z.reshape(-1, 1)).astype(out_dtype, copy=False)
        u = self._solve_g(xbar, dtype=int_dtype)     # (p,)
        z = self.x @ u                              # (n,)

        w_eq = (self.w * z.reshape(-1, 1)).astype(out_dtype, copy=False)
        if y is None:
            return w_eq, None

        y_vec = np.asarray(y, dtype=int_dtype).reshape(-1)      # (n,)
        r_rhs = self.x.T @ (self.w.reshape(-1) * y_vec)         # (p,)
        beta = self._solve_g(r_rhs, dtype=int_dtype)            # (p,)
        resid = (y_vec - (self.x @ beta)).astype(out_dtype, copy=False)  # (n,)

        return w_eq, resid


    def weights_for(self,
                    x_new: NDArray[Any],
                    int_dtype: DTypeLike = np.float64,
                    out_dtype: DTypeLike = np.float64,
                    ) -> NDArray[np.floating]:
        """Compute weights. CURRENTLY NOT USED."""
        x_new = np.asarray(x_new, dtype=int_dtype)       # (p,)
        u = self._solve_g(x_new, dtype=int_dtype)        # (p,)
        z = self.x @ u                                  # (n,)

        return (self.w * z).astype(out_dtype, copy=False)


    def predict(self,
                y: NDArray[Any],
                x_new: NDArray[Any],
                int_dtype: DTypeLike = np.float64,
                out_dtype: DTypeLike = np.float64,
                ) -> NDArray[np.floating]:
        """Predict the outcome as weighted mean. CURRENTLY NOT USED."""
        y = np.asarray(y, dtype=int_dtype)
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            w_eq = self.weights_for(xn,
                                    int_dtype=int_dtype,
                                    out_dtype=int_dtype,
                                    )
            return np.array(w_eq @ y, dtype=out_dtype)
        # batch
        u = self._solve_g(xn.T, dtype=int_dtype)         # (p, m)
        z = self.x @ u                                   # (n, m)
        wz = z * self.w.reshape(-1, 1)                   # (n, m)

        return (wz.T @ y).astype(out_dtype, copy=False)  # (m,)

    def predict_mean(self,
                     y: NDArray[Any],
                     x_new: NDArray[Any], *,
                     avg_w: ArrayLike = None,
                     int_dtype: DTypeLike = np.float64,
                     out_dtype: DTypeLike = np.float64,
                     ) -> NDArray[Any]:
        """Scalar average prediction over x_new (or weighted average with avg_w).
        
        CURRENTLY NOT USED.
        """
        y64 = np.asarray(y, dtype=int_dtype)
        w_eq_mean = self.weights_matrix_for_mean(x_new,
                                                 avg_w=avg_w,
                                                 int_dtype=int_dtype,
                                                 out_dtype=int_dtype,
                                                 )
        return np.array(w_eq_mean @ y64, dtype=out_dtype)

    def weights_matrix(self,
                       x_new: NDArray[Any],
                       int_dtype: DTypeLike = np.float64,
                       out_dtype: DTypeLike = np.float64,
                       ) -> NDArray[np.floating]:
        """Compute new weights to compute average prediction. CURRENTLY NOT USED."""
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            xn = xn.reshape(1, -1)
        u = self._solve_g(xn.T, dtype=int_dtype)        # (p, m)
        z = self.x @ u                                  # (n, m)

        return (z * self.w.reshape(-1, 1)).T.astype(out_dtype, copy=False)

    def projector_matrix(self,
                         int_dtype: DTypeLike = np.float64,
                         out_dtype: DTypeLike = np.float64,
                         ) -> NDArray[np.floating]:
        """Return s = W X (X' W X)^{-1}  so that w_eq(x_new) = s @ x_new.

        Useful when you’ll form many equivalent-weight vectors.
        CURRENTLY NOT USED.
        """
        p = self.x.shape[1]
        inv_g = self._solve_g(np.eye(p, dtype=int_dtype))    # (p, p) via solves
        s = (self.x * self.w.reshape(-1, 1)) @ inv_g         # (n, p)

        return s.astype(out_dtype, copy=False)


def wls_equivalent_weights(x: NDArray[Any],
                           w: NDArray[np.floating],
                           x_new: ArrayLike = None,
                           int_dtype: DTypeLike = np.float64,
                           out_dtype: DTypeLike = np.float64,
                           ) -> NDArray[np.floating]:
    """Compute new weights. CURRENTLY NOT USED."""
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)

    return proj.weights_matrix(x_new, out_dtype=out_dtype)


def wls_predict_fast(x: NDArray[Any],
                     w: NDArray[np.floating],
                     y: NDArray[Any], *,
                     x_new: NDArray[Any] | None = None,
                     int_dtype: DTypeLike = np.float64,
                     out_dtype: DTypeLike = np.float64,
                     ) -> NDArray[Any]:
    """Do fast predictions. CURRENTLY NOT USED."""
    # avoids materializing the (m x n) weights matrix — faster/leaner for big n
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)

    return proj.predict(y, x_new, out_dtype=out_dtype)


def wls_weights_and_pred(x: NDArray[Any],
                         w: NDArray[np.floating],
                         y: NDArray[Any], *,
                         x_new: ArrayLike = None,
                         int_dtype: DTypeLike = np.float64,
                         out_dtype: DTypeLike = np.float64,
                         ) -> tuple[NDArray[np.floating], NDArray[Any]]:
    """Compute predictions and update weights. CURRENTLY NOT USED."""
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)
    w_eq = proj.weights_matrix(x_new, out_dtype=out_dtype)     # shape (m, n)
    yhat = (w_eq @ np.asarray(y, dtype=int_dtype)).astype(out_dtype, copy=False)

    return w_eq, yhat


# Below are corresponding functions using Tensors and GPU (cuda)
def _ridge_solve_beta_cuda(g: torch.Tensor,
                           r: torch.Tensor,
                           lam: torch.Tensor,  # scalar tensor on same device
                           pen_diag: torch.Tensor,
                           ) -> torch.Tensor:
    """Solve (g + lam*diag(pen_diag)) beta = r on GPU.

    Uses Cholesky factorization (assumes SPD in typical ridge settings).
    """
    require_torch()
    # Form A = g with ridge added only on the diagonal: A_ii += lam * pen_diag_i
    a = g.clone()
    a.diagonal().add_(lam * pen_diag)

    # Cholesky solve: a = L L^T
    # l = torch.linalg.cholesky(a)   # pylint: disable=E1102
    l = safe_cholesky(a)
    beta = torch.cholesky_solve(r.unsqueeze(1), l).squeeze(1)

    return beta


def _make_k_folds_cuda(n: int,
                       k: int,
                       *,
                       device: torch.device,
                       shuffle: bool,
                       seed: int,
                       ) -> list[torch.Tensor]:
    """Return list of k index tensors (on device), similar to np.array_split."""
    require_torch()
    if k <= 1:
        raise ValueError('k must be at least 2')
    if n <= 0:
        raise ValueError('n must be positive')

    k = min(k, n)

    idx = torch.arange(n, device=device)
    if shuffle:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        idx = idx[torch.randperm(n, generator=gen, device=device)]

    base = n // k
    rem = n % k
    folds: list[torch.Tensor] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append(idx[start:start + size])
        start += size
    return folds


def ridge_cv_penalty_cuda(x: torch.Tensor,
                          y: torch.Tensor,
                          w: torch.Tensor, *,
                          lambdas: TensorLike = None,
                          k: int = 5,
                          shuffle: bool = True,
                          seed: int = 42,
                          not_penalize_first_k: int = 0,
                          dtype: TorchDType | None = None,
                          ) -> tuple[float, dict[str, Any]]:
    """
    Choose ridge penalty by k-fold CV minimizing weighted MSE on validation folds.

    Returns (best_lambda, info) where info has:
      - 'lambdas': tested lambdas (CPU tensor)
      - 'wmse': mean weighted MSE per lambda (CPU tensor, shape (L,))
      - 'wmse_per_fold': (CPU tensor, shape (L, k))
      - 'best_idx': int
    """
    require_torch()  # Deals with default values to make it save for failed torch import
    if dtype is None:
        dtype = torch.float64

    if x.ndim != 2:
        raise TypeError('x must be 2D')
    n, p = x.shape

    device = x.device

    # Ensure dtype/shape consistency (assume already on GPU; this is cast-only)
    x = x.to(dtype=dtype).contiguous()
    y = y.to(dtype=dtype).reshape(-1)
    w = w.to(dtype=dtype).reshape(-1)

    if y.numel() != n or w.numel() != n:
        raise ValueError('y and w must have length n = x.shape[0]')

    if lambdas is None:
        lambdas = torch.logspace(-6, 3, steps=20, base=10.0, device=device, dtype=dtype)
    else:
        lambdas = lambdas.to(device=device, dtype=dtype).reshape(-1)

    # penalty diagonal (first columns optionally unpenalized)
    pen_diag = torch.ones(p, device=device, dtype=dtype)
    if not_penalize_first_k > 0:
        pen_diag[:not_penalize_first_k] = 0.0

    # build folds on GPU
    folds = _make_k_folds_cuda(n, k, device=device, shuffle=shuffle, seed=seed)
    k_eff = len(folds)

    # precompute per-fold Gram and rhs on TRAIN, and cache VAL pieces
    g_list: list[torch.Tensor] = []
    r_list: list[torch.Tensor] = []
    x_val_list: list[torch.Tensor] = []
    y_val_list: list[torch.Tensor] = []
    w_val_list: list[torch.Tensor] = []

    all_idx = torch.arange(n, device=device)
    for i in range(k_eff):
        val_idx = folds[i]
        # train_idx = all indices except val_idx
        # (boolean mask avoids concatenating k-1 tensors repeatedly)
        mask = torch.ones(n, device=device, dtype=torch.bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]

        xt = x[train_idx]
        yt = y[train_idx]
        wt = w[train_idx]

        # g = X' W X ; r = X' W y
        wx = xt * wt.unsqueeze(1)
        g = xt.transpose(0, 1) @ wx
        r = xt.transpose(0, 1) @ (wt * yt)

        g_list.append(g)
        r_list.append(r)
        x_val_list.append(x[val_idx])
        y_val_list.append(y[val_idx])
        w_val_list.append(w[val_idx])

    ls = int(lambdas.numel())
    wmse = torch.empty(ls, device=device, dtype=torch.float64)
    wmse_per_fold = torch.empty((ls, k_eff), device=device, dtype=torch.float64)

    eps = torch.tensor(1e-300, device=device, dtype=dtype)

    for li in range(ls):
        lam = lambdas[li]  # scalar tensor
        se_sum = torch.zeros((), device=device, dtype=dtype)
        w_sum = torch.zeros((), device=device, dtype=dtype)

        for i in range(k_eff):
            beta = _ridge_solve_beta_cuda(g_list[i], r_list[i], lam, pen_diag)
            y_hat = x_val_list[i] @ beta
            resid = y_val_list[i] - y_hat

            se = (w_val_list[i] * resid.square()).sum()
            wv = w_val_list[i].sum()

            denom = torch.maximum(wv, eps)
            wmse_per_fold[li, i] = (se / denom).to(dtype=torch.float64)

            se_sum = se_sum + se
            w_sum = w_sum + wv

        wmse[li] = (se_sum / torch.maximum(w_sum, eps)).to(dtype=torch.float64)

    # pick smallest wmse (tie-breaking: argmin returns first occurrence => smallest lambda)
    best_idx = int(torch.argmin(wmse).item())
    best_lambda = float(lambdas[best_idx].item())

    # Move diagnostics to CPU to avoid holding GPU memory
    info = {'lambdas': lambdas.detach().cpu(),
            'wmse': wmse.detach().cpu(),
            'wmse_per_fold': wmse_per_fold.detach().cpu(),
            'best_idx': best_idx,
            }
    return best_lambda, info


class RidgeProjectorCuda:
    """Projector for ridge-equivalent weights on GPU.

    Computes equivalent weights for linear ridge predictions:
        xbar' beta = sum_i (w_i * (x_i' u)) * y_i
    where u solves:
        (X' W X + lam * diag(pen_diag)) u = xbar
    """

    def __init__(self,
                 x: torch.Tensor,
                 w: torch.Tensor,
                 *,
                 lam: float | torch.Tensor,
                 dtype: TorchDType | None = None,
                 not_penalize_first_k: int = 1,
                 ) -> None:
        require_torch()
        if dtype is None:
            dtype = torch.float64

        if x.ndim != 2:
            raise ValueError('x must be 2D')
        n, p = x.shape

        if w.ndim == 2 and w.shape[1] == 1:
            w = w.reshape(-1)
        elif w.ndim != 1:
            raise TypeError('w must be 1D or (n, 1)')

        if w.numel() != n:
            raise ValueError('w must have length n = x.shape[0]')

        self.device = x.device
        self.dtype = dtype

        self.x = x.to(dtype=dtype).contiguous()
        w1 = w.to(device=self.device, dtype=dtype).contiguous()
        self.w = w1.reshape(-1, 1)  # (n, 1)

        lam_t = lam if torch.is_tensor(lam) else torch.tensor(lam, device=self.device, dtype=dtype)
        lam_t = lam_t.to(dtype=dtype)

        # penalty diagonal (0 for unpenalized, 1 otherwise)
        pen_diag = torch.ones(p, device=self.device, dtype=dtype)
        if not_penalize_first_k > 0:
            pen_diag[:not_penalize_first_k] = 0.0
        self.pen_diag = pen_diag

        # Gram matrix g = X' W X
        wx = self.x * w1.unsqueeze(1)               # (n, p)
        g = self.x.transpose(0, 1) @ wx             # (p, p)

        # A = g + lam * diag(pen_diag)
        a = g.clone()
        a.diagonal().add_(lam_t * pen_diag)

        # Cholesky factorization (ridge should make SPD for lam>0)
        # If you need a fallback for lam=0 edge cases, add try/except here.
        self._chol = torch.linalg.cholesky(a)  # pylint: disable=E1102

    def _solve_g_cuda(self, xbar: torch.Tensor) -> torch.Tensor:
        """Solve (X'WX + lam*D) u = xbar for u (shape (p,))."""
        b = xbar.to(device=self.device, dtype=self.dtype).reshape(-1, 1)
        u = torch.cholesky_solve(b, self._chol).reshape(-1)

        return u

    def weights_matrix_for_mean_cuda(self,
                                     x_new: torch.Tensor,
                                     y: torch.Tensor | None = None,
                                     *,
                                     avg_w: TensorLike = None,
                                     int_dtype: TorchDType | None = None,
                                     out_dtype: TorchDType | None = None,
                                     ) -> torch.Tensor:
        """Equivalent weights (n, 1); if y is provided also return residuals (n,).
        
        If avg_w is given, uses a weighted average over rows (avg_w >= 0, sums to 1 after
        normalization).
        """
        require_torch()  # Deals with default values to make it save for failed torch import
        if int_dtype is None:
            int_dtype = torch.float64
        if out_dtype is None:
            out_dtype = torch.float64

        xn = x_new.to(device=self.device, dtype=int_dtype)

        if xn.ndim == 1:
            xbar = xn
        else:
            if avg_w is None:
                xbar = xn.mean(dim=0)
            else:
                a = avg_w.to(device=self.device, dtype=int_dtype).reshape(-1)
                s = a.sum()
                if not bool((s > 0).item()):
                    raise ValueError('avg_w must sum to a positive value')
                a = a / s
                xbar = a @ xn  # (p,)

        u = self._solve_g_cuda(xbar)          # (p,)
        z = self.x @ u                        # (n,)
        # (n, 1) and cast

        # return (self.w * z.reshape(-1, 1)).to(dtype=out_dtype)
        # (n, 1) and cast
        w_eq = (self.w * z.reshape(-1, 1)).to(dtype=out_dtype)
        if y is None:
            return w_eq

        y_vec = y.to(device=self.device, dtype=int_dtype).reshape(-1)      # (n,)
        r_rhs = self.x.transpose(0, 1) @ (self.w.reshape(-1) * y_vec)      # (p,)
        beta = self._solve_g_cuda(r_rhs)                                   # (p,)
        resid = (y_vec - (self.x @ beta)).to(dtype=out_dtype)              # (n,)

        return w_eq, resid


def ridge_equivalent_weights_for_mean_cuda(x: torch.Tensor,
                                           w: torch.Tensor,
                                           lam: float,
                                           y: torch.Tensor | None, *,
                                           x_eval: Any | None = None,
                                           int_dtype: TorchDType | None = None,
                                           out_dtype: TorchDType | None = None,
                                           weights_eval: TensorLike = None,
                                           not_penalize_first_k: int = 1,
                                           return_residuals: bool = True,
                                           ):
    """Compute ridge 'equivalent weights' for the mean across rows of x_eval.

    Returns
    -------
      - torch.Tensor of shape (n, 1) if x_eval is a tensor or None
      - list[torch.Tensor] if x_eval is a list of tensors

    """
    require_torch()  # Deals with default values to make it save for failed torch import
    if int_dtype is None:
        int_dtype = torch.float64
    if out_dtype is None:
        out_dtype = torch.float64
    proj = RidgeProjectorCuda(x,
                              w,
                              lam=lam,
                              dtype=int_dtype,
                              not_penalize_first_k=not_penalize_first_k,
                              )
    if x_eval is None:
        x_eval = x.to(dtype=int_dtype)

    if isinstance(x_eval, list):
        weight_list: list[torch.Tensor] = []
        resid_list: list[torch.Tensor] = []
        for x_ev in x_eval:
            weight, resid = proj.weights_matrix_for_mean_cuda(x_ev,
                                                              y=y if return_residuals else None,
                                                              avg_w=weights_eval,
                                                              int_dtype=int_dtype,
                                                              out_dtype=out_dtype,
                                                              )
            weight_list.append(weight)
            resid_list.append(resid)
        return weight_list, resid_list

    return proj.weights_matrix_for_mean_cuda(x_eval,
                                             y=y if return_residuals else None,
                                             avg_w=weights_eval,
                                             int_dtype=int_dtype,
                                             out_dtype=out_dtype,
                                             )


def wls_equivalent_weights_for_mean_cuda(x: torch.Tensor,
                                         w: torch.Tensor, *,
                                         y: torch.Tensor | None = None,
                                         x_eval: Any | list | None = None,
                                         int_dtype: TorchDType | None = None,
                                         out_dtype: TorchDType | None = None,
                                         weights_eval: TensorLike = None,
                                         return_residuals: bool = True,
                                         ):
    """Compute WLS equivalent weights for mean on GPU.

    Returns
    -------
      - torch.Tensor of shape (n, 1) if x_eval is a tensor or None
      - list[torch.Tensor] if x_eval is a list of tensors
    """
    require_torch()
    if int_dtype is None:
        int_dtype = torch.float64
    if out_dtype is None:
        out_dtype = torch.float64

    proj = WLSProjectorCuda(x, w, dtype=int_dtype)

    if x_eval is None:
        x_eval = x.to(dtype=int_dtype)

    if isinstance(x_eval, list):
        weight_list: list[torch.Tensor] = []
        resid_list: list[torch.Tensor] = []
        for x_ev in x_eval:
            weight, resid = proj.weights_matrix_for_mean_cuda(x_ev,
                                                              y=y if return_residuals else None,
                                                              avg_w=weights_eval,
                                                              int_dtype=int_dtype,
                                                              out_dtype=out_dtype,
                                                              )
        weight_list.append(weight)
        resid_list.append(resid)

        return weight_list, resid_list

    return proj.weights_matrix_for_mean_cuda(x_eval,
                                             y=y if return_residuals else None,
                                             avg_w=weights_eval,
                                             int_dtype=int_dtype,
                                             out_dtype=out_dtype,
                                             )


class WLSProjectorCuda:
    """
    Equivalent-weights projector for WLS with sampling weights (GPU).

    x must already include the intercept column.

    For any x_new, yhat = w_eq(x_new)^T y, where
    w_eq(x_new) = W X (X' W X)^{-1} x_new.

    This class provides weights for the *mean* prediction over rows of x_new.
    """

    def __init__(self,
                 x: torch.Tensor,
                 w: torch.Tensor,
                 dtype: TorchDType | None = None,
                 ) -> None:
        require_torch()
        if dtype is None:
            dtype = torch.float64
        if x.ndim != 2:
            raise TypeError('x must be 2D')

        n = x.shape[0]

        if w.ndim == 2 and w.shape[1] == 1:
            w = w.reshape(-1)
        elif w.ndim != 1:
            raise TypeError('w must be 1D or (n, 1)')
        if w.numel() != n:
            raise ValueError('w must have length n = x.shape[0]')

        self.device = x.device
        self.dtype = dtype

        # Work in float64 by default
        self.x = x.to(device=self.device, dtype=dtype).contiguous()
        w1 = w.to(device=self.device, dtype=dtype).contiguous()
        self.w = w1.reshape(-1, 1)  # (n, 1)

        # Gram G = X' W X
        wx = self.x * w1.unsqueeze(1)                 # (n, p)
        g = self.x.transpose(0, 1) @ wx               # (p, p)

        # Prefer Cholesky; fall back to pinv if not SPD
        try:
            self._chol = torch.linalg.cholesky(g)  # pylint: disable=E1102
            self._use_chol = True
        except RuntimeError:
            self._use_chol = False
            # pinv on GPU; can be slower but robust
            self._g_pinv = torch.linalg.pinv(g)  # pylint: disable=E1102

    def _solve_g_cuda(self, b: torch.Tensor) -> torch.Tensor:
        """Solve G u = b for u (shape (p,))."""
        b = b.to(device=self.device, dtype=self.dtype).reshape(-1, 1)
        u = torch.cholesky_solve(b, self._chol) if self._use_chol else self._g_pinv @ b
        return u.reshape(-1)

    def weights_matrix_for_mean_cuda(self,
                                     x_new: torch.Tensor,
                                     y: torch.Tensor | None = None,
                                     *,
                                     avg_w: TensorLike = None,
                                     int_dtype: TorchDType | None = None,
                                     out_dtype: TorchDType | None = None,
                                     ) -> torch.Tensor:
        """Equivalent weights (n, 1); if y is provided also return residuals (n,).
        
        If avg_w is given, uses a weighted average over rows (avg_w >= 0, sums to 1 after
        normalization).
        """
        require_torch()
        if int_dtype is None:
            int_dtype = torch.float64
        if out_dtype is None:
            out_dtype = torch.float64

        xn = x_new.to(device=self.device, dtype=int_dtype)

        # Compute xbar (p,)
        if xn.ndim == 1:
            xbar = xn
        else:
            if avg_w is None:
                xbar = xn.mean(dim=0)
            else:
                a = avg_w.to(device=self.device, dtype=int_dtype).reshape(-1)
                s = a.sum()
                if not bool((s > 0).item()):
                    raise ValueError('avg_w must sum to a positive value')
                a = a / s
                xbar = a @ xn

        u = self._solve_g_cuda(xbar)          # (p,)
        z = self.x @ u                        # (n,)

        # return (self.w * z.reshape(-1, 1)).to(dtype=out_dtype)
        w_eq = (self.w * z.reshape(-1, 1)).to(dtype=out_dtype)
        if y is None:
            return w_eq, None

        y_vec = y.to(device=self.device, dtype=int_dtype).reshape(-1)      # (n,)
        r_rhs = self.x.transpose(0, 1) @ (self.w.reshape(-1) * y_vec)      # (p,)
        beta = self._solve_g_cuda(r_rhs)                                   # (p,)
        resid = (y_vec - (self.x @ beta)).to(dtype=out_dtype)              # (n,)

        return w_eq, resid



def safe_cholesky(a: torch.Tensor,
                  max_tries: int = 8,
                  jitter0: float | None = None
                  ) -> torch.Tensor:
    """Compute safe Cholesky decomposition."""
    # Ensure symmetry (important with GPU/float32 accumulation)
    a = 0.5 * (a + a.T)

    l, info = torch.linalg.cholesky_ex(a)        # pylint: disable=E1102
    if info.item() == 0:
        return l

    # Scale jitter to the matrix magnitude
    diag_mean = a.diagonal().abs().mean()
    if jitter0 is None:
        # Reasonable defaults: larger for float32, smaller for float64
        base = 1e-6 if a.dtype in (torch.float64, torch.complex128) else 1e-4
        jitter = base * (diag_mean.item() if diag_mean.isfinite() else 1.0)
    else:
        jitter = float(jitter0)

    i = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)

    for _ in range(max_tries):
        l, info = torch.linalg.cholesky_ex(a + jitter * i)    # pylint: disable=E1102
        if info.item() == 0:
            return l
        jitter *= 10.0

    # If we get here, it’s genuinely problematic (or extremely ill-conditioned)
    raise torch.linalg.LinAlgError(f'Cholesky failed after {max_tries} jitter attempts; '
                                   'last jitter={jitter}'
                                   )
