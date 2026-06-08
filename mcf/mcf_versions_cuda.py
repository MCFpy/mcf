"""
Created on Mon Dec 15 13:59:59 2025.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

from mcf import mcf_estimation_weighted_regression as mcf_ewr
from mcf.mcf_cuda import require_torch

if TYPE_CHECKING:
    from torch import Tensor
    from torch import dtype as TorchDType
else:
    Tensor = Any
    TorchDType = Any

type TensorLike = Tensor | None


# The following functions will work on the GPU only
def version_wregr_cuda(weights: list[torch.Tensor] | torch.Tensor, *,
                       y_train: torch.Tensor | None = None,
                       d_train: torch.Tensor | None = None,
                       x_train: torch.Tensor | None = None,
                       x_pred: torch.Tensor | None = None,
                       d_sub_only: torch.Tensor | list[int] | None = None,
                       d_all_values: torch.Tensor | list[int] | None = None,
                       tv_min_subtreat: int | None = 10,
                       cv_k: int | None = 5,
                       container_w: list[torch.Tensor] | None = None,
                       container_r: list[torch.Tensor] | None = None,
                       w_index: torch.Tensor | None = None,
                       treat_idx: int = 0,
                       maintreat_idx: int = 0,
                       subtreat_idx: int | None = 0,
                       int_dtype: TorchDType | None = None,
                       out_dtype: TorchDType | None = None,
                       zero_tol: float = 1e-10,
                       ridge: bool = True,
                       penalize_version: bool = False,
                       return_residuals: bool = True,
                       standardize_x: bool = True,
                       ) -> tuple[torch.Tensor,
                                  list[torch.Tensor],
                                  torch.Tensor,
                                  list[torch.Tensor],
                                  int, int,
                                  str,
                                  ]:
    """Estimate new weights based on treatment versions (CUDA-friendly)."""
    require_torch()
    if int_dtype is None:
        int_dtype = torch.float64
    if out_dtype is None:
        out_dtype = torch.float32

    txt = ''
    device = y_train.device
    # --- Ensure weights is a CUDA tensor (only potential CPU->GPU transfer in this function)
    if isinstance(weights, list):
        weights = torch.as_tensor(weights, device=device, dtype=out_dtype)
    else:
        weights = weights.to(device=device, dtype=out_dtype)
    weights = weights.reshape(-1)

    # --- Robust 'length' check for d_sub_only (list or tensor)
    if d_sub_only is None:
        n_sub = 0
    elif torch.is_tensor(d_sub_only):
        n_sub = int(d_sub_only.numel())
    else:
        n_sub = len(d_sub_only)

    if n_sub < 2:
        txt += f'\nMain treatment {maintreat_idx}: No subtreatment'
        return weights, None, None, None, maintreat_idx + 1, 0, txt

    # --- Optional subsampling (IATE)
    if w_index is not None:
        # w_index must be a Long tensor for indexing; assume already on GPU, but enforce dtype
        if not torch.is_tensor(w_index):
            w_index = torch.as_tensor(w_index, device=device)
        w_index = w_index.to(dtype=torch.long, device=device)

        y_train = y_train[w_index]
        d_train = d_train[w_index, :]
        if x_train is not None:
            x_train = x_train[w_index, :]

    # --- x_pred fallback (replace .copy() with .clone())
    if x_train.shape[1] != x_pred.shape[1]:
        x_pred = x_train.clone()
        txt += ('\nPREDICTION AND TRAINING DATA HAVE DIFFERENT # OF COLUMNS. '
                'TRAINING DATA USED FOR PREDICTIONS IN VERSION REGRESSIONS. ')

    if standardize_x and x_train is not None:
        x_train, x_pred = scale_by_sd_of_first_cuda(x_train, x_pred, ddof=0, zero_tol=zero_tol)
    # --- Select within same main treatment AND positive weights
    # Keep main_treat_val as a tensor scalar for GPU compare;
    # only .item() for Python control-flow/text.
    main_treat_val = d_all_values[treat_idx, 0]
    # select_data = (d_train[:, 0] == main_treat_val) & (weights > zero_tol)
    select_data = (torch.isclose(d_train[:, 0], main_treat_val, atol= zero_tol, rtol=zero_tol)
                   & (weights > zero_tol)
                   )
    # First time this main treatment is seen by this function
    if treat_idx == 0:
        first_time_main = True
    else:
        # sync point
        first_time_main = (main_treat_val != d_all_values[treat_idx - 1, 0]).item()
    if first_time_main:
        # Select data
        y_tr = y_train[select_data]
        d_sub_tr = d_train[select_data, 1]
        x_tr = x_train[select_data, :]
        w_tr = weights[select_data].to(dtype=int_dtype)

        # Container: Python list holding CUDA tensors (no GPU<->CPU data copies)
        container_w = [None] * n_sub
        container_r = [None] * n_sub

        # 1) small-cell check (assumed GPU-aware inside)
        subtreat_idx_too_small = too_small_cuda(d_sub_tr, d_sub_only, tv_min_subtreat,
                                                zero_tol=zero_tol
                                                )
        # expected: list[int] (indices), or empty list/None

        # 2) dummies (assumed to return a CUDA tensor)
        d_dummy = dummies_ordered_cuda(d_sub_tr, d_sub_only, dtype=int_dtype, unknown='error')
        # Drop columns with too small cells
        # (replace np.delete with boolean column mask)
        if subtreat_idx_too_small:
            col_mask = torch.ones(d_dummy.shape[1],
                                  device=device, dtype=torch.bool
                                  )
            col_mask[subtreat_idx_too_small] = False
            d_dummy = d_dummy[:, col_mask]

            txt += ('\nThe following subtreatments are too small for version '
                    f'regression: Main_treatment: {main_treat_val.item()} '
                    f'Subtreatments: {" ".join(map(str, subtreat_idx_too_small))}'
                    )
        # Replace np.concat with torch.cat
        regr_tr = torch.cat((d_dummy, x_tr.to(dtype=int_dtype)), dim=1)

        # Construct list of regressors for predicting subeffects (GPU)
        regr_eval_list_o = []
        col_1 = 0
        n_pred = x_pred.shape[0]
        n_dummy_cols = d_dummy.shape[1]

        for sub_idx in range(n_sub):
            if subtreat_idx_too_small and sub_idx in subtreat_idx_too_small:
                continue
            dummy_eval = torch.zeros((n_pred, n_dummy_cols), device=device, dtype=int_dtype)
            dummy_eval[:, col_1] = 1.0
            regr_eval = torch.cat((dummy_eval, x_pred.to(dtype=int_dtype)), dim=1)
            regr_eval_list_o.append(regr_eval)  # no need to clone unless later mutated
            col_1 += 1

        if ridge:
            # Replace np.zeros + np.logspace with torch equivalents on GPU
            grid_length = 30
            if penalize_version:
                grid =  torch.logspace(-5, 2, steps=grid_length, base=10.0,
                                       device=device, dtype=int_dtype
                                       )
                not_penalize_first_k = 1
                # --- training ---
                constant_tr = torch.ones((regr_tr.shape[0], 1),
                                         dtype=regr_tr.dtype, device=regr_tr.device
                                         )
                regr_tr = torch.cat((constant_tr, regr_tr), dim=1)
                # --- eval list ---
                regr_eval_list = [torch.cat((torch.ones((regr_eval.shape[0], 1),
                                                        dtype=regr_eval.dtype,
                                                        device=regr_eval.device), regr_eval),
                                            dim=1)
                                  ]
            else:
                grid = torch.cat((torch.zeros(1, device=device, dtype=int_dtype),
                                  torch.logspace(-5, 2, steps=grid_length, base=10.0,
                                                 device=device, dtype=int_dtype),),
                                 dim=0
                                 )
                not_penalize_first_k = d_dummy.shape[1]
                regr_eval_list = regr_eval_list_o

            best_lam, _ = mcf_ewr.ridge_cv_penalty_cuda(regr_tr, y_tr,
                                                        w_tr.reshape(-1, 1),
                                                        lambdas=grid,
                                                        k=cv_k,
                                                        shuffle=True,
                                                        seed=1234566,
                                                        not_penalize_first_k=not_penalize_first_k,
                                                        dtype=int_dtype,
                                                        )
        else:
            regr_eval_list = regr_eval_list_o
        if ridge:
            weights_tv_list, resid_tv_list = mcf_ewr.ridge_equivalent_weights_for_mean_cuda(
                regr_tr,
                w_tr.reshape(-1, 1),
                y=y_tr if return_residuals else None,
                lam=best_lam,
                x_eval=regr_eval_list,
                int_dtype=int_dtype,
                out_dtype=out_dtype,
                weights_eval=None,
                not_penalize_first_k=not_penalize_first_k,
                return_residuals=return_residuals,
                )
        else:
            weights_tv_list, resid_tv_list = mcf_ewr.wls_equivalent_weights_for_mean_cuda(
                regr_tr,
                w_tr.reshape(-1, 1),
                y=y_tr if return_residuals else None,
                x_eval=regr_eval_list,
                int_dtype=int_dtype,
                out_dtype=out_dtype,
                weights_eval=None,
                return_residuals=return_residuals,
                )
        # Fill container (replace .copy() with .clone())
        col_1 = 0
        for sub_idx in range(n_sub):
            weights_return = weights.clone()
            if return_residuals:
                residuals_return = torch.zeros_like(weights_return, dtype=weights.dtype)
            if not (subtreat_idx_too_small and sub_idx in subtreat_idx_too_small):
                weights_return[select_data] = weights_tv_list[col_1].reshape(-1).to(
                    dtype=weights.dtype)
                if return_residuals:
                    residuals_return[select_data] = resid_tv_list[col_1].reshape(-1).to(
                        dtype=weights.dtype)
                col_1 += 1
            container_w[sub_idx] = weights_return
            if return_residuals:
                container_r[sub_idx] = residuals_return

    weight_new = container_w[subtreat_idx]
    residual_new = container_r[subtreat_idx] if return_residuals else None

    # Update indices
    if subtreat_idx == n_sub - 1:
        maintreat_idx += 1
        subtreat_idx = 0
    else:
        subtreat_idx += 1

    return weight_new, container_w, residual_new, container_r, maintreat_idx, subtreat_idx, txt


def too_small_cuda(d_dat: torch.Tensor,
                   d_sub_values: Sequence[int],
                   tv_min_subtreat: int,
                   zero_tol: float = 1e-10,
                   ) -> list[int]:
    """Return list of indices (into d_sub_values) with too-small cell sizes.

    Notes
    -----
    - d_dat is assumed to be a CUDA tensor (integer-coded treatments).
    - Missing subtreatment values in d_dat are treated as count 0.
    - Work is done on GPU; only the returned indices are copied to CPU.
    """
    if d_dat.numel() == 0:
        # No observations => all subtreatments are "too small"
        return list(range(len(d_sub_values)))

    # Unique values + counts on GPU
    u, c = torch.unique(d_dat, return_counts=True)  # u is sorted by default

    # Subtreat values as GPU tensor
    sub_vals = torch.as_tensor(d_sub_values, device=d_dat.device, dtype=u.dtype)

    if u.numel() == 0:
        counts = torch.zeros_like(sub_vals, dtype=torch.long)
    else:
        # Map each sub_vals entry to its count (0 if missing), fully on GPU
        pos = torch.searchsorted(u, sub_vals)

        # Guard against pos == u.numel() when sub_vals contains larger values
        in_range = pos < u.numel()
        pos_safe = pos.clamp(max=u.numel() - 1)

        # match = in_range & (u[pos_safe] == sub_vals)
        match = in_range & torch.isclose(u[pos_safe], sub_vals, rtol=zero_tol, atol=zero_tol)

        counts = torch.zeros(sub_vals.shape, device=d_dat.device, dtype=c.dtype)
        counts[match] = c[pos_safe[match]]

    # Indices of subtreatments below threshold
    idx_too_small = torch.nonzero(counts < tv_min_subtreat, as_tuple=False).reshape(-1)

    # Return as Python list[int] (small device->host copy of indices only)
    return idx_too_small.tolist()


def dummies_ordered_cuda(d: torch.Tensor,
                         cats: Sequence[int],
                         dtype: TorchDType | None = None,
                         unknown: str = 'error',
                         ) -> torch.Tensor:
    """
    One-hot encode 1D integer tensor d with columns ordered as in cats.

    Parameters
    ----------
    d:
        1D tensor of integer-coded categories (assumed already on CUDA if desired).
    cats:
        Sequence of allowed category values; defines output column order.
    dtype:
        Output dtype (e.g., torch.int8, torch.float32, torch.float64).
    unknown:
        - 'error'  -> raise if d contains values not in cats
        - 'ignore' -> unknown values become all-zeros rows

    Returns
    -------
    out:
        Tensor of shape (n, k) on same device as d.
    """
    if dtype is None:
        dtype = torch.int8

    if d.ndim != 1:
        raise TypeError('d must be 1D')
    if unknown not in ('error', 'ignore'):
        raise ValueError("unknown must be 'error' or 'ignore'")

    device = d.device
    # cats tensor on same device; use d.dtype for exact equality semantics
    cats_t = torch.as_tensor(cats, device=device, dtype=d.dtype)

    if cats_t.ndim != 1:
        raise TypeError('cats must be 1D')

    k = int(cats_t.numel())
    n = int(d.numel())

    # Disallow duplicates in cats
    if k > 0 and torch.unique(cats_t).numel() != cats_t.numel():
        raise ValueError('cats must not contain duplicates')

    # Handle empty cats explicitly (torch.searchsorted + clamp would break)
    if k == 0:
        if unknown == 'error' and n > 0:
            bad = torch.unique(d).tolist()
            raise ValueError(f'Unknown values in d (not in cats): {bad}')

        return torch.zeros((n, 0), device=device, dtype=dtype)

    # Build mapping via sorting + searchsorted, then map back to original cats-order
    order = torch.argsort(cats_t)              # indices that sort cats
    sorted_cats = cats_t[order]
    pos = torch.searchsorted(sorted_cats, d)   # in [0, k]

    in_range = pos < k
    pos_safe = pos.clamp(max=k - 1)            # safe for indexing
    valid = in_range & (sorted_cats[pos_safe] == d)

    if unknown == 'error' and not bool(valid.all().item()):
        bad = torch.unique(d[~valid]).tolist()
        raise ValueError(f'Unknown values in d (not in cats): {bad}')

    out = torch.zeros((n, k), device=device, dtype=dtype)

    if valid.any():
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        cols = order[pos[valid]]               # sorted position -> original cats position
        out[rows, cols] = 1

    return out


def scale_by_sd_of_first_cuda(a: torch.Tensor,
                              b: torch.Tensor, *,
                              ddof: int = 0,
                              zero_tol: float = 1e-10,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale two (N, K) tensors by the column-wise SD of `a` (GPU friendly).

    - If a or b is 1D, reshapes to (1, K).
    - Uses column-wise std of a over dim=0.
    - Avoids divide-by-zero for constant columns by replacing 0 SD with 1.
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('a and b must be 1D or 2D (N, K)')

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b must have the same number of columns (K)')

    # torch.std: correction == ddof (PyTorch 1.8+; in newer versions it's the preferred API)
    sd = a.std(dim=0, correction=ddof)  # shape (K,)

    # Replace zeros with ones to prevent division by zero; keep dtype/device
    # sd_safe = torch.where(sd == 0, torch.ones_like(sd), sd)
    sd_safe = torch.where(torch.isclose(sd, 0, rtol=zero_tol, atol=zero_tol),
                          torch.ones_like(sd), sd
                          )

    return a / sd_safe, b / sd_safe
