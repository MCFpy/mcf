"""
Created on Mon Dec 15 13:59:59 2025.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from __future__ import annotations

from numbers import Integral
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
    from mcf.mcf_init import GenTvCfg
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
                       penalize_version: bool | torch.Tensor = False,
                       return_residuals: bool = True,
                       standardize_x: bool = True,
                       interacted: bool = False,
                       cfg: 'GenTvCfg' | None = None,
                       scores_train: TensorLike = None,
                       scores_pred: TensorLike = None,
                       allow_local_version_scores: bool = True,
                       use_train_score_cache: bool = False,
                       ) -> tuple[torch.Tensor,
                                  list[torch.Tensor],
                                  torch.Tensor,
                                  list[torch.Tensor],
                                  int, int,
                                  str,
                                  ]:
    """Estimate common version weights and outcome-specific residuals on CUDA.

    y_train contains all outcomes in var_cfg.y_name order. cfg.y_tree_index
    selects the outcome for ridge tuning. Residuals have shape (n,) for one
    outcome and (n, n_outcomes) otherwise, with rows aligned to weights after
    applying w_index when supplied.
    """
    require_torch()

    if y_train is None or d_train is None:
        raise ValueError('y_train and d_train must be provided for version regression.')

    if d_train.ndim != 2 or d_train.shape[1] < 2:
        raise ValueError('d_train must be a 2D tensor with at least two columns: '
                         'main treatment and subtreatment.'
                         )
    if d_all_values is None:
        raise ValueError('d_all_values must be provided for version regression.')

    if int_dtype is None:
        int_dtype = torch.float64
    if out_dtype is None:
        out_dtype = torch.float32

    if torch.is_tensor(penalize_version):
        penalize_version = bool(penalize_version.item())

    txt = ''
    device = y_train.device
    with_x = x_train is not None and x_pred is not None

    if isinstance(weights, list):
        weights = torch.as_tensor(weights, device=device, dtype=out_dtype)
    else:
        weights = weights.to(device=device, dtype=out_dtype)
    weights = weights.reshape(-1)

    if not torch.is_tensor(d_all_values):
        d_all_values = torch.as_tensor(d_all_values, device=device)
    d_all_values = d_all_values.to(device=device)

    if d_sub_only is None:
        n_sub = 0
    elif torch.is_tensor(d_sub_only):
        n_sub = int(d_sub_only.numel())
    else:
        n_sub = len(d_sub_only)

    if n_sub < 2:
        txt += f'\nMain treatment {maintreat_idx}: No subtreatment'
        return weights, None, None, None, maintreat_idx + 1, 0, txt

    if y_train.ndim not in (1, 2) or y_train.shape[0] != d_train.shape[0]:
        raise ValueError('y_train must be a vector or matrix with one row per row of d_train.')
    if y_train.ndim == 2:
        if y_train.shape[1] == 0:
            raise ValueError('y_train must contain at least one outcome.')
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)

    no_of_out = 1 if y_train.ndim == 1 else y_train.shape[1]
    y_tree_index = None if cfg is None else cfg.y_tree_index
    if y_tree_index is None and no_of_out == 1:
        y_tree_index = 0
    if (not isinstance(y_tree_index, Integral)
            or not 0 <= y_tree_index < no_of_out):
        raise ValueError('cfg.y_tree_index must identify the tree-building outcome in y_train. '
                         f'Got {y_tree_index!r} for {no_of_out} outcome(s).'
                         )
    y_tree_index = int(y_tree_index)

    if w_index is not None:
        if not torch.is_tensor(w_index):
            w_index = torch.as_tensor(w_index, device=device)
        w_index = w_index.to(dtype=torch.long, device=device)

        y_train = y_train[w_index]
        d_train = d_train[w_index, :]
        if with_x:
            x_train = x_train[w_index, :]

    if weights.shape[0] != d_train.shape[0]:
        raise ValueError('weights and d_train must have the same number of observations '
                         'after applying w_index. Got '
                         f'weights.shape={weights.shape}, d_train.shape={d_train.shape}.'
                         )
    if with_x and x_train.shape[1] != x_pred.shape[1]:
        raise ValueError('Prediction and training data have different numbers of columns in '
                         f'version_wregr_cuda(): x_train.shape={x_train.shape}, '
                         f'x_pred.shape={x_pred.shape}. This usually means that score '
                         'variables were added to one matrix but not the other.'
                         )
    x_score_train_all = x_train if with_x else None
    x_score_pred_all = x_pred if with_x else None

    main_treat_val = d_all_values[treat_idx, 0]
    select_data = (torch.isclose(d_train[:, 0], main_treat_val,
                                 atol=zero_tol, rtol=zero_tol,
                                 )
                   & (weights > zero_tol)
                   )
    if treat_idx == 0:
        first_time_main = True
    else:
        first_time_main = not torch.isclose(main_treat_val,
                                            d_all_values[treat_idx - 1, 0],
                                            atol=zero_tol,
                                            rtol=zero_tol,
                                            ).item()

    if first_time_main:
        # Rescale only when computing this main treatment's cached regression results.
        if standardize_x and with_x:
            x_train, x_pred = scale_by_sd_of_first_cuda(x_train, x_pred,
                                                        ddof=0, zero_tol=zero_tol,
                                                        )
        y_tr = y_train[select_data]
        y_tr_tune = y_tr if y_tr.ndim == 1 else y_tr[:, y_tree_index]
        d_sub_tr = d_train[select_data, 1]
        x_tr = x_train[select_data, :] if with_x else None
        w_tr = weights[select_data].to(dtype=int_dtype)

        container_w, container_r = [None] * n_sub, [None] * n_sub

        subtreat_idx_too_small = too_small_cuda(d_sub_tr, d_sub_only,
                                                tv_min_subtreat,
                                                zero_tol=zero_tol,
                                                )

        if len(subtreat_idx_too_small) == n_sub:
            txt += ('\nAll subtreatments are too small for version regression. '
                    f'Main_treatment: {main_treat_val.item()}. Original weights '
                    'are kept.'
                    )
            for sub_idx in range(n_sub):
                container_w[sub_idx] = weights.clone()
                container_r[sub_idx] = (torch.zeros_like(y_train, dtype=weights.dtype)
                                        if return_residuals else None
                                        )
        else:
            d_dummy = dummies_ordered_cuda(d_sub_tr, d_sub_only, dtype=int_dtype, unknown='error')

            if subtreat_idx_too_small:
                col_mask = torch.ones(d_dummy.shape[1], device=device,
                                      dtype=torch.bool)
                col_mask[subtreat_idx_too_small] = False
                d_dummy = d_dummy[:, col_mask]

                txt += ('\nThe following subtreatments are too small for version '
                        f'regression: Main_treatment: {main_treat_val.item()} '
                        f'Subtreatments: {" ".join(map(str, subtreat_idx_too_small))}'
                        )

            if cfg is not None:
                score_type = cfg.score_type or ()
                if isinstance(score_type, str):
                    score_type = (score_type,)

                ps_version_yes = cfg.addscores and 'prop_version' in score_type
                pg_version_yes = cfg.addscores and 'prog_version' in score_type
                version_yes = ps_version_yes or pg_version_yes

                local_version_scores = allow_local_version_scores and cfg.score_version_local
                global_version_scores = not cfg.score_version_local
                version_scores = version_yes and (local_version_scores or global_version_scores)

                if local_version_scores and version_yes:
                    raise NotImplementedError('Local version scores are not implemented in the '
                                              'CUDA path. Use the CPU path or precompute global '
                                              'version scores.'
                                              )
                if with_x and cfg.addscores and version_scores:
                    if not global_version_scores:
                        raise NotImplementedError('Only global version scores are implemented in '
                                                  'the CUDA path.'
                                                  )
                    if not use_train_score_cache:
                        raise NotImplementedError('CUDA global version scores require '
                                                  'use_train_score_cache=True.'
                                                  )
                    if w_index is None:
                        raise ValueError('CUDA training-score cache requires w_index.')

                    if scores_train is None or scores_pred is None:
                        raise ValueError('Global CUDA version scores require scores_train '
                                         'and scores_pred.'
                                         )
                    if x_score_train_all is None or x_score_pred_all is None:
                        raise ValueError('Version-score variables require x_train and x_pred.')

                    score_rows = w_index[select_data].to(dtype=torch.long)
                    score_tr = scores_train[score_rows, :].to(device=device, dtype=x_tr.dtype)
                    score_pr = scores_pred.to(device=device, dtype=x_pred.dtype)

                    x_tr = torch.cat((x_tr, score_tr), dim=1)
                    x_pred = torch.cat((x_pred, score_pr), dim=1)

            x_tr_use = None if x_tr is None else x_tr.to(dtype=int_dtype)
            regr_tr = build_tv_regressors_train_cuda(d_dummy, x_tr_use, interacted=interacted)

            regr_eval_list_o = []
            col_1 = 0
            n_dummy_cols = d_dummy.shape[1]
            # Average the augmented covariates only after all scores have been added.
            x_pred_use = (None if x_pred is None
                          else x_pred.to(dtype=int_dtype).mean(dim=0, keepdim=True)
                          )

            for sub_idx in range(n_sub):
                if subtreat_idx_too_small and sub_idx in subtreat_idx_too_small:
                    continue

                dummy_eval = torch.zeros((1, n_dummy_cols), device=device, dtype=int_dtype)
                dummy_eval[:, col_1] = 1.0

                regr_eval = build_tv_regressors_eval_cuda(dummy_eval, x_pred_use,
                                                          interacted=interacted,
                                                          )
                regr_eval_list_o.append(regr_eval)
                col_1 += 1

            if ridge:
                grid_length = 30
                if penalize_version:
                    grid = torch.logspace(-5, 2,
                                          steps=grid_length,
                                          base=10.0,
                                          device=device,
                                          dtype=int_dtype,
                                          )
                    not_penalize_first_k = 1

                    constant_tr = torch.ones((regr_tr.shape[0], 1),
                                             dtype=regr_tr.dtype,
                                             device=regr_tr.device,
                                             )
                    regr_tr = torch.cat((constant_tr, regr_tr), dim=1)

                    regr_eval_list = [torch.cat((torch.ones((regr_eval.shape[0], 1),
                                                            dtype=regr_eval.dtype,
                                                            device=regr_eval.device,
                                                            ),
                                                 regr_eval,
                                                 ),
                                                dim=1,
                                                )
                                      for regr_eval in regr_eval_list_o
                                      ]
                else:
                    grid = torch.cat((torch.zeros(1, device=device,
                                                  dtype=int_dtype),
                                      torch.logspace(-5, 2, steps=grid_length,
                                                     base=10.0,
                                                     device=device,
                                                     dtype=int_dtype,
                                                     ),
                                      ),
                                     dim=0,
                                     )
                    not_penalize_first_k = d_dummy.shape[1]
                    regr_eval_list = regr_eval_list_o

                # Use the tree-building outcome to select one common penalty.
                best_lam, _ = mcf_ewr.ridge_cv_penalty_cuda(regr_tr, y_tr_tune,
                                                            w_tr.reshape(-1, 1),
                                                            lambdas=grid, k=cv_k,
                                                            shuffle=True, seed=1234566,
                                                            not_penalize_first_k=
                                                                not_penalize_first_k,
                                                            dtype=int_dtype,
                                                            )
                weights_tv_list, resid_tv_list = (
                    mcf_ewr.ridge_equivalent_weights_for_mean_cuda(
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
                    )
            else:
                regr_eval_list = regr_eval_list_o
                weights_tv_list, resid_tv_list = (
                    mcf_ewr.wls_equivalent_weights_for_mean_cuda(
                        regr_tr,
                        w_tr.reshape(-1, 1),
                        y=y_tr if return_residuals else None,
                        x_eval=regr_eval_list,
                        int_dtype=int_dtype,
                        out_dtype=out_dtype,
                        weights_eval=None,
                        return_residuals=return_residuals,
                        )
                    )
            col_1 = 0
            for sub_idx in range(n_sub):
                weights_return = weights.clone()
                residuals_return = (torch.zeros_like(y_train, dtype=weights.dtype)
                                    if return_residuals else None
                                    )
                if subtreat_idx_too_small and sub_idx in subtreat_idx_too_small:
                    container_w[sub_idx] = weights_return
                    container_r[sub_idx] = residuals_return
                    continue

                weights_return[select_data] = (weights_tv_list[col_1].reshape(-1)
                                               .to(dtype=weights.dtype)
                                               )
                container_w[sub_idx] = weights_return

                if return_residuals:
                    residuals_return[select_data] = resid_tv_list[col_1].to(dtype=weights.dtype)
                    container_r[sub_idx] = residuals_return
                else:
                    container_r[sub_idx] = None

                col_1 += 1

    if container_w is None:
        raise ValueError('container_w must be provided for non-first subtreatment calls.')
    if return_residuals and container_r is None:
        raise ValueError('container_r must be provided when return_residuals=True.')

    weight_new = container_w[subtreat_idx]
    residual_new = container_r[subtreat_idx] if return_residuals else None

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
    Scale two (N, K) tensors by the column-wise SD of `a`.

    Returns (a_scaled, b_scaled).
    """
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('a and b must be 1D or 2D arrays/tensors.')

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b must have the same number of columns (K)')

    sd = a.std(dim=0, correction=ddof)
    sd_safe = torch.where(torch.isclose(sd, torch.zeros_like(sd), atol=zero_tol, rtol=zero_tol),
                          torch.ones_like(sd),
                          sd,
                          )
    return a / sd_safe, b / sd_safe


def build_tv_regressors_train_cuda(d_dummy: torch.Tensor,
                                   x_tr: torch.Tensor | None, *,
                                   interacted: bool,
                                   ) -> torch.Tensor:
    """Build training regressor matrix for treatment-version regression."""
    if x_tr is None:
        return d_dummy

    if interacted:
        interact_tr = (d_dummy[:, :, None] * x_tr[:, None, :]).reshape(d_dummy.shape[0], -1)
        return torch.cat((d_dummy, interact_tr), dim=1)

    return torch.cat((d_dummy, x_tr), dim=1)


def build_tv_regressors_eval_cuda(dummy_eval: torch.Tensor,
                                  x_pred: torch.Tensor | None, *,
                                  interacted: bool,
                                  ) -> torch.Tensor:
    """Build evaluation regressor matrix for treatment-version regression."""
    if x_pred is None:
        return dummy_eval

    if interacted:
        interact_eval = (dummy_eval[:, :, None] * x_pred[:, None, :]).reshape(
            dummy_eval.shape[0], -1
            )
        return torch.cat((dummy_eval, interact_eval), dim=1)

    return torch.cat((dummy_eval, x_pred), dim=1)
