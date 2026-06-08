"""
Created on Mon Dec 15 13:59:59 2025.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd

from mcf import mcf_estimation_weighted_regression as mcf_ewr

if TYPE_CHECKING:
    from mcf.mcf_init import VarCfg, GenTvCfg

type ArrayLike = NDArray[Any] | None


def version_wregr(weights: NDArray[np.floating], *,
                  y_train: ArrayLike = None,
                  d_train: NDArray[np.integer] | None = None,
                  x_train: ArrayLike = None,
                  x_pred: ArrayLike = None,
                  cfg: Any = None,
                  container_w: list[np.floating] | None = None,
                  container_r: list[np.floating] | None = None,
                  w_index: ArrayLike = None,
                  treat_idx: int = 0,  # Running overall index (counting treatments + subtreatments)
                  maintreat_idx: int = 0,
                  subtreat_idx: int | None = 0,
                  int_dtype: DTypeLike = np.float64,
                  out_dtype: DTypeLike = np.float32,
                  zero_tol: float = 1e-10,
                  ridge: bool = True,
                  penalize_version: bool = False,
                  return_residuals: bool = True,
                  standardize_x: bool = True,
                  ) -> tuple[NDArray[np.floating],
                             list[NDArray[np.floating]],
                             NDArray[np.floating],
                             list[NDArray[np.floating]],
                             int, int,
                             str,
                             ]:
    """Estimate new weights based on treatment versions."""
    txt = ''
    d_sub_only = cfg.d_dict['sub_treat'][maintreat_idx]  # Subtreatments
    with_x = x_train is not None and x_pred is not None
    xd_interacted = cfg.specification == 'interacted'

    if not d_sub_only or len(d_sub_only) < 2:   # No subtreatments, regression not needed
        txt += f'\nMain treatment {maintreat_idx}: No subtreatment'

        return weights, None, None, None, maintreat_idx + 1, 0, txt

    if w_index is not None:    # Relevant for IATE estimation
        y_train = y_train[w_index]
        d_train = d_train[w_index, :]
        if with_x:
            x_train = x_train[w_index, :]

    if with_x and x_train.shape[1] != x_pred.shape[1]:
        warn_msg = ('Prediction and training data have different numbers of columns in '
                    f'version_wregr(): x_train.shape={x_train.shape}, '
                    f'x_pred.shape={x_pred.shape}. Training data are used for prediction '
                    'in the version regression fallback.'
                    )
        warned = getattr(version_wregr, '_warned_x_shape_mismatches', set())
        key = (x_train.shape[1], x_pred.shape[1],)

        if key not in warned:
            warnings.warn(warn_msg, RuntimeWarning, stacklevel=2)
            warned.add(key)
            version_wregr._warned_x_shape_mismatches = warned  # pylint: disable=W0212

        x_pred = x_train.copy()
        txt += '\n' + warn_msg

    if standardize_x and with_x:
        x_train, x_pred = scale_by_sd_of_first(x_train, x_pred, ddof=0, zero_tol=zero_tol)

    d_all_values = cfg.d_all_values
    # Select all within same main treatment AND positive weights
    main_treat_val = d_all_values[treat_idx, 0]
    # select_data = (d_train[:, 0] == main_treat_val) & (weights > zero_tol)
    select_data = (np.isclose(d_train[:, 0], main_treat_val, atol=zero_tol, rtol=zero_tol)
                   & (weights > zero_tol)
                   )
    # First time this main treatment is seen by this function
    # if treat_idx == 0 or main_treat_val != d_all_values[treat_idx-1, 0]:
    if treat_idx == 0 or not np.isclose(main_treat_val, d_all_values[treat_idx-1, 0],
                                        atol=zero_tol, rtol=zero_tol
                                        ):
        # In this branch: Compute new weights and keep them in container
        # 0) Select data & predefine container list with [None] * subtreatments

        # This data contains all subtreatments together as needed for regression
        y_tr = y_train[select_data]
        d_sub_tr = d_train[select_data, 1]
        x_tr = x_train[select_data, :] if with_x else None
        w_tr = weights[select_data]

        no_of_sub_treat = len(d_sub_only)   # Number of subtreatments

        container_w = [None] * no_of_sub_treat  # One spot for each subtreatment
        container_r = [None] * no_of_sub_treat  # One spot for each subtreatment
        # 1) Check if treatments are sufficiently full (for non-zero weights).
        #    If not, set weights to average weights,i.e. just take weights as is
        subtreat_idx_too_small = too_small(d_sub_tr, d_sub_only, cfg.tv_min_subtreat)
        # 2) Estimate either regression model or ridge with x_dat + d_dat as regressors
        # a) Define dummy variables for subtreatments
        d_dummy = dummies_ordered(d_sub_tr, d_sub_only, dtype=int_dtype, unknown='error')
        if subtreat_idx_too_small:
            # Drop columns with too small cells
            d_dummy = np.delete(d_dummy, subtreat_idx_too_small, axis=1)
            txt += ('\nThe following subtreatments are too small for version '
                   f'regression: Main_treatment: {main_treat_val} '
                   f'Subtreatments: {" ".join(subtreat_idx_too_small)}'
                   )
        regr_tr = build_tv_regressors_train(d_dummy, x_tr, interacted=xd_interacted)

        # Construct list of regressor to be used for predicting subeffects
        regr_eval_list_o = []
        col_1 = 0
        n_pred = x_pred.shape[0] if with_x else weights.shape[0]
        for sub_idx, _ in enumerate(d_sub_only):
            if sub_idx not in subtreat_idx_too_small:
                dummy_eval = np.zeros((n_pred, d_dummy.shape[1]))
                dummy_eval[:, col_1] = 1.0

                regr_eval = build_tv_regressors_eval(dummy_eval, x_pred, interacted=xd_interacted)

                regr_eval_list_o.append(regr_eval.copy())
                col_1 += 1

        if ridge:
            grid_length = 30
            if penalize_version:
                constant_tr = np.ones((regr_tr.shape[0], 1), dtype=regr_tr.dtype)
                regr_tr = np.concatenate((constant_tr, regr_tr), axis=1)
                regr_eval_list = [np.concatenate((np.ones((regr_eval.shape[0], 1),
                                                          dtype=regr_eval.dtype),
                                                  regr_eval,
                                                  ), axis=1,
                                            )
                                  for regr_eval in regr_eval_list_o
                                  ]
                not_penalize_first_k = 1
                grid = np.logspace(-5, 2, grid_length)
            else:
                regr_eval_list = regr_eval_list_o
                not_penalize_first_k = d_dummy.shape[1]
                grid=np.zeros(grid_length + 1)  # To include 0
                grid[1:] = np.logspace(-5, 2, grid_length)

            # Determine penalty by cross-validation
            best_lam, _ = mcf_ewr.ridge_cv_penalty(regr_tr,
                                                   y_tr,
                                                   w_tr.reshape(-1, 1),
                                                   lambdas=grid,
                                                   k=cfg.cv_k,
                                                   shuffle=True,
                                                   seed=1234566,
                                                   not_penalize_first_k=not_penalize_first_k,
                                                   dtype=int_dtype,
                                                   )
            weights_tv_list, residual_tv_list = mcf_ewr.ridge_equivalent_weights_for_mean(
                regr_tr,
                w_tr.reshape(-1, 1),
                y=y_tr if return_residuals else None,
                lam=best_lam,
                x_eval=regr_eval_list,
                int_dtype=int_dtype, out_dtype=out_dtype,
                weights_eval=None,
                not_penalize_first_k=not_penalize_first_k,
                return_residuals=return_residuals,
                )
        else:
            regr_eval_list = regr_eval_list_o
            weights_tv_list, residual_tv_list = mcf_ewr.wls_equivalent_weights_for_mean(
                regr_tr,
                w_tr.reshape(-1, 1),
                y=y_tr if return_residuals else None,
                x_eval=regr_eval_list,
                int_dtype=int_dtype, out_dtype=out_dtype,
                weights_eval=None,
                return_residuals=return_residuals,
                )
        col_1 = 0
        for sub_idx, _ in enumerate(d_sub_only):
            weights_return = weights.copy()
            residual_return = np.zeros_like(weights_return) if return_residuals else None
            if sub_idx in subtreat_idx_too_small:
                container_w[sub_idx] = weights_return
                container_r[sub_idx] = residual_return if return_residuals else None
            else:
                weights_return[select_data] = weights_tv_list[col_1].reshape(-1)
                container_w[sub_idx] = weights_return
                if return_residuals:
                    residual_return[select_data] = residual_tv_list[col_1].reshape(-1)
                    container_r[sub_idx] = residual_return
                else:
                    container_r[sub_idx] = None

                col_1 += 1

    weight_new = container_w[subtreat_idx]
    residual_new = container_r[subtreat_idx] if return_residuals else None

    # Update indices of treatments
    if subtreat_idx == len(d_sub_only) - 1:    # Last index
        maintreat_idx += 1 # Next main treatment
        subtreat_idx = 0
    else:
        subtreat_idx += 1   # New value for next arrival in this function

    return weight_new, container_w, residual_new, container_r, maintreat_idx, subtreat_idx, txt


def joint_d_values(main_values: list[int],
                   sub_values: list[list[int]],
                   as_string: bool = False,
                   ) -> list[int]:
    """Create a numerical joint label of main and subtreatments."""
    if as_string:
        return [str(m) + '_' + str(s) for m in main_values for s in sub_values[m]]

    return [m * 1000 + s for m in main_values for s in sub_values[m]]


def dummies_ordered(d: NDArray[np.integer],
                    cats: list[int],
                    dtype=np.int8,
                    unknown='error',
                    ) -> NDArray[np.integer]:
    """
    One-hot encode 1D integer array x with columns ordered as in cats.

    unknown:
      - 'error'  -> raise if x contains values not in cats
      - 'ignore' -> unknown values become all-zeros rows
    """
    cats = np.asarray(cats)

    if d.ndim != 1 or cats.ndim != 1:
        raise ValueError('x and cats must both be 1D')
    if np.unique(cats).size != cats.size:
        raise ValueError('cats must not contain duplicates')

    k = cats.size
    n = d.size

    # Build a mapping via sorting + searchsorted, then map back to original
    # cats-order
    order = np.argsort(cats)          # indices that sort cats
    sorted_cats = cats[order]
    pos = np.searchsorted(sorted_cats, d)

    valid = (pos < k) & (sorted_cats[pos] == d)

    if unknown == 'error' and not np.all(valid):
        bad = np.unique(d[~valid])
        raise ValueError(f'Unknown values in d (not in cats): {bad.tolist()}')

    out = np.zeros((n, k), dtype=dtype)
    rows = np.nonzero(valid)[0]
    cols = order[pos[valid]]     # map sorted position -> original cats position
    out[rows, cols] = 1

    return out


def too_small(d_dat: NDArray[np.integer],
              d_sub_values: list[int] | tuple[int, ...],
              tv_min_subtreat: int,
              ) -> list[int]:
    """Return list of indices of treatment with too small values."""   
    u, c = np.unique(d_dat, return_counts=True)
    count_map = dict(zip(u.tolist(), c.tolist()))
    # returns also False if value in d_sub_values is not in data
    subtreat_idx_too_small = [idx for idx, v in enumerate(d_sub_values)
                              if (count_map.get(v, 0) < tv_min_subtreat)
                              ]
    return subtreat_idx_too_small


def expand_dimension(matrix1: ArrayLike,
                     matrix2: ArrayLike,
                     matrix3: ArrayLike,
                     matrix4: ArrayLike,
                     matrix5: ArrayLike, *,
                     gen_tv_cfg: 'GenTvCfg',
                     ) -> tuple[int, NDArray, NDArray, NDArray, NDArray]:
    """Expand array dimension to accomodate treatment versions."""
    no_of_treat = gen_tv_cfg.no_of_treat_all
    d_values = gen_tv_cfg.d_all_values
    no_treat_per_main = gen_tv_cfg.no_of_treat_per_main

    matrix1_new = None if matrix1 is None else np.repeat(matrix1, repeats=no_treat_per_main, axis=1)
    matrix2_new = None if matrix2 is None else np.repeat(matrix2, repeats=no_treat_per_main, axis=1)
    matrix3_new = None if matrix3 is None else np.repeat(matrix3, repeats=no_treat_per_main, axis=1)
    matrix4_new = None if matrix4 is None else np.repeat(matrix4, repeats=no_treat_per_main, axis=1)
    matrix5_new = None if matrix5 is None else np.repeat(matrix5, repeats=no_treat_per_main, axis=1)

    return no_of_treat, d_values, matrix1_new, matrix2_new, matrix3_new, matrix4_new,  matrix5_new


def get_tv_data(data_df: pd.DataFrame | None,
                weights_dic: dict[str, Any],
                var_cfg: 'VarCfg',
                zero_tol: float = 1e-10,
                x_tv_pred_np: NDArray[Any] | None = None,
                ) -> tuple[NDArray[np.integer], NDArray[Any], NDArray[Any]]:
    """Get data for the regression step when using treatment versions."""
    spec = fit_tv_feature_spec(weights_dic['x_tv_dat'], var_cfg, zero_tol=zero_tol)
    x_tv_train = transform_x_tv(weights_dic['x_tv_dat'], spec)

    if x_tv_pred_np is None:
        x_tv_pred = transform_x_tv(data_df[var_cfg.x_name_tv], spec)
    else:
        x_tv_pred = x_tv_pred_np

    d_dat = weights_dic['d_dat']

    return d_dat, x_tv_train, x_tv_pred


def d_dictionary(data_all_df: pd.DataFrame,
                 d_name: tuple[str,...] | list[str],
                 zero_tol: float = 1e-10
                 ) -> dict:
    """Build dictionary if treatment and subtreatmend information."""
    d_np = data_all_df[d_name].to_numpy(dtype=np.float32)
    # Main treatments
    d_main_values = np.unique(d_np[:, 0])
    d_main_values = [int(d) for d in d_main_values]
    # Subtreatments
    versions = [[0]] * len(d_main_values)
    for d_main_idx, d_main_val in enumerate(d_main_values):
        # treat = d_np[:, 0] == d_main_val
        treat = np.isclose(d_np[:, 0], d_main_val, atol=zero_tol, rtol=zero_tol)
        version_vals = list(np.unique(d_np[treat, 1]))
        if len(version_vals) > 1:   # Otherwise, there are no versions
            version_vals = [int(v) for v in version_vals]
            versions[d_main_idx] = version_vals

    d_dict = {'main_treat': d_main_values, 'sub_treat': versions,}
    return d_dict


def versions_per_treatment(d_train: NDArray[np.integer], zero_tol: float = 1e-10,
                           ) -> tuple[list[np.integer], list[list[np.integer] | None]]:
    """Create a list that with the version information for each treatment."""
    # Currently not used.
    if d_train.ndim == 1 or d_train.shape[1] == 1:  # Only 1 column
        d_main_values = list(np.unique(d_train))
        return d_main_values, None   # No treatment versions

    d_main_values = list(np.unique(d_train[:, 0]))
    versions = [None] * len(d_main_values)
    for d_main_idx, d_main_val in enumerate(d_main_values):
        # treat = d_train[:, 0] == d_main_val
        treat = np.isclose(d_train[:, 0], d_main_val, rtol=zero_tol, atol=zero_tol)
        version_vals = list(np.unique(d_train[treat, 1]))
        if len(version_vals) > 1:   # Otherwise, there are no versions
            versions[d_main_idx] = version_vals

    return d_main_values, versions


def check_subtreatments_available(d_df: pd.DataFrame,
                                  d_dict_train: dict,
                                  zero_tol: float = 1e-10,
                                  ) -> None:
    """Check if prediction data have same subtreatments as training."""
    d_dict_pred = d_dictionary(d_df, d_df.columns, zero_tol=zero_tol)
    if d_dict_train != d_dict_pred:
        raise ValueError('Treatment differently coded (or available) for training and prediction. '
                         '\nCorrect, or set ATET and GATET to False (in this case '
                         'treatment information is not needed in the prediction data).'
                         )


def pot_eff_names(d_values: list[np.integer],
                  versions: list[list[Any]] | None,
                  str1: str,
                  str2: str,
                  start: int,
                  ) -> str:
    """Create name string for potential outcomes and effects."""
    # Currently not used
    if versions is None:
        return [str1 + str(val) + str2 for val in enumerate(d_values[start:], start=start)]

    label_list = []
    for main_idx, main_val in enumerate(d_values[start:], start=start):
        version = versions[main_idx]
        if version is None:
            label_list.append(str1 + str(main_val) + str2)
        else:
            for sub_val in version:
                label_list.append(str1 + str(main_val) + '_' + str(sub_val) + str2)

    return label_list


def scale_by_sd_of_first(a: NDArray, b: NDArray, *, ddof: int = 0, zero_tol: float = 1e-10,
                         ) -> tuple[NDArray, NDArray]:
    """
    Scale two (N, K) arrays by the column-wise SD of `a`.

    Returns (a_scaled, b_scaled).
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b must have the same number of columns (K)')

    sd = a.std(axis=0, ddof=ddof)          # shape (K,)
    sd_safe = np.where(np.isclose(sd, 0, atol=zero_tol, rtol=zero_tol), 1.0, sd)
    # avoid divide-by-zero for constant columns

    return a / sd_safe, b / sd_safe


@dataclass(slots=True, frozen=True, kw_only=True)
class TvFeatureSpec:
    """Feature schema for treatment-version regressions."""

    names_tv_ordered: list[str]
    names_tv_unordered: list[str]
    unordered_dummy_columns: list[str]


def fit_tv_feature_spec(x_tv_df: pd.DataFrame,
                        var_cfg: 'VarCfg', *,
                        zero_tol: float = 1e-10,
                        ) -> TvFeatureSpec:
    """Fit treatment-version feature schema on training data."""
    del zero_tol  # kept for API compatibility

    x_name_tv = var_cfg.x_name_tv
    names_ordered = var_cfg.x_name_ord
    names_unordered = var_cfg.x_name_unord

    names_tv_ordered = [name for name in x_name_tv if name in names_ordered]
    names_tv_unordered = [name for name in x_name_tv if name in names_unordered]

    if names_tv_unordered:
        x_unord_df = x_tv_df[names_tv_unordered].astype('category')
        x_unord_dummies = pd.get_dummies(data=x_unord_df, columns=names_tv_unordered,
                                         prefix=names_tv_unordered, drop_first=True, dtype='int8',
                                         )
        unordered_dummy_columns = list(x_unord_dummies.columns)
    else:
        unordered_dummy_columns = []

    return TvFeatureSpec(names_tv_ordered=names_tv_ordered,
                         names_tv_unordered=names_tv_unordered,
                         unordered_dummy_columns=unordered_dummy_columns,
                         )


def transform_x_tv(x_tv_df: pd.DataFrame | None, spec: TvFeatureSpec) -> NDArray[Any] | None:
    """Transform treatment-version features using a training-defined schema."""
    if x_tv_df is None:
        return None

    if spec.names_tv_ordered:
        x_ord = x_tv_df[spec.names_tv_ordered].to_numpy(dtype=np.float32)
    else:
        x_ord = None

    if spec.names_tv_unordered:
        x_unord_df = x_tv_df[spec.names_tv_unordered].astype('category')
        x_unord_dummies = pd.get_dummies(data=x_unord_df, columns=spec.names_tv_unordered,
                                         prefix=spec.names_tv_unordered, drop_first=True,
                                         dtype='int8',
                                         )
        x_unord_dummies = x_unord_dummies.reindex(columns=spec.unordered_dummy_columns,
                                                  fill_value=0,
                                                  )
        x_unord = x_unord_dummies.to_numpy(dtype=np.float32)
    else:
        x_unord = None

    if x_ord is not None and x_unord is not None:
        return np.concatenate((x_ord, x_unord), axis=1)
    if x_ord is not None:
        return x_ord

    return x_unord


def build_tv_regressors_train(d_dummy: NDArray[Any],
                              x_tr: NDArray[Any] | None, *,
                              interacted: bool,
                              ) -> NDArray[Any]:
    """Build training regressor matrix for treatment-version regression."""
    if x_tr is None:
        return d_dummy

    if interacted:
        interact_tr = (d_dummy[:, :, None] * x_tr[:, None, :]).reshape(d_dummy.shape[0], -1)
        return np.concatenate((d_dummy, interact_tr), axis=1)

    return np.concatenate((d_dummy, x_tr), axis=1)


def build_tv_regressors_eval(dummy_eval: NDArray[Any],
                             x_pred: NDArray[Any] | None, *,
                             interacted: bool,
                             ) -> NDArray[Any]:
    """Build evaluation regressor matrix for treatment-version regression."""
    if x_pred is None:
        return dummy_eval

    if interacted:
        interact_eval = (dummy_eval[:, :, None] * x_pred[:, None, :]
                         ).reshape(dummy_eval.shape[0], -1,)

        return np.concatenate((dummy_eval, interact_eval), axis=1)

    return np.concatenate((dummy_eval, x_pred), axis=1)
