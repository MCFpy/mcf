"""
Created on Mon Dec 15 13:59:59 2025.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd
from psutil import cpu_count
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mcf import mcf_estimation_weighted_regression as mcf_ewr
from mcf.mcf_data import dummies_for_unord, get_x_data
from mcf.mcf_estimation import predict_scores, train_scores, get_score_names, dtype_score
from mcf.mcf_general import get_key_values_in_list, to_numpy_big_data
from mcf.mcf_print_stats import print_mcf

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg, GenTvCfg, IntCfg, VarCfg
    from mcf.mcf_main import ModifiedCausalForest

type ArrayLike = NDArray[Any] | None
type Number = int | float | np.integer | np.floating
type ScoreInst = BaseEstimator | float | np.floating | int | np.integer
type ScoreScaler = StandardScaler | None
type WeightContainer = list[NDArray[np.floating] | None] | None


@dataclass(slots=True, frozen=True, kw_only=True)
class GenTvData:
    """Data needed for treatment-version IATE computations."""

    d_train: NDArray[np.integer]
    x_tv_train: NDArray[Any] | None
    x_tv_pred: NDArray[Any] | None
    scores_pred: list[NDArray[Any] | None] | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class TvFeatureSpec:
    """Feature schema for treatment-version regressions."""

    names_tv_ordered: list[str]
    names_tv_unordered: list[str]
    unordered_dummy_columns: list[str]
    names_tv_scores: list[str]
    n_base_features: int


@dataclass(slots=True, kw_only=True)
class TvScores:
    """Hold training data & information to compute prognostic & propensity score for version est."""

    scores_df: pd.DataFrame | None           # Global scores
    scores_np: NDArray[Any] | None = None    # Cached training predictions
    instances: list[ScoreInst] | list[list[ScoreInst]] | None = None
    scalers: list[ScoreScaler] | list[list[ScoreScaler]] | None = None
    names: list[str] | tuple[str, ...] | None = None
    d_values: NDArray[Any] | None = None
    feature_spec: TvFeatureSpec | None = None


def version_wregr(weights: NDArray[np.floating],
                  *,
                  cfg: 'GenTvCfg',
                  y_train: ArrayLike = None,
                  d_train: NDArray[np.integer] | None = None,
                  x_train: ArrayLike = None,
                  x_pred: ArrayLike = None,
                  container_w: WeightContainer = None,
                  container_r: WeightContainer = None,
                  w_index: ArrayLike = None,
                  treat_idx: int = 0,  # Running overall index (counting treatments + subtreatments)
                  maintreat_idx: int = 0,
                  subtreat_idx: int = 0,
                  int_dtype: DTypeLike = np.float64,
                  out_dtype: DTypeLike = np.float32,
                  zero_tol: float = 1e-10,
                  ridge: bool = True,
                  penalize_version: bool = False,
                  return_residuals: bool = True,
                  standardize_x: bool = True,
                  mp_parallel: int=round(cpu_count(logical=True) * 0.8),
                  obs_bigdata: int=100_000,
                  allow_local_version_scores: bool=True,
                  use_train_score_cache: bool = False,
                  scores_pred: ArrayLike = None,
                  ) -> tuple[NDArray[np.floating],
                             WeightContainer,
                             NDArray[np.floating] | None,
                             WeightContainer,
                             int,
                             int,
                             str,
                             ]:
    """Estimate common version weights and outcome-specific residuals.

    y_train contains all outcomes in var_cfg.y_name order. cfg.y_tree_index
    selects the outcome for ridge tuning and local prognostic-score estimation.
    Residuals have shape (n,) for one outcome and (n, n_outcomes) otherwise,
    with rows aligned to weights after applying w_index when supplied.
    scores_pred optionally supplies global version scores in x_pred row order;
    it requires use_train_score_cache=True and an aligned outcome-sample cache.
    With w_index=None, the cache and y_train/d_train must contain the full forest
    outcome sample in the same row order. Otherwise, w_index selects cache rows.
    """
    txt = ''
    d_sub_only = cfg.d_dict['sub_treat'][maintreat_idx]  # Subtreatments
    with_x = x_train is not None and x_pred is not None
    xd_interacted = cfg.specification == 'interacted'

    score_type = cfg.score_type or ()
    if isinstance(score_type, str):
        score_type = (score_type,)
    ps_version_yes = cfg.addscores and 'prop_version' in score_type
    pg_version_yes = cfg.addscores and 'prog_version' in score_type
    version_yes = ps_version_yes or pg_version_yes

    if scores_pred is not None:
        if not version_yes or cfg.score_version_local:
            raise ValueError('Precomputed prediction scores require global version scores.')
        if not use_train_score_cache:
            raise ValueError('Precomputed prediction scores require use_train_score_cache=True.')

    if y_train is None or d_train is None:
        raise ValueError('y_train and d_train must be provided for version regression.')

    if d_train.ndim != 2 or d_train.shape[1] < 2:
        raise ValueError('d_train must be a 2D array with at least two columns: '
                         'main treatment and subtreatment.'
                         )
    if not d_sub_only or len(d_sub_only) < 2:   # No subtreatments, regression not needed
        txt += f'\nMain treatment {maintreat_idx}: No subtreatment'

        return weights, None, None, None, maintreat_idx + 1, 0, txt

    y_train = np.asarray(y_train)
    if y_train.ndim not in (1, 2) or y_train.shape[0] != d_train.shape[0]:
        raise ValueError('y_train must be a vector or matrix with one row per row of d_train.')
    if y_train.ndim == 2:
        if y_train.shape[1] == 0:
            raise ValueError('y_train must contain at least one outcome.')
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)

    no_of_out = 1 if y_train.ndim == 1 else y_train.shape[1]
    y_tree_index = cfg.y_tree_index
    if y_tree_index is None and no_of_out == 1:
        y_tree_index = 0
    if (not isinstance(y_tree_index, (int, np.integer))
            or not 0 <= y_tree_index < no_of_out):
        raise ValueError('cfg.y_tree_index must identify the tree-building outcome in y_train. '
                         f'Got {y_tree_index!r} for {no_of_out} outcome(s).'
                         )
    if w_index is not None:    # Only relevant for IATE estimation; weights is already w_index'ed
        y_train = y_train[w_index]
        d_train = d_train[w_index, :]
        if with_x:
            x_train = x_train[w_index, :]

    if weights.shape[0] != d_train.shape[0]:
        raise ValueError('weights and d_train must have the same number of observations after '
                         f'applying w_index. Got weights.shape={weights.shape}, '
                         f'd_train.shape={d_train.shape}.'
                         )
    if with_x and x_train.shape[1] != x_pred.shape[1]:
        raise ValueError('Prediction and training data have different numbers of columns in '
                         f'version_wregr(): x_train.shape={x_train.shape}, '
                         f'x_pred.shape={x_pred.shape}. This usually means that score variables '
                         'were added to one matrix but not the other.'
                         )
    x_score_train_all = x_train if with_x else None
    x_score_pred_all = x_pred if with_x else None

    d_all_values = cfg.d_all_values
    # Select all within same main treatment AND positive weights
    main_treat_val = d_all_values[treat_idx, 0]
    # select_data = (d_train[:, 0] == main_treat_val) & (weights > zero_tol)
    select_data = (np.isclose(d_train[:, 0], main_treat_val, atol=zero_tol, rtol=zero_tol)
                   & (weights > zero_tol)
                   )
    # First time this main treatment is seen by this function
    if treat_idx == 0 or not np.isclose(main_treat_val, d_all_values[treat_idx-1, 0],
                                        atol=zero_tol, rtol=zero_tol
                                        ):
        # In this branch: Compute new weights and keep them in container
        # 0) Select data & predefine container list with [None] * subtreatments

        # This data contains all subtreatments together as needed for regression
        # Rescale only when computing this main treatment's cached regression results.
        if standardize_x and with_x:
            x_train, x_pred = scale_by_sd_of_first(x_train, x_pred, ddof=0, zero_tol=zero_tol)

        y_tr = y_train[select_data]
        y_tr_tune = y_tr if y_tr.ndim == 1 else y_tr[:, y_tree_index]
        d_sub_tr = d_train[select_data, 1]
        x_tr = x_train[select_data, :] if with_x else None
        x_score_tr = None  # x_score_train_all[select_data, :] if with_x else None
        w_tr = weights[select_data]

        no_of_sub_treat = len(d_sub_only)   # Number of subtreatments

        container_w = [None] * no_of_sub_treat  # One spot for each subtreatment
        container_r = [None] * no_of_sub_treat  # One spot for each subtreatment
        # 1) Check if treatments are sufficiently full (for non-zero weights).
        #    If not, set weights to average weights,i.e. just take weights as is
        subtreat_idx_too_small = too_small(d_sub_tr, d_sub_only, cfg.tv_min_subtreat)

        if len(subtreat_idx_too_small) == no_of_sub_treat:
            txt += ('\nAll subtreatments are too small for version regression. '
                    f'Main_treatment: {main_treat_val}. Original weights are kept.'
                    )
            for sub_idx in range(no_of_sub_treat):
                container_w[sub_idx] = weights.copy()
                container_r[sub_idx] = (np.zeros(y_train.shape, dtype=weights.dtype)
                                        if return_residuals else None
                                        )
        else:
            # existing ridge/WLS code

            # 2) Estimate either regression model or ridge with x_dat + d_dat as regressors
            # a) Define dummy variables for subtreatments
            d_dummy = dummies_ordered(d_sub_tr, d_sub_only, dtype=int_dtype, unknown='error')
            if subtreat_idx_too_small:
                # Drop columns with too small cells
                d_dummy = np.delete(d_dummy, subtreat_idx_too_small, axis=1)
                small_txt = ' '.join(str(idx) for idx in subtreat_idx_too_small)
                txt += ('\nThe following subtreatments are too small for version '
                       f'regression: Main_treatment: {main_treat_val} '
                       f'Subtreatments: {small_txt}'
                       )
            local_version_scores = allow_local_version_scores and cfg.score_version_local
            global_version_scores = not cfg.score_version_local
            version_scores = version_yes and (local_version_scores or global_version_scores)

            scores_train_precomp = None
            if with_x and cfg.addscores and version_scores:
                global_info_score = None

                if global_version_scores:
                    if cfg.scores_version_round is None:
                        raise ValueError('Global version scores requested, but '
                                         'scores_version_round is None.'
                                         )
                    if maintreat_idx >= len(cfg.scores_version_round):
                        raise ValueError('Global version scores do not contain main treatment '
                                         f'{maintreat_idx}. '
                                         f'Length is {len(cfg.scores_version_round)}.'
                                         )
                    global_info_score = cfg.scores_version_round[maintreat_idx]
                    x_score_pred = x_score_pred_all
                    scores_train_precomp = None

                    if use_train_score_cache and global_info_score.scores_np is not None:
                        scores_cache = global_info_score.scores_np
                        if scores_cache.ndim != 2:
                            raise ValueError('The outcome-sample score cache must be 2D.')
                        if w_index is None:
                            if scores_cache.shape[0] != d_train.shape[0]:
                                raise ValueError('The score cache must match the full outcome '
                                                 'sample when w_index is None.'
                                                 )
                            scores_train_precomp = scores_cache[select_data, :]
                        else:
                            score_rows = np.asarray(w_index)[select_data]
                            scores_train_precomp = scores_cache[score_rows, :]
                    else:
                        x_score_tr = x_score_train_all[select_data, :]
                else:
                    x_score_pred = None

                x_tr, x_pred = add_scores_local(y_train=y_tr_tune,
                                                d_train=d_sub_tr,
                                                x_train=x_tr,
                                                x_pred=x_pred,
                                                cfg=cfg,
                                                boot=cfg.boot,
                                                mp_parallel=mp_parallel,
                                                obs_bigdata=obs_bigdata,
                                                zero_tol=zero_tol,
                                                ps_yes=ps_version_yes,
                                                pg_yes=pg_version_yes,
                                                global_version_scores=global_version_scores,
                                                global_info_score=global_info_score,
                                                x_score_train=x_score_tr,
                                                x_score_pred=x_score_pred,
                                                scores_train_precomp=scores_train_precomp,
                                                scores_pred_precomp=scores_pred,
                                                )
            regr_tr = build_tv_regressors_train(d_dummy, x_tr, interacted=xd_interacted)

            # Average the augmented covariates only after all scores have been predicted.
            x_pred_mean = (None if x_pred is None
                           else np.mean(x_pred, axis=0, dtype=int_dtype, keepdims=True)
                           )
            regr_eval_list_o = []
            col_1 = 0
            for sub_idx, _ in enumerate(d_sub_only):
                if sub_idx not in subtreat_idx_too_small:
                    dummy_eval = np.zeros((1, d_dummy.shape[1]))
                    dummy_eval[:, col_1] = 1.0

                    regr_eval = build_tv_regressors_eval(dummy_eval, x_pred_mean,
                                                         interacted=xd_interacted,
                                                         )
                    regr_eval_list_o.append(regr_eval)
                    col_1 += 1

            if ridge:
                grid_length = 30
                if penalize_version:
                    constant_tr = np.ones((regr_tr.shape[0], 1), dtype=regr_tr.dtype)
                    regr_tr = np.concatenate((constant_tr, regr_tr), axis=1)
                    regr_eval_list = [
                        np.concatenate((np.ones((regr_eval.shape[0], 1), dtype=regr_eval.dtype),
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
                    grid = np.zeros(grid_length + 1)  # To include 0
                    grid[1:] = np.logspace(-5, 2, grid_length)

                # Determine penalty by cross-validation
                # Use the tree-building outcome to select one common penalty.
                best_lam, _ = mcf_ewr.ridge_cv_penalty(regr_tr,
                                                       y_tr_tune,
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
            sum_weights_org = np.sum(weights)
            col_1 = 0
            for sub_idx, _ in enumerate(d_sub_only):
                weights_return = weights.copy()
                residual_return = (np.zeros(y_train.shape, dtype=weights_return.dtype)
                                   if return_residuals else None
                                   )
                if sub_idx in subtreat_idx_too_small:
                    container_w[sub_idx] = weights_return
                    container_r[sub_idx] = residual_return if return_residuals else None
                else:
                    weights_return[select_data] = weights_tv_list[col_1].reshape(-1)
                    sum_weights_return = np.sum(weights_return)
                    if np.abs(sum_weights_return) <= zero_tol:
                        raise RuntimeError('Treatment-version regression produced zero-sum weights.'
                                           )
                    weights_return *= sum_weights_org / sum_weights_return
                    container_w[sub_idx] = weights_return
                    if return_residuals:
                        residual_return[select_data] = residual_tv_list[col_1]
                        container_r[sub_idx] = residual_return
                    else:
                        container_r[sub_idx] = None

                    col_1 += 1

    if container_w is None:
        raise ValueError('container_w must be provided for non-first subtreatment calls.')
    if return_residuals and container_r is None:
        raise ValueError('container_r must be provided when return_residuals=True.')
    weight_new = container_w[subtreat_idx]
    residual_new = container_r[subtreat_idx] if return_residuals else None

    # Update indices of treatments
    if subtreat_idx == len(d_sub_only) - 1:    # Last index
        maintreat_idx += 1 # Next main treatment
        subtreat_idx = 0
    else:
        subtreat_idx += 1   # New value for next arrival in this function

    return weight_new, container_w, residual_new, container_r, maintreat_idx, subtreat_idx, txt


def dummies_ordered(d: NDArray[np.integer],
                    cats: list[int],
                    dtype: DTypeLike = np.int8,
                    unknown: str = 'error',
                    ) -> NDArray[Any]:
    """One-hot encode 1D integer array with columns ordered as in cats."""
    cats = np.asarray(cats)

    if d.ndim != 1 or cats.ndim != 1:
        raise ValueError('d and cats must both be 1D')
    if np.unique(cats).size != cats.size:
        raise ValueError('cats must not contain duplicates')
    if unknown not in ('error', 'ignore'):
        raise ValueError("unknown must be either 'error' or 'ignore'")

    k = cats.size
    n = d.size

    order = np.argsort(cats)
    sorted_cats = cats[order]
    pos = np.searchsorted(sorted_cats, d)

    in_range = pos < k
    valid = np.zeros(n, dtype=bool)
    valid[in_range] = sorted_cats[pos[in_range]] == d[in_range]

    if unknown == 'error' and not np.all(valid):
        bad = np.unique(d[~valid])
        raise ValueError(f'Unknown values in d, not in cats: {bad.tolist()}')

    out = np.zeros((n, k), dtype=dtype)
    rows = np.nonzero(valid)[0]
    cols = order[pos[valid]]
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
                     ) -> tuple[int,
                                ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike
                                ]:
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
                *,
                x_tv_pred_np: NDArray[Any] | None = None,
                score_names: list[str] | tuple[str, ...] | None = None,
                zero_tol: float = 1e-10,
                ) -> tuple[NDArray[np.integer], NDArray[Any] | None, NDArray[Any] | None]:
    """Get data for the regression step when using treatment versions."""
    if not var_cfg.x_name_tv:
        return weights_dic['d_dat'], None, None
    if weights_dic['x_tv_df'] is None:
        raise ValueError('Treatment-version covariates were specified, but x_tv_df is None.')

    if score_names is None:
        score_names = []

    required_train = set(var_cfg.x_name_tv)
    missing_train = required_train.difference(weights_dic['x_tv_df'].columns)

    if missing_train:
        raise ValueError('weights_dic[\'x_tv_df\'] is missing treatment-version variables: '
                         f'{sorted(missing_train)}'
                         )
    if x_tv_pred_np is None:
        if data_df is None:
            raise ValueError('data_df must be provided if x_tv_pred_np is None.')

        missing_pred = required_train.difference(data_df.columns)
        if missing_pred:
            raise ValueError('data_df is missing treatment-version variables: '
                             f'{sorted(missing_pred)}'
                             )
    spec = fit_tv_feature_spec(weights_dic['x_tv_df'],
                               var_cfg,
                               zero_tol=zero_tol,
                               score_names=score_names,
                               )
    x_tv_train = transform_x_tv(weights_dic['x_tv_df'], spec)

    if x_tv_pred_np is None:
        x_tv_pred = transform_x_tv(data_df[var_cfg.x_name_tv], spec)
    else:
        x_tv_pred = x_tv_pred_np

    if x_tv_train is not None and x_tv_pred is not None:
        if x_tv_train.shape[1] != x_tv_pred.shape[1]:
            raise ValueError('x_tv_train and x_tv_pred must have the same number of columns. '
                             f'Got x_tv_train.shape={x_tv_train.shape}, '
                             f'x_tv_pred.shape={x_tv_pred.shape}.'
                             )
    d_dat = weights_dic['d_dat']

    return d_dat, x_tv_train, x_tv_pred


def d_dictionary(data_all_df: pd.DataFrame,
                 d_name: tuple[str, ...] | list[str],
                 zero_tol: float = 1e-10,
                 ) -> dict:
    """Build dictionary of main-treatment and subtreatment values."""
    d_np = data_all_df[d_name].to_numpy()

    if d_np.ndim == 1:
        d_np = d_np.reshape(-1, 1)

    d_main_values = [int(d) for d in np.unique(d_np[:, 0])]

    if d_np.shape[1] == 1:
        versions = [[0] for _ in d_main_values]
        return {'main_treat': d_main_values, 'sub_treat': versions}

    versions = [[0] for _ in d_main_values]
    for d_main_idx, d_main_val in enumerate(d_main_values):
        treat = np.isclose(d_np[:, 0], d_main_val, atol=zero_tol, rtol=zero_tol)
        version_vals = list(np.unique(d_np[treat, 1]))
        if len(version_vals) > 1:
            versions[d_main_idx] = [int(v) for v in version_vals]

    return {'main_treat': d_main_values, 'sub_treat': versions}


# def versions_per_treatment(d_train: NDArray[np.integer], zero_tol: float = 1e-10,
#                            ) -> tuple[list[np.integer], list[list[np.integer] | None] | None]:
#     """Create a list that with the version information for each treatment."""
#     # Currently not used.
#     if d_train.ndim == 1 or d_train.shape[1] == 1:  # Only 1 column
#         d_main_values = list(np.unique(d_train))
#         return d_main_values, None   # No treatment versions

#     d_main_values = list(np.unique(d_train[:, 0]))
#     versions = [None] * len(d_main_values)
#     for d_main_idx, d_main_val in enumerate(d_main_values):
#         # treat = d_train[:, 0] == d_main_val
#         treat = np.isclose(d_train[:, 0], d_main_val, rtol=zero_tol, atol=zero_tol)
#         version_vals = list(np.unique(d_train[treat, 1]))
#         if len(version_vals) > 1:   # Otherwise, there are no versions
#             versions[d_main_idx] = version_vals

#     return d_main_values, versions


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
                  ) -> list[str]:
    """Create name string for potential outcomes and effects."""
    # Currently not used
    if versions is None:
        return [str1 + str(main_val) + str2 for main_val in d_values[start:]]

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
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if a.shape[1] != b.shape[1]:
        raise ValueError('a and b must have the same number of columns (K)')

    sd = a.std(axis=0, ddof=ddof)          # shape (K,)
    sd_safe = np.where(np.isclose(sd, 0, atol=zero_tol, rtol=zero_tol), 1.0, sd)
    # avoid divide-by-zero for constant columns

    return a / sd_safe, b / sd_safe


def transform_x_tv(x_tv_df: pd.DataFrame | None, spec: TvFeatureSpec) -> NDArray[Any] | None:
    """Transform treatment-version features using a training-defined schema."""
    if x_tv_df is None:
        return None

    parts = []

    if spec.names_tv_ordered:
        parts.append(x_tv_df[spec.names_tv_ordered].to_numpy(dtype=np.float32))

    if spec.names_tv_unordered:
        x_unord_df = x_tv_df[spec.names_tv_unordered].astype('category')
        x_unord_dummies = pd.get_dummies(data=x_unord_df,
                                         columns=spec.names_tv_unordered,
                                         prefix=spec.names_tv_unordered,
                                         drop_first=False,
                                         dtype='int8',
                                         )
        x_unord_dummies = x_unord_dummies.reindex(columns=spec.unordered_dummy_columns,
                                                  fill_value=0,
                                                  )
        parts.append(x_unord_dummies.to_numpy(dtype=np.float32))

    if spec.names_tv_scores:
        parts.append(x_tv_df[spec.names_tv_scores].to_numpy(dtype=np.float32))

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    return np.concatenate(parts, axis=1)


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


def tv_scores(*,
              train_df: pd.DataFrame | None,
              predict_df: pd.DataFrame,
              forest_dict: dict | None,
              int_cfg: 'IntCfg',
              gen_cfg: 'GenCfg',
              gen_tv_cfg: 'GenTvCfg',
              var_cfg: 'VarCfg',
              data_train_dict: dict,
              var_x_type: dict,
              boot: int,
              first_use: bool = True,
              predict: bool = False,
              ) -> tuple[TvScores | None, list[TvScores] | None]:
    """Generate propensity and prognostic scores."""
    if gen_cfg.with_output and first_use:
        txt = ('=' * 100
               + '\nComputing prognostic & propensity scores for version estimation: '
               + ('Prediction' if predict else 'Training')
               )
        print_mcf(gen_cfg, txt, summary=True)
    else:
        txt = ''

    obs_bigdata = int_cfg.obs_bigdata
    training = train_df is not None
    cv_k = gen_tv_cfg.score_cv_k
    estimators = gen_tv_cfg.score_estimator

    ps_m_yes = 'prop_main' in gen_tv_cfg.score_type
    pg_m_yes = 'prog_main' in gen_tv_cfg.score_type
    main_yes = ps_m_yes or pg_m_yes
    ps_v_yes = 'prop_version' in gen_tv_cfg.score_type and not gen_tv_cfg.score_version_local
    pg_v_yes = 'prog_version' in gen_tv_cfg.score_type and not gen_tv_cfg.score_version_local
    version_yes = ps_v_yes or pg_v_yes

    ps_m_inst = pg_m_inst = pg_m_scalers = None
    scores_main_dc = scores_vers_by_main = None

    main_score_names = []
    if main_yes:
        prop_score_m_name, prog_score_m_name = get_score_names(gen_cfg.d_values)
        if ps_m_yes:
            main_score_names.extend(prop_score_m_name)
        if pg_m_yes:
            main_score_names.extend(prog_score_m_name)

    main_score_names_set = set(main_score_names)
    x_name_tv_base = [name for name in var_cfg.x_name_tv if name not in main_score_names_set]

    if version_yes and not var_cfg.x_name_tv:
        raise ValueError('Version specific propensity and prognostic scores will be computed '
                         'with the variables specified in x_name_tv. However, no such variables '
                         'have been specified.'
                         )
    if main_yes:
        if training:
            scores_main_dc = TvScores(scores_df=None, instances=None, scalers=None, names=None)
        else:  # Prediction
            if forest_dict is None or forest_dict.get('tv_scores_dc') is None:
                raise ValueError('forest_dict must contain trained tv_scores_dc for prediction.')
            old_scores_main_dc = forest_dict['tv_scores_dc']
            if old_scores_main_dc.instances is None or old_scores_main_dc.scalers is None:
                raise ValueError('tv_scores_dc does not contain trained score instances/scalers.')
            scores_main_dc = TvScores(scores_df=None,  # avoids deepcopy() which can be costly
                                      instances=old_scores_main_dc.instances,
                                      scalers=old_scores_main_dc.scalers,
                                      names=old_scores_main_dc.names,
                                      )
    x_name, x_type = get_key_values_in_list(var_x_type)
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]

    if training:
        x_train_df, _ = get_x_data(train_df, x_name)

        if main_yes:
            x_train_main_df = x_train_df.copy()
            if names_unordered:
                x_train_main_df, _ = dummies_for_unord(x_train_main_df,
                                                       names_unordered,
                                                       data_train_dict=data_train_dict,
                                                       bad_vals_exception=False,
                                                       )
            x_train_main_np = to_numpy_big_data(x_train_main_df, obs_bigdata)
        else:
            x_train_main_np = None

        if version_yes:
            x_train_vers_df = x_train_df[x_name_tv_base].copy()

            tv_base_spec = fit_tv_feature_spec(x_train_vers_df,
                                               var_cfg,
                                               score_names=[],
                                               zero_tol=int_cfg.zero_tol,
                                               x_name_tv=x_name_tv_base,
                                               )
            x_train_vers_np = transform_x_tv(x_train_vers_df, tv_base_spec)
        else:
            x_train_vers_np = None

        y_train_np = to_numpy_big_data(train_df[var_cfg.y_tree_name], obs_bigdata)
        d_train_np = to_numpy_big_data(train_df[var_cfg.d_name], obs_bigdata)
        if d_train_np.ndim == 2:
            d_train_main_np = d_train_np[:, 0]
            d_train_vers_np = d_train_np[:, 1] if version_yes else None
        else:
            raise ValueError('Version information missing in treatment.')
        if main_yes:
            ps_m_inst, pg_m_inst, pg_m_scalers = train_scores(y_np=y_train_np,
                                                              x_np=x_train_main_np,
                                                              d_np=d_train_main_np,
                                                              d_values=gen_cfg.d_values,
                                                              cv_k=cv_k,
                                                              boot=boot,
                                                              mp_parallel=gen_cfg.mp_parallel,
                                                              obs_bigdata=obs_bigdata,
                                                              estimators=estimators,
                                                              zero_tol=int_cfg.zero_tol,
                                                              ps_yes=ps_m_yes,
                                                              pg_yes=pg_m_yes,
                                                              )
        if version_yes:
            (ps_v_inst, pg_v_inst, pg_v_scalers, names_vers, d_values_vers_by_main,
             ) = train_score_vers(y_np=y_train_np,
                                  x_np=x_train_vers_np,
                                  d_main_np=d_train_main_np,
                                  d_vers_np=d_train_vers_np,
                                  d_main_values=gen_cfg.d_values,
                                  cv_k=cv_k,
                                  boot=boot,
                                  mp_parallel=gen_cfg.mp_parallel,
                                  obs_bigdata=obs_bigdata,
                                  estimators=estimators,
                                  zero_tol=int_cfg.zero_tol,
                                  ps_yes=ps_v_yes,
                                  pg_yes=pg_v_yes,
                                  )
            scores_vers_by_main = collect_version_scores_by_main(ps_yes=ps_v_yes,
                                                                 pg_yes=pg_v_yes,
                                                                 ps_obj=ps_v_inst,
                                                                 pg_obj=pg_v_inst,
                                                                 pg_scalers=pg_v_scalers,
                                                                 names=names_vers,
                                                                 d_values_by_main
                                                                     =d_values_vers_by_main,
                                                                 )
            # Retain the exact input schema used to fit the version-score models.
            for score_dc in scores_vers_by_main:
                score_dc.feature_spec = tv_base_spec

            # Cache scores for the outcome sample using the retained training schema.
            dtype_vers = dtype_score(obs_bigdata=obs_bigdata,
                                     obs1=len(train_df),
                                     obs2=len(predict_df),
                                     )
            add_train_version_score_cache(scores_by_main=scores_vers_by_main,
                                          data_df=predict_df,
                                          ps_yes=ps_v_yes,
                                          pg_yes=pg_v_yes,
                                          dtype=dtype_vers,
                                          zero_tol=int_cfg.zero_tol,
                                          )
    elif main_yes: # Prediction for main treatments only
        ps_m_inst = scores_main_dc.instances[0]
        pg_m_inst = scores_main_dc.instances[1:]
        pg_m_scalers = scores_main_dc.scalers[1:]

    dtype = dtype_score(obs_bigdata=obs_bigdata,
                        obs1=0 if train_df is None else len(train_df),
                        obs2=0 if predict_df is None else len(predict_df),
                        )
    # Prediction
    x_predict_df, _ = get_x_data(predict_df, x_name)
    if names_unordered:
        x_predict_df, _ = dummies_for_unord(x_predict_df, names_unordered,
                                            data_train_dict=data_train_dict,
                                            bad_vals_exception=False,
                                            )
    x_predict_np = to_numpy_big_data(x_predict_df, obs_bigdata)

    if main_yes:  # Only for main treatment, version will be predicted in prediction part
        ps_m_np, pg_m_np = predict_scores(x_np=x_predict_np,
                                          ps_inst=ps_m_inst,
                                          pg_inst=pg_m_inst,
                                          pg_scalers=pg_m_scalers,
                                          d_values=gen_cfg.d_values,
                                          dtype=dtype,
                                          zero_tol=int_cfg.zero_tol,
                                          ps_yes=ps_m_yes,
                                          pg_yes=pg_m_yes,
                                          )
        # define variables for score prediction
        if training:
            scores_main_dc.instances = collect_instances_scalers(ps_yes=ps_m_yes,
                                                                 pg_yes=pg_m_yes,
                                                                 ps_obj=ps_m_inst,
                                                                 pg_obj=pg_m_inst,
                                                                 )
            scores_main_dc.scalers = collect_instances_scalers(ps_yes=ps_m_yes,
                                                               pg_yes=pg_m_yes,
                                                               ps_obj=None,
                                                               pg_obj=pg_m_scalers,
                                                               )
            scores_main_dc.names = main_score_names

        # Collect scores to be returned in more compact way
        if ps_m_yes and pg_m_yes:
            scores_main = np.concatenate([ps_m_np, pg_m_np], axis=1)
        else:
            scores_main = ps_m_np if ps_m_yes else pg_m_np

        scores_main_dc.scores_df = pd.DataFrame(data=scores_main, columns=scores_main_dc.names)

    if gen_cfg.with_output and first_use:
        txt += ' - COMPLETED'
        print_mcf(gen_cfg, txt, summary=False)

    return scores_main_dc, scores_vers_by_main


def collect_instances_scalers(*,
                              ps_yes: bool,
                              pg_yes: bool,
                              ps_obj: ScoreInst | ScoreScaler,
                              pg_obj: ScoreInst | ScoreScaler,
                              ) -> list[ScoreInst | ScoreScaler]:
    """Collect the information on the instances and scalers into a list with ps_scores first."""
    if not isinstance(pg_obj, (list, tuple)):
        pg_obj = [pg_obj]
    if not isinstance(ps_obj, (list, tuple)):
        ps_obj = [ps_obj]

    if pg_yes and ps_yes:
        return [*ps_obj, *pg_obj]
    if pg_yes:
        return [None, *pg_obj]

    return [*ps_obj, None]


def fit_tv_feature_spec(x_tv_df: pd.DataFrame,
                        var_cfg: 'VarCfg', *,
                        score_names: list[str] | tuple[str, ...],
                        zero_tol: float = 1e-10,
                        x_name_tv: list[str] | tuple[str, ...] | None = None,
                        ) -> TvFeatureSpec:
    """Fit treatment-version feature schema on training data."""
    del zero_tol  # kept for API compatibility

    x_name_tv_use = list(var_cfg.x_name_tv if x_name_tv is None else x_name_tv)
    score_names = [] if score_names is None else list(score_names)
    score_names_set = set(score_names)

    names_tv_ordered = [name for name in x_name_tv_use
                        if name in var_cfg.x_name_ord and name not in score_names_set
                        ]
    names_tv_unordered = [name for name in x_name_tv_use
                          if name in var_cfg.x_name_unord and name not in score_names_set
                          ]
    names_tv_scores = [name for name in x_name_tv_use if name in score_names_set]

    known_names = set(names_tv_ordered).union(names_tv_unordered).union(names_tv_scores)
    unknown_tv_names = [name for name in x_name_tv_use if name not in known_names]

    if unknown_tv_names:
        raise ValueError('Some treatment-version variables are neither ordered, unordered, '
                         f'nor listed in score_names: {unknown_tv_names}'
                         )

    if names_tv_unordered:
        x_unord_df = x_tv_df[names_tv_unordered].astype('category')
        x_unord_dummies = pd.get_dummies(data=x_unord_df,
                                         columns=names_tv_unordered,
                                         prefix=names_tv_unordered,
                                         drop_first=True,
                                         dtype='int8',
                                         )
        unordered_dummy_columns = list(x_unord_dummies.columns)
    else:
        unordered_dummy_columns = []

    n_base_features = len(names_tv_ordered) + len(unordered_dummy_columns)

    return TvFeatureSpec(names_tv_ordered=names_tv_ordered,
                         names_tv_unordered=names_tv_unordered,
                         unordered_dummy_columns=unordered_dummy_columns,
                         names_tv_scores=names_tv_scores,
                         n_base_features=n_base_features,
                         )


def collect_version_scores_by_main(*,
                                   ps_yes: bool,
                                   pg_yes: bool,
                                   ps_obj: list[ScoreInst],
                                   pg_obj: list[list[ScoreInst]],
                                   pg_scalers: list[list[ScoreScaler]],
                                   names: list[list[str]],
                                   d_values_by_main: list[NDArray[Any] | None],
                                   ) -> list[TvScores]:
    """Convert version-score output into one TvScores object per main treatment."""
    scores_by_main = []

    for main_idx, names_main in enumerate(names):
        instances = collect_instances_scalers(ps_yes=ps_yes,
                                              pg_yes=pg_yes,
                                              ps_obj=ps_obj[main_idx],
                                              pg_obj=pg_obj[main_idx],
                                              )
        scalers = collect_instances_scalers(ps_yes=ps_yes,
                                            pg_yes=pg_yes,
                                            ps_obj=None,
                                            pg_obj=pg_scalers[main_idx],
                                            )
        scores_by_main.append(TvScores(scores_df=None,
                                       scores_np=None,
                                       instances=instances,
                                       scalers=scalers,
                                       names=names_main,
                                       d_values=d_values_by_main[main_idx],
                                       )
                              )

    return scores_by_main


def predict_version_scores(data_df: pd.DataFrame,
                            scores_by_main: list[TvScores], *,
                            ps_yes: bool,
                            pg_yes: bool,
                            dtype: DTypeLike,
                            zero_tol: float,
                            scores_precomputed: list[NDArray[Any] | None] | None = None,
                            ) -> list[NDArray[Any] | None]:
    """Predict global version scores in dataframe row order using stored training schemas.

    Return one array per main treatment, or None when that treatment has no scores.
    Within each array, propensity columns precede prognostic columns as in names.
    No models are fitted and no stored outcome-sample caches are modified.
    scores_precomputed must come from these models and match data_df row order.
    When supplied, validate and reuse it without running the models again.
    """
    if not (ps_yes or pg_yes):
        raise ValueError('At least one version-score type must be requested.')

    if scores_precomputed is not None and len(scores_precomputed) != len(scores_by_main):
        raise ValueError('Precomputed version scores need one entry per main treatment.')

    scores_out = []
    previous_spec, x_np = None, None
    for main_idx, score_dc in enumerate(scores_by_main):
        if score_dc.names is None or len(score_dc.names) == 0:
            if (scores_precomputed is not None
                    and scores_precomputed[main_idx] is not None):
                raise ValueError('Unexpected scores for a main treatment without scores.')
            scores_out.append(None)
            continue
        if score_dc.instances is None or score_dc.scalers is None:
            raise ValueError('Version-score prediction requires fitted instances and scalers.')
        if score_dc.d_values is None:
            raise ValueError('Version-score prediction requires stored version values.')

        prop_names, prog_names = get_score_names(score_dc.d_values)
        expected_names = (prop_names if ps_yes else []) + (prog_names if pg_yes else [])
        if list(score_dc.names) != expected_names:
            raise ValueError('Requested score types do not match the fitted score columns.')

        spec = score_dc.feature_spec
        if spec is None:
            raise ValueError('The version-score training schema is missing. '
                             'Recreate the MCF instance and run train() with the updated code.'
                             )
        if spec.names_tv_scores:
            raise ValueError('Version-score model inputs must not include main-score columns.')
        if spec is not previous_spec:
            required = set(spec.names_tv_ordered + spec.names_tv_unordered)
            missing = required.difference(data_df.columns)
            if missing:
                raise ValueError(f'Missing version-score input columns: {sorted(missing)}')
            if scores_precomputed is None:
                x_np = transform_x_tv(data_df, spec)
                if x_np is None or x_np.shape[1] != spec.n_base_features:
                    raise ValueError('Version-score features do not match the stored '
                                     'training schema.')
            previous_spec = spec

        if scores_precomputed is not None:
            scores = scores_precomputed[main_idx]
            if scores is None:
                raise ValueError('Precomputed version scores are missing for a '
                                 'main treatment.')
            scores = np.asarray(scores, dtype=dtype)
            if scores.shape != (len(data_df), len(expected_names)):
                raise ValueError('Precomputed version scores do not match the '
                                 'prediction rows and fitted score columns.')
            scores_out.append(scores)
            continue

        if len(data_df) == 0:
            scores_out.append(np.empty((0, len(expected_names)), dtype=dtype))
            continue

        prop_np, prog_np = predict_scores(x_np=x_np,
                                          ps_inst=score_dc.instances[0],
                                          pg_inst=score_dc.instances[1:],
                                          pg_scalers=score_dc.scalers[1:],
                                          d_values=score_dc.d_values,
                                          dtype=dtype,
                                          zero_tol=zero_tol,
                                          ps_yes=ps_yes,
                                          pg_yes=pg_yes,
                                          )
        if ps_yes and pg_yes:
            scores = np.concatenate((prop_np, prog_np), axis=1)
        else:
            scores = prop_np if ps_yes else prog_np
        if scores.shape != (len(data_df), len(expected_names)):
            raise ValueError('Version-score predictions have an unexpected shape.')
        scores_out.append(scores)

    return scores_out


def add_train_version_score_cache(*,
                                  scores_by_main: list[TvScores],
                                  data_df: pd.DataFrame,
                                  ps_yes: bool,
                                  pg_yes: bool,
                                  dtype: DTypeLike,
                                  zero_tol: float,
                                  ) -> None:
    """Cache global version scores in the supplied outcome-sample row order."""
    scores = predict_version_scores(data_df, scores_by_main,
                                    ps_yes=ps_yes, pg_yes=pg_yes,
                                    dtype=dtype, zero_tol=zero_tol,
                                    )
    for score_dc, scores_np in zip(scores_by_main, scores):
        score_dc.scores_np = scores_np


def version_scores_for_forest(scores_by_main: list[TvScores] | None,
                              forest_dict: dict[str, Any],
                              ) -> list[TvScores] | None:
    """Align cached version scores with this forest's outcome rows without refitting."""
    if scores_by_main is None:
        return None
    if 'tv_score_rows' not in forest_dict:
        raise ValueError('The forest has no version-score row mapping. '
                         'Run train() again with the updated code.'
                         )

    n_y = len(forest_dict['y_train_df'])
    rows = forest_dict['tv_score_rows']
    if rows is not None:
        rows = np.asarray(rows)
        if (rows.ndim != 1 or len(rows) != n_y
                or not np.issubdtype(rows.dtype, np.integer)):
            raise ValueError('Version-score row mapping must have one integer per outcome row.')
        if np.any(rows < 0):
            raise ValueError('Version-score row mapping contains negative positions.')

    aligned_scores = []
    for score_dc in scores_by_main:
        cached = score_dc.scores_np
        if cached is None:
            if score_dc.names:
                raise ValueError('Global version scores are missing their outcome-sample cache.')
            aligned_scores.append(replace(score_dc, scores_np=None))
            continue
        if cached.ndim != 2:
            raise ValueError('The version-score cache must be a two-dimensional array.')

        if rows is None:
            if len(cached) != n_y:
                raise ValueError('The version-score cache does not match the outcome sample.')
            cache_forest = cached
        else:
            if np.any(rows >= len(cached)):
                raise ValueError('Version-score row mapping exceeds the cached outcome sample.')
            cache_forest = cached[rows, :]

        # Only replace the cache; share the already fitted models and scalers.
        aligned_scores.append(replace(score_dc, scores_np=cache_forest))

    return aligned_scores


def tv_scores_predict(mcf_: 'ModifiedCausalForest',
                      data_df: pd.DataFrame,
                      ) -> tuple[list[TvScores], list[str]]:
    """Predict scores taking into account estimation in two rounds."""
    scores_main_dc_list = []
    for round_ in mcf_.cf_cfg.est_rounds:
        forest_dict = mcf_.forest[0][0] if round_ == 'regular' else mcf_.forest[0][1]
        scores_main_dc, _ = tv_scores(predict_df=data_df,
                                      forest_dict=forest_dict,
                                      int_cfg=mcf_.int_cfg,
                                      gen_cfg=mcf_.gen_cfg,
                                      gen_tv_cfg=mcf_.gen_tv_cfg,
                                      var_cfg=mcf_.var_cfg,
                                      data_train_dict=mcf_.data_train_dict,
                                      var_x_type=mcf_.var_x_type,
                                      first_use=round_ == 'regular',
                                      train_df=None, boot=0,
                                      predict=True,
                                      )
        scores_main_dc_list.append(scores_main_dc)

    if not scores_main_dc_list:
        raise ValueError('No estimation rounds available for tv score prediction.')

    names = scores_main_dc_list[0].names
    if names is None:
        raise ValueError('Predicted TvScores do not contain score names.')

    return scores_main_dc_list, list(names)


def add_scores_local(*,
                     cfg: 'GenTvCfg',
                     y_train: NDArray[Any],
                     d_train: NDArray[np.integer] | None,
                     x_train: NDArray[Any],
                     x_pred: NDArray[Any],
                     boot: int,
                     mp_parallel: int,
                     obs_bigdata: int,
                     zero_tol: float,
                     ps_yes: bool,
                     pg_yes: bool,
                     global_version_scores: bool,
                     global_info_score: TvScores | None,
                     x_score_train: NDArray[Any] | None = None,
                     x_score_pred: NDArray[Any] | None = None,
                     scores_train_precomp: NDArray[Any] | None = None,
                     scores_pred_precomp: NDArray[Any] | None = None,
                     ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute local propensity and prognostic scores within one main treatment."""
    def predict_combined_scores(x_np: NDArray[Any],
                                ps_inst: BaseEstimator,
                                pg_inst: list[ScoreInst],
                                pg_scalers: list[ScoreScaler],
                                *,
                                ps_yes: bool,
                                pg_yes: bool,
                                ) -> NDArray[Any]:
        """Predict and concatenate local propensity and prognostic scores."""
        prop_score_np, prog_score_np = predict_scores(x_np=x_np,
                                                      ps_inst=ps_inst,
                                                      pg_inst=pg_inst,
                                                      pg_scalers=pg_scalers,
                                                      d_values=d_values,
                                                      dtype=dtype,
                                                      zero_tol=zero_tol,
                                                      ps_yes=ps_yes,
                                                      pg_yes=pg_yes,
                                                      )
        if pg_yes and ps_yes:
            return np.concatenate((prop_score_np, prog_score_np), axis=1)
        if pg_yes:
            return prog_score_np

        return prop_score_np

    if d_train is None:
        raise ValueError('d_train must be provided for local score computation.')

    if boot < 1:
        raise ValueError('boot must be positive for local score computation.')

    if x_train.ndim != 2 or x_pred.ndim != 2:
        raise ValueError('x_train and x_pred must both be 2D arrays.')

    if x_train.shape[1] != x_pred.shape[1]:
        raise ValueError('x_train and x_pred must have the same number of columns. '
                         f'Got x_train.shape={x_train.shape}, x_pred.shape={x_pred.shape}.'
                         )
    y_train = np.asarray(y_train).reshape(-1)
    d_train = np.asarray(d_train).reshape(-1)

    if x_train.shape[0] != y_train.shape[0] or x_train.shape[0] != d_train.shape[0]:
        raise ValueError('x_train, y_train, and d_train must have the same number of rows. '
                         f'Got x_train.shape={x_train.shape}, y_train.shape={y_train.shape}, '
                         f'd_train.shape={d_train.shape}.'
                         )
    cv_k = cfg.score_cv_k
    estimators = cfg.score_estimator

    d_values_local, counts = np.unique(d_train, return_counts=True)

    if global_version_scores:
        if global_info_score is None:
            raise ValueError('global_info_score must be provided for global version scores.')
        if global_info_score.d_values is None:
            raise ValueError('Global version scores require stored d_values.')

        d_values = np.asarray(global_info_score.d_values)
        if d_values.size < 2:
            raise ValueError('Global version scores require at least two stored version values.')
    else:
        d_values = d_values_local
        if d_values.size < 2:
            raise ValueError('Only single subtreatment available. Cannot compute prognostic and '
                             'propensity scores.'
                             )
    dtype = dtype_score(obs_bigdata=obs_bigdata, obs1=x_train.shape[0], obs2=x_pred.shape[0],)

    prop_names, prog_names = get_score_names(d_values)
    n_score_cols = 0
    if ps_yes:
        n_score_cols += len(prop_names)
    if pg_yes:
        n_score_cols += len(prog_names)

    if scores_pred_precomp is not None:
        if not global_version_scores:
            raise ValueError('Precomputed prediction scores require global version scores.')
        if scores_train_precomp is None:
            raise ValueError('Precomputed prediction scores require aligned training scores.')

        scores_train = np.asarray(scores_train_precomp)
        scores_pred = np.asarray(scores_pred_precomp)
        if scores_train.shape != (x_train.shape[0], n_score_cols):
            raise ValueError('Precomputed training scores have incorrect row or column counts.')
        if scores_pred.shape != (x_pred.shape[0], n_score_cols):
            raise ValueError('Precomputed prediction scores have incorrect row or column counts.')

        # Preserve this regression's existing precision and append scores only once.
        x_train_sc = np.concatenate((x_train, scores_train), axis=1,  # pylint: disable=E1123
                                    dtype=dtype,
                                    )
        x_pred_sc = np.concatenate((x_pred, scores_pred), axis=1,    # pylint: disable=E1123
                                   dtype=dtype,
                                   )

        return x_train_sc, x_pred_sc

    scores_train = np.empty((x_train.shape[0], n_score_cols), dtype=dtype)
    scores_pred = np.zeros((x_pred.shape[0], n_score_cols), dtype=dtype)

    # Ensure that all treatment values are represented in all k-1 fold combinations.
    if global_version_scores:
        n_splits = 1
    else:
        cv_k_requested = 5 if cv_k is None else int(cv_k)
        min_count = int(counts.min())
        n_splits = 1 if min_count < 2 else min(max(cv_k_requested, 2),
                                               min_count,
                                               x_train.shape[0],
                                               )
    if n_splits > 1 and not global_version_scores:
        while True:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234566)
            splits = list(splitter.split(x_train, d_train))
            missing_in_train = False
            for train_idx, _ in splits:
                missing = np.setdiff1d(d_values, np.unique(d_train[train_idx]))
                if missing.size:
                    missing_in_train = True
                    break

            if not missing_in_train:
                break

            if n_splits >= min_count:
                n_splits = 1
                break

            n_splits += 1

    else:  # No cross-validation
        if global_version_scores:
            if global_info_score is None:
                raise ValueError('global_info_score must be provided for global version scores.')
            if global_info_score.instances is None or global_info_score.scalers is None:
                raise ValueError('global_info_score has no trained instances/scalers.')

            ps_inst = global_info_score.instances[0]
            pg_inst = global_info_score.instances[1:]
            pg_scalers = global_info_score.scalers[1:]

            if x_score_pred is None:
                x_score_pred = x_pred
            if scores_train_precomp is None and x_score_train is None:
                x_score_train = x_train

            expected_features = None
            if ps_yes:
                expected_features = getattr(ps_inst, 'n_features_in_', None)
            elif pg_yes:
                for pg_inst_i in pg_inst:
                    expected_features = getattr(pg_inst_i, 'n_features_in_', None)
                    if expected_features is not None:
                        break

            if expected_features is not None:
                if x_score_pred.shape[1] < expected_features:
                    raise ValueError('Global version-score feature mismatch. '
                                     f'Trained model expects {expected_features} features, '
                                     f'but x_score_pred has {x_score_pred.shape[1]}.'
                                     )
                x_score_pred = x_score_pred[:, :expected_features]

                if scores_train_precomp is None:
                    if x_score_train.shape[1] < expected_features:
                        raise ValueError('Global version-score feature mismatch. '
                                         f'Trained model expects {expected_features} features, '
                                         f'but x_score_train has {x_score_train.shape[1]}.'
                                         )
                    x_score_train = x_score_train[:, :expected_features]
        else:
            ps_inst, pg_inst, pg_scalers = train_scores(y_np=y_train,
                                                        x_np=x_train,
                                                        d_np=d_train,
                                                        d_values=d_values,
                                                        cv_k=cv_k,
                                                        boot=boot,
                                                        mp_parallel=mp_parallel,
                                                        obs_bigdata=obs_bigdata,
                                                        estimators=estimators,
                                                        zero_tol=zero_tol,
                                                        ps_yes=ps_yes,
                                                        pg_yes=pg_yes,
                                                        )
            x_score_train = x_train
            x_score_pred = x_pred

        if scores_train_precomp is None:
            scores_train = predict_combined_scores(x_score_train, ps_inst, pg_inst, pg_scalers,
                                                   ps_yes=ps_yes, pg_yes=pg_yes,
                                                   )
        else:
            scores_train = np.asarray(scores_train_precomp)
            if scores_train.ndim != 2:
                raise ValueError('scores_train_precomp must be a 2D array.')

            if scores_train.shape[0] != x_train.shape[0]:
                raise ValueError('scores_train_precomp has wrong number of rows. '
                                 f'Got {scores_train.shape[0]}, expected {x_train.shape[0]}.'
                                 )
            if scores_train.shape[1] != n_score_cols:
                raise ValueError('scores_train_precomp has wrong number of columns. '
                                 f'Got {scores_train.shape[1]}, expected {n_score_cols}.'
                                 )
            scores_train = scores_train.astype(dtype, copy=False)

        scores_pred = predict_combined_scores(x_score_pred, ps_inst, pg_inst, pg_scalers,
                                              ps_yes=ps_yes, pg_yes=pg_yes,
                                              )
        x_train_sc = np.concatenate((x_train, scores_train), axis=1, # pylint: disable=E1123
                                    dtype=dtype,
                                    )
        x_pred_sc = np.concatenate((x_pred, scores_pred), axis=1,    # pylint: disable=E1123
                                   dtype=dtype,
                                   )
        return x_train_sc, x_pred_sc

    for train_idx, valid_idx in splits:
        ps_inst, pg_inst, pg_scalers = train_scores(y_np=y_train[train_idx],
                                                    x_np=x_train[train_idx],
                                                    d_np=d_train[train_idx],
                                                    d_values=d_values,
                                                    cv_k=cv_k,
                                                    boot=boot,
                                                    mp_parallel=mp_parallel,
                                                    obs_bigdata=obs_bigdata,
                                                    estimators=estimators,
                                                    zero_tol=zero_tol,
                                                    ps_yes=ps_yes,
                                                    pg_yes=pg_yes,
                                                    )
        scores_train[valid_idx] = predict_combined_scores(x_train[valid_idx],
                                                          ps_inst,
                                                          pg_inst,
                                                          pg_scalers,
                                                          ps_yes=ps_yes,
                                                          pg_yes=pg_yes,
                                                          )
        scores_pred += predict_combined_scores(x_pred,
                                               ps_inst,
                                               pg_inst,
                                               pg_scalers,
                                               ps_yes=ps_yes,
                                               pg_yes=pg_yes
                                               )
    scores_pred /= n_splits
    x_train_sc = np.concatenate((x_train, scores_train), axis=1,     # pylint: disable=E1123
                                dtype=dtype,
                                )
    x_pred_sc = np.concatenate((x_pred, scores_pred), axis=1,        # pylint: disable=E1123
                               dtype=dtype,
                               )

    return x_train_sc, x_pred_sc


def train_score_vers(*,
                     y_np: NDArray[Number],
                     x_np: NDArray[Number],
                     d_main_np: NDArray[Number],
                     d_vers_np: NDArray[Number],
                     d_main_values: list[Number] | tuple[Number, ...],
                     cv_k: int | None,
                     boot: int,
                     mp_parallel: int,
                     obs_bigdata: int,
                     estimators: list[str] | tuple[str, ...] | None = None,
                     zero_tol: float,
                     ps_yes: bool,
                     pg_yes: bool,
                     ) -> tuple[list[ScoreInst],
                                list[list[ScoreInst]],
                                list[list[ScoreScaler]],
                                list[list[str]],
                                list[NDArray[Any] | None],
                                ]:
    """Collect instances and scalers for propensity and prognostic scores for versions."""
    len_main = len(d_main_values)
    ps_inst, pg_inst = [None] * len_main, [None] * len_main
    pg_scalers, names = [None] * len_main, [None] * len_main
    d_values_by_main = [None] * len_main

    for idx, d_main_value in enumerate(d_main_values):  # main treatment
        in_main = d_main_np == d_main_value
        d_vers_values = np.unique(d_vers_np[in_main])
        d_values_by_main[idx] = d_vers_values
        if d_vers_values.size < 2:
            ps_inst[idx], pg_inst[idx], pg_scalers[idx], names[idx] = [], [], [], []
            continue
        ps_inst[idx], pg_inst[idx], pg_scalers[idx] = train_scores(y_np=y_np[in_main],
                                                                   x_np=x_np[in_main],
                                                                   d_np=d_vers_np[in_main],
                                                                   d_values=d_vers_values,
                                                                   cv_k=cv_k,
                                                                   boot=boot,
                                                                   mp_parallel=mp_parallel,
                                                                   obs_bigdata=obs_bigdata,
                                                                   estimators=estimators,
                                                                   zero_tol=zero_tol,
                                                                   ps_yes=ps_yes,
                                                                   pg_yes=pg_yes,
                                                                   )
        prop_names, prog_names = get_score_names(d_vers_values)
        names_idx = []
        if ps_yes:
            names_idx.extend(prop_names)
        if pg_yes:
            names_idx.extend(prog_names)
        names[idx] = names_idx

    return ps_inst, pg_inst, pg_scalers, names, d_values_by_main
