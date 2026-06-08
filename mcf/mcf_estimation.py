"""
Contains functions for the estimation of various effect.

Created on Mon Jun 19 17:50:33 2023.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike
from scipy.stats import t
from scipy.special import erfc                                              # pylint: disable=E0611

from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_estimation_generic as mcf_est_g
from mcf.mcf_versions import joint_d_values

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg
    from mcf.mcf_init_predict import PCfg


def effect_from_potential(
        pot_y: NDArray[Any],
        pot_y_var: NDArray[Any],
        d_values_in: list[int, float] | NDArray[np.floating | np.integer], *,
        se_yes: bool = True,
        continuous: bool = False,
        return_comparison: bool = True,
        sequential: bool = False,
        sequential_dic: dict | None = None,
        d_sub_values: list[int, float] | NDArray[np.floating | np.integer] | None = None,
        ) -> tuple[NDArray[Any],
                   NDArray[Any] | list[Any] | None,
                   NDArray[Any] | list[Any] | None,
                   NDArray[Any] | list[Any] | None,
                   NDArray[Any] | list[Any] | None,
                   ]:
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.
    d_values_in : List. Treatment values.
    se_yes : Bool. Compute standard errors. Default is True.
    continuous: Bool. Continuous treatment. Default is False.
    return_comparison: Bool. Return an array indicating the treatments that are
                             compared. Default is True.

    Returns
    -------
    est : Numpy array. Point estimates.
    se : Numpy array. Standard error.
    t_val : Numpy array. t-value.
    p_val : Numpy array.
    ...

    """
    d_values = d_values_in if d_sub_values is None else joint_d_values(
        d_values_in, d_sub_values)

    if sequential:
        if continuous:
            raise NotImplementedError('QIATEs are not yet implemented for continuous treatments.')
        # dim of pot_y: (no_of_comparisons, 2)
        est = pot_y[:, 1] - pot_y[:, 0]
        if se_yes:
            var = pot_y_var[:, 1] + pot_y_var[:, 0]
            stderr, t_val, p_val = compute_inference(est, var)
        else:
            stderr = t_val = p_val = None
        if return_comparison:
            m = est.shape[0]  # size by what we actually computed
            d_vals = np.asarray(d_values)
            comparison = np.empty((m, 2), dtype=d_vals.dtype)
            # no_of_comparisons = round(len(d_values) * (len(d_values) - 1) / 2)
            # comparison = np.empty((no_of_comparisons, 2), dtype=np.int32)
            for _, values in sequential_dic.items():
                comparison[values[0], 0] = d_vals[values[2]]
                comparison[values[0], 1] = d_vals[values[1]]
        else:
            comparison = None
    else:
        if continuous:
            # This legazy code for the continuous case is not yet optimized
            # no_of_comparisons = len(d_values) - 1
            # est = np.empty(no_of_comparisons)
            # if se_yes:
            #     var = np.empty_like(est)
            # comparison = [None for _ in range(no_of_comparisons)]
            # j = 0
            # for idx, treat1 in enumerate(d_values):
            #     for jnd, treat2 in enumerate(d_values):
            #         if jnd <= idx:
            #             continue
            #         est[j] = pot_y[jnd] - pot_y[idx]
            #         if se_yes:
            #             var[j] = pot_y_var[jnd] + pot_y_var[idx]
            #         comparison[j] = [treat2, treat1]
            #         j += 1
            #         break
            # if se_yes:
            #     stderr, t_val, p_val = compute_inference(est, var)
            # else:
            #     stderr = t_val = p_val = None
            # Adjacent differences only (same semantics as legacy loop)
            est = pot_y[1:] - pot_y[:-1]
            if se_yes:
                var = pot_y_var[1:] + pot_y_var[:-1]
                # guard tiny negative variance due to roundoff)
                var = np.maximum(var, 0)
                stderr, t_val, p_val = compute_inference(est, var)
            else:
                stderr = t_val = p_val = None

            if return_comparison:
                d_vals = np.asarray(d_values)
                # pairs are (t_{i+1}, t_i)
                comparison = np.column_stack((d_vals[1:], d_vals[:-1])).astype(d_vals.dtype,
                                                                               copy=False
                                                                               )
            else:
                comparison = None

        else:  # Optimized for discrete case
            no_of_treat = len(d_values)
            idx, jnd = np.triu_indices(no_of_treat, k=1)
            est = pot_y[jnd] - pot_y[idx]
            if se_yes:
                var = pot_y_var[jnd] + pot_y_var[idx]
                stderr, t_val, p_val = compute_inference(est, var)
            else:
                stderr = t_val = p_val = None
            if return_comparison:
                d_vals = np.asarray(d_values)
                comparison = np.empty((idx.size, 2), dtype=d_vals.dtype)
                comparison[:, 0] = d_vals[jnd]
                comparison[:, 1] = d_vals[idx]
            else:
                comparison = None

    return est, stderr, t_val, p_val, comparison


def compute_inference_080(estimate: NDArray[Any],
                          variance: NDArray[Any]
                          ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute inference."""
    # This functions requires that RuntimeWarning are turned to exceptions
    # in the calling code. Otherwise it will not use the except block.
    # It is not done here, because of run-time considerations, as when this
    # function is called in large loops it may not be efficient to change
    # warnings too often.
    constant = 0.000001

    # Check if estimate contains NaNs
    mask_est = np.isnan(estimate)
    if mask_est.any():
        estimate = np.where(mask_est,  0, estimate)

    # Check if variance contains NaNs
    mask_var = np.isnan(variance) | (variance < constant)
    if mask_var.any():
        variance = np.where(mask_var,  constant, variance)

    # In case there are still issues with divisions
    try:
        stderr = np.sqrt(variance)
    except RuntimeWarning:
        if len(estimate) == 1:
            try:
                stderr = 100 * np.abs(estimate + constant)
            except RuntimeWarning:
                stderr = 100
        else:
            stderr = np.zeros_like(estimate)
            for idx, est in enumerate(estimate):
                try:
                    stderr[idx] = np.sqrt(variance[idx])
                except RuntimeWarning:
                    stderr[idx] = 100 * np.abs(estimate[idx] + constant)

    try:
        t_val = np.abs(estimate / stderr)
    except RuntimeWarning:
        if len(estimate) == 1:
            t_val = 0
            try:
                stderr = 100 * np.abs(estimate + constant)
            except RuntimeWarning:
                stderr = 100
        else:
            t_val = np.zeros_like(estimate)
            for idx, est in enumerate(estimate):
                try:
                    t_val[idx] = est / stderr[idx]
                except RuntimeWarning:
                    continue
    p_val = t.sf(t_val, 1_000_000) * 2

    return stderr, t_val, p_val


# Newly introduced in 0.9.0
def compute_inference(estimate: NDArray[Any],
                      variance: NDArray[Any], *,
                      var_floor: float = 1e-6
                      ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute inference."""
    est = np.array(estimate, dtype=float, copy=True)  # local copy
    var = np.array(variance, dtype=float, copy=True)

    np.nan_to_num(est, copy=False, nan=0.0)
    np.nan_to_num(var, copy=False, nan=var_floor)
    np.maximum(var, var_floor, out=var)   # clip negatives to floor

    stderr = np.sqrt(var)
    t_val = np.divide(np.abs(est), stderr, out=np.zeros_like(est), where=stderr > 0)
    p_val = erfc(np.abs(t_val) / np.sqrt(2.0))  # two-sided normal tail; more efficient

    return stderr, t_val, p_val


def aggregate_pots(mcf_: 'ModifiedCausalForest',
                   y_pot_f: NDArray[Any] | list[NDArray[Any]],
                   y_pot_var_f: NDArray[Any] | list[NDArray[Any]], *,
                   txt: str,
                   effect_dic: dict,
                   fold: int,
                   pot_is_list: bool = False,
                   title: str = '',
                   ) -> tuple[dict, float]:
    """Aggregate the effects from the independent training data folds."""
    start_time = time()
    first_fold = fold == 0
    last_fold = fold == mcf_.cf_cfg.folds - 1
    if pot_is_list:
        len_list = len(y_pot_f)
    w_text = f'\n\n{title}: Analysis of weights in fold {fold}\n'
    if first_fold:
        if pot_is_list:
            effect_dic = {'y_pot': [0] * len_list, 'y_pot_var': [0] * len_list,
                          'txt_weights': [''] * len_list
                          }
        else:
            effect_dic = {'y_pot': 0, 'y_pot_var': 0, 'txt_weights': ''}
    if pot_is_list:
        y_pot = deepcopy(effect_dic['y_pot'])
        effect_dic['y_pot'] = [y_pot[idx] + val for idx, val in enumerate(y_pot_f)]
        if y_pot_var_f is not None:
            y_pot_var = deepcopy(effect_dic['y_pot_var'])
            effect_dic['y_pot_var'] = [y_pot_var[idx] + val for idx, val in enumerate(y_pot_var_f)]
    else:
        effect_dic['y_pot'] += y_pot_f
        if y_pot_var_f is not None:
            effect_dic['y_pot_var'] += y_pot_var_f
    if last_fold:   # counting folds starts with 0
        if pot_is_list:
            y_pot = deepcopy(effect_dic['y_pot'])
            effect_dic['y_pot'] = [x / (fold + 1) for x in y_pot]
            if y_pot_var_f is not None:
                y_pot_var = deepcopy(effect_dic['y_pot_var'])
                effect_dic['y_pot_var'] = [x / ((fold + 1) ** 2) for x in y_pot_var]
        else:
            effect_dic['y_pot'] /= (fold + 1)
            if y_pot_var_f is not None:
                effect_dic['y_pot_var'] /= ((fold + 1) ** 2)
    if pot_is_list:
        txt_all = deepcopy(effect_dic['txt_weights'])
        effect_dic['txt_weights'] = [txt_all[idx] + w_text + val for idx, val in enumerate(txt)]
    else:
        effect_dic['txt_weights'] += w_text + txt

    return effect_dic, time() - start_time


def average_vals(dict_list: dict) -> dict:
    """Average effect dictionary if needed and remove list around it."""
    def avg_pots(est0: list | NDArray[Any],
                 est1: list | NDArray[Any],
                 var0: list | NDArray[Any] | None,
                 var1: list | NDArray[Any] | None,
                 ) -> tuple[list | NDArray[Any], list | NDArray[Any]]:
        """Average 2 potentials, and their variances."""
        if isinstance(est0, list):
            est = [None] * len(est0)
            if var0 is not None:
                var = [None] * len(est0)
            for idx, est0_ in enumerate(est0):
                est[idx] = (est0_ + est1[idx]) / 2
                if var0 is not None:
                    var[idx] = (var0[idx] + var1[idx]) / 2
        else:
            est = (est0 + est1) / 2
            var = (var0 + var1) / 2 if var0 is not None else None

        return est, var

    def merge_text(txt_list0: list[str] | str, txt_list1: list[str] | str) -> list[str] | str:
        """Merge text."""
        if isinstance(txt_list0, list):
            txt_list = [None] * len(txt_list0)
            for idx, txt0 in enumerate(txt_list0):
                txt_list[idx] = txt0 + '\n' + txt_list1[idx]
        else:
            txt_list = txt_list0 + '\n' + txt_list1

        return txt_list

    if dict_list[0] is None:
        return None

    if len(dict_list) == 1:
        return dict_list[0]

    # Average the respective elements in both elements of the list

    avg_dict = {}
    if dict_list[1] is None:
        avg_dict['y_pot'], avg_dict['y_pot_var'] = avg_pots(dict_list[0]['y_pot'], 0,
                                                            dict_list[0]['y_pot_var'], 0,
                                                            )
        avg_dict['txt_weights'] = merge_text(dict_list[0]['txt_weights'], '')
    else:
        avg_dict['y_pot'], avg_dict['y_pot_var'] = avg_pots(dict_list[0]['y_pot'],
                                                            dict_list[1]['y_pot'],
                                                            dict_list[0]['y_pot_var'],
                                                            dict_list[1]['y_pot_var'],
                                                            )
        avg_dict['txt_weights'] = merge_text(dict_list[0]['txt_weights'],
                                             dict_list[1]['txt_weights'],
                                             )

    return avg_dict


def add_with_list(pot_all: list[NDArray[Any]],
                  pot_add: list[NDArray[Any]]
                  ) -> list[NDArray[Any]]:
    """Add potential to previous if both contained in lists."""
    return [pot_all[idx] + val for idx, val in enumerate(pot_add)]


def weight_var(w0_dat: NDArray[Any],
               y_dat: NDArray[Any],
               cl_dat: NDArray[Any] | None,
               p_cfg: 'PCfg', *,
               weighted: bool = False,
               normalize: bool = True,
               w_for_diff: NDArray[Any] | None = None,
               weights: NDArray[Any] | None = None,
               bootstrap: int = 0,
               keep_some_0: bool = False,
               se_yes: bool = True,
               keep_all: bool = False,
               zero_tol: float = 1e-10,
               sum_tol: float = 1e-8,
               residual_dat: NDArray[Any] | None = None,
               seed: int = 123345,
               min_obs: int = 5,
               ) -> tuple[NDArray[Any] | float, NDArray[Any] | float | None, NDArray[Any],]:
    """Generate the weight-based variance.

    Parameters
    ----------
    w0_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    cl_dat : Numpy array. Cluster indicator.
    p_cfg : PCfg dtaclass. Parameters.
    normalize : Boolean. Normalisation. Default is True.
    w_for_diff : Numpy array. weights used for difference when clustering.
                 Default is None.
    weights : Numpy array. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.
    bootstrap: Int. If > 1: Use bootstrap instead for SE estimation.
    keep_some_0, se_yes, keep_all: Booleans.
    ...

    Returns
    -------
    est, variance, w_ret
    """
    w_dat = w0_dat.copy()

    if p_cfg.cluster_std and (cl_dat is not None) and (bootstrap < 1):
        if not weighted:
            weights = None
        r_dat = None if residual_dat is None else residual_dat.copy()
        w_dat, y_dat, _, _, residual_dat = aggregate_cluster_nonzero_w(
            cl_dat, w_dat, y_dat=y_dat.copy(), r_dat=r_dat,
            norma=normalize, sweights=weights, zero_tol=zero_tol, sum_tol=sum_tol,
            )
        if w_for_diff is not None:
            w_dat = w_dat - w_for_diff

    if not p_cfg.iate_se:
        keep_some_0, bootstrap = False, 0

    if normalize:
        sum_w_dat = np.abs(w_dat.sum())
        if not ((-sum_tol < sum_w_dat < sum_tol) or (1-sum_tol < sum_w_dat < 1+sum_tol)):
            w_dat /= sum_w_dat
    w_ret = np.copy(w_dat)

    if (residual_dat is not None and bootstrap < 2
            and np.any(np.abs(w_dat * residual_dat) > zero_tol)):
        # Versions estimator
        est = np.dot(w_dat, y_dat)
        if se_yes:
            var = np.dot(w_dat**2, residual_dat**2)  # Variance formula for OLS and Ridge regression

            return  est, var, w_ret

        return  est, None, w_ret

    # Early fast-path (no SEs, no bootstrap, no conditional var, no clustering)
    if ((not se_yes) and (bootstrap < 1) and (not p_cfg.cond_var) and (not p_cfg.cluster_std)):
        obs_eff = w_dat.size if keep_all else int(np.count_nonzero(np.abs(w_dat) > zero_tol))

        if obs_eff < min_obs:
            # Not enough observations for decent point estimate
            return 0.0, 1.0, w_ret

        est = float(w_dat @ y_dat)

        return est, None, w_ret

    if keep_all:
        w_nonzero = np.ones_like(w_dat, dtype=bool)
    else:
        w_nonzero = np.abs(w_dat) > zero_tol  # use non-zero only to speed up

    only_copy = np.all(w_nonzero)
    if keep_some_0 and not only_copy:  # to possibly improve variance estimate
        sum_nonzero = w_nonzero.sum()
        obs_all = len(w_dat)
        sum_0 = obs_all - sum_nonzero
        zeros_to_keep = 0.05 * obs_all  # keep to 5% of all obs as zeros
        zeros_to_switch = round(sum_0 - zeros_to_keep)
        if zeros_to_switch <= 2:
            only_copy = True
        else:
            ind_of_0 = np.where(~w_nonzero)[0]
            rng = np.random.default_rng(seed=seed)
            ind_to_true = rng.choice(ind_of_0, size=zeros_to_switch, replace=False)
            w_nonzero[ind_to_true] = True

    if only_copy:
        w_dat2 = w_dat
    else:
        w_dat2, y_dat = w_dat[w_nonzero], y_dat[w_nonzero]

    obs = len(w_dat2)

    if obs < min_obs:  # Not enough observations for decent point estimate
        return 0.0, 1.0, w_ret

    est = np.dot(w_dat2, y_dat)

    if se_yes:
        if bootstrap > 1:
            if p_cfg.cluster_std and (cl_dat is not None) and not only_copy:
                cl_sub = np.round(cl_dat[w_nonzero]).astype(int, copy=False)
                unique_cl_id, inverse = np.unique(cl_sub, return_inverse=True)
                obs_cl = unique_cl_id.size

                # Micro-speedup: precompute indices per cluster
                clusters_idx = [np.where(inverse == j)[0] for j in range(obs_cl)]
            rng = np.random.default_rng(seed=seed)
            est_b = np.empty(bootstrap)
            for b_idx in range(bootstrap):
                if p_cfg.cluster_std and (cl_dat is not None and not only_copy):

                    idx_cl = rng.integers(0, high=obs_cl, size=obs_cl)
                    # concatenate precomputed groups (much faster than per-loop boolean scans)
                    idx = np.concatenate([clusters_idx[j] for j in idx_cl])
                else:
                    idx = rng.integers(0, high=obs, size=obs)
                w_b = w_dat2[idx].copy()
                if normalize:
                    sum_w_b = np.abs(np.sum(w_b))
                    if not ((-sum_tol < sum_w_b < sum_tol) or (1-sum_tol < sum_w_b < 1+sum_tol)):
                        w_b /= sum_w_b
                est_b[b_idx] = np.dot(w_b, y_dat[idx])
            variance = np.var(est_b)
        else:
            if p_cfg.cond_var:
                sort_ind = np.argsort(w_dat2)
                y_s, w_s = y_dat[sort_ind], w_dat2[sort_ind]
                if p_cfg.knn:
                    k = int(round(p_cfg.knn_const * np.sqrt(obs) * 2))
                    k = max(p_cfg.knn_min_k, min(k, obs // 2))
                    exp_y_cond_w, var_y_cond_w = mcf_est_g.moving_avg_mean_var(y_s, k)
                else:
                    band = (mcf_est_g.bandwidth_nw_rule_of_thumb(w_s, zero_tol=zero_tol)
                            * p_cfg.nw_bandw
                            )
                    exp_y_cond_w = mcf_est_g.nadaraya_watson(y_s, w_s, w_s, p_cfg.nw_kern, band)
                    var_y_cond_w = mcf_est_g.nadaraya_watson((y_s - exp_y_cond_w)**2, w_s, w_s,
                                                             p_cfg.nw_kern, band
                                                             )
                variance = np.dot(w_s**2, var_y_cond_w) + obs * np.var(w_s * exp_y_cond_w)
            else:
                variance = obs * np.var(w_dat2 * y_dat)
            variance *= obs / (obs-1)  # Finite sample adjustm.
    else:
        variance = None

    return est, variance, w_ret


def aggregate_cluster_nonzero_w(cl_dat: NDArray[Any],
                                w_dat: NDArray[Any], *,
                                y_dat: NDArray[Any] | None = None,
                                r_dat: NDArray[Any] | None = None,
                                norma: bool = True,
                                w2_dat: NDArray[Any] | None = None,
                                sweights: NDArray[Any] | None = None,
                                y2_compute: bool = False,
                                zero_tol: float = 1e-10,
                                sum_tol: float = 1e-8,
                                ) -> tuple[NDArray[Any], NDArray[Any] | None,
                                           NDArray[Any], NDArray[Any] | None, NDArray[Any] | None,
                                           ]:
    """Aggregate weighted cluster means.

    Parameters
    ----------
    cl_dat : Numpy array. Cluster indicator.
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    ...

    Returns
    -------
    w_agg : Numpy array. Aggregated weights. Normalised to one.
    y_agg : Numpy array. Aggregated outcomes.
    w_agg2 : Numpy array. Aggregated weights. Normalised to one.
    y_agg2 : Numpy array. Aggregated outcomes.
    ....
    """
    cluster_no = np.unique(cl_dat)
    no_cluster = len(cluster_no)
    w_nonzero = np.abs(w_dat) > zero_tol

    if y_dat is None:
        y_agg = None
    else:
        if y_dat.ndim == 1:
            y_dat = np.reshape(y_dat, (-1, 1))
        q_obs = np.size(y_dat, axis=1)
        y_agg = np.zeros((no_cluster, q_obs))

    r_agg = None if r_dat is None else np.zeros((no_cluster, 1))

    y2_agg = np.copy(y_agg) if y2_compute else None
    w_agg = np.zeros(no_cluster)
    if w2_dat is not None:
        w2_agg = np.zeros(no_cluster, dtype=w_dat.dtype)
        w2_nonzero = np.abs(w2_dat) > zero_tol
    else:
        w2_agg = None
        w2_nonzero = False
    for j, cl_ind in enumerate(cluster_no):
        in_cluster = (cl_dat == cl_ind).reshape(-1)
        in_cluster_nonzero = in_cluster & w_nonzero
        in_cluster_non_zero2 = in_cluster & w2_nonzero if y2_compute else None

        if np.any(in_cluster_nonzero):
            w_agg[j] = np.sum(w_dat[in_cluster_nonzero])
            if w2_dat is not None:
                w2_agg[j] = np.sum(w2_dat[in_cluster])
            if r_agg is not None:
                if sweights is None:
                    r_agg[j, 0] = (np.dot(w_dat[in_cluster_nonzero], r_dat[in_cluster_nonzero, 0])
                                   / w_agg[j]
                                   )
                else:
                    r_agg[j, 0] = (np.dot(
                        w_dat[in_cluster_nonzero] * sweights[in_cluster_nonzero].reshape(-1),
                        r_dat[in_cluster_nonzero, 0])
                        / w_agg[j]
                        )
            if y_dat is not None and np.any(in_cluster_nonzero):
                for odx in range(q_obs):
                    if sweights is None:
                        y_agg[j, odx] = (np.dot(w_dat[in_cluster_nonzero],
                                                y_dat[in_cluster_nonzero, odx])
                                         / w_agg[j]
                                         )
                        if y2_agg is not None:
                            y2_agg[j, odx] = (np.dot(w2_dat[in_cluster_non_zero2],
                                                     y_dat[in_cluster_non_zero2, odx])
                                              / w2_agg[j]
                                              )
                    else:
                        y_agg[j, odx] = (np.dot(
                            w_dat[in_cluster_nonzero] * sweights[in_cluster_nonzero].reshape(-1),
                            y_dat[in_cluster_nonzero, odx])
                            / w_agg[j]
                            )
                        if y2_agg is not None:
                            y2_agg[j, odx] = (np.dot(w2_dat[in_cluster_non_zero2]
                                                     * sweights[in_cluster_non_zero2].reshape(-1),
                                                     y_dat[in_cluster_non_zero2, odx]
                                                     ) / w2_agg[j]
                                              )
    if norma:
        sum_w_agg = np.sum(w_agg)
        if not sum_tol < sum_w_agg < sum_tol:
            w_agg = w_agg / sum_w_agg
        if w2_dat is not None:
            sum_w2_agg = np.sum(w2_agg)
            if not 1 - sum_tol < sum_w2_agg < 1 + sum_tol:
                w2_agg = w2_agg / sum_w2_agg

    return w_agg, y_agg, w2_agg, y2_agg, r_agg


def analyse_weights(weights: NDArray[Any],
                    title: str,
                    gen_cfg: 'GenCfg',
                    p_cfg: 'PCfg', *,
                    gen_tv_total_treat: int | None = None,
                    ate: bool = True,
                    continuous: bool = False,
                    no_of_treat_cont: int = None,
                    d_values_cont: NDArray[Any] | None = None,
                    iv: bool = False,
                    zero_tol: float = 1e-10,
                    sum_tol: float = 1e-8,
                    line_length: int = 100,
                    min_obs: int = 5,
                    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any],
                               NDArray[Any], NDArray[Any], NDArray[Any],
                               NDArray[Any], NDArray[Any], NDArray[Any],
                               str,
                               ]:
    """Describe the weights.

    Parameters
    ----------
    weights : Numyp array. Weights.
    title : String. Title for output.
    gen_cfg : GenCfg dataclass. Parameters.
    p_cfg : PCfg dataclass. Parameters.
    ate: Boolean. True if Ate is estimated. Default is True.
    continuous : Boolean. Continuous treatment. Default is False.
    no_of_treat_cont : Int. Number of discretized treatments of continuous
                            treatments used for weights. Default is None.
    d_values_cont : Numpy array. Values of discretized treatments of continuous
                                 treatments used for weights. Default is None.

    Returns
    -------
    nonzero : Numpy array.
    equalzero : Numpy array.
    mean_nonzero : Numpy array.
    std_nonzero : Numpy array.
    gini_all : Numpy array.
    gini_nonzero : Numpy array.
    share_largest_q : Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.
    txt : String. Text to print.

    """
    txt = ''
    if ate:
        txt += '\n' * 2 + '=' * line_length
        if iv:
            txt += '\nAnalysis of weights: ' + title
        else:
            txt += '\nAnalysis of weights (normalised to add to 1): ' + title
    if gen_tv_total_treat is None:
        no_of_treat = no_of_treat_cont if continuous else gen_cfg.no_of_treat
    else:
        no_of_treat = gen_tv_total_treat

    nonzero = np.empty(no_of_treat, dtype=np.uint32)
    equalzero, mean_nonzero = np.empty_like(nonzero), np.empty(no_of_treat)
    std_nonzero, gini_all = np.empty_like(mean_nonzero), np.empty_like(mean_nonzero)
    gini_nonzero = np.empty_like(mean_nonzero)
    share_largest_q = np.empty((no_of_treat, 3))
    sum_larger = np.empty((no_of_treat, len(p_cfg.q_w)))
    obs_larger = np.empty_like(sum_larger)
    sum_weights = np.sum(weights, axis=1)
    for j in range(no_of_treat):
        if not (((1 - sum_tol) < sum_weights[j] < (1 + sum_tol))
                or (-zero_tol < sum_weights[j] < zero_tol)
                ) and not iv:
            w_j = weights[j] / sum_weights[j]
        else:
            w_j = weights[j]
        w_nonzero = w_j[np.abs(w_j) > zero_tol]
        n_nonzero = len(w_nonzero)
        nonzero[j] = n_nonzero
        n_all = len(w_j)
        equalzero[j] = n_all - n_nonzero
        mean_nonzero[j], std_nonzero[j] = np.mean(w_nonzero), np.std(w_nonzero)
        gini_all[j] = mcf_est_g.gini_coeff_all(w_j)
        gini_nonzero[j] = mcf_est_g.gini_coeff_all(w_nonzero)
        w_pos = np.abs(w_nonzero)  # Consider absolute size only
        w_pos_sum = w_pos.sum()
        if n_nonzero > min_obs:
            qqq = np.quantile(w_pos, (0.99, 0.95, 0.9))
            for i in range(3):
                share_largest_q[j, i] = (np.sum(w_pos[w_pos >= (qqq[i] - zero_tol)])
                                         / w_pos_sum * 100
                                         )
            for idx, val in enumerate(p_cfg.q_w):
                sum_larger[j, idx] = np.sum(w_pos[w_pos >= (val - zero_tol)]) / w_pos_sum * 100
                obs_larger[j, idx] = len(w_pos[w_pos >= (val - zero_tol)]) / n_nonzero * 100
        else:
            share_largest_q = np.empty((no_of_treat, 3))
            sum_larger = np.zeros((no_of_treat, len(p_cfg.q_w)))
            obs_larger = np.zeros_like(sum_larger)
            if gen_cfg.with_output:
                txt += '\nLess than 5 observations in some groups.'
    if ate:
        txt += mcf_ps.txt_weight_stat(nonzero=nonzero, equalzero=equalzero,
                                      mean_nonzero=mean_nonzero, std_nonzero=std_nonzero,
                                      gini_all=gini_all, gini_nonzero=gini_nonzero,
                                      share_largest_q=share_largest_q, sum_larger=sum_larger,
                                      obs_larger=obs_larger, gen_cfg=gen_cfg, p_cfg=p_cfg,
                                      continuous=continuous, d_values_cont=d_values_cont,
            )
    return (nonzero, equalzero, mean_nonzero, std_nonzero, gini_all, gini_nonzero, share_largest_q,
            sum_larger, obs_larger, txt,
            )


def predict_proba_aligned(classif, x_np: NDArray[Any],
                          d_values: list[int | float] | NDArray[Any], *,
                          dtype: DTypeLike = np.float32,
                          zero_tol: float = 1e-10,
                          ) -> NDArray[np.floating]:
    """Predict class probabilities aligned to the global treatment-value order."""
    raw = classif.predict_proba(x_np)
    out = np.zeros((raw.shape[0], len(d_values)), dtype=dtype)

    classes = np.asarray(classif.classes_)
    for j, d_val in enumerate(d_values):
        pos = np.flatnonzero(np.isclose(classes, d_val, atol=zero_tol, rtol=zero_tol))
        if pos.size == 1:
            out[:, j] = raw[:, int(pos[0])]
        elif pos.size > 1:
            raise RuntimeError(
                f'Multiple classifier classes match treatment value {d_val}. '
                f'classes_={classes.tolist()}'
            )
        # If pos.size == 0, the class was absent in this fitted sample;
        # keep probability zero for this treatment.

    return out
