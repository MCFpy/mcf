"""
Created on Fri Oct 11 14:13:30 2024.

@author: MLechner

Contains additional functions needed for computing the QIATE.

@author: MLechner
-*- coding: utf-8 -*-
"""

from itertools import compress
from os import remove
from pathlib import Path
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import ray
import pandas as pd
from scipy.stats import norm

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_estimation_functions as mcf_est
from mcf.mcf_estimation_generic_functions import bandwidth_nw_rule_of_thumb
from mcf.mcf_estimation_generic_functions import kernel_proc
from mcf.mcf_general import bound_norm_weights, bound_norm_weights_not_one
from mcf.mcf_general import share_completed
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def qiate_est(mcf_: 'ModifiedCausalForest',
              data_df: pd.DataFrame,
              weights_dic: dict,
              y_pot: NDArray[Any],
              y_pot_var: NDArray | None = None,
              with_output: bool = True,
              iv: bool = False,
              ) -> tuple[NDArray[Any], NDArray[Any] | None,
                         NDArray[Any], NDArray[Any] | None,
                         NDArray[Any], NDArray[Any] | None,
                         dict,
                         str
                         ]:
    """Estimate QIATEs and their standard errors."""
    gen_cfg, int_cfg = mcf_.gen_cfg, mcf_.int_cfg
    p_cfg, var_cfg, p_ba_cfg = mcf_.p_cfg, mcf_.var_cfg, mcf_.p_ba_cfg

    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print('\nComputing QIATE')

    if gen_cfg.d_type == 'continuous':
        raise NotImplementedError(
            'QIATE only implemented for discrete treatments. Maybe (!) support '
            'for continuous treatments will be added in the future.')

    y_dat, weights_all, w_dat, cl_dat = assign_data(weights_dic, gen_cfg, p_cfg)
    d_p, _, w_p, _ = mcf_ate.get_data_for_final_ate_estimation(
        data_df, gen_cfg, p_cfg, var_cfg, ate=False, need_count=False)

    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)

    no_of_treat, t_probs = gen_cfg.no_of_treat, p_cfg.choice_based_probs
    treat_comp_label = get_comparison_labels(no_of_treat, gen_cfg.d_values)
    i_d_val = np.arange(no_of_treat)

    # Define dictionary that relates the respective treatment comparison and
    # outcome variable to the order for this particular IATE (
    # & get that quantile data)
    iate_q_np, iate_q_key_dic = get_iate_order(y_pot, y_pot_var, no_of_treat,
                                               no_of_out, p_cfg)
    if p_cfg.qiate_smooth:
        bandwidth_nw, kernel_nw = get_nadarya_watson_parameters(
            iate_q_np[:, 0, 0],
            p_cfg.qiate_smooth_bandwidth,
            zero_tol=int_cfg.zero_tol,
            )
    else:
        bandwidth_nw = kernel_nw = None

    # Initialize q-variables, comparison specific potential outcomes, weights
    q_values = p_cfg.qiate_quantiles
    no_of_q_values = len(q_values)
    (y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var, y_pot_mopp, y_pot_mopp_var
     ) = initialise_pot_qiate(no_of_q_values, len(iate_q_key_dic), no_of_out)
    # 3rd dimension of y_pot contains both POs of QIATE for respective
    # comparisons
    w_qiate, w_qiate_unc, w_censored, w_median = initialise_weights_qiate(
        no_of_q_values, len(iate_q_key_dic), n_y, no_of_out)

    # Optimisations concerning the use of the weights matrix and ray.
    (files_to_delete, _, weights_all, maxworkers, txt
     ) = optimize_initialize_mp_qiate(weights_all, gen_cfg, int_cfg,
                                      with_output)

    # Define some tuples needed for QIATE-potential outcome computation
    data = (y_dat, cl_dat, w_dat, d_p, w_p,)
    iate_q = (iate_q_np, iate_q_key_dic,)
    dics = (gen_cfg, int_cfg, p_cfg, p_ba_cfg,)
    parameters = (i_d_val, t_probs, no_of_out, bandwidth_nw, kernel_nw,
                  p_cfg.qiate_m_mqiate, p_cfg.qiate_m_opp
                  )
    # Types and dimension of important variables
    # - QIATE Weights (for potential outcomes):
    # -- w_qiate, w_qiate_unc: 5D arrays (no_of_qval,no_of_comp,2,n_y,no_of_out)
    # -- w_censored: 4D array            (no_of_qval, no_of_comp, 2, no_of_out))
    # -- w_median: 4D array              (no_of_comparisons, 2, n_y, no_of_out)
    # Potential outcomes
    # - y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var
    # -- 4D array:                 (no_of_qval, no_of_comparisons, 2, no_of_out)

    # 1st step-(i): Compute weights of median (if needed; no multiprocessing)
    if p_cfg.qiate_m_mqiate or p_cfg.qiate_m_opp:
        parameters_diff = (*parameters[:5], False, False)
        w_qiate_0, w_qiate_unc_0 = w_qiate[0].copy(), w_qiate_unc[0].copy()
        w_censored_0 = w_censored[0].copy()
        qiate_weights_q = (w_qiate_0, w_qiate_unc_0, w_censored_0, None, None)
    else:
        qiate_weights_q = parameters_diff = None
    # 1st step-(ii): Compute weights of quantile difference
    if p_cfg.qiate_m_mqiate:
        _, _, _, _, _, _, w_median, _, _, _ = qiate_q(
            0.5, 0, data, weights_all, iate_q, qiate_weights_q, None, dics,
            parameters_diff, iv=iv, weights_only=True)
    else:
        y_pot_mmed = y_pot_mmed_var = w_median = None

    # Compute weights to get quantile differences later on
    w_mopp_list = [None] * no_of_q_values

    if maxworkers > 1:
        w_qiate_dim = w_qiate[0].shape
        w_qiate_unc_dim = w_qiate_unc[0].shape
        w_censored_dim = w_censored[0].shape

    if p_cfg.qiate_m_opp:
        if maxworkers == 1:
            for q_idx, q_value in enumerate(q_values):
                qiate_weights_q = (w_qiate[q_idx], w_qiate_unc[q_idx],
                                   w_censored[q_idx], None, None)
                _, _, _, _, _, _, w_mopp_list[q_idx], _, _, _ = qiate_q(
                    q_value, q_idx, data, weights_all, iate_q, qiate_weights_q,
                    None, dics, parameters_diff, iv=iv, weights_only=True
                    )
        else:
            data_ref = ray.put(data)
            still_running = [
                ray_qiate_mp.remote(
                    q_value, q_idx, data_ref, weights_all, iate_q,
                    (w_qiate_dim, w_qiate_unc_dim, w_censored_dim, None,
                     None),
                    None, dics, parameters_diff, iv=iv,
                    weights_only=True)
                for q_idx, q_value in enumerate(q_values)
                ]
            idx = 1
            w_qiate0 = w_qiate.copy()
            w_qiate_unc0 = w_qiate_unc.copy()
            w_censored0 = w_censored.copy()
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for results_fut_idx in finished_res:
                    q_idx = results_fut_idx[8]
                    w_qiate0, w_qiate_unc0, w_censored0 = assign_w(
                        w_qiate0, w_qiate_unc0, w_censored0,
                        results_fut_idx, q_idx)
                    w_mopp_list[q_idx] = w_qiate0[q_idx].copy()
                    if gen_cfg.with_output and gen_cfg.verbose:
                        share_completed(idx, no_of_q_values)
                        idx += 1
            del w_qiate0, w_qiate_unc0, w_censored0

    else:
        y_pot_mopp = y_pot_mopp_var = None

    # 2nd step: All quantiles and all effects (i)
    if maxworkers == 1:
        for q_idx, q_value in enumerate(q_values):
            q_idx_opp = no_of_q_values - 1 - q_idx
            qiate_weights_q = (w_qiate[q_idx], w_qiate_unc[q_idx],
                               w_censored[q_idx],
                               w_median, w_mopp_list[q_idx_opp]
                               )
            if p_cfg.qiate_m_mqiate:
                y_pot_mmed_q_idx = y_pot_mmed[q_idx]
                y_pot_mmed_var_q_idx = y_pot_mmed_var[q_idx]
            else:
                y_pot_mmed_q_idx = y_pot_mmed_var_q_idx = None

            if p_cfg.qiate_m_opp:
                y_pot_mopp_q_idx = y_pot_mopp[q_idx]
                y_pot_mopp_var_q_idx = y_pot_mopp_var[q_idx]
            else:
                y_pot_mopp_q_idx = y_pot_mopp_var_q_idx = None

            pot_q = (y_pot[q_idx], y_pot_var[q_idx],
                     y_pot_mmed_q_idx, y_pot_mmed_var_q_idx,
                     y_pot_mopp_q_idx, y_pot_mopp_var_q_idx
                     )
            results_fut_q = qiate_q(
                q_value, q_idx, data, weights_all, iate_q, qiate_weights_q,
                pot_q, dics, parameters, iv
                )
            (y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var,
             y_pot_mopp, y_pot_mopp_var) = assign_pot(
                 y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var,
                 y_pot_mopp, y_pot_mopp_var,
                 results_fut_q,
                 q_idx, p_cfg.qiate_m_mqiate, p_cfg.qiate_m_opp
                 )
            w_qiate, w_qiate_unc, w_censored = assign_w(
                  w_qiate, w_qiate_unc, w_censored, results_fut_q, q_idx)
    else:
        y_pot_dim, y_pot_var_dim = y_pot[0].shape, y_pot_var[0].shape
        if y_pot_mmed is None:
            y_pot_mmed_dim = y_pot_mmed_var_dim = None
        else:
            y_pot_mmed_dim = y_pot_mmed[0].shape
            y_pot_mmed_var_dim = y_pot_mmed_var[0].shape
        if y_pot_mopp is None:
            y_pot_mopp_dim = y_pot_mopp_var_dim = None
        else:
            y_pot_mopp_dim = y_pot_mopp[0].shape
            y_pot_mopp_var_dim = y_pot_mopp_var[0].shape
        data_ref = ray.put(data)
        still_running = [
            ray_qiate_mp.remote(
                q_value, q_idx, data_ref, weights_all, iate_q,
                (w_qiate_dim, w_qiate_unc_dim, w_censored_dim, w_median,
                    w_mopp_list[no_of_q_values - 1 - q_idx]),
                (y_pot_dim, y_pot_var_dim,
                 y_pot_mmed_dim, y_pot_mmed_var_dim,
                 y_pot_mopp_dim, y_pot_mopp_var_dim),
                dics, parameters, iv=iv, weights_only=False)
            for q_idx, q_value in enumerate(q_values)
            ]

        idx = 1
        while len(still_running) > 0:
            finished, still_running = ray.wait(still_running, num_returns=1)
            finished_res = ray.get(finished)
            for results_fut_idx in finished_res:
                q_idx = results_fut_idx[8]
                (y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var,
                 y_pot_mopp, y_pot_mopp_var) = assign_pot(
                     y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var,
                     y_pot_mopp, y_pot_mopp_var,
                     results_fut_idx,
                     q_idx, p_cfg.qiate_m_mqiate, p_cfg.qiate_m_opp
                     )
                w_qiate, w_qiate_unc, w_censored = assign_w(
                    w_qiate, w_qiate_unc, w_censored, results_fut_idx,
                    q_idx)
                if gen_cfg.with_output and gen_cfg.verbose:
                    share_completed(idx, no_of_q_values)
                    idx += 1

        ray_end(int_cfg, weights_all, finished_res, finished)

    if files_to_delete:  # delete temporary files
        for file in files_to_delete:
            remove(file)

    qiate_est_dic = {
        'd_values': gen_cfg.d_values,
        'no_of_out': no_of_out,
        'var_cfg': var_cfg,
        'q_values': q_values,
        'ref_pop_lab': 'All',
        'iate_q_key_dic': iate_q_key_dic,
        'p_cfg': p_cfg,
        'treat_comp_label': treat_comp_label
        }

    return (y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var,
            y_pot_mopp, y_pot_mopp_var, qiate_est_dic, txt)


@ray.remote
def ray_qiate_mp(q_value: list[float] | NDArray[np.floating | float],
                 q_value_idx: Any,
                 data: tuple[NDArray[Any], ...],
                 weights_all: Any,
                 iate_q: tuple[NDArray[Any], dict],
                 qiate_weights_q: tuple[Any, ...],
                 pot_q: tuple[Any, ...],
                 dics: tuple[dict, ...],
                 parameters: tuple[Any],
                 iv: bool = False,
                 weights_only: bool = False,
                 ) -> tuple[Any]:
    """Make function compatible with Ray."""
    return qiate_q(q_value, q_value_idx, data, weights_all, iate_q,
                   qiate_weights_q, pot_q, dics, parameters, multi_ray=True,
                   iv=iv, weights_only=weights_only)


def qiate_q(q_value: list[float] | NDArray[np.floating | float],
            q_value_idx: Any,
            data: tuple[NDArray[Any], ...],
            weights_all: Any,
            iate_q: tuple[NDArray[Any], dict],
            qiate_weights_q: tuple[Any, ...],
            pot_q: tuple[Any, ...],
            dics: tuple[dict, ...],
            parameters: tuple[Any],
            multi_ray: bool = False,
            iv: bool = False,
            weights_only: bool = False,
            ):
    """Compute QIATEs and their variances."""
    # Unpack variables
    y_dat, cl_dat, w_dat, d_p, w_p = data
    iate_q_np, iate_q_key_dic = iate_q
    gen_cfg, int_cfg, p_cfg, p_ba_cfg = dics
    sum_tol = int_cfg.sum_tol
    (i_d_val, t_probs, no_of_out, bandw_q, kernel, qiate_m_mqiate, qiate_m_opp
     ) = parameters
    y_pot_q = y_pot_var_q = y_pot_mmed_q = y_pot_mmed_var_q = None
    y_pot_mopp_q = y_pot_mopp_var_q = None
    if multi_ray:  # Needed because object is otherwise 'read_only'
        (w_qiate_q_dim, w_qiate_unc_q_dim, w_censored_q_dim, w_median, w_opp
         ) = qiate_weights_q
        w_qiate_q = np.zeros(w_qiate_q_dim)
        w_qiate_unc_q = np.zeros(w_qiate_unc_q_dim)
        w_censored_q = np.zeros(w_censored_q_dim)
        if not weights_only:
            (y_pot_q_dim, y_pot_var_q_dim, y_pot_mmed_q_dim,
             y_pot_mmed_var_q_dim, y_pot_mopp_q_dim, y_pot_mopp_var_q_dim
             ) = pot_q
            y_pot_q = np.zeros(y_pot_q_dim)
            y_pot_var_q = np.zeros(y_pot_var_q_dim)

            if y_pot_mmed_q_dim is None:
                y_pot_mmed_q = y_pot_mmed_var_q = None
            else:
                y_pot_mmed_q = np.zeros(y_pot_mmed_q_dim)
                y_pot_mmed_var_q = np.zeros(y_pot_mmed_var_q_dim)

            if y_pot_mopp_q_dim is None:
                y_pot_mopp_q = y_pot_mopp_var_q = None
            else:
                y_pot_mopp_q = np.zeros(y_pot_mopp_q_dim)
                y_pot_mopp_var_q = np.zeros(y_pot_mopp_var_q_dim)

    else:
        (w_qiate_q, w_qiate_unc_q, w_censored_q, w_median, w_opp
         ) = qiate_weights_q

        if not weights_only:
            (y_pot_q, y_pot_var_q, y_pot_mmed_q, y_pot_mmed_var_q,
             y_pot_mopp_q, y_pot_mopp_var_q) = pot_q

    # Define frequently used variables
    d_values = gen_cfg.d_values
    no_of_comp, n_y = len(iate_q_key_dic), len(y_dat)

    weights_q, relevant_data_q, w_q = get_w_rel_q(
        iate_q_np, iate_q_key_dic, q_value, weights_all,
        bandwidth=bandw_q, kernel_str=kernel, no_of_outcomes=no_of_out,
        w_is_csr=int_cfg.weight_as_sparse, zero_tol=int_cfg.zero_tol,
        )
    if gen_cfg.weighted:
        w_p_q = [[w_p[relevant_data_q[comparison][out_idx]]
                  for out_idx in range(no_of_out)]
                 for comparison in range(no_of_comp)]

    # Types and dimension of important variables
    # - QIATE Weights (for potential outcomes):
    # -- w_qiate_q, w_qiate_unc_q: 4D arr (no_of_comparisons, 2, n_y, no_of_out)
    # -- w_censored_q: 3D array           (no_of_comparisons, 2, no_of_out)
    # -- w_median: 4D array               (no_of_comparisons, 2, n_y, no_of_out)
    # -- w_opp: 4D array                  (no_of_comparisons, 2, n_y, no_of_out)
    # Potential outcomes
    # - y_pot_q, y_pot_var_q, y_pot_mmed_q, y_pot_mmed_var_q
    # -- 3D array:                        (no_of_comparisons, 2, no_of_out)

    # Weights: weights_q, relevant_data_q, w_q
    # Type: List (no_of_comparisons) of Lists (no_of_outcomes) of objects
    # - object of weights_q: List of 2 2D arrays (relevant n_p x n_y) (if crs)
    # - array of relevant_data_q: 1D of Boolean (all n_p elements)
    # - w_q:  1D of float (relevant n_p obs) - kernel weights in n_p

    # Step 1: Aggregate weights
    comp_list = list(iate_q_key_dic)
    for out_idx in range(no_of_out):               # Outcomes
        for comp_idx in range(no_of_comp):         # Comparisons
            if int_cfg.weight_as_sparse:
                n_x = weights_q[comp_idx][out_idx][0].shape[0]
            else:
                n_x = len(weights_q[comp_idx][out_idx])
            comp_key = comp_list[comp_idx]
            for n_idx in range(n_x):               # relevant n_p observations
                for t_idx in range(2):             # Both relevant treatments
                    w_q_add = np.zeros((n_y))
                    if int_cfg.weight_as_sparse:
                        weight_i = weights_q[comp_idx][out_idx][t_idx][n_idx, :]
                        w_index = weight_i.col
                        w_i = weight_i.data.copy()
                    else:  # Ind weights > 0
                        w_index = weights_q[
                            comp_idx][out_idx][n_idx][t_idx][0].copy()
                        w_i = weights_q[
                            comp_idx][out_idx][n_idx][t_idx][1].copy()
                    if gen_cfg.weighted:
                        w_i = w_i * w_dat[w_index].reshape(-1)
                    w_i_sum = np.sum(w_i)
                    if not 1-sum_tol < w_i_sum < 1+sum_tol:
                        w_i = w_i / w_i_sum
                    if gen_cfg.weighted:
                        w_i = w_i * w_p_q[n_idx]
                    w_i = w_i * w_q[comp_idx][out_idx][n_idx]
                    if p_cfg.choice_based_sampling:
                        treat = iate_q_key_dic[comp_key][t_idx]
                        i_pos = i_d_val[d_p[n_idx] == d_values[treat]]
                        w_q_add[w_index] = w_i * t_probs[int(i_pos)]
                    else:
                        w_q_add[w_index] = w_i.copy()
                    w_qiate_q[comp_idx, t_idx, :, out_idx] += w_q_add

    # Step 2: Get potential outcomes for particular q_value
            sum_w_qiate_q = np.sum(w_qiate_q, axis=2)
            # TODO: Is Bias adjustment needed or are the weights already bias
            # adjusted? If needed, wo?
            for t_idx in range(2):
                if w_median is not None:
                    w_median_ = w_median[comp_idx, t_idx, :, out_idx]
                else:
                    w_median_ = None
                if w_opp is not None:
                    w_opp_ = w_opp[comp_idx, t_idx, :, out_idx]
                else:
                    w_opp_ = None

                (w_qiate_q[comp_idx, t_idx, :, out_idx],
                 w_diff_med, w_diff_opp,
                 w_censored_q[comp_idx, t_idx, out_idx],
                 w_qiate_unc_q[comp_idx, t_idx, :, out_idx]
                 ) = w_qiate_func(sum_w_qiate_q[comp_idx, t_idx, out_idx],
                                  w_qiate_q[comp_idx, t_idx, :, out_idx],
                                  w_censored_q[comp_idx, t_idx, out_idx],
                                  w_qiate_unc_q[comp_idx, t_idx, :, out_idx],
                                  w_median_, w_opp_,
                                  qiate_m_mqiate, qiate_m_opp,
                                  p_cfg.max_weight_share, iv=iv,
                                  zero_tol=int_cfg.zero_tol,
                                  sum_tol=int_cfg.sum_tol,
                                  p_ba_yes=p_ba_cfg.yes,
                                  )
                if not weights_only:
                    ret = mcf_est.weight_var(
                        w_qiate_q[comp_idx, t_idx, :, out_idx],
                        y_dat[:, out_idx], cl_dat, gen_cfg, p_cfg,
                        weights=w_dat, bootstrap=p_cfg.se_boot_qiate,
                        keep_all=int_cfg.keep_w0, se_yes=p_cfg.qiate_se,
                        normalize=not iv,
                        zero_tol=int_cfg.zero_tol,
                        sum_tol = int_cfg.sum_tol,
                        )
                    y_pot_q[comp_idx, t_idx, out_idx] = ret[0]
                    y_pot_var_q[comp_idx, t_idx, out_idx] = ret[1]

                    if qiate_m_mqiate:
                        ret2 = mcf_est.weight_var(
                            w_diff_med, y_dat[:, out_idx], cl_dat, gen_cfg,
                            p_cfg, normalize=False, weights=w_dat,
                            bootstrap=p_cfg.se_boot_qiate,
                            keep_all=int_cfg.keep_w0,
                            se_yes=p_cfg.qiate_se,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        y_pot_mmed_q[comp_idx, t_idx, out_idx] = ret2[0]
                        y_pot_mmed_var_q[comp_idx, t_idx, out_idx] = ret2[1]

                    if qiate_m_opp:
                        ret2 = mcf_est.weight_var(
                            w_diff_opp, y_dat[:, out_idx], cl_dat, gen_cfg,
                            p_cfg, normalize=False, weights=w_dat,
                            bootstrap=p_cfg.se_boot_qiate,
                            keep_all=int_cfg.keep_w0,
                            se_yes=p_cfg.qiate_se,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        y_pot_mopp_q[comp_idx, t_idx, out_idx] = ret2[0]
                        y_pot_mopp_var_q[comp_idx, t_idx, out_idx] = ret2[1]

    if weights_only:
        return (None, None, None, None, None, None,
                w_qiate_q, w_qiate_unc_q, q_value_idx, w_censored_q)

    if y_pot_mmed_var_q is not None:
        y_pot_mmed_var_q[np.abs(y_pot_mmed_q) < int_cfg.zero_tol] = 0
    if y_pot_mopp_var_q is not None:
        y_pot_mopp_var_q[np.abs(y_pot_mopp_q) < int_cfg.zero_tol] = 0

    return (y_pot_q, y_pot_var_q, y_pot_mmed_q, y_pot_mmed_var_q,
            y_pot_mopp_q, y_pot_mopp_var_q,
            w_qiate_q, w_qiate_unc_q, q_value_idx, w_censored_q)


def assign_pot(y_pot: NDArray[any],
               y_pot_var: NDArray[any] | None,
               y_pot_mmed: NDArray[any],
               y_pot_mmed_var: NDArray[any] | None,
               y_pot_mopp: NDArray[any],
               y_pot_mopp_var: NDArray[any] | None,
               results_fut_q: Any,
               q_idx: int,
               qiate_m_mqiate: NDArray[any] | None,
               qiate_m_opp: NDArray[any] | None,
               ) -> tuple[NDArray[any], NDArray[any] | None,
                          NDArray[any], NDArray[any] | None,
                          NDArray[any], NDArray[any] | None,
                          ]:
    """Reduce repetetive code."""
    y_pot[q_idx] = results_fut_q[0]
    y_pot_var[q_idx] = results_fut_q[1]
    if qiate_m_mqiate:
        y_pot_mmed[q_idx] = results_fut_q[2]
        y_pot_mmed_var[q_idx] = results_fut_q[3]
    if qiate_m_opp:
        y_pot_mopp[q_idx] = results_fut_q[4]
        y_pot_mopp_var[q_idx] = results_fut_q[5]

    return (y_pot, y_pot_var,
            y_pot_mmed, y_pot_mmed_var,
            y_pot_mopp, y_pot_mopp_var)


def assign_w(w_qiate: NDArray[Any],
             w_qiate_unc: NDArray[Any],
             w_censored: NDArray[Any],
             results_fut_q: Any,
             q_idx: int,
             ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Reduce repetetive code."""
    w_qiate[q_idx] = results_fut_q[6]
    w_qiate_unc[q_idx] = results_fut_q[7]
    w_censored[q_idx] = results_fut_q[9]

    return w_qiate, w_qiate_unc, w_censored


def w_qiate_func(sum_w_qiate: NDArray[Any],
                 w_qiate_q: NDArray[Any],
                 w_censored_q: NDArray[Any],
                 w_qiate_unc_q: NDArray[Any],
                 w_mmed: NDArray[Any],
                 w_mopp: NDArray[Any],
                 compute_w_diff_med: bool,
                 compute_w_diff_opp: bool,
                 max_weight_share: float | np.floating,
                 iv: bool = False,
                 zero_tol: float = 1e-15,
                 sum_tol: float = 1e-12,
                 p_ba_yes: bool = False,
                 ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any],
                            NDArray[Any], NDArray[Any]
                            ]:
    """Compute weights for discrete case."""
    if (not 1-sum_tol < sum_w_qiate < 1+sum_tol) and (sum_w_qiate > sum_tol):
        w_qiate_q = w_qiate_q / sum_w_qiate
    w_qiate_unc_q = w_qiate_q

    if max_weight_share < 1:
        if iv:
            w_qiate_q, _, w_censored_q = bound_norm_weights_not_one(
                w_qiate_q, max_weight_share,
                zero_tol=zero_tol, sum_tol=sum_tol,)
        else:
            w_qiate_q, _, w_censored_q = bound_norm_weights(
                w_qiate_q, max_weight_share,
                zero_tol=zero_tol, sum_tol=sum_tol,
                negative_weights_possible=p_ba_yes,
                )
    w_diff_med = w_qiate_unc_q - w_mmed if compute_w_diff_med else None
    w_diff_opp = w_qiate_unc_q - w_mopp if compute_w_diff_opp else None

    return w_qiate_q, w_diff_med, w_diff_opp, w_censored_q, w_qiate_unc_q


def get_w_rel_q(iate_q_np: NDArray[Any],
                iate_q_key_dic: dict,
                q_val: NDArray[Any] | list,
                weights_all_ref: Any,
                no_of_outcomes: int = 1,
                bandwidth: int = 1,
                kernel_str: str = 'epanechikov',
                w_is_csr: bool = False,
                zero_tol: float = 1e-15,
                ) -> tuple[list[list], list[list], list[list]]:
    """Get relevant observations and their weights."""
    # Get neighbouring observation (comparison specific)
    no_of_comparisons = len(iate_q_key_dic)
    weights = [[None for _ in range(no_of_outcomes)]
               for _ in range(no_of_comparisons)]
    w_q = [[None for _ in range(no_of_outcomes)]
           for _ in range(no_of_comparisons)]
    relevant_data = [[None for _ in range(no_of_outcomes)]
                     for _ in range(no_of_comparisons)]

    if isinstance(weights_all_ref, ray.ObjectRef):
        weights_all = ray.get(weights_all_ref)
    else:
        weights_all = weights_all_ref

    kernel = 1 if kernel_str == 'epanechikov' else 2

    for value in iate_q_key_dic.values():   # Treatment comparisons
        for out_idx in range(no_of_outcomes):
            order_dat = iate_q_np[:, value[0], out_idx]
            quantil = np.quantile(order_dat, q_val)

            arg = order_dat - quantil
            if bandwidth is None:
                # If no smoothing use observations closest to quantil
                w_q_val = np.zeros_like(arg)
                no_closest = 1
                for idx in range(no_closest):
                    idx = np.argmin(abs(arg))
                    w_q_val[idx] = 1
                    arg[idx] += 1e100
            else:
                w_q_val = kernel_proc(arg / bandwidth, kernel)

            relevant_data_points = w_q_val > zero_tol
            w_q_val = w_q_val[relevant_data_points]   # Shorten vector
            w_q_val = w_q_val / np.sum(w_q_val) * len(w_q_val)

            if w_is_csr:
                weights[value[0]][out_idx] = [
                    weights_all[t_idx][relevant_data_points, :]
                    for t_idx in value[1:]
                    ]
            else:
                # Remove n_p data that are not needed
                reduce_n = list(
                    compress(weights_all, relevant_data_points))
                # Remove all treatments not relevant for comparison
                n_x = len(reduce_n)
                weights[value[0]][out_idx] = [[reduce_n[n_idx][t_idx]
                                              for t_idx in value[1:]]
                                              for n_idx in range(n_x)]

            w_q[value[0]][out_idx] = w_q_val.copy()
            relevant_data[value[0]][out_idx] = relevant_data_points.copy()

    return weights, relevant_data, w_q


def initialise_weights_qiate(
        no_of_qval: int,
        no_of_comparisons: int,
        n_y: int,
        no_of_out: int,
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Initialize weights."""
    w_qiate = np.zeros((no_of_qval, no_of_comparisons, 2, n_y, no_of_out))
    w_qiate_unc = np.zeros_like(w_qiate)
    w_censored = np.zeros((no_of_qval, no_of_comparisons, 2, no_of_out))
    w_median = np.zeros((no_of_comparisons, 2, n_y, no_of_out))

    return w_qiate, w_qiate_unc, w_censored, w_median


def initialise_pot_qiate(no_of_qval: int,
                         no_of_comparisons: int,
                         no_of_out: int
                         ) -> tuple[NDArray[Any], NDArray[Any],
                                    NDArray[Any], NDArray[Any],
                                    NDArray[Any], NDArray[Any],
                                    ]:
    """Initialise variables that store the potential outcomes."""
    y_pot = np.empty((no_of_qval, no_of_comparisons, 2, no_of_out))
    y_pot_var = np.empty_like(y_pot)
    y_pot_mmed, y_pot_mmed_var = np.empty_like(y_pot), np.empty_like(y_pot)
    y_pot_mopp, y_pot_mopp_var = np.empty_like(y_pot), np.empty_like(y_pot)

    return (y_pot, y_pot_var, y_pot_mmed, y_pot_mmed_var, y_pot_mopp,
            y_pot_mopp_var)


def get_nadarya_watson_parameters(x_dat: NDArray[Any],
                                  qiate_smooth_bandwidth: float,
                                  zero_tol: float = 1e-15,
                                  ) -> tuple[np.floating | float, str]:
    """Smoothing parameters in Nadaraya-Watson estimation."""
    kernel = 'epanechikov'
    bandwidth = bandwidth_nw_rule_of_thumb(x_dat, kernel=1, zero_tol=zero_tol)
    bandwidth *= qiate_smooth_bandwidth

    return bandwidth, kernel


def optimize_initialize_mp_qiate(weights_all: Any,
                                 gen_cfg: Any,
                                 int_cfg: Any,
                                 with_output: bool
                                 ) -> tuple[set, None, Any, int, str]:
    """Define multiprocession and initialize ray (if used)."""
    files_to_delete, save_w_file = set(), None
    txt = ''
    weights_all2 = weights_all

    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        if gen_cfg.mp_automatic:
            maxworkers = mcf_sys.find_no_of_workers(gen_cfg.mp_parallel,
                                                    gen_cfg.sys_share / 2,
                                                    zero_tol=int_cfg.zero_tol,
                                                    )
        else:
            maxworkers = gen_cfg.mp_parallel
        if weights_all2 is None:
            maxworkers = round(maxworkers / 2)
        if not maxworkers > 0:
            maxworkers = 1

    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print('Number of parallel processes (QIATE): ', maxworkers, flush=True)

    if maxworkers > 1:
        if not ray.is_initialized():
            mcf_sys.init_ray_with_fallback(
                maxworkers, int_cfg, gen_cfg,
                mem_object_store=int_cfg.mem_object_store_3,
                ray_err_txt='Ray does not start up in QIATE estimation.')
        if (int_cfg.mem_object_store_3 is not None
                and gen_cfg.with_output and gen_cfg.verbose):
            print('Size of Ray Object Store: ',
                  round(int_cfg.mem_object_store_3/(1024*1024)), " MB")
        weights_all_ref = ray.put(weights_all)
    else:
        weights_all_ref = weights_all

    return files_to_delete, save_w_file, weights_all_ref, maxworkers, txt


def ray_end(int_cfg: Any,
            weights_all_ref: Any,
            finished_res: Any,
            finished: Any,
            ) -> None:
    """End ray according to local rules."""
    if 'refs' in int_cfg.mp_ray_del:
        del weights_all_ref
    if 'rest' in int_cfg.mp_ray_del:
        del finished_res, finished


def assign_data(
        weights_dic: dict,
        gen_cfg: Any,
        p_cfg: dict,
        ) -> tuple[NDArray[Any], Any, NDArray[Any] | None, NDArray[Any] | None]:
    """Assign variables from weight_dic dictionary."""
    y_dat = weights_dic['y_dat_np']
    weights_all = weights_dic['weights']
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    cl_dat = weights_dic['cl_dat_np'] if p_cfg.cluster_std else None

    return y_dat, weights_all, w_dat, cl_dat


def get_comparison_labels(no_of_treat: int,
                          d_values: list[np.floating | np.integer | int | float]
                          | NDArray[np.floating | np.integer],
                          ) -> list[str]:
    """Compute labels for treatment comparisons."""
    jdx = 0
    treat_comp_label = [None for _ in range(
        round(no_of_treat * (no_of_treat - 1) / 2))]
    for t1_idx, t1_lab in enumerate(d_values):
        for t2_idx in range(t1_idx+1, no_of_treat):
            treat_comp_label[jdx] = str(d_values[t2_idx]) + 'vs' + str(t1_lab)
            jdx += 1

    return treat_comp_label


def get_iate_order(y_pot: NDArray[Any],
                   y_pot_var: NDArray[Any] | None,
                   no_of_treat: int,
                   no_of_out: int,
                   p_cfg: Any,
                   ) -> tuple[NDArray[Any], dict]:
    """Compute the position for IATE and also a key to find it later."""
    obs = len(y_pot)
    quantile_key_dic = {}
    quantile_np = np.empty((obs, round(no_of_treat * (no_of_treat - 1) / 2),
                            no_of_out))

    running_idx = 0
    for t1_idx in range(no_of_treat - 1):
        str1 = str(t1_idx) + '_'
        for t2_idx in range(t1_idx + 1, no_of_treat):
            key_str = str1 + str(t2_idx)
            for o_idx in range(no_of_out):
                iate = y_pot[:, t2_idx, o_idx] - y_pot[:, t1_idx, o_idx]
                quantile_np[:, running_idx, o_idx] = (
                    (np.argsort(np.argsort(iate)) + 0.5) / obs)

                if p_cfg.qiate_bias_adjust:
                    iate_se = np.sqrt(y_pot_var[:, t2_idx, o_idx]
                                      + y_pot_var[:, t1_idx, o_idx])
                    quantile_np[:, running_idx, o_idx] = bias_adjust_qiate(
                        quantile_np[:, running_idx, o_idx], iate, iate_se,
                        draws=p_cfg.qiate_bias_adjust_draws,
                        simex=False
                        )

                quantile_key_dic[key_str] = (running_idx, t1_idx, t2_idx, )
            running_idx += 1

    return quantile_np, quantile_key_dic


def bias_adjust_qiate(iate_quantile_np: NDArray[Any],
                      iate_np: NDArray[Any],
                      iate_se_np: NDArray[Any],
                      draws: int = 100,
                      simex: bool = False,
                      ) -> NDArray[Any]:
    """Compute bias adjustment."""
    OLD_SIMULATION = False

    obs = iate_quantile_np.shape[0]
    rng_qiate = np.random.default_rng(seed=125435)

    if simex:
        est_error_simul_np = rng_qiate.normal(loc=0, scale=1, size=(obs, draws))
        grid = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5,)
        len_grid = len(grid)
        iate_simul_rank = np.empty((len(iate_np), len_grid, draws))

        for g_idx, noise_level in enumerate(grid):
            iate_est_simul = iate_np.reshape(-1, 1) + (est_error_simul_np
                                                       * np.sqrt(noise_level))
            for h in range(draws):
                iate_simul_rank[:, g_idx, h] = iate_est_simul[:, h].argsort(
                    axis=0).argsort(axis=0) + 0.5  # .reshape(-1)

        iate_simul_rank_mean = np.mean(iate_simul_rank, axis=-1)
        iate_rank_np = iate_quantile_np * obs

        rank_iate_adj_np = simex_diff_rank_adjustment(iate_rank_np,
                                                      iate_se_np,
                                                      iate_simul_rank_mean,
                                                      grid)
        iate_quantile_adj_np = np.clip(rank_iate_adj_np / obs, 0, 1)

    elif OLD_SIMULATION:
        est_error_simul_np = rng_qiate.normal(loc=0, scale=1, size=(obs, draws))
        iate_simul_rank = np.empty_like(est_error_simul_np)   # OBS x DRAWS
        iate_est_simul = iate_np.reshape(-1, 1) + est_error_simul_np
        for h in range(est_error_simul_np.shape[1]):
            iate_simul_rank[:, h] = iate_est_simul[:, h].argsort(
                axis=0).argsort(axis=0).reshape(-1) + 0.5

        iate_simul_rank_mean = np.mean(iate_simul_rank, axis=1).reshape(-1, 1)

        for h in range(est_error_simul_np.shape[1]):
            iate_simul_rank[:, h] = iate_est_simul[:, h].argsort(
                axis=0).argsort(axis=0).reshape(-1) + 0.5
        iate_simul_rank_mean = np.mean(iate_simul_rank, axis=1).reshape(-1, 1)
        iate_quantile_adj_np = np.clip(
            delta_quantile_adjustment(iate_quantile_np,
                                      iate_simul_rank_mean / obs),
            0, 1)
    else:
        iate_bias_adjust_np = bias_adjust_value(iate_np, iate_quantile_np,
                                                iate_se_np,
                                                )
        iate_quantile_adj_np = switch_ranks(iate_bias_adjust_np, iate_np)

    return iate_quantile_adj_np


def bias_adjust_value(iate_np: NDArray[Any],
                      iate_quantile_np: NDArray[Any],
                      iate_se_np: NDArray[Any],
                      tolerance_0_1: float | np.floating = 1e-10
                      ) -> NDArray[Any]:
    """Bias adjustment of IATE keeping quantile position."""
    obs = len(iate_np)
    bias = norm.ppf(iate_quantile_np.clip(min=tolerance_0_1,
                                          max=1 - tolerance_0_1),
                    loc=np.zeros(obs),
                    scale=iate_se_np)
    iate_bias_adjust_np = iate_np - bias

    return iate_bias_adjust_np


def switch_ranks(iate_bias_adjust_np: NDArray[Any],
                 iate_np: NDArray[Any],
                 ) -> NDArray[Any]:
    """Find quantiles of bias adjusted IATEs in old distribution of IATEs."""
    iate_bias_adjust_np_sorted = np.sort(iate_bias_adjust_np)
    indices = np.searchsorted(iate_bias_adjust_np_sorted, iate_np, side='right')
    quantile_iate_adj_np = (indices + 0.5) / len(iate_np)

    return quantile_iate_adj_np


def simex_diff_rank_adjustment(iate_est_rank: NDArray[Any],
                               iate_est_se: NDArray[Any] | None,
                               iate_simul_rank: NDArray[Any],
                               grid: list[float | np.floating],
                               ) -> NDArray[Any]:
    """Adjust ranks by regression."""
    x_dim = 6

    x_var = np.empty((len(iate_est_rank), x_dim, len(grid)))
    y_var = np.empty((len(iate_est_rank), len(grid),))
    ones = np.ones(len(iate_est_rank))

    for gidx, noise_level in enumerate(grid):
        y_var[:, gidx] = iate_est_rank - iate_simul_rank[:, gidx]
        x_var[:, :, gidx] = make_x_simex_diff(x_dim, ones, iate_est_se,
                                              noise_level)

    x_var_pred = make_x_simex_diff(x_dim, ones, iate_est_se, -1)

    x_list = [x_var[:, :, gidx] for gidx, _ in enumerate(grid)]
    x_varf = np.concatenate(x_list, axis=0)

    y_list = [y_var[:, gidx] for gidx, _ in enumerate(grid)]
    y_varf = np.concatenate(y_list, axis=0)
    # Does OLS make sense for a difference in rank regression?
    beta = np.linalg.solve(x_varf.T @ x_varf, x_varf.T @ y_varf)

    delta = x_var_pred @ beta

    rank_adjusted = iate_est_rank - delta

    return rank_adjusted


def make_x_simex_diff(x_dim: int,
                      ones: NDArray[Any],
                      iate_est_se: NDArray[Any],
                      noise_level: float | np.floating,
                      ) -> NDArray[Any]:
    """Make matrices with independent variables."""
    x_var = np.zeros((len(iate_est_se), x_dim))
    x_var[:, 0] = ones
    x_var[:, 1] = noise_level
    if x_dim > 2:
        x_var[:, 2] = noise_level ** 2
    if x_dim > 3:
        x_var[:, 3] = iate_est_se * noise_level
    if x_dim > 4:
        x_var[:, 4] = iate_est_se * (noise_level ** 2)
    if x_dim > 5:
        x_var[:, 5] = (iate_est_se * noise_level) ** 2

    return x_var


def delta_quantile_adjustment(est_q: NDArray[Any],
                              simul_q: NDArray[Any]
                              ) -> NDArray[Any]:
    """Compute adjusted quantile positions of estimated IATEs."""
    delta = simul_q - est_q.reshape(-1, 1)
    new_quantil = est_q.reshape(-1, 1) - delta

    return new_quantil.reshape(-1)


def qiate_effects_print(mcf_: 'ModifiedCausalForest',
                        effect_dic: dict,
                        effect_m_med_dic: dict,
                        effect_m_opp_dic: dict,
                        qiate_est_dic: dict,
                        special_txt: str | None = None
                        ) -> tuple[NDArray[Any], NDArray[Any] | None,
                                   NDArray[Any], NDArray[Any] | None,
                                   NDArray[Any], NDArray[Any] | None,
                                   list[Path]
                                   ]:
    """Compute effects of QIATEs and print them."""
    continuous = False
    p_cfg, int_cfg, gen_cfg = mcf_.p_cfg, mcf_.int_cfg, mcf_.gen_cfg
    # x_value_unord_org = mcf_.data_train_dict['unique_values_dict']
    q_values = p_cfg.qiate_quantiles
    no_of_qval = len(q_values)
    mid_value = round(no_of_qval / 2)
    y_pot, y_pot_var = effect_dic['y_pot'], effect_dic['y_pot_var']

    m_med_yes = effect_m_med_dic is not None
    m_opp_yes = effect_m_opp_dic is not None
    figure_list = []

    if m_med_yes:
        y_pot_m_med = effect_m_med_dic['y_pot']
        y_pot_m_med_var = effect_m_med_dic['y_pot_var']
    else:
        y_pot_m_med = y_pot_m_med_var = None

    if m_opp_yes:
        y_pot_m_opp = effect_m_opp_dic['y_pot']
        y_pot_m_opp_var = effect_m_opp_dic['y_pot_var']
    else:
        y_pot_m_opp = y_pot_m_opp_var = None

    if special_txt is not None:
        mcf_ps.print_mcf(mcf_.gen_cfg, '\n' + '=' * 100 + special_txt,
                         summary=True, non_summary=False)
    # Get parameters and info computed in 'gate_est'
    treat_comp_labels = (qiate_est_dic['treat_comp_label'],      # True label
                         qiate_est_dic['iate_q_key_dic'].keys())  # Key
    no_of_out, var_cfg = qiate_est_dic['no_of_out'], qiate_est_dic['var_cfg']
    p_cfg = qiate_est_dic['p_cfg']
    effect_type_label = ('QIATE', 'QIATE(q) - QIATE(0.5)',
                         'QIATE(q) - QIATE(1-q)', 'Distribution')

    old_filters = warnings.filters.copy()
    warnings.filterwarnings('error', category=RuntimeWarning)

    qiate = np.empty((no_of_qval, no_of_out, len(treat_comp_labels[0])))
    qiate_se = np.empty_like(qiate)
    qiate_mmed, qiate_mmed_se = np.empty_like(qiate), np.empty_like(qiate)
    qiate_mopp, qiate_mopp_se = np.empty_like(qiate), np.empty_like(qiate)
    txt = ''

    # -- 4D array:   y_pot,        (no_of_qval, no_of_comparisons, 2, no_of_out)
    for o_idx in range(no_of_out):
        if gen_cfg.with_output:
            txt += ('\n' + '-' * 100 + '\nOutcome variable: '
                    f'{var_cfg.y_name[o_idx]}      ')

        if gen_cfg.with_output:
            txt += 'Reference population: All'
            txt += '\n' + '- ' * 50
        ret_qiate = [None for _ in range(no_of_qval)]
        ret_qiate_mmed = [None for _ in range(no_of_qval)]
        ret_qiate_mopp = [None for _ in range(no_of_qval)]
        for q_idx, _ in enumerate(q_values):
            ret = mcf_est.effect_from_potential(
                y_pot[q_idx][:, :, o_idx], y_pot_var[q_idx][:, :, o_idx],
                gen_cfg.d_values, continuous=continuous, sequential=True,
                sequential_dic=qiate_est_dic['iate_q_key_dic'])
            ret_qiate[q_idx] = ret
            qiate[q_idx, o_idx, :] = ret[0]
            qiate_se[q_idx, o_idx, :] = ret[1]

            if m_med_yes:
                ret = mcf_est.effect_from_potential(
                    y_pot_m_med[q_idx][:, :, o_idx],
                    y_pot_m_med_var[q_idx][:, :, o_idx],
                    gen_cfg.d_values, continuous=continuous, sequential=True,
                    sequential_dic=qiate_est_dic['iate_q_key_dic'])
                ret_qiate_mmed[q_idx] = ret
                qiate_mmed[q_idx, o_idx, :] = ret[0]
                qiate_mmed_se[q_idx, o_idx, :] = ret[1]
            else:
                qiate_mmed = qiate_mmed_se = ret_qiate_mmed = None

            if m_opp_yes:
                ret = mcf_est.effect_from_potential(
                    y_pot_m_opp[q_idx][:, :, o_idx],
                    y_pot_m_opp_var[q_idx][:, :, o_idx],
                    gen_cfg.d_values, continuous=continuous, sequential=True,
                    sequential_dic=qiate_est_dic['iate_q_key_dic'])
                ret_qiate_mopp[q_idx] = ret
                qiate_mopp[q_idx, o_idx, :] = ret[0]
                qiate_mopp_se[q_idx, o_idx, :] = ret[1]
            else:
                qiate_mopp = qiate_mopp_se = ret_qiate_mopp = None

        if gen_cfg.with_output:
            txt += ('\nQuantile Individualized Average Treatment Effects\n'
                    + '- ' * 50)
            txt += f'\nOutcome: {var_cfg.y_name[o_idx]} Ref. pop.: All\n'
            txt += mcf_ps.print_effect_z(
                ret_qiate, ret_qiate_mmed, q_values, 'QIATE', 'ATE',
                print_output=False, gates_minus_previous=False, qiate=True,
                gmopp_r=ret_qiate_mopp, z_values_org=None
                )
            txt += '\n' + mcf_ps.print_se_info(
                p_cfg.cluster_std, p_cfg.se_boot_gate
                )
            txt += mcf_ps.print_minus_ate_info(gen_cfg.weighted,
                                               print_it=False)
    med_f = med_f_se = None
    if gen_cfg.with_output:
        for o_idx, o_lab in enumerate(var_cfg.y_name):
            for t_idx, t_lab in enumerate(treat_comp_labels[0]):
                for e_idx, e_lab in enumerate(effect_type_label):
                    figure_disc = figure_cont = None
                    if e_idx == 0:
                        effects = qiate[:, o_idx, t_idx]
                        ste = qiate_se[:, o_idx, t_idx]
                        med_f = qiate[mid_value, o_idx, t_idx]
                        med_f_se = qiate_se[mid_value, o_idx, t_idx]
                        cdf_pdf = False
                    elif e_idx == 1:
                        med_f, med_f_se = 0, None
                        if m_med_yes:
                            effects = qiate_mmed[:, o_idx, t_idx]
                            ste = qiate_mmed_se[:, o_idx, t_idx]
                        else:
                            effects = ste = None
                        cdf_pdf = False
                    elif e_idx == 2:
                        if m_opp_yes:
                            effects = qiate_mopp[:, o_idx, t_idx]
                            middle = round((len(effects) - 1) / 2)
                            qiate_mopp[middle, o_idx, t_idx] = 0
                            qiate_mopp_se[middle, o_idx, t_idx] = 0.001
                            ste = qiate_mopp_se[:, o_idx, t_idx]
                        else:
                            effects = ste = None
                        cdf_pdf = False
                    else:
                        effects = qiate[:, o_idx, t_idx]
                        ste = qiate_se[:, o_idx, t_idx]
                        med_f, med_f_se = 0, None
                        cdf_pdf = True

                    if not continuous and effects is not None and e_idx != 2:
                        figure_disc = make_qiate_figures_discr(
                            e_lab + ' ' + ' ' + o_lab + ' ' + t_lab, q_values,
                            effects, ste, int_cfg, p_cfg, med_f, med_f_se,
                            cdf_pdf=cdf_pdf)

                    figure_file = (figure_disc if figure_cont is None
                                   else figure_cont)
                    vs0 = int(t_lab[-1]) == int(gen_cfg.d_values[0])
                    if vs0 and e_idx != 2 and (
                            ((m_med_yes and e_idx == 1)
                             or (not m_med_yes and e_idx == 0))):
                        figure_list.append(figure_file)

    if gen_cfg.with_output:
        txt += '-' * 100
        mcf_ps.print_mcf(gen_cfg, txt, summary=True, non_summary=True)

    warnings.filters = old_filters
    # warnings.resetwarnings()

    return (qiate, qiate_se, qiate_mmed, qiate_mmed_se,
            qiate_mopp, qiate_mopp_se, figure_list)


def make_qiate_figures_discr(
        titel: str,
        q_values: list,
        effects: NDArray[Any],
        stderr: NDArray[Any] | None,
        int_cfg: Any,
        p_cfg: Any,
        med: int | float | None = 0,
        med_se: int | float | None = None,
        cdf_pdf: bool = False
        ) -> Path | None:
    """Generate the figures for QIATE results (discrete treatments).

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    q_vals : List. Values of z-variables.
    effects : 1D Numpy array. Effects for all z-values.
    stderr : 1D Numpy array. Standard errors for all effects.
    int_cfg : Instance of Dataclass. Parameters.
    p_cfg : PCfg dataclass. Parameters.
    Additional keyword parameters.
    """
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')

    file_name_jpeg = p_cfg.paths['qiate_fig_pfad_jpeg'] / f'{titel_f}.jpeg'
    file_name_pdf = p_cfg.paths['qiate_fig_pfad_pdf'] / f'{titel_f}.pdf'
    file_name_csv = p_cfg.paths['qiate_fig_pfad_csv'] / f'{titel_f}plotdat.csv'

    with_se = not np.all((stderr - 0.001) < 1e-8)
    if cdf_pdf:
        qiate_str_y = 'Distribution'
        qiate_str, label_y = 'cdf_pdf', 'pdf & cdf'
        label_cdf, label_pdf = 'cdf', 'pdf'
        med_label = '_nolegend_'
        pdf_smooth, cdf, values = make_cdf_pdf(effects, q_values)
        stderr, med, med_se = 0, None, None

    else:
        if med_se is None:
            qiate_str_y = 'QIATE(q) - QIATE(0.5)'
            qiate_str = 'QIATE(q)-QIATE(0.5)'
            label_m, label_y = 'QIATE(q)-QIATE(0.5)', 'QIATE(q)-QIATE(0.5)'
            med_label = '_nolegend_'
        else:
            qiate_str_y = 'QIATE(q)'
            label_m = 'QIATE(q)'
            med_label, qiate_str, label_y = 'QIATE(0.5)', 'QIATE(q)', 'QIATE(q)'
        med = med * np.ones((len(q_values), 1))

        cint = norm.ppf(p_cfg.ci_level + 0.5 * (1 - p_cfg.ci_level))
        upper, lower = effects + stderr * cint, effects - stderr * cint
        label_ci = f'{p_cfg.ci_level:2.0%}-CI'
        if med is not None and med_se is not None:
            med_upper, med_lower = med + med_se * cint, med - med_se * cint

    figs, axs = plt.subplots()

    file_name_f_jpeg = (p_cfg.paths['qiate_fig_pfad_jpeg']
                        / f'{titel_f}fill.jpeg')
    file_name_f_pdf = p_cfg.paths['qiate_fig_pfad_pdf'] / f'{titel_f}fill.pdf'
    if cdf_pdf:
        axs.plot(values, cdf, color='b', label=label_cdf)
        axs.plot(values, pdf_smooth, color='r', label=label_pdf)
    else:
        axs.plot(q_values, effects, label=label_m, color='b')
        if with_se:
            axs.fill_between(q_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
        line_med = '-r'
        if med is not None:
            label_med = 'QIATE(0.5)'
            if med_se is not None and with_se:
                axs.fill_between(q_values, med_upper.reshape(-1),
                                 med_lower.reshape(-1), alpha=0.3, color='r',
                                 label=label_ci)
        else:
            label_med = '_nolegend_'
        axs.plot(q_values, med, line_med, label=label_med)
    axs.set_ylabel(label_y)
    axs.legend(loc=int_cfg.legend_loc, shadow=True,
               fontsize=int_cfg.fontsize)
    titel_tmp = titel[:-4] + ' ' + titel[-4:]
    titel_tmp = titel_tmp.replace('vs', ' vs ')
    axs.set_title(titel_tmp)
    if cdf_pdf:
        axs.set_xlabel('IATEs')
    else:
        axs.set_xlabel('Quantiles')
    mcf_sys.delete_file_if_exists(file_name_f_jpeg)
    mcf_sys.delete_file_if_exists(file_name_f_pdf)
    figs.savefig(file_name_f_jpeg, dpi=int_cfg.dpi)
    figs.savefig(file_name_f_pdf, dpi=int_cfg.dpi)

    if int_cfg.show_plots:
        plt.show()
    plt.close()

    if not cdf_pdf and with_se:
        e_line, u_line, l_line = '_-', 'v-', '^-'
        connect_y, connect_x = np.empty(2), np.empty(2)
        fig, axe = plt.subplots()

        for idx, i_lab in enumerate(q_values):
            connect_y[0], connect_y[1] = upper[idx], lower[idx]
            connect_x[0], connect_x[1] = i_lab, i_lab
            axe.plot(connect_x, connect_y, 'b-', linewidth=0.7)
        axe.plot(q_values, effects, e_line + 'b', label=qiate_str)
        axe.set_ylabel(qiate_str_y)
        label_u = f'Upper {p_cfg.ci_level:2.0%}-CI'
        label_l = f'Lower {p_cfg.ci_level:2.0%}-CI'
        axe.plot(q_values, upper, u_line + 'b', label=label_u)
        axe.plot(q_values, lower, l_line + 'b', label=label_l)
        axe.plot(q_values, med, '-' + 'r', label=med_label)
        if med_se is not None:
            axe.plot(q_values, med_upper, '--' + 'r', label=label_u)
            axe.plot(q_values, med_lower, '--' + 'r', label=label_l)
        axe.legend(loc=int_cfg.legend_loc, shadow=True,
                   fontsize=int_cfg.fontsize
                   )
        titel_tmp = titel[:-4] + ' ' + titel[-4:]
        titel_tmp = titel_tmp.replace('vs', ' vs ')
        axe.set_title(titel_tmp)
        axe.set_xlabel('Quantiles')
        mcf_sys.delete_file_if_exists(file_name_jpeg)
        mcf_sys.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=int_cfg.dpi)
        fig.savefig(file_name_pdf, dpi=int_cfg.dpi)
        if int_cfg.show_plots:
            plt.show()
        plt.close()

        # Write values to file
        effects = effects.reshape(-1, 1)
        upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
        q_values_np = np.array(q_values, copy=True).reshape(-1, 1)
        if med_se is not None:
            med_upper = med_upper.reshape(-1, 1)
            med_lower = med_lower.reshape(-1, 1)
            effects_et_al = np.concatenate(
                (upper, effects, lower, med, med_upper, med_lower, q_values_np),
                axis=1)
            cols = ['upper', 'QIATE(q)', 'lower', 'QIATE(0.5)', 'med_upper',
                    'med_lower', 'q_values']
        else:
            cols = ['upper', 'QIATE(q)', 'lower', 'QIATE(0.5)', 'q_values']
            effects_et_al = np.concatenate((upper, effects, lower, med,
                                            q_values_np), axis=1)
        datasave = pd.DataFrame(data=effects_et_al, columns=cols)
        mcf_sys.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)

    file_name = file_name_jpeg if file_name_f_jpeg is None else file_name_f_jpeg

    return file_name


def make_cdf_pdf(values: NDArray[Any],
                 q_values: NDArray[Any]
                 ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Use data from a cdf to approximate a pdf."""
    cdf = q_values.copy()
    pdf = np.zeros_like(q_values)
    area = 0
    for q_idx, q_val in enumerate(q_values):
        if q_idx == 0:
            p_delta = q_val
            values_delta = np.abs(values[1] - values[0])
        else:
            p_delta = q_val - q_values[q_idx-1]
            values_delta = np.abs(values[q_idx] - values[q_idx-1])
        if values_delta == 0:
            values_delta = 0.01   # Step in cdf, pdf is infinity
        pdf[q_idx] = p_delta / values_delta

        area += pdf[q_idx] * p_delta
    pdf /= area

    # Smooth pdf with moving average
    number = max(round(len(pdf) / 100 * 10), 1)
    pdf_smooth = np.convolve(pdf, np.ones(number) / number, mode='same')
    # Adjustments for bounding points
    for idx in range(number):
        pdf_smooth[idx] = np.mean(pdf[:idx+1])
        pdf_smooth[-idx-1] = np.mean(pdf[-idx-1:])

    return pdf_smooth, cdf, values
