"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the ATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from mcf.mcf_bias_adjustment_functions import (
    get_ba_data_prediction, bias_correction_wols, get_weights_eval_ba
    )
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def ate_est(mcf_: 'ModifiedCausalForest',
            data_df: pd.DataFrame,
            weights_dic: dict,
            balancing_test: bool = False,
            w_ate_only: bool = False,
            with_output: bool = True,
            iv: bool = False,
            pred_alloc: bool = False,
            ) -> NDArray[Any]:
    """Estimate ATEs and their standard errors.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame. Prediction data.
    weights_dic : Dict.
               Contains weights and numpy data.
    balancing_test: Boolean,  optional.
               Default is False.
    w_ate_only : Boolean, optional.
               Only weights are needed as output. Default is False.
    with_output : Boolean, optional.
               Output printed if True. Default is True.
    iv : Boolean, optional.
               Local average treatment effect estimation. True will prevent
               weights from being forced to be positive and normalized. Default
               is False.
    pred_alloc : Boolean, optional.
               Evaluate all-in-1-treatment allocations if True.

    Returns
    -------
    w_ate_1dim : Numpy array. Weights used for ATE computation.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.
    effect_list : List of strings with name of effects (same order as ATE)

    """
    gen_cfg, ct_cfg, int_cfg = mcf_.gen_cfg, mcf_.ct_cfg, mcf_.int_cfg
    p_ba_cfg = mcf_.p_ba_cfg
    zero_tol = int_cfg.zero_tol
    sum_tol = int_cfg.sum_tol
    txt = ''
    print_output = not w_ate_only and with_output
    if balancing_test:
        var_cfg, p_cfg = deepcopy(mcf_.var_cfg), deepcopy(mcf_.p_cfg)
        var_cfg.y_name = var_cfg.x_name_balance_test
        p_cfg.atet, p_cfg.gatet = 0, 0
        y_dat = weights_dic['x_bala_np']
    else:
        var_cfg, p_cfg = mcf_.var_cfg, mcf_.p_cfg
        y_dat = weights_dic['y_dat_np']
    if gen_cfg.d_type == 'continuous':
        continuous = True
        p_cfg.atet = p_cfg.gatet = False
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
    else:
        continuous = False
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None

    if p_ba_cfg.yes:
        # Extract information for bias adjustment
        ba_data = get_ba_data_prediction(weights_dic, p_ba_cfg)
        # No need for weighting if even if method is 'weighted_observable'
        # because all score should be included (for the ATE)
        # However, weighting is needed for ATET
    else:
        ba_data = None

    d_p, _, w_p, n_p = get_data_for_final_ate_estimation(
        data_df, gen_cfg, p_cfg, var_cfg, ate=True,
        need_count=int_cfg.weight_as_sparse or iv)
    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)
    if (d_p is not None) and (p_cfg.atet or p_cfg.gatet):
        no_of_ates = no_of_treat + 1      # Compute ATEs, ATET, ATENT
    else:
        p_cfg.atet, no_of_ates = False, 1
    t_probs = p_cfg.choice_based_probs
    # Step 1: Aggregate weights
    if gen_cfg.with_output and gen_cfg.verbose and print_output:
        if balancing_test:
            print('\n\nComputing balancing tests (in ATE-like fashion)')
        elif pred_alloc:
            print('\n\nComputing average outcomes for all-in-1-treatment '
                  'allocations')
        else:
            print('\n\nComputing ATEs')
    w_ate = np.zeros((no_of_ates, no_of_treat, n_y))
    if p_cfg.ate_no_se_only:
        w_ate_export = None
    else:
        w_ate_export = np.zeros_like(w_ate)
    ind_d_val = np.arange(no_of_treat)
    weights = weights_dic['weights']
    if int_cfg.weight_as_sparse:
        for i in range(n_p):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i_csr = weights[t_ind][i, :]   # copy, but still sparse
                if gen_cfg.weighted:
                    w_i_csr = w_i_csr.multiply(w_dat.reshape(-1))
                sum_wi = w_i_csr.sum()
                if not iv and (sum_wi <= zero_tol):
                    txt = f'\nEmpty leaf. Observation: {i}'
                    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                    raise RuntimeError(txt)
                if not iv and not (1-sum_tol) < sum_wi < (1+sum_tol):
                    w_i_csr = w_i_csr.multiply(1 / sum_wi)
                if gen_cfg.weighted:
                    w_i_csr = w_i_csr.multiply(w_p[i])
                if p_cfg.choice_based_sampling:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_i_csr = w_i_csr.multiply(t_probs[int(i_pos)])
                w_add[t_ind, :] = w_i_csr.todense()
            w_ate[0, :, :] += w_add
            if p_cfg.atet:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    else:
        for i, weight_i in enumerate(weights):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i = weight_i[t_ind][1].copy()
                if gen_cfg.weighted:
                    w_i = w_i * w_dat[weight_i[t_ind][0]].reshape(-1)
                if not iv:
                    sum_wi = np.sum(w_i)
                    if np.abs(sum_wi) <= sum_tol:
                        txt = (f'\nZero weight. Index: {weight_i[t_ind][0]}'
                               f'd_value: {t_ind}\nWeights: {w_i}')
                        mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                        raise RuntimeError(txt)
                if not iv and (not (1-sum_tol) < sum_wi < (1+sum_tol)):
                    w_i = w_i / sum_wi
                if gen_cfg.weighted:
                    w_i = w_i * w_p[i]
                if p_cfg.choice_based_sampling:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_add[t_ind, weight_i[t_ind][0]] = w_i * t_probs[
                        int(i_pos)]
                else:
                    w_add[t_ind, weight_i[t_ind][0]] = w_i
            w_ate[0, :, :] += w_add
            if p_cfg.atet:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    # Step 2: Get potential outcomes
    sumw = np.sum(w_ate, axis=2)
    if iv:
        sumw = np.ones_like(sumw) * n_p

    if p_ba_cfg.yes and no_of_ates > 1 and p_ba_cfg.adj_method == 'w_obs':
        weights_eval = get_weights_eval_ba(w_ate, no_of_treat,
                                           zero_tol=int_cfg.zero_tol
                                           )
    else:
        weights_eval = None

    for a_idx in range(no_of_ates):
        if p_ba_cfg.yes and weights_eval is not None:
            ba_data.weights_eval = weights_eval[a_idx, :].copy()

        for ta_idx in range(no_of_treat):
            if not iv and (-zero_tol < sumw[a_idx, ta_idx] < zero_tol):
                if gen_cfg.with_output and print_output:
                    txt += (f'\nTreatment: {ta_idx}, ATE number: {a_idx})'
                            f'\nATE weights: {w_ate[a_idx, ta_idx, :]}')
                if w_ate_only:
                    sumw[a_idx, ta_idx] = 1
                    if gen_cfg.with_output:
                        txt += 'ATE weights are all zero.'
                    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                else:
                    txt += (f'\nATE weights: {w_ate[a_idx, ta_idx, :]}'
                            'ATE weights are all zero. Not good.'
                            ' Redo statistic without this variable.'
                            ' \nOr try to use more bootstraps.'
                            ' \nOr Sample may be too small.'
                            '\nOr Problem may be with CBGATE only.')
                    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                    raise RuntimeError(txt)
            if not continuous:
                w_ate[a_idx, ta_idx, :] /= sumw[a_idx, ta_idx]

            if p_ba_cfg.yes:
                w_ate[a_idx, ta_idx, :] = bias_correction_wols(
                    w_ate[a_idx, ta_idx, :],
                    ba_data,
                    int_dtype=np.float64, out_dtype=np.float32,
                    pos_weights_only=p_ba_cfg.pos_weights_only,
                    zero_tol=int_cfg.zero_tol,
                    )
            if not p_cfg.ate_no_se_only:
                w_ate_export[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :]

            if (p_cfg.max_weight_share < 1) and not continuous:
                if iv:
                    (w_ate[a_idx, ta_idx, :], _, share
                     ) = mcf_gp.bound_norm_weights_not_one(
                         w_ate[a_idx, ta_idx, :], p_cfg.max_weight_share,
                         zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                         )
                else:
                    (w_ate[a_idx, ta_idx, :], _, share
                     ) = mcf_gp.bound_norm_weights(
                         w_ate[a_idx, ta_idx, :], p_cfg.max_weight_share,
                         zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                         negative_weights_possible=p_ba_cfg.yes,
                         )
                if gen_cfg.with_output:
                    txt += ('\nShare of weights censored at'
                            f'{p_cfg.max_weight_share*100:8.3f}%: '
                            f'{share*100:8.3f}%  ATE type: {a_idx:2} '
                            f'Treatment: {ta_idx:2}'
                            )
    if w_ate_only:
        return w_ate_export, None, None, None
    if continuous:
        # Use the larger grid for estimation. This means that the 'missing
        # weights have to generated from existing weights (linear interpol.)
        i_w01, i_w10 = ct_cfg.w_to_dr_int_w01, ct_cfg.w_to_dr_int_w10
        index_full = ct_cfg.w_to_dr_index_full
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
    else:
        no_of_treat_dr = None

    no_of_treat_1dim = no_of_treat_dr if continuous else no_of_treat
    pot_y = np.empty((no_of_ates, no_of_treat_1dim, no_of_out))
    if p_cfg.ate_no_se_only:
        pot_y_var = None
    else:
        pot_y_var = np.empty_like(pot_y)
    if p_cfg.cluster_std:
        cl_dat = weights_dic['cl_dat_np']
        if p_cfg.se_boot_ate < 1:
            w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim,
                                   len(np.unique(cl_dat))))
        else:
            w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim, n_y))
    else:
        cl_dat = None
        w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim, n_y))
    # Normalize weights
    for a_idx in range(no_of_ates):
        for t_idx in range(no_of_treat):
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            w_ate_cont = w_ate[a_idx, t_idx, :]
                        else:
                            w_ate_cont = (w10 * w_ate[a_idx, t_idx, :]
                                          + w01 * w_ate[a_idx, t_idx+1, :])
                        w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
                        if p_cfg.max_weight_share < 1:
                            if iv:
                                (w_ate_cont, _, share
                                 ) = mcf_gp.bound_norm_weights_not_one(
                                     w_ate_cont, p_cfg.max_weight_share,
                                     zero_tol=int_cfg.zero_tol,
                                     sum_tol=int_cfg.sum_tol,
                                     )
                            else:
                                (w_ate_cont, _, share
                                 ) = mcf_gp.bound_norm_weights(
                                     w_ate_cont, p_cfg.max_weight_share,
                                     zero_tol=int_cfg.zero_tol,
                                     sum_tol=int_cfg.sum_tol,
                                     negative_weights_possible=p_ba_cfg.yes,
                                     )
                        ret = mcf_est.weight_var(
                            w_ate_cont, y_dat[:, o_idx], cl_dat, gen_cfg,
                            p_cfg, weights=w_dat,
                            bootstrap=p_cfg.se_boot_ate,
                            keep_all=int_cfg.keep_w0,
                            se_yes=not p_cfg.ate_no_se_only,
                            zero_tol = int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y[a_idx, ti_idx, o_idx] = ret[0]
                        if not p_cfg.ate_no_se_only:
                            pot_y_var[a_idx, ti_idx, o_idx] = ret[1]
                        if o_idx == 0:
                            w_ate_1dim[a_idx, ti_idx, :] = (
                                ret[2] if p_cfg.cluster_std else w_ate_cont
                                )
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_ate[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_cfg, p_cfg, weights=w_dat,
                        bootstrap=p_cfg.se_boot_ate,
                        keep_all=int_cfg.keep_w0,
                        se_yes=not p_cfg.ate_no_se_only,
                        normalize=not iv,
                        zero_tol = int_cfg.zero_tol,
                        sum_tol = int_cfg.sum_tol,
                        )
                    pot_y[a_idx, t_idx, o_idx] = ret[0]
                    if not p_cfg.ate_no_se_only:
                        pot_y_var[a_idx, t_idx, o_idx] = ret[1]
                    if o_idx == 0:
                        w_ate_1dim[a_idx, t_idx, :] = (
                            ret[2] if p_cfg.cluster_std
                            else w_ate[a_idx, t_idx, :]
                            )
    if gen_cfg.with_output and print_output:
        if continuous:
            no_of_treat, d_values = no_of_treat_dr, d_values_dr
        if not balancing_test:
            (_, _, _, _, _, _, _, _, _, txt_weight) = mcf_est.analyse_weights(
                w_ate_1dim[0, :, :], 'Weights to compute ATE', gen_cfg, p_cfg,
                ate=True, continuous=continuous, no_of_treat_cont=no_of_treat,
                d_values_cont=d_values, iv=iv,
                zero_tol=int_cfg.zero_tol,
                )
            txt += txt_weight

    return w_ate_export, pot_y, pot_y_var, txt


def ate_effects_print(mcf_: 'ModifiedCausalForest',
                      effect_dic: dict,
                      y_pred_lc: NDArray[Any],
                      balancing_test: bool = False,
                      pred_alloc: bool = False,
                      extra_title: str = ''
                      ) -> tuple[NDArray[Any], NDArray[Any], list]:
    """Compute ate's from potential outcomes and print them."""
    gen_cfg, ct_cfg, p_cfg = mcf_.gen_cfg, mcf_.ct_cfg, mcf_.p_cfg
    var_cfg, int_cfg = mcf_.var_cfg, mcf_.int_cfg
    y_pot, y_pot_var = effect_dic['y_pot'], effect_dic['y_pot_var']
    no_of_ates = y_pot.shape[0]
    if isinstance(y_pred_lc, (pd.Series, pd.DataFrame)):
        lc_yes = True
        y_pred_lc_ate = np.mean(y_pred_lc, axis=0)
    else:
        lc_yes, y_pred_lc_ate = False, None
    continuous = gen_cfg.d_type == 'continuous'
    d_values = ct_cfg.d_values_dr_np if continuous else gen_cfg.d_values
    no_of_treat = len(d_values)
    y_name = var_cfg.x_name_balance_test if balancing_test else var_cfg.y_name

    if gen_cfg.with_output:
        txt = '\n' * 2 + '=' * 100
        if balancing_test:
            txt += '\nBalancing Tests\n' + '-' * 100
        elif pred_alloc:
            txt += '\nAll are allocated into 1 treatment\n' + '-' * 100
        else:
            txt += ('\nAverage Treatment Effects Estimation '
                    + extra_title + '\n' + '-' * 100)
        txt += '\n'
        if pred_alloc:
            txt += 'Average Outcome'
        else:
            txt += 'Potential outcomes'
        txt += '\n' + '-' * 100
        if p_cfg.se_boot_ate > 1:
            txt += ('\nBootstrap standard errors with '
                    '{p_cfg.se_boot_ate:<6} replications')
        for o_idx, out_name in enumerate(y_name):
            txt += '\nOutcome variable: ' + out_name
            for a_idx in range(no_of_ates):
                if a_idx == 0:
                    txt += '    Reference population: All'
                else:
                    txt += ('\n   Reference population: Treatment group:'
                            f' {d_values[a_idx-1]}')
                if pred_alloc:
                    txt += '\nTreatment    Average Outcome  SE of AO     '
                else:
                    if mcf_.p_cfg.ate_no_se_only:
                        txt += '\nTreatment  Potential Outcome'
                    else:
                        txt += '\nTreatment  Potential Outcome  SE of PO     '
                if lc_yes:
                    txt += '  Uncentered Outcome'
                for t_idx in range(no_of_treat):
                    txt += '\n' + (f'{d_values[t_idx]:>10.5f} '
                                   if continuous else f'{d_values[t_idx]:<9} ')
                    txt += f' {y_pot[a_idx, t_idx, o_idx]:>12.6f}'
                    if not mcf_.p_cfg.ate_no_se_only:
                        sqrt_var = np.sqrt(y_pot_var[a_idx, t_idx, o_idx])
                        txt += f'   {sqrt_var:>12.6f}'
                    if lc_yes:
                        y_adjust = (y_pot[a_idx, t_idx, o_idx]
                                    + y_pred_lc_ate.iloc[o_idx])
                        txt += f'       {y_adjust:>12.6f}'
        if pred_alloc:
            txt += '\n' + '-' * 100 + '\nComparison of all-in-1 allocations'
        else:
            txt += '\n' + '-' * 100 + '\nTreatment effects (ATE, ATETs)'
        txt += '\n' + '-' * 100
    if continuous:  # only comparison to zero
        ate = np.empty((len(y_name), no_of_ates, no_of_treat - 1))
    else:
        ate = np.empty((len(y_name), no_of_ates,
                        round(no_of_treat * (no_of_treat - 1) / 2)))
    ate_se = np.empty_like(ate)
    if balancing_test and gen_cfg.with_output:
        ate_t = np.empty_like(ate)

    old_filters = warnings.filters.copy()
    warnings.filterwarnings('error', category=RuntimeWarning)
    for o_idx, out_name in enumerate(y_name):
        if gen_cfg.with_output:
            txt += f'\nOutcome variable: {out_name}'
        for a_idx in range(no_of_ates):
            if gen_cfg.with_output:
                if a_idx == 0:
                    txt += '   Reference population: All'
                    label_ate = 'ATE'
                else:
                    txt += ('   Reference population: Treatment group '
                            f'{d_values[a_idx-1]}')
                    label_ate = 'ATET' + str(d_values[a_idx-1])
                txt += '\n' + '- ' * 50
            pot_y_ao = y_pot[a_idx, :, o_idx]
            if mcf_.p_cfg.ate_no_se_only:
                pot_y_var_ao = None
            else:
                pot_y_var_ao = y_pot_var[a_idx, :, o_idx]
            (est, stderr, t_val, p_val, effect_list
             ) = mcf_est.effect_from_potential(
                pot_y_ao, pot_y_var_ao, d_values, continuous=continuous,
                se_yes=not mcf_.p_cfg.ate_no_se_only
                )
            ate[o_idx, a_idx], ate_se[o_idx, a_idx] = est, stderr
            if balancing_test and gen_cfg.with_output:
                ate_t[o_idx, a_idx] = t_val
            if gen_cfg.with_output:
                txt += mcf_ps.print_effect(est, stderr, t_val, p_val,
                                           effect_list,
                                           continuous=continuous)
                mcf_ps.effect_to_csv(
                    est, stderr, t_val, p_val, effect_list,
                    path=p_cfg.paths['ate_iate_fig_pfad_csv'],
                    label=label_ate+out_name
                    )
                txt += mcf_ps.print_se_info(p_cfg.cluster_std, p_cfg.se_boot_ate
                                            )
                if continuous and not balancing_test and a_idx == 0:
                    dose_response_figure(out_name, var_cfg.d_name[0], est,
                                         stderr, d_values[1:], int_cfg, p_cfg,
                                         with_output=gen_cfg.with_output
                                         )
    warnings.filters = old_filters
    # warnings.resetwarnings()
    if balancing_test and gen_cfg.with_output:
        average_t = np.mean(ate_t)
        txt += '\nVariables investigated for balancing test:'
        for name in var_cfg.x_name_balance_test:
            txt += f' {name}'
        txt += '\n' + '- ' * 50 + '\n' + 'Balancing test summary measure'
        txt += ' (average t-value of ATEs):'
        txt += f' {average_t:6.2f}' + '\n' + '-' * 100
    if gen_cfg.with_output:
        mcf_ps.print_mcf(gen_cfg, txt, summary=True, non_summary=False)
        mcf_ps.print_mcf(gen_cfg, effect_dic['txt_weights'] + txt,
                         summary=False)

    return ate, ate_se, effect_list


def get_data_for_final_ate_estimation(data_df: pd.DataFrame,
                                      gen_cfg: Any,
                                      p_cfg: dict,
                                      var_cfg: Any,
                                      ate: bool = True,
                                      need_count: bool = False
                                      ) -> tuple[NDArray[Any],
                                                 NDArray[Any],
                                                 NDArray[Any],
                                                 int
                                                 ]:
    """Get data needed to compute final weight based estimates.

    Parameters
    ----------
    data_df : DataFrame. Contains prediction data.
    var_cfg : VarCfg dataclass. Variables.
    p_cfg : Dict. Parameters.
    ate : Boolean, optional. ATE or GATE estimation. Default is True (ATE).
    need_count: Boolean. Need to count number of observations.

    Returns
    -------
    d_at : Numpy array. Treatment.
    z_dat : Numpy array. Heterogeneity variables.
    w_dat : Numpy array. External sampling weights.
    obs: Int. Number of observations.

    """
    obs = len(data_df.index) if need_count else None
    w_dat = (data_df[var_cfg.w_name].to_numpy()     # pylint: disable=E1136
             if gen_cfg.weighted else None)
    if ((var_cfg.d_name[0] in data_df.columns)
            and (p_cfg.atet or p_cfg.gatet or p_cfg.choice_based_sampling)):
        d_dat = np.int32(np.round(
            data_df[var_cfg.d_name].to_numpy()))  # pylint: disable=E1136
    else:
        d_dat = None
    if (ate is False) and (not var_cfg.z_name == []):
        z_dat = data_df[var_cfg.z_name].to_numpy()  # pylint: disable=E1136
    else:
        z_dat = None

    return d_dat, z_dat, w_dat, obs


def dose_response_figure(y_name: str, d_name: str,
                         effects: NDArray[Any],
                         stderr: NDArray[Any],
                         d_values: list | NDArray[Any],
                         int_cfg: Any,
                         p_cfg: dict,
                         with_output: bool = True,
                         ) -> None:
    """Plot the average dose response curve."""
    titel = 'Dose response relative to non-treated: ' + y_name + ' ' + d_name
    file_title = 'DR_rel_treat0' + y_name + d_name
    cint = norm.ppf(p_cfg.ci_level + 0.5 * (1 - p_cfg.ci_level))
    upper = effects + stderr * cint
    lower = effects - stderr * cint
    label_ci = f'{p_cfg.ci_level:2.0%}-CI'
    label_m, label_0, line_0 = 'ADR', '_nolegend_', '_-k'
    zeros = np.zeros_like(effects)
    file_name_jpeg = p_cfg.ate_iate_fig_pfad_jpeg / f'{file_title}.jpeg'
    file_name_pdf = p_cfg.ate_iate_fig_pfad_pdf / f'{file_title}.pdf'
    file_name_csv = p_cfg.ate_iate_fig_pfad_csv / f'{file_title}plotdat.csv'
    fig, axs = plt.subplots()
    axs.set_title(titel.replace('vs', ' vs '))
    axs.set_ylabel("Average dose response (relative to 0)")
    axs.set_xlabel('Treatment level')
    axs.plot(d_values, effects, label=label_m, color='b')
    axs.plot(d_values, zeros, line_0, label=label_0)
    axs.fill_between(d_values, upper, lower, alpha=0.3, color='b',
                     label=label_ci)
    axs.legend(loc=int_cfg.legend_loc, shadow=True, fontsize=int_cfg.fontsize)
    if with_output:
        mcf_sys.delete_file_if_exists(file_name_jpeg)
        mcf_sys.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=int_cfg.dpi)
        fig.savefig(file_name_pdf, dpi=int_cfg.dpi)
        if int_cfg.show_plots:
            plt.show()
        plt.close()

        upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
        effects = effects.reshape(-1, 1)
        d_values_np = np.array(d_values, copy=True).reshape(-1, 1)
        effects_et_al = np.concatenate((upper, effects, lower, d_values_np),
                                       axis=1)
        cols = ['upper', 'effects', 'lower', 'd_values']
        datasave = pd.DataFrame(data=effects_et_al, columns=cols)
        mcf_sys.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)
