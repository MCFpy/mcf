"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the ATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mcf import mcf_bias_adjustment as mcf_ba
from mcf import mcf_versions as mcf_tv
from mcf import mcf_estimation as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats as mcf_ps
from mcf.mcf_ate import get_data_for_final_ate_estimation
from mcf.mcf_effect_helpers import single_treatment_pos

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg, IntCfg, CtGrid, GenTvCfg, PBaCfg, VarCfg
    from mcf.mcf_init_predict import PCfg


def ate_weights(mcf_: 'ModifiedCausalForest',  *,
                data_df: pd.DataFrame,
                weights_dic: dict[str, Any],
                balancing_test: bool = False,
                w_ate_only: bool = False,
                with_output: bool = True,
                iv: bool = False,
                pred_alloc: bool = False,
                ) -> tuple[NDArray[Any], float]:
    """Accumulate weights for ATEs estimation.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame.
               Prediction data.
    w_ate : Numpy array.
               Weights, so far accumulated.
    weights_dic : Dict.
               Contains weights and numpy data.
    balancing_test: Boolean,  optional. Default is False.
    w_ate_only : Boolean, optional.
               Only weights are needed as output. Default is False.
    with_output : Boolean, optional.
               Output printed if True. Default is True.
    iv : Boolean, optional.
               Local average treatment effect estimation. True will prevent
               weights from being forced to be positive and normalized. Default is False.
    pred_alloc : Boolean, optional.
               Evaluate all-in-1-treatment allocations if True.

    Returns
    -------
    w_ate_1dim : Numpy array. Weights used for ATE computation.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.
    effect_list : List of strings with name of effects (same order as ATE)

    """
    time_start = time()
    gen_cfg, ct_cfg, int_cfg, gen_tv_cfg, _, zero_tol, sum_tol = _assign_cfg_tol(mcf_)

    if gen_cfg.with_output and gen_cfg.verbose and not w_ate_only and with_output:
        _print_titles(balancing_test, pred_alloc, 'Accumulating weights for')

    var_cfg, p_cfg, y_dat, w_dat = _change_cfg_dat(mcf_.var_cfg, mcf_.p_cfg, weights_dic, gen_cfg,
                                                   balancing_test,
                                                   )
    p_cfg.atet, p_cfg.gatet, _, no_of_treat, d_values = _continuous(gen_cfg, ct_cfg, p_cfg)

    d_p, _, w_p, n_p = get_data_for_final_ate_estimation(data_df, gen_cfg, p_cfg, var_cfg, ate=True,
                                                         need_count=int_cfg.weight_as_sparse or iv,
                                                         )
    n_y = len(y_dat)
    no_of_ates, p_cfg.atet = _ates(d_p, p_cfg, no_of_treat)
    t_probs = p_cfg.choice_based_probs

    # Step 1: Aggregate weights
    w_ate = np.zeros((no_of_ates, no_of_treat, n_y))

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
                    d_value = d_p[i, 0] if gen_tv_cfg.yes else d_p[i]
                    i_pos = single_treatment_pos(d_value, d_values, ind_d_val, zero_tol=zero_tol)
                    w_i_csr = w_i_csr.multiply(t_probs[i_pos])
                w_add[t_ind, :] = w_i_csr.todense()
            w_ate[0, :, :] += w_add
            if p_cfg.atet:
                d_value = d_p[i, 0] if gen_tv_cfg.yes else d_p[i]
                t_pos = single_treatment_pos(d_value, d_values, ind_d_val, zero_tol=zero_tol)
                w_ate[t_pos + 1, :, :] += w_add
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
                        txt = (f'\nZero weight. Index: {weight_i[t_ind][0]} '
                               f'd_value: {t_ind}\nWeights: {w_i}'
                               )
                        mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                        raise RuntimeError(txt)
                if not iv and (not (1-sum_tol) < sum_wi < (1+sum_tol)):
                    w_i = w_i / sum_wi
                if gen_cfg.weighted:
                    w_i = w_i * w_p[i]
                if p_cfg.choice_based_sampling:
                    d_value = d_p[i, 0] if gen_tv_cfg.yes else d_p[i]
                    i_pos = single_treatment_pos(d_value, d_values, ind_d_val, zero_tol=zero_tol)
                    w_add[t_ind, weight_i[t_ind][0]] = w_i * t_probs[i_pos]
                else:
                    w_add[t_ind, weight_i[t_ind][0]] = w_i
            w_ate[0, :, :] += w_add
            if p_cfg.atet:
                d_value = d_p[i, 0] if gen_tv_cfg.yes else d_p[i]
                t_pos = single_treatment_pos(d_value, d_values, ind_d_val, zero_tol=zero_tol)
                w_ate[t_pos + 1, :, :] += w_add

    return w_ate, time() - time_start


def ate_estimate_pot(mcf_: 'ModifiedCausalForest', *,
                     data_df: pd.DataFrame,
                     w_ate: NDArray | None,
                     weights_dic: dict[str, Any],
                     balancing_test: bool = False,
                     w_ate_only: bool = False,
                     with_output: bool = True,
                     iv: bool = False,
                     pred_alloc: bool = False,
                     ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], str, float]:
    """Estimate ATEs and their standard errors.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame. Prediction data.
    weights_dic : Dict.
               Contains last chunk of weights and numpy data (here only used to access data).
    balancing_test: Boolean,  optional. Default is False.
    w_ate_only : Boolean, optional.
               Only weights are needed as output. Default is False.
    with_output : Boolean, optional.
               Output printed if True. Default is True.
    iv : Boolean, optional.
               Local average treatment effect estimation. True will prevent
               weights from being forced to be positive and normalized. Default is False.
    pred_alloc : Boolean, optional.
               Evaluate all-in-1-treatment allocations if True.

    Returns
    -------
    w_ate_1dim : Numpy array. Weights used for ATE computation.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.
    effect_list : List of strings with name of effects (same order as ATE)
    dtime: time used.

    """
    time_start = time()
    gen_cfg, ct_cfg, int_cfg, gen_tv_cfg, p_ba_cfg, zero_tol, _ = _assign_cfg_tol(mcf_)
    results_container_w, results_container_r, txt = None, None, ''

    print_output = gen_cfg.with_output and not w_ate_only and with_output

    if print_output and gen_cfg.verbose:
        _print_titles(balancing_test, pred_alloc, 'Computing')

    var_cfg, p_cfg, y_dat, w_dat = _change_cfg_dat(mcf_.var_cfg, mcf_.p_cfg, weights_dic, gen_cfg,
                                                   balancing_test,
                                                   )
    p_cfg.atet, p_cfg.gatet, continuous, no_of_treat, d_values = _continuous(gen_cfg, ct_cfg, p_cfg)

    if gen_tv_cfg.yes:
        # Obtain the data for the later regression step (treatment versions)
        d_train, x_tv_train, x_tv_pred = mcf_tv.get_tv_data(
            data_df, weights_dic, mcf_.var_cfg, zero_tol=zero_tol,
            )
    else:
        d_train = x_tv_train = x_tv_pred = None
    if p_ba_cfg.yes:
        # Extract information for bias adjustment
        ba_data = mcf_ba.get_ba_data_prediction(weights_dic, p_ba_cfg)
        # No need for weighting if even if method is 'weighted_observable' because all score should
        # be included (for the ATE). However, weighting is needed for ATET.
    else:
        ba_data = None

    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)

    d_p, _, _, n_p = get_data_for_final_ate_estimation(data_df, gen_cfg, p_cfg, var_cfg, ate=True,
                                                       need_count=int_cfg.weight_as_sparse or iv,
                                                       )
    no_of_ates, p_cfg.atet = _ates(d_p, p_cfg, no_of_treat)

    sumw = np.sum(w_ate, axis=2)
    if iv:
        sumw = np.ones_like(sumw) * n_p

    if p_ba_cfg.yes and no_of_ates > 1 and p_ba_cfg.adj_method == 'w_obs':
        weights_eval = mcf_ba.get_weights_eval_ba(w_ate, no_of_treat, zero_tol=zero_tol)
    else:
        weights_eval = None

    if gen_tv_cfg.yes:
        # Expand to new treatment dimension; redefine d_values, no_of_treat
        no_of_treat, d_values, weights_eval, sumw, w_ate, _, _ = mcf_tv.expand_dimension(
            weights_eval, sumw, w_ate, None, None, gen_tv_cfg=gen_tv_cfg,
            )
        version_res_dat = np.zeros_like(w_ate)
    else:
        version_res_dat = None

    w_ate_export = None if p_cfg.ate_no_se_only else np.zeros_like(w_ate)

    for a_idx in range(no_of_ates):
        if p_ba_cfg.yes and weights_eval is not None:
            ba_data.weights_eval = weights_eval[a_idx, :].copy()

        # maintreat_idx, subtreat_idx = 0, 0 if gen_tv_cfg.yes else (None, None)
        if gen_tv_cfg.yes:
            maintreat_idx, subtreat_idx = 0, 0
        else:
            maintreat_idx, subtreat_idx = None, None

        for ta_idx in range(no_of_treat):
            if not iv and (-zero_tol < sumw[a_idx, ta_idx] < zero_tol):
                if print_output:
                    txt += (f'\nTreatment: {ta_idx}, ATE number: {a_idx})'
                            f'\nATE weights: {w_ate[a_idx, ta_idx, :]}'
                            )
                if w_ate_only:
                    sumw[a_idx, ta_idx] = 1
                    if gen_cfg.with_output:
                        txt += 'ATE weights are all zero.'
                    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                else:
                    txt += (f'\nATE weights: {w_ate[a_idx, ta_idx, :]} '
                            'ATE weights are all zero. Not good. Redo statistic without this '
                            'variable. '
                            '\nOr try to use more bootstraps.'
                            '\nOr Sample may be too small.'
                            '\nOr Problem may be with CBGATE only.'
                            )
                    mcf_ps.print_mcf(gen_cfg, txt, summary=True)
                    raise RuntimeError(txt)

            if gen_tv_cfg.yes:
                (w_ate[a_idx, ta_idx, :], results_container_w, res_tv, results_container_r,
                 maintreat_idx, subtreat_idx, txt_tv,
                 ) = mcf_tv.version_wregr(w_ate[a_idx, ta_idx, :],
                                          y_train=y_dat[:, 0], d_train=d_train, x_train=x_tv_train,
                                          x_pred=x_tv_pred,
                                          cfg=gen_tv_cfg,
                                          container_w=results_container_w,
                                          container_r=results_container_r,
                                          treat_idx=ta_idx,
                                          maintreat_idx=maintreat_idx, subtreat_idx=subtreat_idx,
                                          int_dtype=np.float64, out_dtype=np.float32,
                                          zero_tol=zero_tol,
                                          ridge=gen_tv_cfg.estimator == 'ridge',
                                          penalize_version
                                              =gen_tv_cfg.penalize_version[maintreat_idx],
                                          return_residuals=True, standardize_x=True,
                                          )
                txt += txt_tv
                if res_tv is not None:
                    version_res_dat[a_idx, ta_idx, :] = res_tv
            if not continuous:
                w_ate[a_idx, ta_idx, :] /= sumw[a_idx, ta_idx]

            if p_ba_cfg.yes:
                w_ate[a_idx, ta_idx, :] = mcf_ba.bias_correction_wregr(w_ate[a_idx, ta_idx, :],
                                                                       y_dat[:, 0],
                                                                       ba_data,
                                                                       int_dtype=np.float64,
                                                                       out_dtype=np.float32,
                                                                       pos_weights_only
                                                                         =p_ba_cfg.pos_weights_only,
                                                                       zero_tol=int_cfg.zero_tol,
                                                                       ridge=p_ba_cfg.ridge,
                                                                       cv_k=p_ba_cfg.cv_k,
                                                                       )
            if not p_cfg.ate_no_se_only:
                w_ate_export[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :]

            if (p_cfg.max_weight_share < 1) and not continuous:
                if iv:
                    w_ate[a_idx, ta_idx, :], _, share = mcf_gp.bound_norm_weights_not_one(
                         w_ate[a_idx, ta_idx, :], p_cfg.max_weight_share,
                         zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                         )
                else:
                    w_ate[a_idx, ta_idx, :], _, share = mcf_gp.bound_norm_weights(
                         w_ate[a_idx, ta_idx, :],
                         max_weight_share=p_cfg.max_weight_share,
                         zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                         negative_weights_possible=p_ba_cfg.yes or gen_tv_cfg.yes,
                         )
                if gen_cfg.with_output:
                    txt += (f'\nShare of weights censored at {p_cfg.max_weight_share*100:8.3f}%: '
                            f'{share*100:8.3f}%  ATE type: {a_idx:2} Treatment: {ta_idx:2}'
                            )
    if w_ate_only:
        return w_ate_export, None, None, None, time() - time_start
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
            w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim, len(np.unique(cl_dat))))
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
                                          + w01 * w_ate[a_idx, t_idx+1, :]
                                          )
                        w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
                        if p_cfg.max_weight_share < 1:
                            if iv:
                                w_ate_cont, _, share = mcf_gp.bound_norm_weights_not_one(
                                     w_ate_cont, p_cfg.max_weight_share,
                                     zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                     )
                            else:
                                w_ate_cont, _, share = mcf_gp.bound_norm_weights(
                                     w_ate_cont, max_weight_share=p_cfg.max_weight_share,
                                     zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                     negative_weights_possible=p_ba_cfg.yes or gen_tv_cfg.yes,
                                     )
                        vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                        ret = mcf_est.weight_var(w_ate_cont, y_dat[:, o_idx], cl_dat,  p_cfg,
                                                 residual_dat=vres, weighted=gen_cfg.weighted,
                                                 weights=w_dat, bootstrap=p_cfg.se_boot_ate,
                                                 keep_all=int_cfg.keep_w0,
                                                 se_yes=not p_cfg.ate_no_se_only,
                                                 zero_tol=int_cfg.zero_tol,
                                                 sum_tol=int_cfg.sum_tol, seed=123345, min_obs=5,
                                                 )
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y[a_idx, ti_idx, o_idx] = ret[0]
                        if not p_cfg.ate_no_se_only:
                            pot_y_var[a_idx, ti_idx, o_idx] = ret[1]
                        if o_idx == 0:
                            w_ate_1dim[a_idx, ti_idx, :] = (ret[2] if p_cfg.cluster_std
                                                            else w_ate_cont
                                                            )
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                    ret = mcf_est.weight_var(w_ate[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat, p_cfg,
                                             residual_dat=vres, weights=w_dat,
                                             weighted=gen_cfg.weighted, bootstrap=p_cfg.se_boot_ate,
                                             keep_all=int_cfg.keep_w0,
                                             se_yes=not p_cfg.ate_no_se_only, normalize=not iv,
                                             zero_tol = int_cfg.zero_tol, sum_tol = int_cfg.sum_tol,
                                             seed=123345, min_obs=5,
                                             )
                    pot_y[a_idx, t_idx, o_idx] = ret[0]
                    if not p_cfg.ate_no_se_only:
                        pot_y_var[a_idx, t_idx, o_idx] = ret[1]
                    if o_idx == 0:
                        w_ate_1dim[a_idx, t_idx, :] = (ret[2] if p_cfg.cluster_std
                                                       else w_ate[a_idx, t_idx, :]
                                                       )
    if print_output:
        if continuous:
            no_of_treat, d_values = no_of_treat_dr, d_values_dr
        if not balancing_test:
            ( _, _, _, _, _, _, _, _, _, txt_weight
             ) = mcf_est.analyse_weights(w_ate_1dim[0, :, :], 'Weights to compute ATE', gen_cfg,
                                        p_cfg,
                                        ate=True, continuous=continuous,
                                        no_of_treat_cont=no_of_treat,
                                        d_values_cont=d_values, iv=iv, zero_tol=int_cfg.zero_tol,
                                        )
            txt += txt_weight

    return w_ate_export, pot_y, pot_y_var, txt, time() - time_start


def _assign_cfg_tol(mcf_: 'ModifiedCausalForest'
                    ) -> tuple['GenCfg', 'CtGrid', 'IntCfg', 'GenTvCfg', 'PBaCfg', bool]:
    """Assign short-cuts for cfg instances."""
    return(mcf_.gen_cfg,
           mcf_.ct_cfg,
           mcf_.int_cfg,
           mcf_.gen_tv_cfg,
           mcf_.p_ba_cfg,
           mcf_.int_cfg.zero_tol,
           mcf_.int_cfg.sum_tol,
           )


def _print_titles(balancing_test: bool, pred_alloc: bool, txt: str) -> None:
    """Print titles."""
    if balancing_test:
        print(f'\n\n{txt} balancing tests (in ATE-like fashion)')
    elif pred_alloc:
        print(f'\n\n{txt} average outcomes for all-in-1-treatment allocations')
    else:
        print(f'\n\n{txt} ATEs')


def _change_cfg_dat(var_cfg_in: 'VarCfg',
                    p_cfg_in: 'PCfg',
                    weights_dic: dict[str, Any],
                    gen_cfg: 'GenCfg',
                    balancing_test: bool,
                    ) -> tuple['VarCfg', 'PCfg', 'GenCfg', NDArray, NDArray]:
    """Make some changes for non-ATE estimation and return data."""
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    if balancing_test:
        var_cfg, p_cfg = deepcopy(var_cfg_in), deepcopy(p_cfg_in)
        var_cfg.y_name = var_cfg.x_name_balance_test
        p_cfg.atet, p_cfg.gatet = False, False

        y_dat = weights_dic['x_bala_np']
    else:
        var_cfg, p_cfg = var_cfg_in, p_cfg_in
        y_dat = weights_dic['y_dat_np']

    return var_cfg, p_cfg, y_dat, w_dat


def _continuous(gen_cfg: 'GenCfg',
                ct_cfg: 'CtGrid',
                p_cfg: 'PCfg',
                ) -> tuple[bool, bool, bool, int, int | float]:
    """Change parameters to estimate for continuous treatments and return info."""
    if gen_cfg.d_type == 'continuous':
        continuous = True
        atet = gatet = False
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
    else:
        continuous = False
        atet, gatet = p_cfg.atet, p_cfg.gatet
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values

    return atet, gatet, continuous, no_of_treat, d_values


def _ates(d_p: NDArray, p_cfg: 'PCfg', no_of_treat: int) -> tuple[int, bool]:
    if (d_p is not None) and (p_cfg.atet or p_cfg.gatet):
        no_of_ates = no_of_treat + 1      # Compute ATEs, ATET, ATENT
        atet = p_cfg.atet
    else:
        atet, no_of_ates = False, 1

    return no_of_ates, atet
