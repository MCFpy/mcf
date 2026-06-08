"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from collections import namedtuple
from copy import deepcopy
# from itertools import chain, compress, repeat
from os import remove
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from mcf import mcf_effect_helpers as he
from mcf.mcf_ate import get_data_for_final_ate_estimation
from mcf.mcf_ate_low_memory import ate_weights
from mcf.mcf_bias_adjustment import (bias_correction_wregr, get_ba_data_prediction,
                                     get_weights_eval_ba,
                                     )
from mcf.mcf_estimation import analyse_weights, weight_var
from mcf.mcf_gate import w_gate_func, w_gate_cont_funct
from mcf.mcf_general import split_dataframe
from mcf.mcf_print_stats import del_added_chars, txt_weight_stat
from mcf.mcf_versions import joint_d_values, get_tv_data, expand_dimension, version_wregr
from mcf.mcf_weight import get_weights_mp

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import CtGrid, GenCfg, IntCfg, GenTvCfg, PBaCfg #, VarCfg
    from mcf.mcf_init_predict import PCfg
    from mcf.mcf_bias_adjustment import BaData

type ArrayLike = NDArray[Any] | None


def prepare_gate_context(mcf_: 'ModifiedCausalForest', *,
                         data_df: DataFrame,
                         gate_type: str = 'GATE',
                         paras_cbgate: tuple[dict, dict, bool, str] | None = None,
                         z_name_cbgate: list[str] | str | None = None,
                         ) -> he.GateContext:
    """Prepare full-sample gate context for low-memory chunk computation."""
    gen_cfg, int_cfg, p_cfg = mcf_.gen_cfg, mcf_.int_cfg, mcf_.p_cfg
    var_cfg, var_x_values = deepcopy(mcf_.var_cfg), deepcopy(mcf_.var_x_values)

    if gate_type not in ('GATE', 'CBGATE', 'BGATE'):
        raise ValueError(f'Wrong GATE type: {gate_type!r}')

    if gate_type != 'GATE':
        if paras_cbgate is None or z_name_cbgate is None:
            raise ValueError('paras_cbgate and z_name_cbgate required for BGATE/CBGATE.')
        var_cfg, var_x_values, smooth_yes, z_name_smooth = deepcopy(paras_cbgate)
        var_cfg.z_name = z_name_cbgate
    else:
        if p_cfg.gates_smooth:
            if isinstance(var_cfg.z_name, str):
                var_cfg.z_name = [var_cfg.z_name]
            var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                                var_x_values, p_cfg,
                                                                                )
        else:
            smooth_yes, z_name_smooth = False, None

    z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                 smooth_yes=smooth_yes,
                                                                 z_name_smooth=z_name_smooth,
                                                                 )
    var_cfg.z_name = z_name_l

    _, z_p, _, _ = get_data_for_final_ate_estimation(data_df, gen_cfg, p_cfg, var_cfg,
                                                     ate=False, need_count=False,
                                                     )
    kernel_bandwidth_l = [he.gate_kernel_bandwidth(z_p, z_name_j, z_smooth,
                                                   zero_tol=int_cfg.zero_tol,
                                                   bandwidth_factor=p_cfg.gates_smooth_bandwidth,
                                                   )
                          for z_name_j, z_smooth in enumerate(z_smooth_l)
                          ]
    return he.GateContext(z_name_l=z_name_l, z_values_l=z_values_l, z_smooth_l=z_smooth_l,
                          kernel_bandwidth_l=kernel_bandwidth_l, var_x_values=var_x_values,
                          smooth_yes=smooth_yes, z_name_smooth=z_name_smooth,
                          )


def gate_estimate_pot(mcf_: 'ModifiedCausalForest', *,
                      data_df: DataFrame,
                      w_gate_all: list[ArrayLike],
                      w_atemain: ArrayLike,
                      gate_type: str = 'GATE',
                      paras_cbgate: tuple[dict, dict, bool, str] | None = None,
                      z_name_cbgate: list[str] | str | None = None,
                      weights_dic_data: dict[str, Any],
                      gate_context: he.GateContext | None = None,
                      with_output: bool = True,
                      iv: bool = False,
                      ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,
                                 dict[str, Any], list[str], float,
                                 ]:
    # (pot_gate, pot_gate_var, pot_gate_mate, pot_gate_mate_var, gate_est_dic, txt_gate, dtime
    """Estime potential outcomes and their variances for GATEs."""
    start_time = time()
    if gate_type not in ('GATE', 'CBGATE', 'BGATE'):
        raise ValueError(f'Wrong GATE Type ({gate_type}) (Should be GATE, CBGATE, or BGATE')
    gen_cfg, ct_cfg, int_cfg, p_cfg= mcf_.gen_cfg, mcf_.ct_cfg, mcf_.int_cfg, mcf_.p_cfg
    p_ba_cfg, gen_tv_cfg = mcf_.p_ba_cfg, mcf_.gen_tv_cfg
    zero_tol, sum_tol = int_cfg.zero_tol, int_cfg.sum_tol

    txt = ''
    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print('\nComputing', gate_type)

    var_cfg, w_ate = deepcopy(mcf_.var_cfg), deepcopy(w_atemain)

    # Get training and prediction data
    y_dat = weights_dic_data['y_dat_np']
    w_dat = weights_dic_data['w_dat_np'] if gen_cfg.weighted else None
    cl_dat = weights_dic_data['cl_dat_np'] if p_cfg.cluster_std else None
    no_of_out = len(var_cfg.y_name)

    if gate_context is not None:
        var_x_values = deepcopy(gate_context.var_x_values)
        smooth_yes = gate_context.smooth_yes
        z_name_smooth = gate_context.z_name_smooth
        z_name_l = gate_context.z_name_l
        z_values_l = gate_context.z_values_l
        z_smooth_l = gate_context.z_smooth_l
        kernel_bandwidth_l = gate_context.kernel_bandwidth_l
        var_cfg.z_name = z_name_l

    elif gate_type != 'GATE':
        var_cfg, var_x_values, smooth_yes, z_name_smooth = deepcopy(paras_cbgate)
        var_cfg.z_name = z_name_cbgate

        z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                     smooth_yes=smooth_yes,
                                                                     z_name_smooth=z_name_smooth,
                                                                     )
        var_cfg.z_name = z_name_l
        kernel_bandwidth_l = None

    else:
        var_x_values = deepcopy(mcf_.var_x_values)
        if p_cfg.gates_smooth:
            if isinstance(var_cfg.z_name, str):
                var_cfg.z_name = [var_cfg.z_name]

            var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                                var_x_values, p_cfg,
                                                                                )
        else:
            smooth_yes, z_name_smooth = False, None

        z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                     smooth_yes=smooth_yes,
                                                                     z_name_smooth=z_name_smooth,
                                                                     )
        var_cfg.z_name = z_name_l
        kernel_bandwidth_l = None

    d_p, z_p, _, _ = get_data_for_final_ate_estimation(data_df, gen_cfg, p_cfg, var_cfg,
                                                       ate=False, need_count=False,
                                                         )
    if gen_tv_cfg.yes:
        # Obtain the data for the later regression step (treatment versions)
        d_train, x_tv_train, x_tv_pred = get_tv_data(data_df, weights_dic_data,
                                                     mcf_.var_cfg, zero_tol=zero_tol,
                                                     )
        # Namedtuple is easy to pass through all function and access it
        Tv_data = namedtuple('TV_data', ['d_train', 'x_tv_train', 'x_tv_pred'])
        gen_tv_data = Tv_data(d_train, x_tv_train, x_tv_pred,)
    else:
        gen_tv_data = None

    ba_data = get_ba_data_prediction(weights_dic_data, p_ba_cfg) if p_ba_cfg.yes else None

    if gen_cfg.d_type == 'continuous':
        continuous = True
        p_cfg.atet = p_cfg.gatet = False
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
    else:
        continuous = False
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        no_of_treat_dr, d_values_dr = no_of_treat, d_values

    treat_comp_label = he.make_treat_comp_label(d_values, continuous=continuous)

    ref_pop_lab = ['All']
    if p_cfg.gatet:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab.append(str(lab))

    if p_cfg.gatet and (d_p is not None):
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        p_cfg.gatet, no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]

    if p_cfg.gates_minus_previous:
        w_ate = None
    elif not iv and w_ate is not None:
        w_ate = he.normalize_ate_weights(w_ate,
                                         no_of_tgates=no_of_tgates, no_of_treat=no_of_treat,
                                         zero_tol=zero_tol, sum_tol=sum_tol,
                                         )
    files_to_delete = set()

    pot_all, pot_var_all, pot_mate_all, pot_mate_var_all, txt_all = [], [], [], [], []
    for z_name_j, z_name in enumerate(var_cfg.z_name):
        txt_z_name = txt + ' '
        if gen_cfg.with_output and gen_cfg.verbose and with_output:
            z_name_ = del_added_chars(z_name, prime=True)
            print(f'{z_name_j + 1} ({len(var_cfg.z_name)}) {z_name_}')
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]

        if kernel_bandwidth_l is None:
            kernel, bandw_z = he.gate_kernel_bandwidth(z_p, z_name_j, z_smooth,
                                                       zero_tol=int_cfg.zero_tol,
                                                       bandwidth_factor
                                                           =p_cfg.gates_smooth_bandwidth,
                                                       )
        else:
            kernel, bandw_z = kernel_bandwidth_l[z_name_j]

        no_of_zval = len(z_values)

        (pot, pot_var, pot_mate, pot_mate_var
         ) = he.init_gate_pot_arrays(no_of_zval=no_of_zval, no_of_tgates=no_of_tgates,
                                     no_of_treat_dr=no_of_treat_dr, no_of_out=no_of_out,
                                     gen_tv_yes=gen_tv_cfg.yes,
                                     no_of_treat_per_main=gen_tv_cfg.no_of_treat_per_main,
                                     )
        w_gate_j = w_gate_all[z_name_j].copy()
        w_gate_unc = np.zeros_like(w_gate_j)
        w_censored = np.zeros((no_of_zval, no_of_tgates, no_of_treat))

        # Post normalisation for smooth variables
        if z_smooth:
            for zj_idx, z_val in enumerate(z_values):
                smooth_norm_factor = he.gate_smooth_norm_factor(z_p[:, z_name_j], z_val,
                                                                bandwidth=bandw_z, kernel=kernel,
                                                                sum_tol=int_cfg.sum_tol,
                                                                )
                w_gate_j[zj_idx, :, :, :] *= smooth_norm_factor

        if gen_tv_cfg.yes:
            treat_main = gen_tv_cfg.no_of_treat_per_main
            w_gate_j = np.repeat(w_gate_j, repeats=treat_main, axis=2)
            w_gate_unc = np.repeat(w_gate_unc, repeats=treat_main, axis=2)
            w_censored = np.repeat(w_censored, repeats=treat_main, axis=2)

        for zj_idx in range(no_of_zval):
            if p_cfg.gates_minus_previous:
                if zj_idx > 0:
                    w_ate = w_gate_unc[zj_idx-1, :, :, :]
                else:
                    w_ate = w_gate_unc[zj_idx, :, :, :]
            results_fut_zj = gate_zj_pot(z_val=z_values[zj_idx], y_dat=y_dat, cl_dat=cl_dat,
                                         w_dat=w_dat, z_p=z_p, z_name_j=z_name_j,
                                         w_gate_zj=w_gate_j[zj_idx, :, :, :],
                                         w_gate_unc_zj=w_gate_unc[zj_idx, :, :, :],
                                         w_censored_zj=w_censored[zj_idx, :, :],
                                         w_ate=w_ate,
                                         pot_zj=pot[zj_idx, :, :, :],
                                         pot_var_zj=pot_var[zj_idx, :, :, :],
                                         pot_mate_zj=pot_mate[zj_idx, :, :, :],
                                         pot_mate_var_zj=pot_mate_var[zj_idx, :, :, :],
                                         no_of_tgates=no_of_tgates, no_of_out=no_of_out,
                                         ct_cfg=ct_cfg, gen_cfg=gen_cfg, int_cfg=int_cfg,
                                         p_cfg=p_cfg, p_ba_cfg=p_ba_cfg, ba_data=ba_data,
                                         gen_tv_cfg=gen_tv_cfg, gen_tv_data=gen_tv_data,
                                         bandw_z=bandw_z, kernel=kernel, smooth_it=z_smooth,
                                         continuous=continuous, iv=iv,
                                         )
            pot, pot_var, pot_mate, pot_mate_var = he.assign_pot(
                 pot, pot_var, pot_mate, pot_mate_var,
                 results_fut_zj=results_fut_zj, zj_idx=zj_idx,
                 )
            w_gate_j, w_gate_unc, w_censored = he.assign_w(w_gate_j, w_gate_unc, w_censored,
                                                           results_fut_zj, zj_idx,
                                                           )
        # Describe weights
        if gen_cfg.with_output:
            for a_idx in range(no_of_tgates):
                w_st = np.zeros((6, no_of_treat))
                share_largest_q = np.zeros((no_of_treat, 3))
                sum_larger = np.zeros((no_of_treat, len(p_cfg.q_w)))
                obs_larger = np.zeros_like(sum_larger)
                if gen_tv_cfg.yes:
                    w_censored_all = np.zeros(w_censored.shape[2])
                else:
                    w_censored_all = np.zeros(no_of_treat)
                for zj_idx in range(no_of_zval):
                    ret = analyse_weights(w_gate_j[zj_idx, a_idx, :, :], None, gen_cfg, p_cfg,
                                          ate=False, continuous=continuous,
                                          no_of_treat_cont=no_of_treat, d_values_cont=d_values,
                                          iv=iv, zero_tol=int_cfg.zero_tol,
                                          )
                    for idx in range(6):
                        w_st[idx] += ret[idx] / no_of_zval
                    share_largest_q += ret[6] / no_of_zval
                    sum_larger += ret[7] / no_of_zval
                    obs_larger += ret[8] / no_of_zval
                    w_censored_all += w_censored[zj_idx, a_idx, :]
                if gate_type == 'GATE':
                    if z_name in mcf_.data_train_dict['prime_values_dict']:
                        z_name_label = mcf_.data_train_dict['prime_values_dict'][z_name]
                    else:
                        z_name_label = z_name
                    txt_z_name += '\n' + '=' * 100
                    txt_z_name += ('\nAnalysis of weights (normalised to add to 1): '
                                   f'{gate_type} for {z_name_label} '
                                   f'(stats are averaged over {no_of_zval} groups).'
                                   )
                    if p_cfg.gatet:
                        txt += f'\nTarget population: {ref_pop_lab[a_idx]:<4}'
                    txt_z_name += txt_weight_stat(nonzero=w_st[0], equalzero=w_st[1],
                                                  mean_nonzero=w_st[2], std_nonzero=w_st[3],
                                                  gini_all=w_st[4], gini_nonzero=w_st[5],
                                                  share_largest_q=share_largest_q,
                                                  sum_larger=sum_larger, obs_larger=obs_larger,
                                                  gen_cfg=gen_cfg, p_cfg=p_cfg,
                                                  share_censored=w_censored_all,
                                                  continuous=continuous, d_values_cont=d_values,
                                                  )  # Discretized weights if cont
            txt_z_name += '\n'

        pot_all.append(pot)
        pot_var_all.append(pot_var)
        pot_mate_all.append(pot_mate)
        pot_mate_var_all.append(pot_mate_var)
        txt_all.append(txt_z_name)
        if files_to_delete:  # delete temporary files
            for file in files_to_delete:
                remove(file)
    var_x_values_unord_org = mcf_.data_train_dict['unique_values_dict']

    est_dic = {'continuous': continuous,
               'd_values': d_values,
               'd_values_dr': d_values_dr,
               'treat_comp_label': treat_comp_label,
               'no_of_out': no_of_out,
               'var_cfg': var_cfg,
               'var_x_values': var_x_values,
               'var_x_values_unord_org': var_x_values_unord_org,
               'smooth_yes': smooth_yes,
               'z_name_smooth': z_name_smooth,
               'ref_pop_lab': ref_pop_lab,
               'z_p': z_p,
               'no_of_tgates': no_of_tgates,
               'p_cfg': p_cfg,
               }
    if gen_tv_cfg.yes:
        est_dic['d_values_dr'] = joint_d_values(gen_tv_cfg.d_dict['main_treat'],
                                                gen_tv_cfg.d_dict['sub_treat'],
                                                as_string=False,
                                                )
        est_dic['d_values'] = est_dic['d_values_dr']
        est_dic['treat_comp_label'] = he.make_treat_comp_label(est_dic['d_values_dr'],
                                                               continuous=continuous,
                                                               )
    return (pot_all, pot_var_all, pot_mate_all, pot_mate_var_all, est_dic, txt_all,
            time() - start_time,
            )


def gate_zj_pot( *, z_val: int | float,
                y_dat: NDArray[Any],
                cl_dat: ArrayLike,
                w_dat: ArrayLike,
                z_p: NDArray[Any],
                z_name_j: str,
                w_gate_zj: NDArray[Any],
                w_gate_unc_zj: ArrayLike,
                w_censored_zj: ArrayLike,
                w_ate: ArrayLike,
                pot_zj: NDArray[Any],
                pot_var_zj: ArrayLike,
                pot_mate_zj: ArrayLike,
                pot_mate_var_zj: ArrayLike,
                no_of_tgates: int,
                no_of_out: int,
                ct_cfg: 'CtGrid', gen_cfg: 'GenCfg', int_cfg: 'IntCfg', p_cfg: 'PCfg',
                p_ba_cfg: 'PBaCfg', ba_data: 'BaData',
                gen_tv_cfg: 'GenTvCfg', gen_tv_data: Any,
                bandw_z: float,
                kernel: int,
                smooth_it: bool = False,
                continuous: bool = False,
                iv: bool = False,
                ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, None,
                           ArrayLike,
                           ]:
    """Compute Gates and their variances for MP."""
    w_diff, results_container_w, results_container_r = None, None, None
    gen_tv, p_ba = gen_tv_cfg.yes, p_ba_cfg.yes

    if continuous:
        no_of_treat = ct_cfg.ct_grid_w
        i_w01 = ct_cfg.ct_w_to_dr_int_w01
        i_w10 = ct_cfg.ct_w_to_dr_int_w10
        index_full = ct_cfg.ct_w_to_dr_index_full
    else:
        no_of_treat = gen_cfg.no_of_treat

    _, relevant_z,  _ = he.get_w_rel_z(z_p[:, z_name_j], z_val,
                                       weights_all=None, smooth_it=smooth_it, bandwidth=bandw_z,
                                       kernel=kernel, w_is_csr=int_cfg.weight_as_sparse,
                                       sum_tol=int_cfg.sum_tol,
                                       normalize_smooth_z=False,
                                       relevant_data_points_only=True,
                                       )
    if gen_tv:
        d_tv_train = gen_tv_data.d_train
        x_tv_train = gen_tv_data.x_tv_train
        x_tv_pred = None if x_tv_train is None else gen_tv_data.x_tv_pred[relevant_z, :]

    # Bias adjustment (optional)
    if p_ba:
        if p_ba_cfg.adj_method == 'w_obs':
            weights_eval = get_weights_eval_ba(w_gate_zj, no_of_treat, zero_tol=int_cfg.zero_tol)
        else:
            weights_eval = None

        for a_idx in range(no_of_tgates):
            if p_ba_cfg.adj_method == 'w_obs':
                ba_data.weights_eval = weights_eval[a_idx, :].copy()

            for t_idx in range(no_of_treat):
                w_gate_zj[a_idx, t_idx, :] = bias_correction_wregr(w_gate_zj[a_idx, t_idx, :],
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
    else:
        weights_eval = None

    # Get potential outcomes for particular z_value
    sum_wgate = None if continuous else np.sum(w_gate_zj, axis=2)
    if iv and sum_wgate is not None:
        sum_wgate = np.full_like(sum_wgate, fill_value=len(z_p), dtype=float)

    if gen_tv:
        # Expand to new treatment dimension; redefine d_values, no_of_treat
        no_of_treat, _, _, _, _, _, _, = expand_dimension(None, None, None, None, None,
                                                          gen_tv_cfg=gen_tv_cfg,
                                                          )
        version_res_dat = np.zeros_like(w_ate)

        for a_idx in range(no_of_tgates):
            maintreat_idx, subtreat_idx = 0, 0
            for t_idx in range(no_of_treat):
                (w_gate_zj[a_idx, t_idx, :], results_container_w, res_tv, results_container_r,
                 maintreat_idx, subtreat_idx, _
                 ) = version_wregr(w_gate_zj[a_idx, t_idx, :],
                                   y_train=y_dat[:, 0], d_train=d_tv_train, x_train=x_tv_train,
                                   x_pred=x_tv_pred,
                                   cfg=gen_tv_cfg,
                                   container_w=results_container_w, container_r=results_container_r,
                                   treat_idx=t_idx,
                                   maintreat_idx=maintreat_idx, subtreat_idx=subtreat_idx,
                                   int_dtype=np.float64, out_dtype=np.float32,
                                   zero_tol=int_cfg.zero_tol,
                                   ridge=gen_tv_cfg.estimator == 'ridge',
                                   penalize_version=gen_tv_cfg.penalize_version[maintreat_idx],
                                   return_residuals=True, standardize_x=True,
                                   )
                if res_tv is not None:
                    version_res_dat[a_idx, t_idx, :] = res_tv
    else:
        version_res_dat = None
    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                (w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj
                 ) = w_gate_func(a_idx, t_idx, sum_wgate[a_idx, t_idx],
                                 w_gate_zj=w_gate_zj, w_censored_zj=w_censored_zj,
                                 w_gate_unc_zj=w_gate_unc_zj, w_ate=w_ate, p_cfg=p_cfg,
                                 with_output=gen_cfg.with_output, p_ba_cfg_yes=p_ba,
                                 gen_tv_cfg_yes=gen_tv, iv=iv, zero_tol=int_cfg.zero_tol,
                                 sum_tol=int_cfg.sum_tol,
                                 )
            for o_idx in range(no_of_out):
                if continuous:  # Continuous treatment
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj, w_censored_zj
                         ) = w_gate_cont_funct(t_idx, a_idx,
                                               no_of_treat=no_of_treat, w_gate_zj=w_gate_zj,
                                               w10=w10, w01=w01, i=i, w_gate_unc_zj=w_gate_unc_zj,
                                               w_censored_zj=w_censored_zj,
                                               max_weight_share=int_cfg.max_weight_share,
                                               iv=iv, zero_tol = int_cfg.zero_tol,
                                               sum_tol=int_cfg.sum_tol,
                                               )
                        vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                        ret = weight_var(w_gate_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                         residual_dat=vres,
                                         normalize=True,
                                         weighted=gen_cfg.weighted, weights=w_dat,
                                         bootstrap=p_cfg.se_boot_gate, keep_all=int_cfg.keep_w0,
                                         zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                         seed=123345, min_obs=5,
                                         )
                        ti_idx = index_full[t_idx, i]                     # pylint: disable=E1136
                        pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if gen_cfg.with_output:
                            w_diff_cont = he.w_diff_cont_func(t_idx, a_idx,
                                                              no_of_treat=no_of_treat,
                                                              w_gate_cont=w_gate_cont_unc,
                                                              w_ate=w_ate, w10=w10, w01=w01,
                                                              )
                            vres = (None if version_res_dat is None
                                    else version_res_dat[a_idx, t_idx, :]
                                    )
                            ret2 = weight_var(w_diff_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                              residual_dat=vres,
                                              normalize=False,
                                              weighted=gen_cfg.weighted, weights=w_dat,
                                              bootstrap=p_cfg.se_boot_gate,
                                              keep_all=int_cfg.keep_w0, zero_tol=int_cfg.zero_tol,
                                              sum_tol=int_cfg.sum_tol, seed=123345, min_obs=5,
                                              )
                            pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                    ret = weight_var(w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat, p_cfg,
                                     residual_dat=vres,
                                     normalize=not iv,
                                     weighted=gen_cfg.weighted, weights=w_dat,
                                     bootstrap=p_cfg.se_boot_gate,
                                     keep_all=int_cfg.keep_w0, zero_tol=int_cfg.zero_tol,
                                     sum_tol=int_cfg.sum_tol, seed=123345, min_obs=5,
                                     )
                    pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if gen_cfg.with_output:
                        ret2 = weight_var(w_diff, y_dat[:, o_idx], cl_dat, p_cfg,
                                          residual_dat=None,
                                          normalize=False,
                                          weighted=gen_cfg.weighted, weights=w_dat,
                                          bootstrap=p_cfg.se_boot_gate, keep_all=int_cfg.keep_w0,
                                          zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                          seed=123345, min_obs=5,
                                          )
                        pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]

    return (pot_zj, pot_var_zj, pot_mate_zj, pot_mate_var_zj, w_gate_zj, w_gate_unc_zj,
            None, w_censored_zj,   # None added to have common format used by assign_w
            )


def gate_weights(mcf_: 'ModifiedCausalForest', *,
                 data_df: DataFrame,
                 weights_dic: dict[str, Any],
                 gate_type: str = 'GATE',
                 paras_cbgate: tuple[dict, dict, bool, str] | None = None,
                 z_name_cbgate: list[str] | str | None = None,
                 gate_context: he.GateContext | None = None,
                 with_output: bool = True,
                 iv: bool = False,
                 ) -> tuple[list[NDArray[Any]], float]:
    """Estimate weights für chunk of prediction data for GATEs."""
    time_start = time()
    gen_cfg, ct_cfg, int_cfg, p_cfg = mcf_.gen_cfg, mcf_.ct_cfg, mcf_.int_cfg, mcf_.p_cfg
    var_cfg, var_x_values = deepcopy(mcf_.var_cfg), deepcopy(mcf_.var_x_values)

    if gate_type not in ('GATE', 'CBGATE', 'BGATE'):
        raise ValueError(f'Wrong GATE type: {gate_type!r}')

    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print(f'\nComputing {gate_type}s')

    weights_all = weights_dic['weights']
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    n_y = len(weights_dic['y_dat_np'])

    if gate_context is not None:
        var_x_values = deepcopy(gate_context.var_x_values)
        smooth_yes = gate_context.smooth_yes
        z_name_smooth = gate_context.z_name_smooth
        z_name_l = gate_context.z_name_l
        z_values_l = gate_context.z_values_l
        z_smooth_l = gate_context.z_smooth_l
        kernel_bandwidth_l = gate_context.kernel_bandwidth_l

    elif gate_type != 'GATE':
        if paras_cbgate is None or z_name_cbgate is None:
            raise ValueError('paras_cbgate and z_name_cbgate required for BGATE/CBGATE.')
        var_cfg, var_x_values, smooth_yes, z_name_smooth = deepcopy(paras_cbgate)
        var_cfg.z_name = z_name_cbgate
        z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                     smooth_yes=smooth_yes,
                                                                     z_name_smooth=z_name_smooth,
                                                                     )
        kernel_bandwidth_l = None

    else:
        if p_cfg.gates_smooth:
            if isinstance(var_cfg.z_name, str):
                var_cfg.z_name = [var_cfg.z_name]
            var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                                var_x_values, p_cfg,
                                                                                )
        else:
            smooth_yes, z_name_smooth = False, None

        z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                     smooth_yes=smooth_yes,
                                                                     z_name_smooth=z_name_smooth,
                                                                     )
        kernel_bandwidth_l = None

    var_cfg.z_name = z_name_l

    # Get prediction data for chunk
    d_p, z_p, w_p, _ = get_data_for_final_ate_estimation(data_df,
                                                         gen_cfg, p_cfg, var_cfg,
                                                         ate=False, need_count=False,
                                                         )
    if d_p is not None:
        d_p = d_p if d_p.ndim == 1 else d_p[:, 0].reshape(-1)   # Only main treatments relevant

    if gen_cfg.d_type == 'continuous':
        continuous = True
        p_cfg.atet = p_cfg.gatet = False
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
    else:
        continuous = False
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values

    i_d_val = np.arange(no_of_treat)
    ref_pop_lab = ['All']
    if p_cfg.gatet:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab.append(str(lab))

    if p_cfg.gatet and (d_p is not None):
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        p_cfg.gatet, no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = p_cfg.choice_based_probs
    w_gate_all = []
    for z_name_j, _ in enumerate(var_cfg.z_name):
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]

        if kernel_bandwidth_l is None:
            kernel, bandw_z = he.gate_kernel_bandwidth(z_p, z_name_j, z_smooth,
                                                       zero_tol=int_cfg.zero_tol,
                                                       bandwidth_factor
                                                           =p_cfg.gates_smooth_bandwidth,
                                                       )
        else:
            kernel, bandw_z = kernel_bandwidth_l[z_name_j]

        no_of_zval = len(z_values)
        w_gate0_dim = (no_of_treat, n_y)
        w_gate_j = np.zeros((no_of_zval, no_of_tgates, no_of_treat, n_y))
        for zj_idx in range(no_of_zval):
            w_gate_j[zj_idx, :, :, :] = gate_zj_weights(z_val=z_values[zj_idx],
                                                        w_dat=w_dat,
                                                        z_p=z_p, d_p=d_p, w_p=w_p,
                                                        z_name_j=z_name_j,
                                                        weights_all=weights_all,
                                                        w_gate0_dim=w_gate0_dim,
                                                        w_gate_zj=w_gate_j[zj_idx, :, :, :],
                                                        i_d_val=i_d_val,
                                                        t_probs=t_probs,
                                                        ct_cfg=ct_cfg, gen_cfg=gen_cfg,
                                                        int_cfg=int_cfg, p_cfg=p_cfg,
                                                        bandw_z=bandw_z, kernel=kernel,
                                                        smooth_it=z_smooth, continuous=continuous,
                                                        iv=iv,
                                                        )
        w_gate_all.append(w_gate_j)

    return w_gate_all, time() - time_start


def bgate_est_low_memory(mcf_: 'ModifiedCausalForest', *,
                         data_df: DataFrame,
                         forest_dic: dict,
                         gate_type: str = 'CBGATE',
                         iv_tuple: Any = None,
                         ) -> tuple[list[NDArray[Any]], list[NDArray[Any]],
                                    list[NDArray[Any]], list[NDArray[Any]],
                                    dict[str, Any], list[str], str]:
    """BGATE and CBGATE estimation with low memory consumption."""
    if gate_type not in ('CBGATE', 'BGATE'):
        raise ValueError(f'Wrong gate_type: {gate_type!r}')

    if iv_tuple is not None:
        raise NotImplementedError('Low-memory BGATE/CBGATE IV path not implemented yet.')

    mcf_wgt = deepcopy(mcf_)
    mcf_wgt.gen_cfg.mp_parallel = 1
    mcf_wgt.gen_cfg.with_output = False

    gen_cfg, int_cfg = mcf_.gen_cfg, mcf_.int_cfg
    x_name_mcf= mcf_.cf_cfg.x_name_mcf
    p_cfg, var_cfg = deepcopy(mcf_.p_cfg), deepcopy(mcf_.var_cfg)
    var_x_values = deepcopy(mcf_.var_x_values)
    var_x_type = deepcopy(mcf_.var_x_type)
    bgate = gate_type == 'BGATE'

    if gen_cfg.with_output:
        txt_1 = '\n' + '=' * 100 + f'\nComputing {gate_type}'
        print(txt_1)

    if isinstance(var_cfg.z_name, str):
        var_cfg.z_name = [var_cfg.z_name]
    if not var_cfg.z_name:
        raise ValueError(f'{gate_type}: No heterogeneity variables specified.')

    if bgate and not var_cfg.x_name_balance_bgate:
        raise ValueError('BGATE: No balancing variables specified.')

    if p_cfg.gatet:
        p_cfg.gatet = p_cfg.atet = False
        if gen_cfg.with_output:
            txt_1 += f'\nNo treatment specific effects for {gate_type}.'

    if p_cfg.gates_smooth:
        var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                            var_x_values, p_cfg,
                                                                            )
    else:
        smooth_yes, z_name_smooth = False, None

    eva_values = he.ref_vals_cbgate(data_df, var_x_type, var_x_values,
                                    no_eva_values=p_cfg.gate_no_evalu_points,
                                    )
    y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all, txt_all = [], [], [], [], []
    text_sim_all = ''
    first_run = True
    gate_est_dic_all = None

    for v_j, vname in enumerate(var_cfg.z_name):  # Here one may use multiprocessing
        if gen_cfg.with_output:
            txt = txt_1[:]
            if gen_cfg.verbose:
                z_name_ = del_added_chars(vname, prime=True)
                print(f'{v_j + 1} ({len(var_cfg.z_name)}) {z_name_}')
        else:
            txt = ''
        if vname not in x_name_mcf:
            raise ValueError(f'Heterogeneity variable {vname} for {gate_type} NOT used for '
                             'splitting. Add to splitting variables.'
                             )
        if bgate:
            (data_df_new, z_values, matches, txt_sim
             ) = he.ref_data_bgate(data_df, vname, p_cfg, eva_values,
                                   deepcopy(var_cfg.x_name_balance_bgate),
                                   with_output_verbose=gen_cfg.with_output and gen_cfg.verbose,
                                   zero_tol=int_cfg.zero_tol,
                                   )
            var_dupl_mult = he.adjust_var_mult_duplicates(matches)
        else:
            (data_df_new, z_values, txt_sim
             ) = he.ref_data_cbgate(data_df, vname, p_cfg, eva_values,
                                    with_output_verbose=gen_cfg.with_output and gen_cfg.verbose,
                                    )
            var_dupl_mult = None

        vname_ = del_added_chars(vname, prime=True)
        text_sim_all += f'\n{vname_}: ' + txt_sim
        txt += txt_sim
        var_x_values[vname] = z_values[:]
        paras_cbgate_v = (var_cfg, var_x_values, smooth_yes, z_name_smooth)
        gate_context_v = prepare_gate_context(mcf_, data_df=data_df_new, gate_type=gate_type,
                                              paras_cbgate=paras_cbgate_v, z_name_cbgate=[vname],
                                              )
        data_df_split, _ = split_dataframe(data_df_new,
                                           max_chunk_size=mcf_.low_mem_cfg.max_chunksize,
                                           reset_index=True,
                                           )
        w_ate_v = w_gate_v = weights_dic_data_v = None

        for data_df_chunk in data_df_split:
            weights_dic_chunk, _ = get_weights_mp(data_df_chunk, forest_dic,
                                                  reg_round=True, cf_cfg=mcf_wgt.cf_cfg,
                                                  ct_cfg=mcf_wgt.ct_cfg, gen_cfg=mcf_wgt.gen_cfg,
                                                  int_cfg=mcf_wgt.int_cfg,
                                                  gen_tv_cfg_yes=mcf_wgt.gen_tv_cfg.yes,
                                                  p_cfg=p_cfg, var_cfg=var_cfg,
                                                  print_progress=False, with_output=False,
                                                  )
            w_ate_chunk, _ = ate_weights(mcf_wgt, data_df=data_df_chunk,
                                         weights_dic=weights_dic_chunk, balancing_test=False,
                                         w_ate_only=False, with_output=False, iv=mcf_wgt.gen_cfg.iv,
                                         pred_alloc=False,
                                         )
            w_gate_chunk, _ = gate_weights(mcf_wgt, data_df=data_df_chunk,
                                           weights_dic=weights_dic_chunk, gate_type=gate_type,
                                           paras_cbgate=paras_cbgate_v, z_name_cbgate=[vname],
                                           gate_context=gate_context_v,
                                           with_output=False, iv=mcf_wgt.gen_cfg.iv,
                                           )
            w_ate_v, _ = he.accumulate_weights(w_ate_v, w_ate_chunk)

            if w_gate_v is None:
                w_gate_v = [None] * len(w_gate_chunk)
            for jdx, gate_j in enumerate(w_gate_chunk):
                w_gate_v[jdx], _ = he.accumulate_weights(w_gate_v[jdx], gate_j)

            if weights_dic_data_v is None:
                weights_dic_data_v = weights_dic_chunk

        (y_pot_gate_z, y_pot_var_gate_z, y_pot_mate_gate_z, y_pot_mate_var_gate_z, gate_est_dic_z,
         txt_p, _) = gate_estimate_pot(mcf_,
                                       data_df=data_df_new, w_gate_all=w_gate_v, w_atemain=w_ate_v,
                                       gate_type=gate_type, paras_cbgate=paras_cbgate_v,
                                       z_name_cbgate=[vname],
                                       weights_dic_data=weights_dic_data_v,
                                       gate_context=gate_context_v,
                                       with_output=False, iv=False,
                                       )
        if bgate and var_dupl_mult is not None:
            try:
                y_pot_var_gate_z[0] *= var_dupl_mult
            except ValueError:
                pass
            try:
                y_pot_mate_var_gate_z[0] *= var_dupl_mult
            except ValueError:
                pass
        txt += txt_p[0]

        txt_all.append(txt)
        y_pot_all.append(y_pot_gate_z[0])
        y_pot_var_all.append(y_pot_var_gate_z[0])
        y_pot_mate_all.append(y_pot_mate_gate_z[0])
        y_pot_mate_var_all.append(y_pot_mate_var_gate_z[0])

        if first_run:
            gate_est_dic_all = deepcopy(gate_est_dic_z)
            gate_est_dic_all['z_p'] = np.full((len(gate_est_dic_z['z_p']), len(var_cfg.z_name)),
                                              np.nan,
                                              )
            gate_est_dic_all['var_cfg'].z_name = []
            first_run = False

        obs = len(gate_est_dic_z['z_p'])
        obs_old = gate_est_dic_all['z_p'].shape[0]

        if obs > obs_old:
            new_array = np.full((obs, len(var_cfg.z_name)), np.nan)
            new_array[:obs_old, :] = gate_est_dic_all['z_p']
            gate_est_dic_all['z_p'] = new_array

        gate_est_dic_all['z_p'][:obs, v_j] = gate_est_dic_z['z_p'].reshape(-1)
        gate_est_dic_all['var_cfg'].z_name.append(vname)

        gate_est_dic_all['var_x_values'][vname] = z_values[:]
        gate_est_dic_all['smooth_yes'] = smooth_yes
        gate_est_dic_all['z_name_smooth'] = z_name_smooth

    return (y_pot_all, y_pot_var_all,y_pot_mate_all, y_pot_mate_var_all, gate_est_dic_all, txt_all,
            text_sim_all,
            )

def gate_zj_weights(*, z_val: int | float,
                    w_dat: ArrayLike,
                    z_p: NDArray[Any],
                    d_p: ArrayLike,
                    w_p: ArrayLike,
                    z_name_j: str,
                    weights_all: list[list[NDArray[Any]]] | list[Any],
                    w_gate0_dim: tuple[int, int],
                    w_gate_zj: NDArray[Any],
                    i_d_val: NDArray[Any],
                    t_probs: NDArray[Any],
                    ct_cfg: 'CtGrid',
                    gen_cfg: 'GenCfg',
                    int_cfg: 'IntCfg',
                    p_cfg: 'PCfg',
                    bandw_z: float,
                    kernel: int,
                    smooth_it: bool = False,
                    continuous: bool = False,
                    iv: bool = False,
                    ) -> NDArray[Any]:
    """Compute weights for specific Gates."""
    zero_tol, sum_tol = int_cfg.zero_tol, int_cfg.sum_tol

    d_values = ct_cfg.ct_grid_w_val if continuous else gen_cfg.d_values

    weights, relevant_z,  w_z_val = he.get_w_rel_z(z_p[:, z_name_j], z_val,
                                                   weights_all=weights_all, smooth_it=smooth_it,
                                                   bandwidth=bandw_z, kernel=kernel,
                                                   w_is_csr=int_cfg.weight_as_sparse,
                                                   sum_tol=int_cfg.sum_tol,
                                                   normalize_smooth_z=False,
                                                   )
    d_p_z = d_p[relevant_z] if d_p is not None else None
    w_p_z = w_p[relevant_z] if gen_cfg.weighted and w_p is not None else None

    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)

    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_cfg.weight_as_sparse:
                weight_i = weights[t_idx][n_idx:n_idx + 1, :].tocsr()
                w_index = weight_i.indices.copy()
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_cfg.weighted:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if abs(w_i_sum) > zero_tol and not (1-sum_tol < w_i_sum < 1+sum_tol) and not iv:
                w_i = w_i / w_i_sum
            if gen_cfg.weighted:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_cfg.choice_based_sampling:
                if d_p_z is None:
                    raise ValueError('Choice-based sampling requires prediction treatments d_p.')
                i_pos = he.single_treatment_pos(d_p_z[n_idx], d_values, i_d_val, zero_tol=zero_tol)
                w_gadd[t_idx, w_index] = w_i * t_probs[i_pos]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()

        w_gate_zj[0, :, :] += w_gadd
        if p_cfg.gatet:
            if d_p_z is None:
                raise ValueError('GATET requires prediction treatments d_p.')
            t_pos = he.single_treatment_pos(d_p_z[n_idx], d_values, i_d_val, zero_tol=zero_tol)
            w_gate_zj[t_pos+1, :, :] += w_gadd

    return w_gate_zj
