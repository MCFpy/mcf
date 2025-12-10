"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from copy import deepcopy
from itertools import chain, compress, repeat
from os import remove
from typing import Any, NamedTuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import ray

from mcf import mcf_ate_functions as mcf_ate
from mcf.mcf_bias_adjustment_functions import (
    get_ba_data_prediction, bias_correction_wols, get_weights_eval_ba,
    )
from mcf import mcf_estimation_functions as mcf_est
from mcf.mcf_estimation_generic_functions import bandwidth_nw_rule_of_thumb
from mcf.mcf_estimation_generic_functions import kernel_proc
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_weight_functions as mcf_w
from mcf import mcf_iv_functions_add as mcf_iv_add

type ArrayLike = NDArray[Any] | None

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def gate_est(mcf_: 'ModifiedCausalForest',
             data_df: DataFrame,
             weights_dic: dict,
             w_atemain: ArrayLike,
             gate_type: str = 'GATE',
             z_name_cbgate: str = None,
             with_output: bool = True,
             paras_cbgate: tuple[dict, dict, bool, str] | None = None,
             iv: bool = False,
             ) -> tuple[list[NDArray[Any]], list[NDArray[Any]],
                        list[NDArray[Any]], list[NDArray[Any]],
                        dict, str
                        ]:
    """Estimate GATE(T)s, BGATE and CBGATE and their standard errors."""
    if gate_type not in ('GATE', 'CBGATE', 'BGATE'):
        raise ValueError(f'Wrong GATE specifified ({gate_type}). gate_type must'
                         'be on of the following: GATE, CBGATE, BGATE')
    gen_cfg, ct_cfg, int_cfg = mcf_.gen_cfg, mcf_.ct_cfg, mcf_.int_cfg
    p_cfg, var_cfg = mcf_.p_cfg, deepcopy(mcf_.var_cfg)
    p_ba_cfg = mcf_.p_ba_cfg
    sum_tol = int_cfg.sum_tol

    if gate_type != 'GATE':
        (var_cfg, var_x_values, smooth_yes, z_name_smooth) = deepcopy(
            paras_cbgate)
        var_cfg.z_name = z_name_cbgate
    else:
        var_x_values = deepcopy(mcf_.var_x_values)
        if p_cfg.gates_smooth:
            var_cfg, var_x_values, smooth_yes, z_name_smooth = addsmoothvars(
                data_df, var_cfg, var_x_values, p_cfg)
        else:
            smooth_yes, z_name_smooth = False, None
    var_x_type = deepcopy(mcf_.var_x_type)
    w_ate = deepcopy(w_atemain)
    txt = ''
    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print('\nComputing', gate_type)
    y_dat = weights_dic['y_dat_np']
    weights_all = weights_dic['weights']
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    cl_dat = weights_dic['cl_dat_np'] if p_cfg.cluster_std else None
    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)
    d_p, z_p, w_p, _ = mcf_ate.get_data_for_final_ate_estimation(
        data_df, gen_cfg, p_cfg, var_cfg, ate=False, need_count=False)

    if p_ba_cfg.yes:
        # Extract information for bias adjustment
        ba_data = get_ba_data_prediction(weights_dic, p_ba_cfg)
    else:
        ba_data = None

    z_type_l = [None for _ in range(len(var_cfg.z_name))]
    z_values_l = z_type_l[:]
    z_smooth_l = [False for _ in range(len(var_cfg.z_name))]

    if gen_cfg.d_type == 'continuous':
        continuous = True
        p_cfg.atet = p_cfg.gatet = False
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
        treat_comp_label = [None for _ in range(round(no_of_treat_dr - 1))]
    else:
        continuous = False
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
        treat_comp_label = [None for _ in range(
            round(no_of_treat * (no_of_treat - 1) / 2))]
    i_d_val = np.arange(no_of_treat)
    ref_pop_lab = ['All']
    if p_cfg.gatet:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab += str(lab)
    for zj_idx, z_name in enumerate(var_cfg.z_name):
        z_type_l[zj_idx] = var_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = var_x_values[z_name]
        if smooth_yes:
            z_smooth_l[zj_idx] = z_name in z_name_smooth
    if (d_p is not None) and p_cfg.gatet:
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        p_cfg.gatet, no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = p_cfg.choice_based_probs

    jdx = 0
    for t1_idx, t1_lab in enumerate(d_values):
        for t2_idx in range(t1_idx+1, no_of_treat):
            treat_comp_label[jdx] = str(d_values[t2_idx]) + 'vs' + str(t1_lab)
            jdx += 1
        if continuous:
            break
    if p_cfg.gates_minus_previous:
        w_ate = None
    else:
        if not iv:
            w_ate_sum = np.sum(w_ate, axis=2)
            for a_idx in range(no_of_tgates):  # Weights for ATE are normalized
                for t_idx in range(no_of_treat):
                    if not ((1-sum_tol) < w_ate_sum[a_idx, t_idx] < (1+sum_tol)
                            ):
                        w_ate[a_idx, t_idx, :] = (w_ate[a_idx, t_idx, :]
                                                  / w_ate_sum[a_idx, t_idx])
    files_to_delete, save_w_file = set(), None

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
        print('Number of parallel processes (GATE): ', maxworkers, flush=True)

    if maxworkers > 1:
        if not ray.is_initialized():
            mcf_sys.init_ray_with_fallback(
                maxworkers, int_cfg, gen_cfg,
                mem_object_store=int_cfg.mem_object_store_3,
                ray_err_txt='Ray did not start in in GATE estimation.'
                )
        if (int_cfg.mem_object_store_3 is not None
                and gen_cfg.with_output and gen_cfg.verbose):
            print('Size of Ray Object Store: ',
                  round(int_cfg.mem_object_store_3/(1024*1024)), " MB"
                  )
        weights_all_ref = ray.put(weights_all)
        y_dat_ref = ray.put(y_dat)
        z_p_ref = ray.put(z_p)
        if d_p is not None and len(d_p) > 1:
            d_p_ref = ray.put(d_p)
        else:
            d_p_ref = d_p
    else:
        weights_all_ref = y_dat_ref = z_p_ref = d_p_ref = None
    y_pot_all, y_pot_var_all, y_pot_mate_all = [], [], []
    y_pot_mate_var_all, txt_all = [], []
    for z_name_j, z_name in enumerate(var_cfg.z_name):
        txt_z_name = txt + ' '
        if gen_cfg.with_output and gen_cfg.verbose and with_output:
            z_name_ = mcf_ps.del_added_chars(z_name, prime=True)
            print(z_name_j + 1, '(', len(var_cfg.z_name), ')', z_name_,
                  flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        if z_smooth:
            kernel = 1  # Epanechikov
            bandw_z = bandwidth_nw_rule_of_thumb(z_p[:, z_name_j],
                                                 zero_tol=int_cfg.zero_tol
                                                 )
            bandw_z = bandw_z * p_cfg.gates_smooth_bandwidth
        else:
            kernel = bandw_z = None
        no_of_zval = len(z_values)
        y_pot = np.empty((no_of_zval, no_of_tgates, no_of_treat_dr, no_of_out))
        y_pot_var = np.empty_like(y_pot)
        y_pot_mate, y_pot_mate_var = np.empty_like(y_pot), np.empty_like(y_pot)
        w_gate = np.zeros((no_of_zval, no_of_tgates, no_of_treat, n_y))
        w_gate_unc = np.zeros_like(w_gate)
        w_censored = np.zeros((no_of_zval, no_of_tgates, no_of_treat))
        w_gate0_dim = (no_of_treat, n_y)
        if (maxworkers == 1) or p_cfg.gates_minus_previous:
            for zj_idx in range(no_of_zval):
                if p_cfg.gates_minus_previous:
                    if zj_idx > 0:
                        w_ate = w_gate_unc[zj_idx-1, :, :, :]
                    else:
                        w_ate = w_gate_unc[zj_idx, :, :, :]
                results_fut_zj = gate_zj(
                    z_values[zj_idx], zj_idx, y_dat, cl_dat, w_dat, z_p, d_p,
                    w_p, z_name_j, weights_all, w_gate0_dim,
                    w_gate[zj_idx, :, :, :], w_gate_unc[zj_idx, :, :, :],
                    w_censored[zj_idx, :, :], w_ate, y_pot[zj_idx, :, :, :],
                    y_pot_var[zj_idx, :, :, :], y_pot_mate[zj_idx, :, :, :],
                    y_pot_mate_var[zj_idx, :, :, :], i_d_val, t_probs,
                    no_of_tgates, no_of_out, ct_cfg, gen_cfg, int_cfg, p_cfg,
                    p_ba_cfg, ba_data,
                    bandw_z, kernel, z_smooth, continuous, iv)
                y_pot, y_pot_var, y_pot_mate, y_pot_mate_var = assign_pot(
                     y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                     results_fut_zj, zj_idx)
                w_gate, w_gate_unc, w_censored = assign_w(
                     w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx)
        else:
            still_running = [ray_gate_zj_mp.remote(
                     z_values[zj_idx], zj_idx, y_dat_ref, cl_dat,
                     w_dat, z_p_ref, d_p_ref, w_p, z_name_j, weights_all_ref,
                     w_gate0_dim, w_ate, i_d_val, t_probs,
                     no_of_tgates, no_of_out, ct_cfg, gen_cfg, int_cfg, p_cfg,
                     p_ba_cfg, ba_data,
                     n_y, bandw_z, kernel, save_w_file, z_smooth, continuous,
                     iv=iv, sum_tol=sum_tol)
                for zj_idx in range(no_of_zval)]
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for results_fut_idx in finished_res:
                    (y_pot, y_pot_var, y_pot_mate, y_pot_mate_var
                     ) = assign_pot(
                         y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                         results_fut_idx, results_fut_idx[6])
                    w_gate, w_gate_unc, w_censored = assign_w(
                        w_gate, w_gate_unc, w_censored, results_fut_idx,
                        results_fut_idx[6])
        if gen_cfg.with_output:
            # Describe weights
            for a_idx in range(no_of_tgates):
                w_st = np.zeros((6, no_of_treat))
                share_largest_q = np.zeros((no_of_treat, 3))
                sum_larger = np.zeros((no_of_treat, len(p_cfg.q_w)))
                obs_larger = np.zeros_like(sum_larger)
                w_censored_all = np.zeros(no_of_treat)
                for zj_idx in range(no_of_zval):
                    ret = mcf_est.analyse_weights(
                        w_gate[zj_idx, a_idx, :, :], None, gen_cfg, p_cfg,
                        ate=False,
                        continuous=continuous,
                        no_of_treat_cont=no_of_treat,
                        d_values_cont=d_values,
                        iv=iv,
                        zero_tol=int_cfg.zero_tol,
                        )
                    for idx in range(6):
                        w_st[idx] += ret[idx] / no_of_zval
                    share_largest_q += ret[6] / no_of_zval
                    sum_larger += ret[7] / no_of_zval
                    obs_larger += ret[8] / no_of_zval
                    w_censored_all += w_censored[zj_idx, a_idx, :]
                if gate_type == 'GATE':
                    if z_name in mcf_.data_train_dict['prime_values_dict']:
                        z_name_label = mcf_.data_train_dict['prime_values_dict'
                                                            ][z_name]
                    else:
                        z_name_label = z_name
                    txt_z_name += '\n' + '=' * 100
                    txt_z_name += (
                        '\nAnalysis of weights (normalised to add to 1): '
                        f'{gate_type} for {z_name_label} '
                        f'(stats are averaged over {no_of_zval} groups).')
                    if p_cfg.gatet:
                        txt += f'\nTarget population: {ref_pop_lab[a_idx]:<4}'
                    txt_z_name += mcf_ps.txt_weight_stat(
                        w_st[0], w_st[1], w_st[2], w_st[3], w_st[4], w_st[5],
                        share_largest_q, sum_larger, obs_larger, gen_cfg,
                        p_cfg, w_censored_all, continuous=continuous,
                        d_values_cont=d_values)  # Discretized weights if cont
            txt_z_name += '\n'
        y_pot_all.append(y_pot)
        y_pot_var_all.append(y_pot_var)
        y_pot_mate_all.append(y_pot_mate)
        y_pot_mate_var_all.append(y_pot_mate_var)
        txt_all.append(txt_z_name)
        if files_to_delete:  # delete temporary files
            for file in files_to_delete:
                remove(file)
    var_x_values_unord_org = mcf_.data_train_dict['unique_values_dict']
    gate_est_dic = {
        'continuous': continuous,
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
    if maxworkers > 1:
        if 'refs' in int_cfg.mp_ray_del:
            del weights_all_ref, y_dat_ref, z_p_ref, d_p_ref
        if 'rest' in int_cfg.mp_ray_del:
            del finished_res, finished

    return (y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all,
            gate_est_dic, txt_all
            )


def bgate_est(mcf_: 'ModifiedCausalForest',
              data_df: DataFrame,
              weights_dic: dict,
              w_ate: ArrayLike,
              forest_dic: dict,
              gate_type='CBGATE',
              iv_tuple=None
              ) -> tuple[list[NDArray[Any]], list[NDArray[Any]],
                         list[NDArray[Any]], list[NDArray[Any]],
                         dict, str, dict
                         ]:
    """Compute CBGATE & BGATE for single variables keeping others constant."""
    var_cfg, gen_cfg = mcf_.var_cfg, mcf_.gen_cfg
    var_x_type = deepcopy(mcf_.var_x_type)
    var_x_values = deepcopy(mcf_.var_x_values)
    p_cfg, var_cfg = deepcopy(mcf_.p_cfg), deepcopy(mcf_.var_cfg)
    x_name_mcf = mcf_.cf_cfg.x_name_mcf
    bgate = gate_type == 'BGATE'
    if var_cfg.z_name is None or var_cfg.z_name == []:
        raise ValueError(f'Something wrong with {var_cfg.z_name}')
    if bgate:
        if (var_cfg.x_name_balance_bgate is None
                or var_cfg.x_name_balance_bgate == []):
            raise ValueError(
                f'Something wrong with {var_cfg.x_name_balance_bgate}')
    txt = ''
    if gen_cfg.with_output:
        txt_1 = '\n' + '=' * 100 + f'\nComputing {gate_type}'
        print(txt_1)
    if p_cfg.gatet:
        p_cfg.gatet = p_cfg.atet = False
        if gen_cfg.with_output:
            txt_1 += f'\nNo treatment specific effects for {gate_type}.'
    # Add continues variables
    if p_cfg.gates_smooth:
        var_cfg, var_x_values, smooth_yes, z_name_smooth = addsmoothvars(
            data_df, var_cfg, var_x_values, p_cfg)
    else:
        smooth_yes, z_name_smooth = False, None
    eva_values = ref_vals_cbgate(data_df, var_x_type, var_x_values,
                                 no_eva_values=p_cfg.gate_no_evalu_points
                                 )
    if gen_cfg.with_output and gen_cfg.verbose:
        print(f'\n{gate_type} variable under investigation: ', end=' ')
    y_pot_all, y_pot_var_all, y_pot_mate_all, txt_all = [], [], [], []
    y_pot_mate_var_all = []
    first_run = True
    text_sim_all = ''
    for v_j, vname in enumerate(var_cfg.z_name):
        if gen_cfg.with_output:
            txt = txt_1[:]
        if vname not in x_name_mcf:
            raise ValueError(f'Heterogeneity variable for {gate_type} NOT used'
                             ' for splitting. Perhaps turn off {type_txt}.')
        if gen_cfg.with_output and gen_cfg.verbose:
            vname_ = mcf_ps.del_added_chars(vname, prime=True)
            print(mcf_ps.del_added_chars(vname_, prime=True), end=' ')
        if bgate:
            data_df_new, z_values, matches, txt_sim = ref_data_bgate(
                data_df.copy(), vname, p_cfg, eva_values,
                var_cfg.x_name_balance_bgate[:],
                with_output_verbose=(gen_cfg.with_output
                                     and gen_cfg.verbose
                                     ),
                )
        else:
            data_df_new, z_values, txt_sim = ref_data_cbgate(
                data_df.copy(), vname, p_cfg, eva_values,
                with_output_verbose=(gen_cfg.with_output
                                     and gen_cfg.verbose
                                     )
                )
        vname_ = mcf_ps.del_added_chars(vname, prime=True)
        text_sim_all += f'\n{vname_}: ' + txt_sim
        txt += txt_sim
        var_x_values[vname] = z_values[:]

        if bgate:
            # Compute correction factor for duplicates in matching
            var_dupl_mult = adjust_var_mult_duplicates(matches)

        if iv_tuple is None:
            iv = False
            weights_dic = mcf_w.get_weights_mp(mcf_, data_df_new, forest_dic,
                                               'regular', with_output=False)
        else:
            iv = True
            mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic = iv_tuple
            iate_1st_dic, _ = mcf_iv_add.iate_1st_stage_all_folds_rounds(
                mcf_, mcf_1st, data_df_new, True
                )
            # Compute weights of reduced form & 1st stage & final estimation
            _, _, weights_dic = mcf_iv_add.get_weights_iv_local(
                mcf_, mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic,
                iate_1st_dic, None, data_df_new, round_='regular',
                local_effects=True, no_1st_weights=True)

        (w_ate, _, _, _) = mcf_ate.ate_est(
            mcf_, data_df_new, weights_dic, with_output=False, iv=iv)
        (y_pot_gate_z, y_pot_var_gate_z, y_pot_mate_gate_z,
         y_pot_mate_var_gate_z, gate_est_dic_z, txt_p) = gate_est(
             mcf_, data_df_new, weights_dic, w_ate, gate_type=gate_type,
             z_name_cbgate=[vname], with_output=False,
             paras_cbgate=(var_cfg, var_x_values, smooth_yes, z_name_smooth),
             iv=iv,
             )

        if bgate:
            # Multiply variances (if they exist) with correction factor
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
            gate_est_dic_all['z_p'] = []
            gate_est_dic_all['var_cfg'].z_name = []
            first_run = False
            gate_est_dic_all['z_p'] = np.full(
                (len(gate_est_dic_z['z_p']), len(var_cfg.z_name)), np.nan)
        obs = len(gate_est_dic_z['z_p'])
        obs_old = len(gate_est_dic_all['z_p'])
        if obs > obs_old:
            new_array = np.full((obs, len(var_cfg.z_name)), np.nan)
            new_array[:obs_old, :] = gate_est_dic_all['z_p'].copy()
            gate_est_dic_all['z_p'] = new_array
        gate_est_dic_all['z_p'][:obs, v_j] = gate_est_dic_z['z_p'].reshape(-1)
        gate_est_dic_all['var_cfg'].z_name.append(vname)
    gate_est_dic_all['smooth_yes'] = smooth_yes
    gate_est_dic_all['z_name_smooth'] = z_name_smooth

    return (y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all,
            gate_est_dic_all, txt_all, text_sim_all
            )


def ref_data_cbgate(data_df: DataFrame,
                    z_name: str,
                    p_cfg: dict,
                    eva_values: dict,
                    with_output_verbose: bool = True,
                    ) -> tuple[DataFrame, dict, str]:
    """Create reference samples for covariates (CBGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = p_cfg.bgate_sample_share / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if with_output_verbose:
                txt += f' CBGATE: {share:5.2%} random sample drawn'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_all_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_all_df.loc[:, z_name] = new_values_z
    if with_output_verbose:
        txt += ('\nCBGATEs minus ATE are evaluated at fixed z-feature values'
                ' (equally weighted).'
                )
    return data_all_df, eva_values, txt


def ref_data_bgate(data_df: DataFrame,
                   z_name: str,
                   p_cfg: Any,
                   eva_values: dict,
                   x_name_balance_bgate: list[str],
                   with_output_verbose: bool = True,
                   ) -> tuple[DataFrame, dict, list[int, str]]:
    """Create reference samples for covariates (BGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if z_name in x_name_balance_bgate:
        x_name_balance_bgate.remove(z_name)
    if z_name.endswith('catv') and z_name[:-4] in x_name_balance_bgate:
        x_name_balance_bgate.remove(z_name[:-4])
    if z_name + 'catv' in x_name_balance_bgate:  # the continuous variable
        x_name_balance_bgate.remove(z_name + 'catv')
    if not x_name_balance_bgate:
        raise ValueError('BGATE {z_name}: No variables left for balancing.')
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = p_cfg.bgate_sample_share / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if with_output_verbose:
                txt += f' BGATE: {share:5.2%} random sample drawn.'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_new_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_new_df.loc[:, z_name] = new_values_z
    if with_output_verbose:
        txt += ('\nBGATEs are balanced with respect to '
                f'{" ".join(x_name_balance_bgate)}'
                '\nBGATEs minus ATE are evaluated at fixed z-feature values'
                ' (equally weighted).')
    # Until here these are the identical steps to CBGATE. Next, observations in
    # data_all_df get substituted by their nearest neighbours in terms of
    # x_name_balance_bgate conditional on the z_name variables.
    # Get duplicates
    data_new_b_np = data_new_df[x_name_balance_bgate].to_numpy(copy=True)
    data_new_z_np = data_new_df[z_name].to_numpy(copy=True)
    data_org_b_np = data_df[x_name_balance_bgate].to_numpy()
    data_org_z_np = data_df[z_name].to_numpy()
    data_org_np = data_df.to_numpy()
    if data_org_b_np.shape[1] > 1:
        bz_cov_inv = invcovariancematrix(data_org_b_np)
        bz_cov_inv[-1, -1] *= 10
    # Give much more additional weight to the z-related component in matching
    incr = 0.2 * np.std(data_new_z_np)
    collect_matches = []
    for idx, z_value in enumerate(data_new_z_np):
        z_true = data_org_z_np == z_value
        test_val = 0
        for _ in range(5):
            if np.count_nonzero(z_true) > 10:
                break
            test_val += incr
            lower_ok = data_org_z_np >= z_value - test_val
            upper_ok = data_org_z_np <= z_value + test_val
            lower_upper_ok = lower_ok & upper_ok
            z_true = np.zeros_like(data_org_z_np, dtype=bool)
            z_true[lower_upper_ok] = True
        if not np.any(z_true):
            z_true = np.ones_like(data_org_z_np, dtype=bool)
        data_org_np_condz = data_org_np[z_true]
        data_org_b_np_condz = data_org_b_np[z_true]
        diff = data_org_b_np_condz - data_new_b_np[idx, :]
        if data_org_b_np.shape[1] > 1:
            dist = np.sum((diff @ bz_cov_inv) * diff, axis=1)
        else:
            dist = diff**2
        match_neigbour_idx = np.argmin(dist)
        data_new_df.iloc[idx] = data_org_np_condz[match_neigbour_idx, :]
        collect_matches.append(match_neigbour_idx)

    return data_new_df, eva_values, collect_matches, txt


def invcovariancematrix(data_np: NDArray[Any]) -> NDArray[Any]:
    """Compute inverse of covariance matrix and adjust for missing rank."""
    k = np.shape(data_np)
    if k[1] > 1:
        cov_x = np.cov(data_np, rowvar=False)
        rank_not_ok, counter = True, 0
        while rank_not_ok:
            if counter == 20:
                cov_x *= np.eye(k[1])
            if counter > 20:
                cov_inv = np.eye(k[1])
                break
            if np.linalg.matrix_rank(cov_x) < k[1]:
                cov_x += 0.5 * np.diag(cov_x) * np.eye(k[1])
                counter += 1
            else:
                cov_inv = np.linalg.inv(cov_x)
                rank_not_ok = False

    return cov_inv


def ref_vals_cbgate(data_df: DataFrame,
                    var_x_type: dict,
                    var_x_values: dict,
                    no_eva_values: int = 50,
                    ) -> dict:
    """Compute reference values for moderated gates and balanced gates."""
    evaluation_values = {}
    obs = len(data_df)
    for vname in var_x_type.keys():
        ddf = data_df[vname]
        if not var_x_values[vname]:
            no_eva_values = min(no_eva_values, obs)
            quas = np.linspace(0.01, 0.99, no_eva_values)
            eva_val = np.unique(ddf.quantile(quas)).tolist()
        else:
            eva_val = var_x_values[vname].copy()
        evaluation_values.update({vname: eva_val})

    return evaluation_values


def addsmoothvars(data_df: DataFrame,
                  var_cfg: Any,
                  var_x_values: dict,
                  p_cfg: Any
                  ) -> tuple[dict, dict, bool, list[str]]:
    """
    Find variables for which to smooth gates and evaluation points.

    Parameters
    ----------
    data_df: DataFrame. Prediction data.
    var_cfg : VarCfg Dataclass. Variables.
    var_x_values : Dict. Variables
    p_cfg : PCfg. Controls.

    Returns
    -------
    var_cfg_new : VarCfg Dataclass. Updated variables.
    var_x_values_new : Dict. Updated with evaluation points.
    smooth_yes : Bool. Indicator if smoothing will happen.
    ...
    """
    smooth_yes, z_name = False, var_cfg.z_name
    z_name_add = [name[:-4] for name in z_name if (name.endswith('catv')
                                                   and (len(name) > 4))]
    if z_name_add:
        smooth_yes = True
        var_cfg_new = deepcopy(var_cfg)
        var_x_values_new = deepcopy(var_x_values)
        data_np = data_df[z_name_add].to_numpy()
        for idx, name in enumerate(z_name_add):
            var_x_values_new[name] = smooth_gate_eva_values(
                data_np[:, idx], p_cfg.gates_smooth_no_evalu_points
                )
            var_cfg_new.z_name.append(name)
    else:
        var_cfg_new, var_x_values_new = var_cfg, var_x_values

    return var_cfg_new, var_x_values_new, smooth_yes, z_name_add


def smooth_gate_eva_values(z_dat: NDArray[Any],
                           no_eva_values: dict
                           ):
    """
    Get the evaluation points.

    Parameters
    ----------
    z_dat : Numpy 1D array. Data.
    no_eva_values : Int.

    Returns
    -------
    eva_values : List of numpy.float. Evaluation values.

    """
    unique_vals = np.unique(z_dat)
    obs = len(unique_vals)
    if no_eva_values >= obs:
        return list(unique_vals)
    quas = np.linspace(0.01, 0.99, no_eva_values)

    return np.unique(np.quantile(z_dat, quas)).tolist()


@ray.remote
def ray_gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p, z_name_j,
                   weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
                   no_of_tgates, no_of_out, ct_cfg, gen_cfg, int_cfg, p_cfg,
                   p_ba_cfg, ba_data,
                   n_y, bandw_z, kernel, save_w_file=None, smooth_it=False,
                   continuous=False, iv=False, sum_tol=1e-12,
                   ):
    """Make function compatible with Ray."""
    return gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                      z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val,
                      t_probs, no_of_tgates, no_of_out, ct_cfg, gen_cfg,
                      int_cfg, p_cfg,
                      p_ba_cfg, ba_data,
                      n_y, bandw_z, kernel, save_w_file,
                      smooth_it, continuous, iv, sum_tol=sum_tol,
                      )


def gate_zj(z_val: int | float,
            zj_idx: int | np.integer,
            y_dat: NDArray[Any],
            cl_dat: ArrayLike,
            w_dat: ArrayLike,
            z_p: NDArray[Any],
            d_p: NDArray[Any],
            w_p: NDArray[Any],
            z_name_j: str,
            weights_all: list[list[NDArray[Any]]] | list[Any],
            w_gate0_dim: list[int, int],
            w_gate_zj: NDArray[Any],
            w_gate_unc_zj: NDArray[Any],
            w_censored_zj: NDArray[Any],
            w_ate: NDArray[Any],
            y_pot_zj: NDArray[Any],
            y_pot_var_zj: NDArray[Any],
            y_pot_mate_zj: NDArray[Any],
            y_pot_mate_var_zj: NDArray[Any],
            i_d_val: NDArray[Any],
            t_probs: NDArray[Any],
            no_of_tgates: int,
            no_of_out: int,
            ct_cfg: Any,
            gen_cfg: Any,
            int_cfg: Any,
            p_cfg: Any,
            p_ba_cfg: Any,
            ba_data: NamedTuple,
            bandw_z: float,
            kernel: int,
            smooth_it: bool = False,
            continuous: bool = False,
            iv: bool = False,
            ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any],
                       NDArray[Any], NDArray[Any], NDArray[Any],
                       int | np.integer,
                       NDArray[Any],
                       ]:
    """Compute Gates and their variances for MP."""
    w_diff = None
    sum_tol = int_cfg.sum_tol
    if continuous:
        no_of_treat, d_values = ct_cfg.ct_grid_w, ct_cfg.ct_grid_w_val
        i_w01 = ct_cfg.ct_w_to_dr_int_w01
        i_w10 = ct_cfg.ct_w_to_dr_int_w10
        index_full = ct_cfg.ct_w_to_dr_index_full
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
    weights, relevant_z,  w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=int_cfg.weight_as_sparse,
        sum_tol=int_cfg.sum_tol,
        )
    if p_cfg.gatet:
        d_p_z = d_p[relevant_z]
    if gen_cfg.weighted:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)
    # Step 1: Aggregate weights
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_cfg.weight_as_sparse:
                weight_i = weights[t_idx][n_idx, :]
                w_index = weight_i.col
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_cfg.weighted:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not (1-sum_tol < w_i_sum < 1+sum_tol) and not iv:
                w_i = w_i / w_i_sum
            if gen_cfg.weighted:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_cfg.choice_based_sampling:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if p_cfg.gatet:
            t_pos_i = i_d_val[d_p_z[n_idx] == d_values]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd

    # Bias adjustment (optional)
    if p_ba_cfg.yes:
        if p_ba_cfg.adj_method == 'w_obs':
            weights_eval = get_weights_eval_ba(w_gate_zj, no_of_treat,
                                               zero_tol=int_cfg.zero_tol
                                               )
        else:
            weights_eval = None

        for a_idx in range(no_of_tgates):
            if p_ba_cfg.adj_method == 'w_obs':
                ba_data.weights_eval = weights_eval[a_idx, :].copy()

            for t_idx in range(no_of_treat):
                w_gate_zj[a_idx, t_idx, :] = bias_correction_wols(
                    w_gate_zj[a_idx, t_idx, :],
                    ba_data,
                    int_dtype=np.float64, out_dtype=np.float32,
                    pos_weights_only=p_ba_cfg.pos_weights_only,
                    zero_tol=int_cfg.zero_tol,
                    )

    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    if iv:
        sum_wgate = np.ones_like(sum_wgate) * n_x

    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx], w_gate_zj,
                    w_censored_zj, w_gate_unc_zj, w_ate, p_cfg,
                    gen_cfg.with_output, p_ba_cfg_yes=p_ba_cfg.yes, iv=iv,
                    zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                    )
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             int_cfg.max_weight_share, iv=iv,
                             zero_tol = int_cfg.zero_tol,
                             sum_tol = int_cfg.sum_tol,
                             )
                        ret = mcf_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, gen_cfg,
                            p_cfg, weights=w_dat,
                            bootstrap=p_cfg.se_boot_gate,
                            keep_all=int_cfg.keep_w0,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if gen_cfg.with_output:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = mcf_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, gen_cfg,
                                p_cfg, normalize=False, weights=w_dat,
                                bootstrap=p_cfg.se_boot_gate,
                                keep_all=int_cfg.keep_w0,
                                zero_tol=int_cfg.zero_tol,
                                )
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_cfg, p_cfg, weights=w_dat,
                        bootstrap=p_cfg.se_boot_gate,
                        keep_all=int_cfg.keep_w0, normalize=not iv,
                        zero_tol=int_cfg.zero_tol,
                        )
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if gen_cfg.with_output:
                        ret2 = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat,
                            gen_cfg, p_cfg, normalize=False, weights=w_dat,
                            bootstrap=p_cfg.se_boot_gate,
                            keep_all=int_cfg.keep_w0,
                            zero_tol=int_cfg.zero_tol,
                            )
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]

    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj
            )


def gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
               z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
               no_of_tgates, no_of_out, ct_cfg, gen_cfg, int_cfg, p_cfg,
               p_ba_cfg, ba_data,
               n_y,
               bandw_z, kernel, save_w_file=None, smooth_it=False,
               continuous=False, iv=False, sum_tol: float = 1e-12
               ):
    """Compute Gates and their variances for MP."""
    w_diff = None
    if continuous:
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
        i_w01 = ct_cfg.w_to_dr_int_w01
        i_w10 = ct_cfg.w_to_dr_int_w10
        index_full = ct_cfg.w_to_dr_index_full
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    if save_w_file is not None:
        weights_all = mcf_sys.save_load(save_w_file, save=False,
                                        output=gen_cfg.with_output
                                        )
    w_gate_zj = np.zeros((no_of_tgates, no_of_treat, n_y))
    w_gate_unc_zj = np.zeros_like(w_gate_zj)
    w_censored_zj = np.zeros((no_of_tgates, no_of_treat))
    y_pot_zj = np.empty((no_of_tgates, no_of_treat_dr, no_of_out))
    y_pot_var_zj = np.empty_like(y_pot_zj)
    y_pot_mate_zj = np.empty_like(y_pot_zj)
    y_pot_mate_var_zj = np.empty_like(y_pot_zj)
    # Step 1: Aggregate weights
    weights, relevant_z, w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=int_cfg.weight_as_sparse,
        sum_tol=int_cfg.sum_tol,
        )
    if p_cfg.gatet:
        d_p_z = d_p[relevant_z]
    if gen_cfg.weighted:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_cfg.weight_as_sparse:
                weight_i = weights[t_idx][n_idx, :]
                w_index = weight_i.col
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_cfg.weighted:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not ((1-sum_tol) < w_i_sum < (1+sum_tol) or iv):
                w_i = w_i / w_i_sum
            if gen_cfg.weighted:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_cfg.choice_based_sampling:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if p_cfg.gatet:
            t_pos_i = i_d_val[d_p_z[n_idx] == d_values]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd

    # Bias adjustment (optional)
    if p_ba_cfg.yes:
        if p_ba_cfg.adj_method == 'w_obs':
            weights_eval = get_weights_eval_ba(w_gate_zj, no_of_treat,
                                               zero_tol=int_cfg.zero_tol
                                               )
        else:
            weights_eval = None

        for a_idx in range(no_of_tgates):
            if p_ba_cfg.adj_method == 'w_obs':
                ba_data.weights_eval = weights_eval[a_idx, :].copy()
            for t_idx in range(no_of_treat):
                w_gate_zj[a_idx, t_idx, :] = bias_correction_wols(
                    w_gate_zj[a_idx, t_idx, :],
                    ba_data,
                    int_dtype=np.float64, out_dtype=np.float32,
                    pos_weights_only=p_ba_cfg.pos_weights_only,
                    zero_tol=int_cfg.zero_tol,
                    )

    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    if iv:
        sum_wgate = np.ones_like(sum_wgate) * n_x
    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx], w_gate_zj,
                    w_censored_zj, w_gate_unc_zj, w_ate, p_cfg,
                    gen_cfg.with_output, p_ba_cfg_yes=p_ba_cfg.yes, iv=iv,
                    zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                    )
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             p_cfg.max_weight_share,
                             p_ba_cfg_yes= p_ba_cfg.yes, iv=iv,
                             zero_tol=int_cfg.zero_tol,
                             sum_tol=int_cfg.sum_tol,
                             )
                        ret = mcf_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, gen_cfg,
                            p_cfg, weights=w_dat,
                            bootstrap=p_cfg.se_boot_gate,
                            keep_all=int_cfg.keep_w0,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if gen_cfg.with_output:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = mcf_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, gen_cfg,
                                p_cfg, normalize=False, weights=w_dat,
                                bootstrap=p_cfg.se_boot_gate,
                                keep_all=int_cfg.keep_w0,
                                zero_tol=int_cfg.zero_tol,
                                sum_tol = int_cfg.sum_tol,
                                )
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_cfg, p_cfg, weights=w_dat,
                        bootstrap=p_cfg.se_boot_gate,
                        keep_all=int_cfg.keep_w0,
                        normalize=not iv,
                        zero_tol=int_cfg.zero_tol,
                        sum_tol = int_cfg.sum_tol,
                        )
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if gen_cfg.with_output:
                        ret2 = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat, gen_cfg,
                            p_cfg, normalize=False, weights=w_dat,
                            bootstrap=p_cfg.se_boot_gate,
                            keep_all=int_cfg.keep_w0,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol = int_cfg.sum_tol,
                            )
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]

    # if w_gate_zj.nbytes > 1e+9 and int_cfg.ray_or_dask != 'ray':
    #     # otherwise tuple gets too large for MP
    #     save_name_w = 'wtemp' + str(zj_idx) + '.npy'
    #     save_name_wunc = 'wunctemp' + str(zj_idx) + '.npy'
    #     np.save(save_name_w, w_gate_zj, fix_imports=False)
    #     np.save(save_name_wunc, w_gate_unc_zj, fix_imports=False)
    #     w_gate_zj = w_gate_unc_zj = None
    # else:
    #     save_name_w = save_name_wunc = None
    save_name_w = save_name_wunc = None

    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj, save_name_w,
            save_name_wunc)


def get_w_rel_z(z_dat: NDArray[Any],
                z_val: int | float,
                weights_all: list[list[NDArray[Any]]] | list[Any],
                smooth_it: bool,
                bandwidth: int | float = 1,
                kernel: int = 1,
                w_is_csr: bool = False,
                sum_tol: float = 1e-12,
                ) -> tuple[list[list[NDArray[Any]]] | list[Any],
                           NDArray[Any], NDArray[Any]
                           ]:
    """
    Get relevant observations and their weights.

    Parameters
    ----------
    z_dat : 1D Numpy array. Data.
    z_val : Int or float. Evaluation point.
    weights_all : List of lists of lists. MCF weights.
    smooth_it : Bool. Use smoothing (True) or select data.
    bandwidth : Float. Bandwidth for weights. Default is 1.
    kernel : Int. 1: Epanechikov. 2: Normal. Default is 1.
    w_is_csr : Boolean. If weights are saved as sparse csv matrix.

    Returns
    -------
    weights : List of list of list. Relevant observations.
    relevant_data_points : 1D Numpy array of Bool. True if data will be used.
    w_z_val : Numpy array. Weights.

    """
    if smooth_it:
        w_z_val = kernel_proc((z_dat - z_val) / bandwidth, kernel)
        relevant_data_points = w_z_val > sum_tol
        w_z_val = w_z_val[relevant_data_points]
        w_z_val = w_z_val / np.sum(w_z_val) * len(w_z_val)  # Normalise
    else:
        relevant_data_points = np.isclose(z_dat, z_val)  # Creates tuple
        w_z_val = None
    if w_is_csr:
        iterator = len(weights_all)
        weights = [weights_all[t_idx][relevant_data_points, :] for t_idx in
                   range(iterator)]
    else:
        weights = list(compress(weights_all, relevant_data_points))

    return weights, relevant_data_points, w_z_val


def w_gate_func(
        a_idx: int | np.integer,
        t_idx: int | np.integer,
        sum_wgate: float | np.floating,
        w_gate_zj: NDArray[Any],
        w_censored_zj: NDArray[Any],
        w_gate_unc_zj: NDArray[Any],
        w_ate: NDArray[Any],
        p_cfg: Any,
        with_output: bool = True,
        p_ba_cfg_yes: bool = False,
        iv: bool = False,
        zero_tol: float = 1e-15,
        sum_tol: float = 1e-12,
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any],]:
    """Compute weights for discrete case."""
    if iv or ((not 1-sum_tol < sum_wgate < 1+sum_tol) and sum_wgate > sum_tol):
        w_gate_zj[a_idx, t_idx, :] /= sum_wgate

    w_gate_unc_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :]

    if p_cfg.max_weight_share < 1:
        if iv:
            (w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx]
             ) = mcf_gp.bound_norm_weights_not_one(w_gate_zj[a_idx, t_idx, :],
                                                   p_cfg.max_weight_share,
                                                   zero_tol=zero_tol,
                                                   sum_tol=sum_tol,
                                                   )
        else:
            (w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx]
             ) = mcf_gp.bound_norm_weights(
                 w_gate_zj[a_idx, t_idx, :],
                 p_cfg.max_weight_share,
                 zero_tol=zero_tol,
                 sum_tol=sum_tol,
                 negative_weights_possible=p_ba_cfg_yes,
                 )
    if with_output and w_ate is not None:
        w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx, t_idx, :]
    else:
        w_diff = None

    return w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj


def w_gate_cont_funct(
        t_idx: int | np.integer,
        a_idx: int | np.integer,
        no_of_treat: int,
        w_gate_zj: NDArray[Any],
        w10: float | np.floating,
        w01: float | np.floating,
        i: int,
        w_gate_unc_zj: NDArray[Any],
        w_censored_zj: NDArray[Any],
        max_weight_share: float,
        p_ba_cfg_yes: bool = False,
        iv: bool = False,
        zero_tol: float = 1e-15,
        sum_tol: float = 1e-12,
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any],]:
    """Approximate weights for continuous treatments."""
    if t_idx == (no_of_treat - 1):  # last element, no inter
        w_gate_cont = w_gate_zj[a_idx, t_idx, :]
    else:
        w_gate_cont = (w10 * w_gate_zj[a_idx, t_idx, :]
                       + w01 * w_gate_zj[a_idx, t_idx+1, :])
    sum_wgate = np.sum(w_gate_cont)
    if not ((-sum_tol < sum_wgate < sum_tol)
            or (1-sum_tol < sum_wgate < 1+sum_tol)
            or iv
            ):
        w_gate_cont = w_gate_cont / sum_wgate
    if i == 0:
        w_gate_unc_zj[a_idx, t_idx, :] = w_gate_cont
    w_gate_cont_unc = w_gate_cont.copy()
    if max_weight_share < 1:
        if iv:
            w_gate_cont, _, w_censored = mcf_gp.bound_norm_weights_not_one(
                w_gate_cont, max_weight_share,
                zero_tol=zero_tol, sum_tol=sum_tol,
                )
        else:
            w_gate_cont, _, w_censored = mcf_gp.bound_norm_weights(
                w_gate_cont, max_weight_share,
                zero_tol=zero_tol, sum_tol=sum_tol,
                negative_weights_possible=p_ba_cfg_yes,
                )
        if i == 0:
            w_censored_zj[a_idx, t_idx] = w_censored

    return w_gate_cont, w_gate_cont_unc, w_gate_unc_zj, w_censored_zj


def w_diff_cont_func(t_idx: int | np.integer,
                     a_idx: int | np.integer,
                     no_of_treat: int,
                     w_gate_cont: NDArray[Any],
                     w_ate: NDArray[Any],
                     w10: float | np.floating,
                     w01: float | np.floating,
                     ) -> tuple[NDArray[Any]]:
    """Compute weights for difference in continuous case."""
    w_ate_cont = w_ate[a_idx, t_idx, :] if t_idx == no_of_treat - 1 else (
        w10 * w_ate[a_idx, t_idx, :] + w01 * w_ate[a_idx, t_idx+1, :])
    w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
    w_diff_cont = w_gate_cont - w_ate_cont

    return w_diff_cont


def assign_pot(y_pot: NDArray[Any],
               y_pot_var: NDArray[Any],
               y_pot_mate: NDArray[Any],
               y_pot_mate_var: NDArray[Any],
               results_fut_zj: Any,
               zj_idx: int | np.integer,
               ):
    """Reduce repetetive code."""
    y_pot[zj_idx, :, :, :] = results_fut_zj[0]
    y_pot_var[zj_idx, :, :, :] = results_fut_zj[1]
    y_pot_mate[zj_idx, :, :, :] = results_fut_zj[2]
    y_pot_mate_var[zj_idx, :, :, :] = results_fut_zj[3]

    return y_pot, y_pot_var, y_pot_mate, y_pot_mate_var


def assign_w(w_gate: NDArray[Any],
             w_gate_unc: NDArray[Any],
             w_censored: NDArray[Any],
             results_fut_zj: Any,
             zj_idx,
             ):
    """Reduce repetetive code."""
    w_gate[zj_idx, :, :, :] = results_fut_zj[4]
    w_gate_unc[zj_idx, :, :, :] = results_fut_zj[5]
    w_censored[zj_idx, :, :] = results_fut_zj[7]

    return w_gate, w_gate_unc, w_censored


def adjust_var_mult_duplicates(matches: NDArray[Any] | list[Any]) -> float:
    """Compute multiplier of variance adjusting for duplicate matches."""
    matches_np = np.array(matches)
    unique_values, counts = np.unique(matches_np, return_counts=True)
    n_unique = len(unique_values)
    d_i = counts - 1
    var_dupl_mult = ((n_unique + np.sum(2 * d_i + d_i**2))
                     / (n_unique + np.sum(d_i)))

    return var_dupl_mult
