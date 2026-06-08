"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from collections import namedtuple
from copy import deepcopy
from os import remove
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
try:
    import ray
except ImportError:
    ray = None

from mcf import mcf_ate
from mcf import mcf_bias_adjustment as mcf_ba
from mcf import mcf_estimation as mcf_est
from mcf import mcf_effect_helpers as he
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_iv_add
from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_versions as mcf_tv
from mcf import mcf_weight as mcf_w
from mcf import mcfoptp_parallel_backend_ray_classical as mcf_ray
from mcf.mcfoptp_parallel_backend_forest_executor import (forest_executor_with_shared,
                                                          map_task_batches,
                                                          )
from mcf.mcfoptp_parallel_backends_base import TaskSpec

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg, IntCfg, GenTvCfg, PBaCfg, CtGrid
    from mcf.mcf_init_predict import PCfg

type ArrayLike = NDArray[Any] | None


def gate_est(mcf_: 'ModifiedCausalForest',
             data_df: DataFrame,
             weights_dic: dict,
             w_atemain: ArrayLike, *,
             gate_type: str = 'GATE',
             z_name_cbgate: list[str] | tuple[str] | str | None = None,
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
    p_ba_cfg, gen_tv_cfg = mcf_.p_ba_cfg, mcf_.gen_tv_cfg
    zero_tol, sum_tol = int_cfg.zero_tol, int_cfg.sum_tol

    if gate_type != 'GATE':
        var_cfg, var_x_values, smooth_yes, z_name_smooth = deepcopy(paras_cbgate)
        var_cfg.z_name = z_name_cbgate
    else:
        var_x_values = deepcopy(mcf_.var_x_values)
        if p_cfg.gates_smooth:
            if isinstance(var_cfg.z_name, str):
                var_cfg.z_name = [var_cfg.z_name]
            var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                                var_x_values, p_cfg
                                                                                )
        else:
            smooth_yes, z_name_smooth = False, None
    w_ate = deepcopy(w_atemain)
    old_ray_tasks_used = False
    txt = ''
    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        print('\nComputing', gate_type)
    y_dat = weights_dic['y_dat_np']
    weights_all = weights_dic['weights']
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    cl_dat = weights_dic['cl_dat_np'] if p_cfg.cluster_std else None
    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)

    z_name_l, z_values_l, z_smooth_l = he.prepare_gate_z_context(var_cfg.z_name, var_x_values,
                                                                 smooth_yes=smooth_yes,
                                                                 z_name_smooth=z_name_smooth,
                                                                 )
    var_cfg.z_name = z_name_l
    d_p, z_p, w_p, _ = mcf_ate.get_data_for_final_ate_estimation(data_df,
                                                                 gen_cfg, p_cfg, var_cfg,
                                                                 ate=False, need_count=False,
                                                                 )
    if gen_tv_cfg.yes:
        # Obtain the data for the later regression step (treatment versions)
        d_train, x_tv_train, x_tv_pred = mcf_tv.get_tv_data(data_df, weights_dic,
                                                            mcf_.var_cfg, zero_tol=zero_tol,
                                                            )
        # Namedtuple is easy to pass through all function and access it
        Tv_data = namedtuple('TV_data', ['d_train', 'x_tv_train', 'x_tv_pred'])
        gen_tv_data = Tv_data(d_train, x_tv_train, x_tv_pred,)
    else:
        gen_tv_data = None

    ba_data = mcf_ba.get_ba_data_prediction(weights_dic, p_ba_cfg) if p_ba_cfg.yes else None

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

    i_d_val = np.arange(no_of_treat)
    ref_pop_lab = ['All']
    if p_cfg.gatet:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab.append(str(lab))

    if (d_p is not None) and p_cfg.gatet:
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        p_cfg.gatet, no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = p_cfg.choice_based_probs

    if p_cfg.gates_minus_previous:
        w_ate = None
    elif not iv and w_ate is not None:
        w_ate = he.normalize_ate_weights(w_ate,
                                         no_of_tgates=no_of_tgates, no_of_treat=no_of_treat,
                                         zero_tol=zero_tol, sum_tol=sum_tol,
                                         )
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

    # print_runtime_info(gen_cfg, int_cfg, maxworkers, txt_method='GATE')

    ray_err_txt = 'Ray did not start in in GATE estimation.'
    weights_all_ref = y_dat_ref = z_p_ref = d_p_ref = None

    if maxworkers > 1 and int_cfg.mp_use_old_ray:
        if ray is None or ray_gate_zj_mp is None:
            raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                              'Install ray or use the new backend with mp_use_old_ray=False.'
                              )
        if not ray.is_initialized():
            mcf_ray.init_ray_with_fallback(maxworkers, gen_cfg,
                                           mem_object_store=int_cfg.mem_object_store_3,
                                           mem_object_store_2=int_cfg.mem_object_store_2,
                                           ray_err_txt=ray_err_txt,
                                           )
        mcf_ray.print_object_store(gen_cfg, int_cfg.mem_object_store_3)
        weights_all_ref, y_dat_ref, z_p_ref = mcf_ray.ray_put_all(weights_all, y_dat, z_p)
        d_p_ref = mcf_ray.ray_put_all(d_p) if d_p is not None and len(d_p) > 1 else d_p

    y_pot_all, y_pot_var_all, y_pot_mate_all = [], [], []
    y_pot_mate_var_all, txt_all = [], []
    for z_name_j, z_name in enumerate(var_cfg.z_name):
        txt_z_name = txt + ' '
        if gen_cfg.with_output and gen_cfg.verbose and with_output:
            z_name_ = mcf_ps.del_added_chars(z_name, prime=True)
            print(z_name_j + 1, '(', len(var_cfg.z_name), ')', z_name_, flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        kernel, bandw_z = he.gate_kernel_bandwidth(z_p, z_name_j, z_smooth,
                                                   zero_tol=int_cfg.zero_tol,
                                                   bandwidth_factor=p_cfg.gates_smooth_bandwidth,
                                                   )
        no_of_zval = len(z_values)

        (y_pot, y_pot_var, y_pot_mate, y_pot_mate_var
         ) = he.init_gate_pot_arrays(no_of_zval=no_of_zval, no_of_tgates=no_of_tgates,
                                     no_of_treat_dr=no_of_treat_dr, no_of_out=no_of_out,
                                     gen_tv_yes=gen_tv_cfg.yes,
                                     no_of_treat_per_main=gen_tv_cfg.no_of_treat_per_main,
                                     )
        w_gate = np.zeros((no_of_zval, no_of_tgates, no_of_treat, n_y))
        w_gate_unc = np.zeros_like(w_gate)
        w_censored = np.zeros((no_of_zval, no_of_tgates, no_of_treat))
        w_gate0_dim = (no_of_treat, n_y)

        if gen_tv_cfg.yes:
            treat_main = gen_tv_cfg.no_of_treat_per_main
            w_gate = np.repeat(w_gate, repeats=treat_main, axis=2)
            w_gate_unc = np.repeat(w_gate_unc, repeats=treat_main, axis=2)
            w_censored = np.repeat(w_censored, repeats=treat_main, axis=2)

        kw_args= {'i_d_val': i_d_val, 't_probs': t_probs, 'no_of_tgates': no_of_tgates,
                  'no_of_out': no_of_out, 'ct_cfg': ct_cfg, 'gen_cfg': gen_cfg, 'int_cfg': int_cfg,
                  'p_cfg': p_cfg, 'p_ba_cfg': p_ba_cfg, 'ba_data': ba_data,
                  'gen_tv_cfg': gen_tv_cfg, 'gen_tv_data': gen_tv_data,  'bandw_z': bandw_z,
                  'kernel': kernel,  'smooth_it': z_smooth, 'continuous': continuous, 'iv': iv,
                  'bgate_cbgate': gate_type in ('CBGATE', 'BGATE'),
                  }
        if maxworkers == 1 or p_cfg.gates_minus_previous:
            for zj_idx in range(no_of_zval):
                if p_cfg.gates_minus_previous:
                    if zj_idx > 0:
                        w_ate = w_gate_unc[zj_idx-1, :, :, :]
                    else:
                        w_ate = w_gate_unc[zj_idx, :, :, :]
                results_fut_zj = gate_zj(
                    z_values[zj_idx], zj_idx, y_dat, cl_dat, w_dat,
                    z_p=z_p, d_p=d_p, w_p=w_p, z_name_j=z_name_j, weights_all=weights_all,
                    w_gate0_dim=w_gate0_dim, w_gate_zj=w_gate[zj_idx, :, :, :],
                    w_gate_unc_zj=w_gate_unc[zj_idx, :, :, :],
                    w_censored_zj=w_censored[zj_idx, :, :], w_ate=w_ate,
                    y_pot_zj=y_pot[zj_idx, :, :, :], y_pot_var_zj=y_pot_var[zj_idx, :, :, :],
                    y_pot_mate_zj=y_pot_mate[zj_idx, :, :, :],
                    y_pot_mate_var_zj=y_pot_mate_var[zj_idx, :, :, :],
                    **kw_args,
                    )
                y_pot, y_pot_var, y_pot_mate, y_pot_mate_var = he.assign_pot(
                     y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                     results_fut_zj=results_fut_zj, zj_idx=zj_idx,
                     )
                w_gate, w_gate_unc, w_censored = he.assign_w(w_gate, w_gate_unc, w_censored,
                                                             results_fut_zj, zj_idx,
                                                             )
        else:
            if int_cfg.mp_use_old_ray:
                old_ray_tasks_used = True
                if ray is None:
                    raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                      'Install ray or use the new backend with '
                                      'mp_use_old_ray=False.'
                                      )
                still_running = [ray_gate_zj_mp.remote(
                         z_values[zj_idx], zj_idx, y_dat_ref, cl_dat, w_dat,
                         z_p=z_p_ref, d_p=d_p_ref, w_p=w_p, z_name_j=z_name_j,
                         weights_all=weights_all_ref, w_gate0_dim=w_gate0_dim, w_ate=w_ate,
                         n_y=n_y, save_w_file=save_w_file, sum_tol=sum_tol, zero_tol=zero_tol,
                         **kw_args,
                         )
                    for zj_idx in range(no_of_zval)]
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running, num_returns=1)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        (y_pot, y_pot_var, y_pot_mate, y_pot_mate_var
                         ) = he.assign_pot(y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                                           results_fut_zj=results_fut_idx,
                                           zj_idx=results_fut_idx[6],
                                           )
                        w_gate, w_gate_unc, w_censored = he.assign_w(
                            w_gate, w_gate_unc, w_censored, results_fut_idx, results_fut_idx[6]
                            )
            else:
                shared_gate_data = {'y_dat': y_dat, 'z_p': z_p, 'd_p': d_p,
                                    'weights_all': weights_all,
                                    }
                with forest_executor_with_shared(int_cfg=int_cfg, maxworkers=maxworkers,
                                                 shared_obj=shared_gate_data,
                                                 shared_name='gate_zj_mp_data',
                                                 ray_err_txt=ray_err_txt,
                                                 fail_txt='Failed to make executor in gate_zj_mp.'
                                                 ) as (executor, data_handle, maxworkers):
                    tasks = [TaskSpec(func=gate_zj_mp_backend,
                                      kwargs={'z_val': z_values[zj_idx],
                                              'zj_idx': zj_idx,
                                              'cl_dat': cl_dat,
                                              'w_dat': w_dat,
                                              'data': data_handle,
                                              'z_name_j': z_name_j,
                                              'w_gate0_dim': w_gate0_dim,
                                              'w_p': w_p,
                                              'w_ate': w_ate,
                                              'n_y': n_y,
                                              'save_w_file': save_w_file,
                                              'sum_tol': sum_tol,
                                              'zero_tol': zero_tol,
                                              'kw_args': kw_args,
                                              },
                                      name=f'gate_zj_{zj_idx}',
                                      )
                             for zj_idx in range(no_of_zval)
                             ]
                    for results_fut_idx in map_task_batches(executor=executor, tasks=tasks,
                                                            int_cfg=int_cfg,
                                                            maxworkers=maxworkers,
                                                            min_worker_waves = (
                                                                4 if int_cfg.mp_backend == 'joblib'
                                                                else 1
                                                                ),
                                                            ):
                        y_pot, y_pot_var, y_pot_mate, y_pot_mate_var = he.assign_pot(
                            y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                            results_fut_zj=results_fut_idx, zj_idx=results_fut_idx[6],
                            )
                        w_gate, w_gate_unc, w_censored = he.assign_w(w_gate, w_gate_unc, w_censored,
                                                                     results_fut_idx,
                                                                     results_fut_idx[6],
                                                                     )
        if gen_cfg.with_output:
            # Describe weights
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
                    ret = mcf_est.analyse_weights(w_gate[zj_idx, a_idx, :, :], None, gen_cfg, p_cfg,
                                                  ate=False, continuous=continuous,
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
                    txt_z_name += mcf_ps.txt_weight_stat(
                        nonzero=w_st[0], equalzero=w_st[1], mean_nonzero=w_st[2],
                        std_nonzero=w_st[3], gini_all=w_st[4], gini_nonzero=w_st[5],
                        share_largest_q=share_largest_q, sum_larger=sum_larger,
                        obs_larger=obs_larger, gen_cfg=gen_cfg, p_cfg=p_cfg,
                        share_censored=w_censored_all, continuous=continuous,
                        d_values_cont=d_values,
                        )  # Discretized weights if cont
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

    gate_est_dic = {'continuous': continuous,
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
        gate_est_dic['d_values_dr'] = mcf_tv.joint_d_values(gen_tv_cfg.d_dict['main_treat'],
                                                            gen_tv_cfg.d_dict['sub_treat'],
                                                            as_string=False,
                                                            )
        gate_est_dic['d_values'] = gate_est_dic['d_values_dr']
        gate_est_dic['treat_comp_label'] = he.make_treat_comp_label(gate_est_dic['d_values_dr'],
                                                                    continuous=continuous,
                                                                    )
    if maxworkers > 1 and int_cfg.mp_use_old_ray and old_ray_tasks_used:
        (weights_all_ref, y_dat_ref, z_p_ref, d_p_ref, finished_res, finished
         ) = mcf_ray.ray_del_refs(weights_all_ref, y_dat_ref, z_p_ref, d_p_ref,
                                  f1=finished_res, f2=finished, mp_ray_del=int_cfg.mp_ray_del,
                                  )
    return y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all, gate_est_dic, txt_all


def gate_zj_mp_backend(*, z_val: int | float,
                       zj_idx: int | np.integer,
                       cl_dat: ArrayLike,
                       w_dat: ArrayLike,
                       data: dict[str, Any],
                       z_name_j: str,
                       w_gate0_dim: list[int],
                       w_p: NDArray[Any],
                       w_ate: NDArray[Any],
                       n_y: int,
                       save_w_file: str | None,
                       sum_tol: float,
                       zero_tol: float,
                       kw_args: dict[str, Any],
                       ) -> Any:
    """Backend-agnostic wrapper for one z-value specific GATE task."""
    return gate_zj_mp(z_val, zj_idx, data['y_dat'], cl_dat, w_dat,
                      z_p=data['z_p'], d_p=data['d_p'], w_p=w_p, z_name_j=z_name_j,
                      weights_all=data['weights_all'], w_gate0_dim=w_gate0_dim, w_ate=w_ate,
                      n_y=n_y, save_w_file=save_w_file, sum_tol=sum_tol, zero_tol=zero_tol,
                      **kw_args,
                      )


def bgate_est(mcf_: 'ModifiedCausalForest',
              data_df: DataFrame,
              weights_dic: dict,
              w_ate: ArrayLike,
              forest_dic: dict, *,
              gate_type='CBGATE',
              iv_tuple=None
              ) -> tuple[list[NDArray[Any]], list[NDArray[Any]],
                         list[NDArray[Any]], list[NDArray[Any]],
                         dict, str, dict
                         ]:
    """Compute CBGATE & BGATE for single variables keeping others constant."""
    gen_cfg = mcf_.gen_cfg
    var_x_type = deepcopy(mcf_.var_x_type)
    var_x_values = deepcopy(mcf_.var_x_values)
    p_cfg, var_cfg = deepcopy(mcf_.p_cfg), deepcopy(mcf_.var_cfg)
    x_name_mcf = mcf_.cf_cfg.x_name_mcf
    bgate = gate_type == 'BGATE'
    if var_cfg.z_name is None or var_cfg.z_name == []:
        raise ValueError(f'Something wrong with {var_cfg.z_name}')
    if bgate:
        if (var_cfg.x_name_balance_bgate is None or var_cfg.x_name_balance_bgate == []):
            raise ValueError(f'Something wrong with {var_cfg.x_name_balance_bgate}')

    if gen_cfg.with_output:
        txt_1 = '\n' + '=' * 100 + f'\nComputing {gate_type}'
        print(txt_1)
    if p_cfg.gatet:
        p_cfg.gatet = p_cfg.atet = False
        if gen_cfg.with_output:
            txt_1 += f'\nNo treatment specific effects for {gate_type}.'
    # Add continues variables
    if p_cfg.gates_smooth:
        if isinstance(var_cfg.z_name, str):
            var_cfg.z_name = [var_cfg.z_name]
        var_cfg, var_x_values, smooth_yes, z_name_smooth = he.addsmoothvars(data_df, var_cfg,
                                                                            var_x_values, p_cfg
                                                                            )
    else:
        smooth_yes, z_name_smooth = False, None
    eva_values = he.ref_vals_cbgate(data_df, var_x_type, var_x_values,
                                    no_eva_values=p_cfg.gate_no_evalu_points,
                                    )
    if gen_cfg.with_output and gen_cfg.verbose:
        print(f'\n{gate_type} variable under investigation: ', end=' ')
    y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all, txt_all = [], [], [], [], []
    first_run = True
    text_sim_all = ''
    for v_j, vname in enumerate(var_cfg.z_name):
        if gen_cfg.with_output:
            txt = txt_1[:]
        else:
            txt = ''
        if vname not in x_name_mcf:
            raise ValueError(f'Heterogeneity variable {vname} for {gate_type} NOT used for '
                             'splitting. Add to splitting variables.'
                             )
        if gen_cfg.with_output and gen_cfg.verbose:
            vname_ = mcf_ps.del_added_chars(vname, prime=True)
            print(mcf_ps.del_added_chars(vname_, prime=True), end=' ')
        if bgate:
            data_df_new, z_values, matches, txt_sim = he.ref_data_bgate(
                data_df.copy(), vname, p_cfg, eva_values,
                var_cfg.x_name_balance_bgate[:],
                with_output_verbose=gen_cfg.with_output and gen_cfg.verbose,
                zero_tol = mcf_.int_cfg.zero_tol,
                )
        else:
            data_df_new, z_values, txt_sim = he.ref_data_cbgate(
                data_df.copy(), vname, p_cfg, eva_values,
                with_output_verbose=gen_cfg.with_output and gen_cfg.verbose
                )
        vname_ = mcf_ps.del_added_chars(vname, prime=True)
        text_sim_all += f'\n{vname_}: ' + txt_sim
        txt += txt_sim
        var_x_values[vname] = z_values[:]

        if bgate:
            # Compute correction factor for duplicates in matching
            var_dupl_mult = he.adjust_var_mult_duplicates(matches)

        gen_cfg_weight_weight = deepcopy(gen_cfg)
        gen_cfg_weight_weight.with_output = gen_cfg_weight_weight.verbose = False
        if iv_tuple is None:
            iv = False
            weights_dic, _ = mcf_w.get_weights_mp(data_df_new, forest_dic,
                                                  reg_round='regular',
                                                  cf_cfg=mcf_.cf_cfg, ct_cfg=mcf_.ct_cfg,
                                                  gen_cfg=gen_cfg_weight_weight,
                                                  int_cfg=mcf_.int_cfg,
                                                  p_cfg=p_cfg, var_cfg=var_cfg,
                                                  gen_tv_cfg_yes=mcf_.gen_tv_cfg.yes,
                                                  print_progress=False, with_output=False,
                                                  )
        else:
            iv = True
            mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic = iv_tuple
            iate_1st_dic, _ = mcf_iv_add.iate_1st_stage_all_folds_rounds(mcf_, mcf_1st, data_df_new,
                                                                         True
                                                                         )
            # Compute weights of reduced form & 1st stage & final estimation
            _, _, weights_dic = mcf_iv_add.get_weights_iv_local(
                mcf_, mcf_1st=mcf_1st, mcf_redf=mcf_redf, forest_1st_dic=forest_1st_dic,
                forest_redf_dic=forest_redf_dic, iate_1st_dic=iate_1st_dic, iate_eff_1st_dic=None,
                data_df=data_df_new, round_='regular', local_effects=True, no_1st_weights=True
                )
        w_ate, _, _, _ = mcf_ate.ate_est(mcf_, data_df_new, weights_dic, with_output=False, iv=iv)
        (y_pot_gate_z, y_pot_var_gate_z, y_pot_mate_gate_z,
         y_pot_mate_var_gate_z, gate_est_dic_z, txt_p
         ) = gate_est(mcf_, data_df_new, weights_dic, w_ate, gate_type=gate_type,
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


# @ray.remote
def _ray_gate_zj_mp_impl(z_val, zj_idx, y_dat, cl_dat, w_dat, *, z_p, d_p, w_p, z_name_j,
                         weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
                         no_of_tgates, no_of_out, ct_cfg, gen_cfg, int_cfg, p_cfg,
                         p_ba_cfg, ba_data, gen_tv_cfg, gen_tv_data,
                         n_y, bandw_z, kernel, save_w_file=None, smooth_it=False,
                         continuous: bool = False, iv: bool =False, sum_tol: float = 1e-8,
                         zero_tol: float = 1e-10, bgate_cbgate: bool = False,
                         ):
    """Make function compatible with Ray."""
    return gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat,
                      z_p=z_p, d_p=d_p, w_p=w_p, z_name_j=z_name_j, weights_all=weights_all,
                      w_gate0_dim=w_gate0_dim, w_ate=w_ate, i_d_val=i_d_val,
                      t_probs=t_probs, no_of_tgates=no_of_tgates, no_of_out=no_of_out,
                      ct_cfg=ct_cfg, gen_cfg=gen_cfg, int_cfg=int_cfg, p_cfg=p_cfg,
                      p_ba_cfg=p_ba_cfg, ba_data=ba_data, gen_tv_cfg=gen_tv_cfg,
                      gen_tv_data=gen_tv_data, n_y=n_y, bandw_z=bandw_z, kernel=kernel,
                      save_w_file=save_w_file, smooth_it=smooth_it, continuous=continuous, iv=iv,
                      sum_tol=sum_tol, zero_tol=zero_tol, bgate_cbgate=bgate_cbgate,
                      )


ray_gate_zj_mp = ray.remote(_ray_gate_zj_mp_impl) if ray is not None else None


def gate_zj(z_val: int | float,
            zj_idx: int | np.integer,
            y_dat: NDArray[Any],
            cl_dat: ArrayLike,
            w_dat: ArrayLike, *,
            z_p: NDArray[Any], d_p: NDArray[Any], w_p: NDArray[Any],
            z_name_j: str,
            weights_all: list[list[NDArray[Any]]] | list[Any],
            w_gate0_dim: tuple[int, int],
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
            ct_cfg: 'CtGrid', gen_cfg: 'GenCfg', int_cfg: 'IntCfg', p_cfg: 'PCfg',
            p_ba_cfg: 'PBaCfg',
            ba_data: Any,
            gen_tv_cfg: 'GenTvCfg',
            gen_tv_data: Any,
            bandw_z: float,
            kernel: int,
            smooth_it: bool = False,
            continuous: bool = False,
            iv: bool = False,
            bgate_cbgate: bool = False,                    # pylint: disable=unused-argument
            ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any],
                       NDArray[Any], NDArray[Any], NDArray[Any],
                       int | np.integer,
                       NDArray[Any],
                       ]:
    """Compute Gates and their variances for MP."""
    zero_tol, sum_tol = int_cfg.zero_tol, int_cfg.sum_tol
    w_diff, results_container_w, results_container_r = None, None, None
    gen_tv, p_ba = gen_tv_cfg.yes, p_ba_cfg.yes

    treat_main = gen_tv_cfg.no_of_treat_per_main if gen_tv else None

    if continuous:
        no_of_treat, d_values = ct_cfg.ct_grid_w, ct_cfg.ct_grid_w_val
        i_w01 = ct_cfg.ct_w_to_dr_int_w01
        i_w10 = ct_cfg.ct_w_to_dr_int_w10
        index_full = ct_cfg.ct_w_to_dr_index_full
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values

    weights, relevant_z,  w_z_val = he.get_w_rel_z(z_p[:, z_name_j], z_val,
                                                   weights_all=weights_all, smooth_it=smooth_it,
                                                   bandwidth=bandw_z, kernel=kernel,
                                                   w_is_csr=int_cfg.weight_as_sparse,
                                                   sum_tol=int_cfg.sum_tol,
                                                   normalize_smooth_z=True,
                                                   )
    if gen_tv:
        d_tv_train = gen_tv_data.d_train
        x_tv_train = gen_tv_data.x_tv_train
        # if bgate_cbgate:
        #     x_tv_pred = x_tv_train.copy()
        #     # txt += ('Versions: Training (instead of prediction) data '
        #     #         'used for BGATE and CBGATE'
        #     #         )
        # else:
        x_tv_pred = None if x_tv_train is None else gen_tv_data.x_tv_pred[relevant_z, :]

    d_p_z = d_p[relevant_z] if d_p is not None else None
    w_p_z = w_p[relevant_z] if gen_cfg.weighted and w_p is not None else None

    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)
    # Step 1: Aggregate weights

    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_cfg.weight_as_sparse:
                # weight_i = weights[t_idx].getrow(n_idx).tocsr()
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
                d_value = d_p_z[n_idx, 0] if gen_tv else d_p_z[n_idx]
                i_pos = he.single_treatment_pos(d_value, d_values, i_d_val, zero_tol=zero_tol)
                w_gadd[t_idx, w_index] = w_i * t_probs[i_pos]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()

        if gen_tv:
            w_gadd = np.repeat(w_gadd, repeats=treat_main, axis=0)

        w_gate_zj[0, :, :] += w_gadd
        if p_cfg.gatet:
            if d_p_z is None:
                raise ValueError('GATET requires prediction treatments d_p.')
            d_value = d_p_z[n_idx, 0] if gen_tv else d_p_z[n_idx]
            t_pos = he.single_treatment_pos(d_value, d_values, i_d_val, zero_tol=zero_tol,)
            w_gate_zj[t_pos+1, :, :] += w_gadd

    # Bias adjustment (optional)
    if p_ba:
        if p_ba_cfg.adj_method == 'w_obs':
            weights_eval = mcf_ba.get_weights_eval_ba(w_gate_zj, no_of_treat,
                                                      zero_tol=int_cfg.zero_tol
                                                      )
        else:
            weights_eval = None

        for a_idx in range(no_of_tgates):
            if p_ba_cfg.adj_method == 'w_obs':
                ba_data.weights_eval = weights_eval[a_idx, :].copy()

            for t_idx in range(no_of_treat):
                w_gate_zj[a_idx, t_idx, :] = mcf_ba.bias_correction_wregr(
                    w_gate_zj[a_idx, t_idx, :],
                    y_dat[:, 0],
                    ba_data,
                    int_dtype=np.float64, out_dtype=np.float32,
                    pos_weights_only=p_ba_cfg.pos_weights_only,
                    zero_tol=int_cfg.zero_tol,
                    ridge=p_ba_cfg.ridge,
                    cv_k=p_ba_cfg.cv_k,
                    )
    else:
        weights_eval = None
    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    if iv:
        sum_wgate = np.ones_like(sum_wgate) * n_x

    if gen_tv:
        # Expand to new treatment dimension; redefine d_values, no_of_treat
        no_of_treat, d_values, _, _, _, _, _, = mcf_tv.expand_dimension(None, None, None, None,
                                                                        None,
                                                                        gen_tv_cfg=gen_tv_cfg,
                                                                        )
        version_res_dat = np.zeros_like(w_ate)

        for a_idx in range(no_of_tgates):
            maintreat_idx, subtreat_idx = 0, 0
            for t_idx in range(no_of_treat):
                (w_gate_zj[a_idx, t_idx, :], results_container_w, res_tv, results_container_r,
                 maintreat_idx, subtreat_idx, _
                 ) = mcf_tv.version_wregr(w_gate_zj[a_idx, t_idx, :],
                                          y_train=y_dat[:, 0],
                                          d_train=d_tv_train,
                                          x_train=x_tv_train,
                                          x_pred=x_tv_pred,
                                          cfg=gen_tv_cfg,
                                          container_w=results_container_w,
                                          container_r=results_container_r,
                                          treat_idx=t_idx,
                                          maintreat_idx=maintreat_idx,
                                          subtreat_idx=subtreat_idx,
                                          int_dtype=np.float64, out_dtype=np.float32,
                                          zero_tol=int_cfg.zero_tol,
                                          ridge=gen_tv_cfg.estimator == 'ridge',
                                          penalize_version=gen_tv_cfg.penalize_version[maintreat_idx
                                                                                       ],
                                          return_residuals=True,
                                          standardize_x=True,
                                          )
                if res_tv is not None:
                    version_res_dat[a_idx, t_idx, :] = res_tv
    else:
        version_res_dat = None
    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx],
                    w_gate_zj=w_gate_zj, w_censored_zj=w_censored_zj, w_gate_unc_zj=w_gate_unc_zj,
                    w_ate=w_ate, p_cfg=p_cfg, with_output=gen_cfg.with_output, p_ba_cfg_yes=p_ba,
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
                                               sum_tol = int_cfg.sum_tol,
                                               )
                        vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                        ret = mcf_est.weight_var(w_gate_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                                 residual_dat=vres,
                                                 weighted=gen_cfg.weighted, weights=w_dat,
                                                 bootstrap=p_cfg.se_boot_gate,
                                                 keep_all=int_cfg.keep_w0,
                                                 zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                                 seed=123345, min_obs=5,
                                                 )
                        ti_idx = index_full[t_idx, i]                     # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if gen_cfg.with_output:
                            w_diff_cont = he.w_diff_cont_func(t_idx, a_idx,
                                                              no_of_treat=no_of_treat,
                                                              w_gate_cont=w_gate_cont_unc,
                                                              w_ate=w_ate, w10=w10, w01=w01,
                                                              )
                            vres = (None if version_res_dat is None
                                    else version_res_dat[a_idx, t_idx, :]
                                    )
                            ret2 = mcf_est.weight_var(w_diff_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                                      residual_dat=vres,
                                                      weighted=gen_cfg.weighted, normalize=False,
                                                      weights=w_dat,
                                                      bootstrap=p_cfg.se_boot_gate,
                                                      keep_all=int_cfg.keep_w0,
                                                      zero_tol=int_cfg.zero_tol,
                                                      sum_tol=int_cfg.sum_tol,
                                                      seed=123345, min_obs=5,
                                                      )
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                    ret = mcf_est.weight_var(w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                                             p_cfg,
                                             residual_dat=vres,
                                             weighted=gen_cfg.weighted, weights=w_dat,
                                             bootstrap=p_cfg.se_boot_gate,
                                             keep_all=int_cfg.keep_w0, normalize=not iv,
                                             zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                             seed=123345, min_obs=5,
                                             )
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if gen_cfg.with_output:
                        ret2 = mcf_est.weight_var(w_diff, y_dat[:, o_idx], cl_dat, p_cfg,
                                                  residual_dat=None,
                                                  weighted=gen_cfg.weighted, normalize=False,
                                                  weights=w_dat,
                                                  bootstrap=p_cfg.se_boot_gate,
                                                  keep_all=int_cfg.keep_w0,
                                                  zero_tol=int_cfg.zero_tol,
                                                  sum_tol=int_cfg.sum_tol,
                                                  seed=123345, min_obs=5,
                                                  )
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]

    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj, w_gate_zj, w_gate_unc_zj,
            zj_idx, w_censored_zj,
            )


def gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, *, z_p, d_p, w_p,
               z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
               no_of_tgates, no_of_out,
               ct_cfg, gen_cfg, int_cfg, p_cfg,
               p_ba_cfg, ba_data, gen_tv_cfg, gen_tv_data,
               n_y, bandw_z, kernel,
               save_w_file=None, smooth_it=False,
               continuous=False, iv=False, sum_tol: float = 1e-8, zero_tol: float = 1e-10,
               bgate_cbgate=False,  # pylint: disable=unused-argument
               ):
    """Compute Gates and their variances for MP."""
    w_diff, results_container_w, results_container_r = None, None, None
    p_ba, gen_tv = p_ba_cfg.yes, gen_tv_cfg.yes

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
        weights_all = mcf_sys.save_load(save_w_file, save=False, output=gen_cfg.with_output)
    w_gate_zj = np.zeros((no_of_tgates, no_of_treat, n_y))
    w_gate_unc_zj = np.zeros_like(w_gate_zj)
    w_censored_zj = np.zeros((no_of_tgates, no_of_treat))
    y_pot_zj = np.empty((no_of_tgates, no_of_treat_dr, no_of_out))
    y_pot_var_zj = np.empty_like(y_pot_zj)
    y_pot_mate_zj = np.empty_like(y_pot_zj)
    y_pot_mate_var_zj = np.empty_like(y_pot_zj)

    if gen_tv:
        treat_main = gen_tv_cfg.no_of_treat_per_main
        y_pot_zj = np.repeat(y_pot_zj, repeats=treat_main, axis=1)
        y_pot_var_zj = np.repeat(y_pot_var_zj, repeats=treat_main, axis=1)
        y_pot_mate_zj = np.repeat(y_pot_mate_zj, repeats=treat_main, axis=1)
        y_pot_mate_var_zj = np.repeat(y_pot_mate_var_zj, repeats=treat_main, axis=1)
        w_gate_zj = np.repeat(w_gate_zj, repeats=treat_main, axis=1)
        w_gate_unc_zj = np.repeat(w_gate_unc_zj, repeats=treat_main, axis=1)
        w_censored_zj = np.repeat(w_censored_zj, repeats=treat_main, axis=1)
    else:
        treat_main = version_res_dat = None

    # Step 1: Aggregate weights
    weights, relevant_z, w_z_val = he.get_w_rel_z(z_p[:, z_name_j], z_val,
                                                  weights_all=weights_all, smooth_it=smooth_it,
                                                  bandwidth=bandw_z, kernel=kernel,
                                                  w_is_csr=int_cfg.weight_as_sparse,
                                                  sum_tol=int_cfg.sum_tol,
                                                  normalize_smooth_z=True,
                                                  )
    if gen_tv:
        d_tv_train = gen_tv_data.d_train
        x_tv_train = gen_tv_data.x_tv_train
        x_tv_pred = None if x_tv_train is None else gen_tv_data.x_tv_pred[relevant_z, :]

    d_p_z = d_p[relevant_z] if d_p is not None else None
    w_p_z = w_p[relevant_z] if gen_cfg.weighted and w_p is not None else None

    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_cfg.weight_as_sparse:
                # weight_i = weights[t_idx].getrow(n_idx).tocsr()
                weight_i = weights[t_idx][n_idx:n_idx + 1, :].tocsr()
                w_index = weight_i.indices.copy()
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_cfg.weighted:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if abs(w_i_sum) > zero_tol and not ((1-sum_tol) < w_i_sum < (1+sum_tol) or iv):
                w_i = w_i / w_i_sum
            if gen_cfg.weighted:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_cfg.choice_based_sampling:
                if d_p_z is None:
                    raise ValueError('Choice-based sampling requires prediction treatments d_p.')
                d_value = d_p_z[n_idx, 0] if gen_tv else d_p_z[n_idx]
                i_pos = he.single_treatment_pos(d_value, d_values, i_d_val, zero_tol=zero_tol)
                w_gadd[t_idx, w_index] = w_i * t_probs[i_pos]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()

        if gen_tv:
            treat_main = gen_tv_cfg.no_of_treat_per_main
            w_gadd = np.repeat(w_gadd, repeats=treat_main, axis=0)
        w_gate_zj[0, :, :] += w_gadd

        if p_cfg.gatet:
            if d_p_z is None:
                raise ValueError('GATET requires prediction treatments d_p.')
            d_value = d_p_z[n_idx, 0] if gen_tv else d_p_z[n_idx]
            t_pos = he.single_treatment_pos(d_value, d_values, i_d_val, zero_tol=zero_tol,)
            w_gate_zj[t_pos+1, :, :] += w_gadd

    # Bias adjustment (optional)
    if p_ba:
        if p_ba_cfg.adj_method == 'w_obs':
            weights_eval = mcf_ba.get_weights_eval_ba(w_gate_zj, no_of_treat,
                                                      zero_tol=int_cfg.zero_tol,
                                                      )
        else:
            weights_eval = None

        for a_idx in range(no_of_tgates):
            if p_ba_cfg.adj_method == 'w_obs':
                ba_data.weights_eval = weights_eval[a_idx, :].copy()
            for t_idx in range(no_of_treat):
                w_gate_zj[a_idx, t_idx, :] = mcf_ba.bias_correction_wregr(
                    w_gate_zj[a_idx, t_idx, :],
                    y_dat[:, 0],
                    ba_data,
                    int_dtype=np.float64, out_dtype=np.float32,
                    pos_weights_only=p_ba_cfg.pos_weights_only,
                    zero_tol=int_cfg.zero_tol,
                    ridge=p_ba_cfg.ridge,
                    cv_k=p_ba_cfg.cv_k,
                    )
    else:
        weights_eval = None
    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    if iv:
        sum_wgate = np.ones_like(sum_wgate) * n_x

    if gen_tv:
        # Expand to new treatment dimension; redefine d_values, no_of_treat
        no_of_treat, d_values, _, _, _, _, _, = mcf_tv.expand_dimension(
            None, None, None, None, None, gen_tv_cfg=gen_tv_cfg,
            )
        version_res_dat = np.zeros_like(w_gate_zj)

        for a_idx in range(no_of_tgates):
            maintreat_idx, subtreat_idx = 0, 0
            for t_idx in range(no_of_treat):
                (w_gate_zj[a_idx, t_idx, :], results_container_w, res_tv, results_container_r,
                 maintreat_idx, subtreat_idx, _,
                 ) = mcf_tv.version_wregr(w_gate_zj[a_idx, t_idx, :],
                                          y_train=y_dat[:, 0],
                                          d_train=d_tv_train,
                                          x_train=x_tv_train,
                                          x_pred=x_tv_pred,
                                          cfg=gen_tv_cfg,
                                          container_w=results_container_w,
                                          container_r=results_container_r,
                                          treat_idx=t_idx,
                                          maintreat_idx=maintreat_idx,
                                          subtreat_idx=subtreat_idx,
                                          int_dtype=np.float64, out_dtype=np.float32,
                                          zero_tol=int_cfg.zero_tol,
                                          ridge=gen_tv_cfg.estimator == 'ridge',
                                          penalize_version=gen_tv_cfg.penalize_version[maintreat_idx
                                                                                       ],
                                          return_residuals=True,
                                          standardize_x=True,
                                          )
                if res_tv is not None:
                    version_res_dat[a_idx, t_idx, :] = res_tv
    else:
        version_res_dat = None

    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx],
                    w_gate_zj=w_gate_zj, w_censored_zj=w_censored_zj, w_gate_unc_zj=w_gate_unc_zj,
                    w_ate=w_ate, p_cfg=p_cfg, with_output=gen_cfg.with_output,
                    p_ba_cfg_yes=p_ba, gen_tv_cfg_yes=gen_tv, iv=iv, zero_tol=int_cfg.zero_tol,
                    sum_tol=int_cfg.sum_tol,
                    )
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj, w_censored_zj
                         ) = w_gate_cont_funct(t_idx, a_idx,
                                               no_of_treat=no_of_treat, w_gate_zj=w_gate_zj,
                                               w10=w10, w01=w01, i=i, w_gate_unc_zj=w_gate_unc_zj,
                                               w_censored_zj=w_censored_zj,
                                               max_weight_share=p_cfg.max_weight_share,
                                               p_ba_cfg_yes=p_ba_cfg.yes, iv=iv,
                                               zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                                               )
                        vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                        ret = mcf_est.weight_var(w_gate_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                                 residual_dat=vres,
                                                 weighted=gen_cfg.weighted, weights=w_dat,
                                                 bootstrap=p_cfg.se_boot_gate,
                                                 keep_all=int_cfg.keep_w0,
                                                 zero_tol=int_cfg.zero_tol,
                                                 sum_tol = int_cfg.sum_tol,
                                                 seed=123345, min_obs=5,
                                                 )
                        ti_idx = index_full[t_idx, i]                      # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if gen_cfg.with_output:
                            w_diff_cont = he.w_diff_cont_func(t_idx, a_idx,
                                                              no_of_treat=no_of_treat,
                                                              w_gate_cont=w_gate_cont_unc,
                                                              w_ate=w_ate, w10=w10, w01=w01,
                                                              )
                            ret2 = mcf_est.weight_var(w_diff_cont, y_dat[:, o_idx], cl_dat, p_cfg,
                                                      residual_dat=None,
                                                      weighted=gen_cfg.weighted, normalize=False,
                                                      weights=w_dat,
                                                      bootstrap=p_cfg.se_boot_gate,
                                                      keep_all=int_cfg.keep_w0,
                                                      zero_tol=int_cfg.zero_tol,
                                                      sum_tol = int_cfg.sum_tol,
                                                      seed=123345, min_obs=5,
                                                      )
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    vres = None if version_res_dat is None else version_res_dat[a_idx, t_idx, :]
                    ret = mcf_est.weight_var(w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                                             p_cfg,
                                             residual_dat=vres,
                                             weighted=gen_cfg.weighted, weights=w_dat,
                                             bootstrap=p_cfg.se_boot_gate,
                                             keep_all=int_cfg.keep_w0,
                                             normalize=not iv,
                                             zero_tol=int_cfg.zero_tol, sum_tol = int_cfg.sum_tol,
                                             seed=123345, min_obs=5,
                                             )
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if gen_cfg.with_output:
                        ret2 = mcf_est.weight_var(w_diff, y_dat[:, o_idx], cl_dat, p_cfg,
                                                  residual_dat=None,
                                                  weighted=gen_cfg.weighted, normalize=False,
                                                  weights=w_dat,
                                                  bootstrap=p_cfg.se_boot_gate,
                                                  keep_all=int_cfg.keep_w0,
                                                  zero_tol=int_cfg.zero_tol,
                                                  sum_tol = int_cfg.sum_tol,
                                                  seed=123345, min_obs=5,
                                                  )
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]

    save_name_w = save_name_wunc = None

    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj, w_gate_zj, w_gate_unc_zj,
            zj_idx, w_censored_zj, save_name_w, save_name_wunc
            )


def w_gate_func(a_idx: int | np.integer,
                t_idx: int | np.integer,
                sum_wgate: float | np.floating, *,
                w_gate_zj: NDArray[Any],
                w_censored_zj: NDArray[Any],
                w_gate_unc_zj: NDArray[Any],
                w_ate: NDArray[Any],
                p_cfg: 'PCfg',
                with_output: bool = True,
                p_ba_cfg_yes: bool = False,
                gen_tv_cfg_yes: bool = False,
                iv: bool = False,
                zero_tol: float = 1e-10,
                sum_tol: float = 1e-8,
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
             ) = mcf_gp.bound_norm_weights(w_gate_zj[a_idx, t_idx, :],
                                           max_weight_share=p_cfg.max_weight_share,
                                           zero_tol=zero_tol, sum_tol=sum_tol,
                                           negative_weights_possible=p_ba_cfg_yes or gen_tv_cfg_yes,
                                           )
    if with_output and w_ate is not None:
        w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx, t_idx, :]
    else:
        w_diff = None

    return w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj


def w_gate_cont_funct(t_idx: int | np.integer,
                      a_idx: int | np.integer, *,
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
                      zero_tol: float = 1e-10,
                      sum_tol: float = 1e-8,
                      ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any],]:
    """Approximate weights for continuous treatments."""
    if t_idx == (no_of_treat - 1):  # last element, no inter
        w_gate_cont = w_gate_zj[a_idx, t_idx, :]
    else:
        w_gate_cont = (w10 * w_gate_zj[a_idx, t_idx, :]
                       + w01 * w_gate_zj[a_idx, t_idx+1, :])
    sum_wgate = np.sum(w_gate_cont)
    if not ((-sum_tol < sum_wgate < sum_tol) or (1-sum_tol < sum_wgate < 1+sum_tol) or iv):
        w_gate_cont = w_gate_cont / sum_wgate
    if i == 0:
        w_gate_unc_zj[a_idx, t_idx, :] = w_gate_cont
    w_gate_cont_unc = w_gate_cont.copy()
    if max_weight_share < 1:
        if iv:
            w_gate_cont, _, w_censored = mcf_gp.bound_norm_weights_not_one(
                w_gate_cont, max_weight_share, zero_tol=zero_tol, sum_tol=sum_tol,
                )
        else:
            w_gate_cont, _, w_censored = mcf_gp.bound_norm_weights(
                w_gate_cont,
                max_weight_share=max_weight_share, zero_tol=zero_tol, sum_tol=sum_tol,
                negative_weights_possible=p_ba_cfg_yes,
                )
        if i == 0:
            w_censored_zj[a_idx, t_idx] = w_censored

    return w_gate_cont, w_gate_cont_unc, w_gate_unc_zj, w_censored_zj
