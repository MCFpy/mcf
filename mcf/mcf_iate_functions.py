"""
Procedures needed for IATE estimation.

Created on Thu Dec 8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from concurrent import futures
from copy import deepcopy
from dask.distributed import Client, as_completed

import numpy as np
import pandas as pd

import ray

from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_general_purpose as mcf_gp
from mcf import mcf_iate_add_functions as mcf_iate_add


def iate_est_mp(weights, data_file, y_dat, cl_dat, w_dat, v_dict, c_in_dict,
                w_ate=None, balancing_test=False, save_predictions=True,
                pot_y_prev=None, lc_forest=None, var_x_type=None):
    """
    Estimate IATE and their standard errors, plot & save them, MP version.

    Parameters
    ----------
    weights : List of lists. For every obs, positive weights are saved.
              Alternative: Sparse csr-Matrix.
    pred_data : String. csv-file with data to make predictions for.
    y : Numpy array. All outcome variables.
    cl : Numpy array. Cluster variable.
    w : Numpy array. Sampling weights.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    w_ate: Numpy array. Weights of ATE estimation. Default is None.
    balancing_test : Bool. Balancing test. Default is False.
    save_predictions : Bool. Default is True.
    pot_y_prev : Numpy array or None. Potential outcomes from previous
          estimations. Default is None.
    lc_forest : RandomForestRegressor (sklearn.ensemble). Default is None.
          Contains estimated RF used for centering outcomes
    var_x_type : Dict. Names and type of features as used for centering.
          Default is None.

    Returns
    -------
    post_estimation_file : String. Name of files with predictions.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var: Numpy array. Standard errors of potential outcomes.
    iate: Numpy array. IATEs.
    iate_se: Numpy array. Standard errors of IATEs.
    (names_pot_iate, names_pot_iate0): Tuple of list of strings.
           names_pot_iate: List of strings: All names of IATEs in file.
           names_pot_iate0: Only those names related to first category.
    """
    def warn_text(c_dict):
        if c_dict['with_output'] and c_dict['verbose']:
            print('If prediction file is large, this step may take long. If ',
                  'nothing seems to happen, it may be worth to try do the ',
                  'estimation without sparse weight matrix. This needs more '
                  'memory, but could be substantially faster ',
                  '(weight_as_sparse = False).')
    if pot_y_prev is None:  # 1nd round of estimations
        c_dict = c_in_dict
        reg_round = True
    else:
        c_dict = deepcopy(c_in_dict)
        c_dict['iate_se_flag'] = False
        c_dict['se_boot_iate'] = False
        reg_round = False
    if c_dict['with_output'] and c_dict['verbose'] and save_predictions:
        print('\nComputing IATEs 1/2 (potential outcomes)')

    n_x = weights[0].shape[0] if c_dict['weight_as_sparse'] else len(weights)
    n_y, no_of_out = len(y_dat), len(v_dict['y_name'])
    if c_dict['d_type'] == 'continuous':
        no_of_treat, d_values = c_dict['ct_grid_w'], c_dict['ct_grid_w_val']
        d_values_dr = c_dict['ct_d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    pot_y = np.empty((n_x, no_of_treat_dr, no_of_out))
    larger_0 = np.zeros(no_of_treat_dr)
    equal_0, mean_pos = np.zeros_like(larger_0), np.zeros_like(larger_0)
    std_pos, gini_all = np.zeros_like(larger_0), np.zeros_like(larger_0)
    gini_pos, share_censored = np.zeros_like(larger_0), np.zeros_like(larger_0)
    share_largest_q = np.zeros((no_of_treat_dr, 3))
    sum_larger = np.zeros((no_of_treat_dr, len(c_dict['q_w'])))
    obs_larger = np.zeros_like(sum_larger)
    if c_dict['iate_se_flag']:
        pot_y_var, pot_y_m_ate = np.empty_like(pot_y), np.empty_like(pot_y)
        pot_y_m_ate_var = np.empty_like(pot_y)
    else:
        pot_y_var = pot_y_m_ate = pot_y_m_ate_var = w_ate = None
    if w_ate is not None:
        w_ate = w_ate[0, :, :]
    if not c_dict['w_yes']:
        w_dat = None
    no_of_cluster = None
    if c_dict['iate_se_flag'] and c_dict['cluster_std']:
        no_of_cluster = len(np.unique(cl_dat))
    l1_to_9 = [None] * n_x
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if c_dict['weight_as_sparse']:
        iterator = len(weights)
    if maxworkers == 1:
        for idx in range(n_x):
            if c_dict['weight_as_sparse']:
                weights_idx = [weights[t_idx].getrow(idx) for
                               t_idx in range(iterator)]
            else:
                weights_idx = weights[idx]
            ret_all_i = iate_func1_for_mp(
                idx, weights_idx, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
                no_of_out, n_y, c_dict)
            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
             share_censored) = assign_ret_all_i(
                 pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                 share_censored, ret_all_i, n_x, idx)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, n_x)
    else:
        if c_dict['obs_by_obs']:  # this is currently not used, too slow
            if c_dict['_ray_or_dask'] == 'ray':
                if c_dict['mem_object_store_3'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_3']/(1024*1024)), " MB")
                if c_dict['weight_as_sparse']:
                    still_running = [ray_iate_func1_for_mp.remote(
                        idx, [weights[t_idx].getrow(idx) for t_idx in
                              range(iterator)], cl_dat, no_of_cluster, w_dat,
                        w_ate, y_dat, no_of_out, n_y, c_dict)
                        for idx in range(n_x)]
                    warn_text(c_dict)
                else:
                    still_running = [ray_iate_func1_for_mp.remote(
                        idx, weights[idx], cl_dat, no_of_cluster, w_dat,
                        w_ate, y_dat, no_of_out, n_y, c_dict)
                        for idx in range(n_x)]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                         l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, ret_all_i, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, n_x)
                        jdx += 1
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    if c_dict['weight_as_sparse']:
                        ret_fut = [clt.submit(
                            iate_func1_for_mp, idx,
                            [weights[t_idx].getrow(idx) for t_idx in
                             range(iterator)], cl_dat, no_of_cluster, w_dat,
                            w_ate, y_dat, no_of_out, n_y, c_dict)
                            for idx in range(n_x)]
                        warn_text(c_dict)
                    else:
                        ret_fut = [clt.submit(
                            iate_func1_for_mp, idx, weights[idx], cl_dat,
                            no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                            c_dict) for idx in range(n_x)]
                    jdx = 0
                    for _, res in as_completed(ret_fut, with_results=True):
                        jdx += 1
                        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                         l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, res, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, n_x)
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    if c_dict['weight_as_sparse']:
                        ret_fut = {fpp.submit(
                            iate_func1_for_mp, idx,
                            [weights[t_idx].getrow(idx) for t_idx in
                             range(iterator)], cl_dat, no_of_cluster, w_dat,
                            w_ate, y_dat, no_of_out, n_y, c_dict):
                                idx for idx in range(n_x)}
                        warn_text(c_dict)
                    else:
                        ret_fut = {fpp.submit(
                            iate_func1_for_mp, idx, weights[idx], cl_dat,
                            no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                            c_dict): idx for idx in range(n_x)}
                    for jdx, frv in enumerate(futures.as_completed(ret_fut)):
                        ret_all_i = frv.result()
                        del ret_fut[frv]
                        del frv
                        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                         l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, ret_all_i, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, n_x)
        else:
            rows_per_split = c_dict['max_elements_per_split'] / n_y
            no_of_splits = round(n_x / rows_per_split)
            no_of_splits = min(max(no_of_splits, maxworkers), n_x)
            if c_dict['with_output'] and c_dict['verbose']:
                print('IATE-1: Avg. number of obs per split:',
                      f'{n_x / no_of_splits:5.2f}.',
                      ' Number of splits: ', no_of_splits)
            obs_idx_list = np.array_split(np.arange(n_x), no_of_splits)
            if c_dict['_ray_or_dask'] == 'ray':
                if c_dict['mem_object_store_3'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_3']/(1024*1024)), " MB")
                if c_dict['weight_as_sparse']:
                    still_running = [ray_iate_func1_for_mp_many_obs.remote(
                        idx, [weights[t_idx][idx, :] for t_idx in
                              range(iterator)], cl_dat, no_of_cluster,
                        w_dat, w_ate, y_dat, no_of_out, n_y, c_dict)
                        for idx in obs_idx_list]
                    warn_text(c_dict)
                else:
                    still_running = [ray_iate_func1_for_mp_many_obs.remote(
                        idx, [weights[idxx] for idxx in idx], cl_dat,
                        no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                        c_dict) for idx in obs_idx_list]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i_list in finished_res:
                        for ret_all_i in ret_all_i_list:
                            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, ret_all_i, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, no_of_splits)
                        jdx += 1
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    if c_dict['weight_as_sparse']:
                        ret_fut = [clt.submit(
                            iate_func1_for_mp_many_obs, idx,
                            [weights[t_idx][idx, :] for t_idx in
                             range(iterator)], cl_dat, no_of_cluster, w_dat,
                            w_ate, y_dat, no_of_out, n_y, c_dict)
                                for idx in obs_idx_list]
                    else:
                        ret_fut = [clt.submit(
                            iate_func1_for_mp_many_obs, idx,
                            [weights[idxx] for idxx in idx], cl_dat,
                            no_of_cluster, w_dat, w_ate, y_dat, no_of_out,
                            n_y, c_dict) for idx in obs_idx_list]
                    jdx = 0
                    for _, res_l in as_completed(ret_fut, with_results=True):
                        jdx += 1
                        for res in res_l:
                            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, res, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, no_of_splits)
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    if c_dict['weight_as_sparse']:
                        ret_fut = {fpp.submit(
                            iate_func1_for_mp_many_obs, idx,
                            [weights[t_idx][idx, :] for t_idx in
                             range(iterator)], cl_dat, no_of_cluster, w_dat,
                            w_ate, y_dat, no_of_out, n_y, c_dict):
                                idx for idx in obs_idx_list}
                    else:
                        ret_fut = {fpp.submit(
                            iate_func1_for_mp_many_obs, idx,
                            [weights[idxx] for idxx in idx], cl_dat,
                            no_of_cluster, w_dat, w_ate, y_dat, no_of_out,
                            n_y, c_dict): idx for idx in obs_idx_list}
                    for jdx, frv in enumerate(futures.as_completed(ret_fut)):
                        ret_all_i_list = frv.result()
                        del ret_fut[frv]
                        del frv
                        for ret_all_i in ret_all_i_list:
                            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored) = assign_ret_all_i(
                             pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                             l1_to_9, share_censored, ret_all_i, n_x)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, no_of_splits)
    if reg_round:
        for idx in range(n_x):
            larger_0 += l1_to_9[idx][0]
            equal_0 += l1_to_9[idx][1]
            mean_pos += l1_to_9[idx][2]
            std_pos += l1_to_9[idx][3]
            gini_all += l1_to_9[idx][4]
            gini_pos += l1_to_9[idx][5]
            share_largest_q += l1_to_9[idx][6]
            sum_larger += l1_to_9[idx][7]
            obs_larger += l1_to_9[idx][8]
        if c_dict['with_output'] and (not balancing_test) and save_predictions:
            print('\n')
            print('=' * 80)
            print('Analysis of weights (normalised to add to 1): ', 'IATE',
                  '(stats are averaged over all effects)')
            mcf_ate.print_weight_stat(
                larger_0 / n_x, equal_0 / n_x, mean_pos / n_x, std_pos / n_x,
                gini_all / n_x, gini_pos / n_x, share_largest_q / n_x,
                sum_larger / n_x, obs_larger / n_x, c_dict, share_censored,
                continuous=c_dict['d_type'] == 'continuous',
                d_values_cont=d_values_dr)
    else:
        pot_y = 0.5 * pot_y_prev + 0.5 * pot_y
    if c_dict['with_output'] and c_dict['verbose'] and save_predictions:
        print('\nComputing IATEs 2/2 (effects)')
    if c_dict['d_type'] == 'continuous':
        dim_3 = round(no_of_treat_dr - 1)
    else:
        dim_3 = round(no_of_treat * (no_of_treat - 1) / 2)
    iate = np.empty((n_x, no_of_out, dim_3, 2))
    if c_dict['iate_se_flag']:
        iate_se, iate_p = np.empty_like(iate), np.empty_like(iate)
    else:
        iate_se = iate_p = None
    # obs x outcome x effects x type_of_effect
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(n_x):
            if c_dict['iate_se_flag']:
                ret_all_idx = iate_func2_for_mp(
                    idx, no_of_out, pot_y[idx], pot_y_var[idx],
                    pot_y_m_ate[idx], pot_y_m_ate_var[idx], c_dict,
                    d_values_dr, no_of_treat_dr)
            else:
                ret_all_idx = iate_func2_for_mp(
                    idx, no_of_out, pot_y[idx], None, None, None, c_dict,
                    d_values_dr, no_of_treat_dr)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, n_x)
            iate[idx, :, :, :] = ret_all_idx[1]
            if c_dict['iate_se_flag']:
                iate_se[idx, :, :, :] = ret_all_idx[2]
                iate_p[idx, :, :, :] = ret_all_idx[3]
            if idx == n_x-1:
                effect_list = ret_all_idx[4]
    else:
        if c_dict['_ray_or_dask'] == 'ray':
            if c_dict['mem_object_store_3'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=maxworkers, include_dashboard=False,
                        object_store_memory=c_dict['mem_object_store_3'])
                if c_dict['with_output'] and c_dict['verbose']:
                    print("Size of Ray Object Store: ", round(
                        c_dict['mem_object_store_3']/(1024*1024)), " MB")
            if c_dict['iate_se_flag']:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, pot_y[idx], pot_y_var[idx],
                    pot_y_m_ate[idx], pot_y_m_ate_var[idx], c_dict,
                    d_values_dr, no_of_treat_dr)
                    for idx in range(n_x)]
            else:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, pot_y[idx], None, None, None, c_dict,
                    d_values_dr, no_of_treat_dr) for idx in range(n_x)]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i2 in finished_res:
                    iix = ret_all_i2[0]
                    iate[iix, :, :, :] = ret_all_i2[1]
                    if c_dict['iate_se_flag']:
                        iate_se[iix, :, :, :] = ret_all_i2[2]
                        iate_p[iix, :, :, :] = ret_all_i2[3]
                    if jdx == n_x-1:
                        effect_list = ret_all_i2[4]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, n_x)
                    jdx += 1
            if 'rest' in c_dict['_mp_ray_del']:
                del finished_res, finished
            if c_dict['_mp_ray_shutdown']:
                ray.shutdown()
        elif c_dict['_ray_or_dask'] == 'dask':
            with Client(n_workers=maxworkers) as clt:
                if c_dict['iate_se_flag']:
                    ret_fut = [clt.submit(
                        iate_func2_for_mp, idx, no_of_out, pot_y[idx],
                        pot_y_var[idx], pot_y_m_ate[idx], pot_y_m_ate_var[idx],
                        c_dict, d_values_dr, no_of_treat_dr)
                        for idx in range(n_x)]
                else:
                    ret_fut = [clt.submit(
                        iate_func2_for_mp, idx, no_of_out, pot_y[idx],
                        None, None, None, c_dict, d_values_dr, no_of_treat_dr)
                        for idx in range(n_x)]
                jdx = 0
                for _, res in as_completed(ret_fut, with_results=True):
                    jdx += 1
                    iix, iate[iix, :, :, :] = res[0], res[1]
                    if c_dict['iate_se_flag']:
                        iate_se[iix, :, :, :] = res[2]
                        iate_p[iix, :, :, :] = res[3]
                    if jdx == n_x-1:
                        effect_list = res[4]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, n_x)
        else:
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
                ret_fut = {fpp.submit(
                    iate_func2_for_mp, idx, no_of_out, pot_y[idx],
                    pot_y_var[idx], pot_y_m_ate[idx], pot_y_m_ate_var[idx],
                    c_dict, d_values_dr, no_of_treat_dr):
                        idx for idx in range(n_x)}
                for jdx, frv in enumerate(futures.as_completed(ret_fut)):
                    ret_all_i2 = frv.result()
                    del ret_fut[frv]
                    del frv
                    iix, iate[iix, :, :, :] = ret_all_i2[0], ret_all_i2[1]
                    if c_dict['iate_se_flag']:
                        iate_se[iix, :, :, :] = ret_all_i2[2]
                        iate_p[iix, :, :, :] = ret_all_i2[3]
                    if jdx == n_x-1:
                        effect_list = ret_all_i2[4]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, n_x)
    if c_dict['with_output'] and save_predictions:
        mcf_iate_add.print_iate(
            iate, iate_se, iate_p, effect_list, v_dict, c_dict)
    # Add results to data file
    pot_y_np = np.empty((n_x, no_of_out * no_of_treat_dr))
    if c_dict['iate_se_flag']:
        pot_y_se_np = np.empty_like(pot_y_np)
    if c_dict['d_type'] == 'continuous':
        dim = round(no_of_out * (no_of_treat_dr - 1))
    else:
        dim = round(no_of_out * no_of_treat * (no_of_treat - 1) / 2)
    iate_np = np.empty((n_x, dim))
    if c_dict['iate_se_flag']:
        iate_se_np = np.empty_like(iate_np)
        iate_mate_np = np.empty_like(iate_np)
        iate_mate_se_np = np.empty_like(iate_np)
    jdx = j2dx = jdx_unlc = 0
    name_pot, name_eff, name_eff0 = [], [], []
    if c_dict['l_centering_uncenter']:
        name_pot_unlc = []
        pot_y_unlc_np = np.empty((n_x, no_of_treat_dr))
        if isinstance(v_dict['y_tree_name'], list):
            y_tree_name = v_dict['y_tree_name'][0]
        else:
            y_tree_name = v_dict['y_tree_name']
    else:
        name_pot_unlc = y_tree_name = name_pot_y_unlc = None
    for o_idx, o_name in enumerate(v_dict['y_name']):
        for t_idx, t_name in enumerate(d_values_dr):
            name_pot += [o_name + str(t_name)]
            pot_y_np[:, jdx] = pot_y[:, t_idx, o_idx]
            if o_name == y_tree_name and c_dict['l_centering_uncenter']:
                name_pot_unlc += [o_name + str(t_name) + '_un_lc']
                pot_y_unlc_np[:, jdx_unlc] = pot_y_np[:, jdx].copy()
                jdx_unlc += 1
            if c_dict['iate_se_flag']:
                pot_y_se_np[:, jdx] = np.sqrt(pot_y_var[:, t_idx, o_idx])
            jdx += 1
        for t2_idx, t2_name in enumerate(effect_list):
            name_eff += [o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            if t2_name[1] == d_values_dr[0]:   # Usually, control
                name_eff0 += [
                    o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            iate_np[:, j2dx] = iate[:, o_idx, t2_idx, 0]
            if c_dict['iate_se_flag']:
                iate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 0]
                iate_mate_np[:, j2dx] = iate[:, o_idx, t2_idx, 1]
                iate_mate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 1]
            j2dx += 1
    if reg_round:
        name_pot_y = [s + '_pot' for s in name_pot]
        if c_dict['l_centering_uncenter']:
            name_pot_y_unlc = [s + '_pot' for s in name_pot_unlc]
        name_iate = [s + '_iate' for s in name_eff]
        name_iate0 = [s + '_iate' for s in name_eff0]
    else:
        name_pot_y = [s + '_pot_eff' for s in name_pot]
        if c_dict['l_centering_uncenter']:
            name_pot_y_unlc = [s + '_pot_eff' for s in name_pot_unlc]
        name_iate = [s + '_iate_eff' for s in name_eff]
        name_iate0 = [s + '_iate_eff' for s in name_eff0]
    if c_dict['iate_se_flag']:
        name_pot_y_se = [s + '_pot_se' for s in name_pot]
        name_iate_se = [s + '_iate_se' for s in name_eff]
        name_iate_mate = [s + '_iatemate' for s in name_eff]
        name_iate_mate_se = [s + '_iatemate_se' for s in name_eff]
        name_iate_se0 = [s + '_iate_se' for s in name_eff0]
        name_iate_mate0 = [s + '_iatemate' for s in name_eff0]
        name_iate_mate_se0 = [s + '_iatemate_se' for s in name_eff0]
    else:
        name_pot_y_se = name_iate_se = name_iate_mate = None
        name_iate_mate_se = name_iate_se0 = name_iate_mate0 = None
        name_iate_mate_se0 = None
    if (c_dict['with_output'] and save_predictions) or c_dict[
            '_return_iate_sp']:
        pot_y_df = pd.DataFrame(data=pot_y_np, columns=name_pot_y)
        iate_df = pd.DataFrame(data=iate_np, columns=name_iate)
        if c_dict['iate_se_flag']:
            pot_y_se_df = pd.DataFrame(data=pot_y_se_np, columns=name_pot_y_se)
            iate_se_df = pd.DataFrame(data=iate_se_np, columns=name_iate_se)
            iate_mate_df = pd.DataFrame(data=iate_mate_np,
                                        columns=name_iate_mate)
            iate_mate_se_df = pd.DataFrame(data=iate_mate_se_np,
                                           columns=name_iate_mate_se)
        if reg_round:
            data_df = pd.read_csv(data_file)
        else:
            data_df = pd.read_csv(c_dict['pred_sample_with_pred'])
        if c_dict['iate_se_flag']:
            df_list = [data_df, pot_y_df, pot_y_se_df, iate_df, iate_se_df,
                       iate_mate_df, iate_mate_se_df]
        else:
            df_list = [data_df, pot_y_df, iate_df]
        if c_dict['l_centering_uncenter']:
            x_pred_np = find_x_to_uncenter(data_df, var_x_type)
            try:
                y_x_lc = lc_forest.predict(x_pred_np)
            except RuntimeError:
                y_x_lc = np.empty(len(x_pred_np)) * np.nan
                if c_dict['with_output']:
                    print('Uncentering not successful.')
            pot_y_unlc_np += np.reshape(y_x_lc, (-1, 1))
            pot_y_unlc_df = pd.DataFrame(data=pot_y_unlc_np,
                                         columns=name_pot_y_unlc)
            df_list.append(pot_y_unlc_df)
        data_pred_new = pd.concat(df_list, axis=1)
        if ((c_dict['with_output'] and save_predictions)
            or (save_predictions and reg_round and c_dict['iate_eff_flag'])
            or (save_predictions and c_dict['iate_eff_flag']
                and c_dict['_return_iate_sp'])):
            gp.delete_file_if_exists(c_dict['pred_sample_with_pred'])
            data_pred_new.to_csv(c_dict['pred_sample_with_pred'], index=False)
        if c_dict['with_output']:
            gp.print_descriptive_stats_file(
                c_dict['pred_sample_with_pred'], 'all',
                c_dict['print_to_file'])
    names_pot_iate = {
        'names_pot_y': name_pot_y, 'names_pot_y_se': name_pot_y_se,
        'names_iate': name_iate, 'names_iate_se': name_iate_se,
        'names_iate_mate': name_iate_mate,
        'names_iate_mate_se': name_iate_mate_se,
        'names_pot_y_uncenter': name_pot_y_unlc}
    names_pot_iate0 = {
        'names_pot_y': name_pot_y, 'names_pot_y_se': name_pot_y_se,
        'names_iate': name_iate0, 'names_iate_se': name_iate_se0,
        'names_iate_mate': name_iate_mate0,
        'names_iate_mate_se': name_iate_mate_se0,
        'names_pot_y_uncenter': name_pot_y_unlc}
    if not c_dict['_return_iate_sp']:
        data_pred_new = None
    return (c_dict['pred_sample_with_pred'], pot_y, pot_y_var, iate, iate_se,
            (names_pot_iate, names_pot_iate0), data_pred_new)


def find_x_to_uncenter(data_df, var_x_type):
    "Find correct x to uncenter potential outcomes."
    x_names = var_x_type.keys()
    x_df = data_df[x_names]
    # This part must be identical to the same part in local centering!
    # To Do: Use same function in both.
    names_unordered = [x_name for x_name in x_names if var_x_type[x_name] > 0]
    if names_unordered:  # List is not empty
        x_dummies = pd.get_dummies(x_df, columns=names_unordered)
        x_df = pd.concat([x_df[names_unordered], x_dummies], axis=1)
    return x_df.to_numpy()


def assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                     share_censored, ret_all_i, n_x, idx=None):
    """Use to avoid duplicate code."""
    if idx is None:
        idx = ret_all_i[0]
    pot_y[idx, :, :] = ret_all_i[1]
    if pot_y_var is not None:
        pot_y_var[idx, :, :] = ret_all_i[2]
        pot_y_m_ate[idx, :, :] = ret_all_i[3]
        pot_y_m_ate_var[idx, :, :] = ret_all_i[4]
    l1_to_9[idx] = ret_all_i[5]
    share_censored += ret_all_i[6] / n_x
    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored)


@ray.remote
def ray_iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                          pot_y_m_ate_var_i, c_dict, d_values, no_of_treat):
    """Make function compatible with Ray."""
    return iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i,
                             pot_y_m_ate_i, pot_y_m_ate_var_i, c_dict,
                             d_values, no_of_treat)


def iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                      pot_y_m_ate_var_i, c_dict, d_values, no_of_treat):
    """
    Do computations for IATE with MP. Second chunck.

    Parameters
    ----------
    i : Int. Counter.
    no_of_out : Int. Number of outcomes.
    pot_y_i : Numpy array.
    pot_y_var_i : Numpy array.
    pot_y_m_ate_i : Numpy array.
    pot_y_m_ate_var_i : Numpy array.
    c_dict : Dict. Parameters.

    Returns
    -------
    iate_i : Numpy array.
    iate_se_i : Numpy array.
    iate_p_i : Numpy array.
    effect_list : List.
    """
    # obs x outcome x effects x type_of_effect
    if c_dict['d_type'] == 'continuous':
        dim = (no_of_out, no_of_treat - 1, 2)
    else:
        dim = (no_of_out, round(no_of_treat * (no_of_treat - 1) / 2), 2)
    iate_i = np.empty(dim)
    if c_dict['iate_se_flag']:
        iate_se_i = np.empty(dim)  # obs x outcome x effects x type_of_effect
        iate_p_i = np.empty(dim)
        iterator = 2
    else:
        iate_se_i = iate_p_i = None
        iterator = 1
    for o_i in range(no_of_out):
        for jdx in range(iterator):
            if jdx == 0:
                pot_y_ao = pot_y_i[:, o_i]
                pot_y_var_ao = (pot_y_var_i[:, o_i] if c_dict['iate_se_flag']
                                else None)
            else:
                pot_y_ao = pot_y_m_ate_i[:, o_i]
                pot_y_var_ao = pot_y_m_ate_var_i[:, o_i]
            ret = mcf_gp.effect_from_potential(
                pot_y_ao, pot_y_var_ao, d_values,
                se_yes=c_dict['iate_se_flag'],
                continuous=c_dict['d_type'] == 'continuous')
            if c_dict['iate_se_flag']:
                (iate_i[o_i, :, jdx], iate_se_i[o_i, :, jdx], _,
                 iate_p_i[o_i, :, jdx], effect_list) = ret
            else:
                (iate_i[o_i, :, jdx], _, _, _, effect_list) = ret
    return idx, iate_i, iate_se_i, iate_p_i, effect_list


@ray.remote
def ray_iate_func1_for_mp_many_obs(
        idx_list, weights_list, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
        no_of_out, n_y, c_dict):
    """Compute IATE for several obs in one loop (MP)."""
    return iate_func1_for_mp_many_obs(
        idx_list, weights_list, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
        no_of_out, n_y, c_dict)


def iate_func1_for_mp_many_obs(idx_list, weights_list, cl_dat, no_of_cluster,
                               w_dat, w_ate, y_dat, no_of_out, n_y, c_dict):
    """Compute IATE for several obs in one loop (MP)."""
    ret_all = []
    if c_dict['weight_as_sparse']:
        iterator = len(weights_list)
    for i, idx_org in enumerate(idx_list):
        if c_dict['weight_as_sparse']:
            weights_i = [weights_list[t_idx].getrow(i)
                         for t_idx in range(iterator)]
        else:
            weights_i = weights_list[i]
        ret = iate_func1_for_mp(idx_org, weights_i, cl_dat, no_of_cluster,
                                w_dat, w_ate, y_dat, no_of_out, n_y, c_dict)
        ret_all.append(ret)
    return ret_all


@ray.remote
def ray_iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat, w_ate,
                          y_dat, no_of_out, n_y, c_dict):
    """Make function useful for Ray."""
    return iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat,
                             w_ate, y_dat, no_of_out, n_y, c_dict)


def iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat, w_ate,
                      y_dat, no_of_out, n_y, c_dict):
    """
    Compute function to be looped over observations for Multiprocessing.

    Parameters
    ----------
    idx : Int. Counter.
    weights_i : List of int. Indices of non-zero weights.
                Alternative: Sparse csr matrix
    cl_dat : Numpy vector. Cluster variable.
    no_of_cluster : Int. Number of clusters.
    w_dat : Numpy vector. Sampling weights.
    w_ate : Numpy array. Weights for ATE.
    y_dat : Numpy array. Outcome variable.
    no_of_out : Int. Number of outcomes.
    n_y : Int. Length of outcome data.
    c_dict : Dict. Parameters.

    Returns
    -------
    idx: Int. Counter.
    pot_y_i: Numpy array.
    pot_y_var_i: Numpy array.
    pot_y_m_ate_i: Numpy array.
    pot_y_m_ate_var_i: Numpy array.
    l1_to_9: Tuple of lists.
    """
    def get_walli(w_index, n_y, w_i):
        w_all_i = np.zeros(n_y)
        w_all_i[w_index] = w_i
        w_all_i_unc = np.zeros_like(w_all_i)
        w_all_i_unc[w_index] = w_i_unc
        return w_all_i, w_all_i_unc

    if (c_dict['with_output'] and (idx == 0)
            and c_dict['_ray_or_dask'] != 'ray' and c_dict['verbose']):
        print('Starting to compute IATE - procedure 1', flush=True)
    if c_dict['d_type'] == 'continuous':
        continuous = True
        no_of_treat = c_dict['ct_grid_w']
        i_w01 = c_dict['ct_w_to_dr_int_w01']
        i_w10 = c_dict['ct_w_to_dr_int_w10']
        index_full = c_dict['ct_w_to_dr_index_full']
        d_values_dr = c_dict['ct_d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
    else:
        continuous = False
        d_values_dr = c_dict['d_values']
        no_of_treat = no_of_treat_dr = c_dict['no_of_treat']
    pot_y_i = np.empty((no_of_treat_dr, no_of_out))
    share_i = np.zeros(no_of_treat_dr)
    if c_dict['iate_se_flag']:
        pot_y_var_i = np.empty_like(pot_y_i)
        pot_y_m_ate_i = np.empty_like(pot_y_i)
        pot_y_m_ate_var_i = np.empty_like(pot_y_i)
        cluster_std = c_dict['cluster_std']
    else:
        pot_y_var_i = pot_y_m_ate_i = pot_y_m_ate_var_i = w_ate = None
        cluster_std = False
    w_add = (np.zeros((no_of_treat_dr, no_of_cluster)) if cluster_std
             else np.zeros((no_of_treat_dr, n_y)))
    if c_dict['iate_se_flag']:
        w_add_unc = np.zeros((no_of_treat_dr, n_y))
    for t_idx in range(no_of_treat):
        extra_weight_p1 = continuous and t_idx < no_of_treat-1
        if c_dict['weight_as_sparse']:
            w_index = weights_i[t_idx].indices
            w_i_t = weights_i[t_idx].data
            if extra_weight_p1:
                w_index_p1 = weights_i[t_idx+1].indices
                w_index_both = np.unique(np.concatenate((w_index, w_index_p1)))
                w_i = np.zeros(n_y)
                w_i[w_index] = w_i_t
                w_i_p1 = np.zeros_like(w_i)
                w_i_p1[w_index_p1] = weights_i[t_idx+1].data
                w_i = w_i[w_index_both]
                w_i_p1 = w_i_p1[w_index_both]
            else:
                w_index_both = w_index
                w_i = w_i_t
        else:
            w_index = weights_i[t_idx][0]    # Indices of non-zero weights
            w_i_t = weights_i[t_idx][1].copy()
            if extra_weight_p1:
                w_index_p1 = weights_i[t_idx+1][0]
                w_index_both = np.unique(np.concatenate((w_index, w_index_p1)))
                w_i = np.zeros(n_y)
                w_i[w_index] = w_i_t
                w_i_p1 = np.zeros_like(w_i)
                w_i_p1[w_index_p1] = weights_i[t_idx+1][1].copy()
                w_i = w_i[w_index_both]
                w_i_p1 = w_i_p1[w_index_both]
            else:
                w_index_both = w_index
                w_i = w_i_t
        if c_dict['w_yes']:
            w_t = w_dat[w_index].reshape(-1)
            w_i = w_i * w_t
            if extra_weight_p1:
                w_t_p1 = w_dat[w_index_p1].reshape(-1)
                w_i_p1 = w_i_p1 * w_t_p1
        else:
            w_t = None
            if extra_weight_p1:
                w_t_p1 = None
        w_i_sum = np.sum(w_i)
        if (not (1-1e-10) < w_i_sum < (1+1e-10)) and not continuous:
            w_i = w_i / w_i_sum
        w_i_unc = np.copy(w_i)
        if c_dict['max_weight_share'] < 1 and not continuous:
            w_i, _, share_i[t_idx] = mcf_gp.bound_norm_weights(
                w_i, c_dict['max_weight_share'])
        if extra_weight_p1:
            w_i_unc_p1 = np.copy(w_i_p1)
        if cluster_std:
            w_all_i, w_all_i_unc = get_walli(w_index, n_y, w_i)
            cl_i = cl_dat[w_index]
            if extra_weight_p1:
                w_all_i_p1, w_all_i_unc_p1 = get_walli(w_index_p1, n_y, w_i_p1)
                cl_i_both = cl_dat[w_index_both]
        else:
            cl_i = cl_i_both = None
        for o_idx in range(no_of_out):
            if continuous:
                for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                    y_dat_cont = y_dat[w_index_both, o_idx]
                    if extra_weight_p1:
                        w_i_cont = w10 * w_i + w01 * w_i_p1
                        w_i_unc_cont = w10 * w_i_unc + w01 * w_i_unc_p1
                        w_t_cont = (None if w_t is None
                                    else w10 * w_t + w01 * w_t_p1)
                        cl_i_cont = cl_i_both
                    else:
                        w_i_cont, w_t_cont, cl_i_cont = w_i, w_t, cl_i_both
                        w_i_unc_cont = w_i_unc
                    w_i_cont = w_i_cont / np.sum(w_i_cont)
                    if w_t_cont is not None:
                        w_t_cont = w_t_cont / np.sum(w_t_cont)
                    w_i_unc_cont = w_i_unc_cont / np.sum(w_i_unc_cont)
                    if c_dict['max_weight_share'] < 1:
                        w_i_cont, _, share_cont = mcf_gp.bound_norm_weights(
                            w_i_cont, c_dict['max_weight_share'])
                        if i == 0:
                            share_i[t_idx] = share_cont
                    ret = gp_est.weight_var(
                        w_i_cont, y_dat_cont, cl_i_cont, c_dict,
                        weights=w_t_cont, se_yes=c_dict['iate_se_flag'],
                        bootstrap=c_dict['se_boot_iate'])
                    ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                    pot_y_i[ti_idx, o_idx] = ret[0]
                    if c_dict['iate_se_flag']:
                        pot_y_var_i[ti_idx, o_idx] = ret[1]
                    if cluster_std:
                        w_cont = (w10 * w_all_i + w01 * w_all_i_p1
                                  if extra_weight_p1 else w_all_i)
                        ret2 = gp_est.aggregate_cluster_pos_w(
                            cl_dat, w_cont, y_dat[:, o_idx], sweights=w_dat)
                        if o_idx == 0:
                            w_add[ti_idx, :] = np.copy(ret2[0])
                            if c_dict['iate_se_flag']:
                                if w_ate is None:
                                    w_diff = (w10 * w_all_i_unc
                                              + w01 * w_all_i_unc_p1)
                                else:
                                    if extra_weight_p1:
                                        w_ate_cont = (w10 * w_ate[t_idx, :] +
                                                      w01 * w_ate[t_idx+1, :])
                                        w_ate_cont /= np.sum(w_ate_cont)
                                        w_diff = w_all_i_unc - w_ate_cont
                                    else:
                                        w_diff = w_all_i_unc - w_ate[t_idx, :]
                        ret = gp_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat, c_dict,
                            norm=False, weights=w_dat,
                            bootstrap=c_dict['se_boot_iate'],
                            se_yes=c_dict['iate_se_flag'])
                    else:
                        if o_idx == 0:
                            w_add[ti_idx, w_index_both] = ret[2]
                            if c_dict['iate_se_flag']:
                                w_i_unc_sum = np.sum(w_i_unc_cont)
                                if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                                    w_add_unc[ti_idx, w_index_both] = (
                                        w_i_unc_cont / w_i_unc_sum)
                                else:
                                    w_add_unc[ti_idx, w_index_both
                                              ] = w_i_unc_cont
                                if w_ate is None:
                                    w_diff = w_add_unc[ti_idx, :]
                                else:
                                    if extra_weight_p1:
                                        w_ate_cont = (w10 * w_ate[t_idx, :] +
                                                      w01 * w_ate[t_idx+1, :])
                                        w_ate_cont /= np.sum(w_ate_cont)
                                        w_diff = (w_add_unc[ti_idx, :]
                                                  - w_ate_cont)
                                    else:
                                        w_diff = (w_add_unc[ti_idx, :]
                                                  - w_ate[t_idx, :])
                        if c_dict['iate_se_flag']:
                            ret = gp_est.weight_var(
                                w_diff, y_dat[:, o_idx], None, c_dict,
                                norm=False, weights=w_dat,
                                bootstrap=c_dict['se_boot_iate'],
                                se_yes=c_dict['iate_se_flag'])
                    if c_dict['iate_se_flag']:
                        pot_y_m_ate_i[ti_idx, o_idx] = ret[0]
                        pot_y_m_ate_var_i[ti_idx, o_idx] = ret[1]
                    if not extra_weight_p1:
                        break
            else:  # discrete treatment
                ret = gp_est.weight_var(
                    w_i, y_dat[w_index, o_idx], cl_i, c_dict, weights=w_t,
                    se_yes=c_dict['iate_se_flag'],
                    bootstrap=c_dict['se_boot_iate'])
                pot_y_i[t_idx, o_idx] = ret[0]
                if c_dict['iate_se_flag']:
                    pot_y_var_i[t_idx, o_idx] = ret[1]
                if cluster_std:
                    ret2 = gp_est.aggregate_cluster_pos_w(
                        cl_dat, w_all_i, y_dat[:, o_idx], sweights=w_dat)
                    if o_idx == 0:
                        w_add[t_idx, :] = np.copy(ret2[0])
                        if c_dict['iate_se_flag']:
                            if w_ate is None:
                                w_diff = w_all_i_unc  # Dummy if no w_ate
                            else:
                                w_diff = w_all_i_unc - w_ate[t_idx, :]
                    ret = gp_est.weight_var(
                        w_diff, y_dat[:, o_idx], cl_dat, c_dict, norm=False,
                        weights=w_dat, bootstrap=c_dict['se_boot_iate'],
                        se_yes=c_dict['iate_se_flag'])
                else:
                    if o_idx == 0:
                        w_add[t_idx, w_index] = ret[2]
                        if c_dict['iate_se_flag']:
                            w_i_unc_sum = np.sum(w_i_unc)
                            if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                                w_add_unc[t_idx, w_index] = (w_i_unc
                                                             / w_i_unc_sum)
                            else:
                                w_add_unc[t_idx, w_index] = w_i_unc
                            if w_ate is None:
                                w_diff = w_add_unc[t_idx, :]
                            else:
                                w_diff = w_add_unc[t_idx, :] - w_ate[t_idx, :]
                    if c_dict['iate_se_flag']:
                        ret = gp_est.weight_var(
                            w_diff, y_dat[:, o_idx], None, c_dict, norm=False,
                            weights=w_dat, bootstrap=c_dict['se_boot_iate'],
                            se_yes=c_dict['iate_se_flag'])
                if c_dict['iate_se_flag']:
                    pot_y_m_ate_i[t_idx, o_idx] = ret[0]
                    pot_y_m_ate_var_i[t_idx, o_idx] = ret[1]
    l1_to_9 = mcf_ate.analyse_weights_ate(
        w_add, None, c_dict, ate=False, continuous=continuous,
        no_of_treat_cont=no_of_treat_dr, d_values_cont=d_values_dr)
    return (idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
            l1_to_9, share_i)
