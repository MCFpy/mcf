"""
Procedures needed for IATE estimation.

Created on Thu Dec 8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
from concurrent import futures
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as sct
import matplotlib.pyplot as plt
import ray
from mcf import mcf_ate_functions as mcf_ate
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_mcf as gp_mcf


def iate_est_mp(weights, data_file, y_dat, cl_dat, w_dat, v_dict, c_dict,
                w_ate=None, balancing_test=False, save_predictions=True):
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
    v : Dict. Variables.
    c : Dict. Parameters.
    w_ate: Numpy array. Weights of ATE estimation. Default = None.
    balancing_test : Bool. Balancing test. Default = False.
    save_predictions : Bool. save_predictions = True.

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
    if c_dict['with_output'] and c_dict['verbose'] and save_predictions:
        print('\nComputing IATEs 1/2 (potential outcomes)')
    if c_dict['weight_as_sparse']:
        n_x = weights[0].shape[0]
    else:
        n_x = len(weights)
    n_y = len(y_dat)
    no_of_out = len(v_dict['y_name'])
    larger_0 = np.zeros(c_dict['no_of_treat'])
    equal_0 = np.zeros(c_dict['no_of_treat'])
    mean_pos = np.zeros(c_dict['no_of_treat'])
    std_pos = np.zeros(c_dict['no_of_treat'])
    gini_all = np.zeros(c_dict['no_of_treat'])
    gini_pos = np.zeros(c_dict['no_of_treat'])
    share_censored = np.zeros(c_dict['no_of_treat'])
    share_largest_q = np.zeros((c_dict['no_of_treat'], 3))
    sum_larger = np.zeros((c_dict['no_of_treat'], len(c_dict['q_w'])))
    obs_larger = np.zeros((c_dict['no_of_treat'], len(c_dict['q_w'])))
    pot_y = np.empty((n_x, c_dict['no_of_treat'], no_of_out))
    pot_y_var = np.empty((n_x, c_dict['no_of_treat'], no_of_out))
    pot_y_m_ate = np.empty((n_x, c_dict['no_of_treat'], no_of_out))
    pot_y_m_ate_var = np.empty((n_x, c_dict['no_of_treat'], no_of_out))
    if w_ate is not None:
        w_ate = w_ate[0, :, :]
    if not c_dict['w_yes']:
        w_dat = None
    if c_dict['cluster_std']:
        no_of_cluster = len(np.unique(cl_dat))
    else:
        no_of_cluster = None
    l1_to_9 = [None] * n_x
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = gp_mcf.find_no_of_workers(c_dict['no_parallel'],
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
        if c_dict['obs_by_obs']:
            if c_dict['mp_with_ray']:
                if c_dict['mem_object_store_3'] is None:
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_3']/(1024*1024)), " MB")
                if c_dict['weight_as_sparse']:
                    tasks = [ray_iate_func1_for_mp.remote(
                        idx, [weights[t_idx].getrow(idx) for t_idx in
                              range(iterator)], cl_dat, no_of_cluster, w_dat,
                        w_ate, y_dat, no_of_out, n_y, c_dict)
                        for idx in range(n_x)]
                else:
                    tasks = [ray_iate_func1_for_mp.remote(
                        idx, weights[idx], cl_dat, no_of_cluster, w_dat,
                        w_ate, y_dat, no_of_out, n_y, c_dict)
                        for idx in range(n_x)]
                still_running = list(tasks)
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
                del finished, still_running, tasks
                ray.shutdown()
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
            no_of_splits = max(no_of_splits, maxworkers)
            no_of_splits = min(no_of_splits, n_x)
            if c_dict['with_output'] and c_dict['verbose']:
                print('IATE-1: Avg. number of obs per split: {:5.2f}'.format(
                    n_x / no_of_splits))
            obs_idx_list = np.array_split(np.arange(n_x), no_of_splits)
            if c_dict['mp_with_ray']:
                if c_dict['mem_object_store_3'] is None:
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_3']/(1024*1024)), " MB")
                if c_dict['weight_as_sparse']:
                    tasks = [ray_iate_func1_for_mp_many_obs.remote(
                        idx,
                        [weights[t_idx][idx, :] for t_idx in range(iterator)],
                        cl_dat, no_of_cluster, w_dat, w_ate, y_dat, no_of_out,
                        n_y, c_dict) for idx in obs_idx_list]
                else:
                    tasks = [ray_iate_func1_for_mp_many_obs.remote(
                        idx, [weights[idxx] for idxx in idx], cl_dat,
                        no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                        c_dict) for idx in obs_idx_list]
                still_running = list(tasks)
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
                del finished, still_running        
                ray.shutdown()
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
            sum_larger / n_x, obs_larger / n_x, c_dict, share_censored)
    if c_dict['with_output'] and c_dict['verbose'] and save_predictions:
        print('\nComputing IATEs 2/2 (effects)')
    dim_3 = round(c_dict['no_of_treat'] * (c_dict['no_of_treat'] - 1) / 2)
    iate = np.empty((n_x, no_of_out, dim_3, 2))     # obs x outcome x
    iate_se = np.empty((n_x, no_of_out, dim_3, 2))  # effects x type_of_effect
    iate_p = np.empty((n_x, no_of_out, dim_3, 2))
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = gp_mcf.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(n_x):
            ret_all_idx = iate_func2_for_mp(
                idx, no_of_out, pot_y[idx], pot_y_var[idx], pot_y_m_ate[idx],
                pot_y_m_ate_var[idx], c_dict)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, n_x)
            iate[idx, :, :, :] = ret_all_idx[1]
            iate_se[idx, :, :, :] = ret_all_idx[2]
            iate_p[idx, :, :, :] = ret_all_idx[3]
            if idx == n_x-1:
                effect_list = ret_all_idx[4]
    else:
        if c_dict['mp_with_ray']:
            if c_dict['mem_object_store_3'] is None:
                ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                ray.init(num_cpus=maxworkers, include_dashboard=False,
                         object_store_memory=c_dict['mem_object_store_3'])
                if c_dict['with_output'] and c_dict['verbose']:
                    print("Size of Ray Object Store: ", round(
                        c_dict['mem_object_store_3']/(1024*1024)), " MB")
            tasks = [ray_iate_func2_for_mp.remote(
                idx, no_of_out, pot_y[idx], pot_y_var[idx], pot_y_m_ate[idx],
                pot_y_m_ate_var[idx], c_dict) for idx in range(n_x)]
            still_running = list(tasks)
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i2 in finished_res:
                    iix = ret_all_i2[0]
                    iate[iix, :, :, :] = ret_all_i2[1]
                    iate_se[iix, :, :, :] = ret_all_i2[2]
                    iate_p[iix, :, :, :] = ret_all_i2[3]
                    if jdx == n_x-1:
                        effect_list = ret_all_i2[4]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, n_x)
                    jdx += 1
            del finished, still_running, tasks
            ray.shutdown()
        else:
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
                ret_fut = {fpp.submit(
                    iate_func2_for_mp, idx, no_of_out, pot_y[idx],
                    pot_y_var[idx], pot_y_m_ate[idx], pot_y_m_ate_var[idx],
                    c_dict): idx for idx in range(n_x)}
                for jdx, frv in enumerate(futures.as_completed(ret_fut)):
                    ret_all_i2 = frv.result()
                    del ret_fut[frv]
                    del frv
                    iix = ret_all_i2[0]
                    iate[iix, :, :, :] = ret_all_i2[1]
                    iate_se[iix, :, :, :] = ret_all_i2[2]
                    iate_p[iix, :, :, :] = ret_all_i2[3]
                    if jdx == n_x-1:
                        effect_list = ret_all_i2[4]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, n_x)
    if c_dict['with_output'] and save_predictions:
        print_iate(iate, iate_se, iate_p, effect_list, v_dict)
    # Add results to data file
    pot_y_np = np.empty((n_x, no_of_out * c_dict['no_of_treat']))
    pot_y_se_np = np.empty((n_x, no_of_out * c_dict['no_of_treat']))
    dim = round(no_of_out * c_dict['no_of_treat'] * (
        c_dict['no_of_treat'] - 1) / 2)
    iate_np = np.empty((n_x, dim))
    iate_se_np = np.empty((n_x, dim))
    iate_mate_np = np.empty((n_x, dim))
    iate_mate_se_np = np.empty((n_x, dim))
    name_pot = []
    name_eff = []
    name_eff0 = []
    jdx = 0
    j2dx = 0
    for o_idx, o_name in enumerate(v_dict['y_name']):
        for t_idx, t_name in enumerate(c_dict['d_values']):
            name_pot += [o_name + str(t_name)]
            pot_y_np[:, jdx] = pot_y[:, t_idx, o_idx]
            pot_y_se_np[:, jdx] = np.sqrt(pot_y_var[:, t_idx, o_idx])
            jdx += 1
        for t2_idx, t2_name in enumerate(effect_list):
            name_eff += [o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            if t2_name[1] == c_dict['d_values'][0]:   # Usually, control
                name_eff0 += [
                    o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            iate_np[:, j2dx] = iate[:, o_idx, t2_idx, 0]
            iate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 0]
            iate_mate_np[:, j2dx] = iate[:, o_idx, t2_idx, 1]
            iate_mate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 1]
            j2dx += 1
    name_pot_y = [s + '_pot' for s in name_pot]
    name_pot_y_se = [s + '_pot_se' for s in name_pot]
    name_iate = [s + '_iate' for s in name_eff]
    name_iate_se = [s + '_iate_se' for s in name_eff]
    name_iate_mate = [s + '_iatemate' for s in name_eff]
    name_iate_mate_se = [s + '_iatemate_se' for s in name_eff]
    name_iate0 = [s + '_iate' for s in name_eff0]
    name_iate_se0 = [s + '_iate_se' for s in name_eff0]
    name_iate_mate0 = [s + '_iatemate' for s in name_eff0]
    name_iate_mate_se0 = [s + '_iatemate_se' for s in name_eff0]
    if c_dict['with_output'] and save_predictions:
        pot_y_df = pd.DataFrame(data=pot_y_np, columns=name_pot_y)
        pot_y_se_df = pd.DataFrame(data=pot_y_se_np, columns=name_pot_y_se)
        iate_df = pd.DataFrame(data=iate_np, columns=name_iate)
        iate_se_df = pd.DataFrame(data=iate_se_np, columns=name_iate_se)
        iate_mate_df = pd.DataFrame(data=iate_mate_np, columns=name_iate_mate)
        iate_mate_se_df = pd.DataFrame(data=iate_mate_se_np,
                                       columns=name_iate_mate_se)
        data_df = pd.read_csv(data_file)
        df_list = [data_df, pot_y_df, pot_y_se_df, iate_df, iate_se_df,
                   iate_mate_df, iate_mate_se_df]
        data_file_new = pd.concat(df_list, axis=1)
        gp.delete_file_if_exists(c_dict['pred_sample_with_pred'])
        data_file_new.to_csv(c_dict['pred_sample_with_pred'], index=False)
        if c_dict['with_output']:
            gp.print_descriptive_stats_file(
                c_dict['pred_sample_with_pred'], 'all',
                c_dict['print_to_file'])
    names_pot_iate = {'names_pot_y': name_pot_y,
                      'names_pot_y_se': name_pot_y_se,
                      'names_iate': name_iate,
                      'names_iate_se': name_iate_se,
                      'names_iate_mate': name_iate_mate,
                      'names_iate_mate_se': name_iate_mate_se}
    names_pot_iate0 = {'names_pot_y': name_pot_y,
                       'names_pot_y_se': name_pot_y_se,
                       'names_iate': name_iate0,
                       'names_iate_se': name_iate_se0,
                       'names_iate_mate': name_iate_mate0,
                       'names_iate_mate_se': name_iate_mate_se0}
    return (c_dict['pred_sample_with_pred'], pot_y, pot_y_var, iate, iate_se,
            (names_pot_iate, names_pot_iate0))


def assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                     share_censored, ret_all_i, n_x, idx=None):
    """Use to avoid duplicate code."""
    if idx is None:
        idx = ret_all_i[0]
    pot_y[idx, :, :] = ret_all_i[1]
    pot_y_var[idx, :, :] = ret_all_i[2]
    pot_y_m_ate[idx, :, :] = ret_all_i[3]
    pot_y_m_ate_var[idx, :, :] = ret_all_i[4]
    l1_to_9[idx] = ret_all_i[5]
    share_censored += ret_all_i[6] / n_x
    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored)


@ray.remote
def ray_iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                          pot_y_m_ate_var_i, c_dict):
    """Make function compatible with Ray."""
    return iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i,
                             pot_y_m_ate_i, pot_y_m_ate_var_i, c_dict)


def iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                      pot_y_m_ate_var_i, c_dict):
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
    c : Dict. Parameters.

    Returns
    -------
    iate_i : Numpy array.
    iate_se_i : Numpy array.
    iate_p_i : Numpy array.
    effect_list : List.
    """
    dim = (no_of_out, round(c_dict['no_of_treat'] * (
                          c_dict['no_of_treat'] - 1) / 2), 2)
    iate_i = np.empty(dim)
    iate_se_i = np.empty(dim)  # obs x outcome x effects x type_of_effect
    iate_p_i = np.empty(dim)
    for o_i in range(no_of_out):
        for jdx in range(2):
            if jdx == 0:
                pot_y_ao = pot_y_i[:, o_i]
                pot_y_var_ao = pot_y_var_i[:, o_i]
            else:
                pot_y_ao = pot_y_m_ate_i[:, o_i]
                pot_y_var_ao = pot_y_m_ate_var_i[:, o_i]
            (iate_i[o_i, :, jdx], iate_se_i[o_i, :, jdx], _,
             iate_p_i[o_i, :, jdx], effect_list
             ) = gp_mcf.effect_from_potential(
                 pot_y_ao, pot_y_var_ao, c_dict['d_values'])
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
    if c_dict['with_output'] and (idx == 0) and not c_dict[
            'mp_with_ray'] and c_dict['verbose']:
        print('Starting to compute IATE - procedure 1', flush=True)
    pot_y_i = np.empty((c_dict['no_of_treat'], no_of_out))
    share_i = np.zeros(c_dict['no_of_treat'])
    pot_y_var_i = np.empty((c_dict['no_of_treat'], no_of_out))
    pot_y_m_ate_i = np.empty((c_dict['no_of_treat'], no_of_out))
    pot_y_m_ate_var_i = np.empty((c_dict['no_of_treat'], no_of_out))
    if c_dict['cluster_std']:
        w_add = np.zeros((c_dict['no_of_treat'], no_of_cluster))
    else:
        w_add = np.zeros((c_dict['no_of_treat'], n_y))
    w_add_unc = np.zeros((c_dict['no_of_treat'], n_y))
    for t_idx in range(c_dict['no_of_treat']):
        if c_dict['weight_as_sparse']:
            w_index = weights_i[t_idx].indices
            # w_i = weights_i[t_idx].data.copy() Copy already made into func
            w_i = weights_i[t_idx].data
        else:
            w_index = weights_i[t_idx][0]  # Indices of non-zero weights
            w_i = weights_i[t_idx][1].copy()
        if c_dict['w_yes']:
            w_t = w_dat[w_index].reshape(-1)
            w_i = w_i * w_t
        else:
            w_t = None
        w_i_sum = np.sum(w_i)
        if not (1-1e-10) < w_i_sum < (1+1e-10):
            w_i = w_i / w_i_sum
        w_i_unc = np.copy(w_i)
        if c_dict['max_weight_share'] < 1:
            w_i, _, share_i[t_idx] = gp_mcf.bound_norm_weights(
                w_i, c_dict['max_weight_share'])
        if c_dict['cluster_std']:
            cl_i = cl_dat[w_index]
            w_all_i = np.zeros(n_y)
            w_all_i_unc = np.zeros(n_y)
            w_all_i[w_index] = w_i
            w_all_i_unc[w_index] = w_i_unc
        else:
            cl_i = 0
        for o_idx in range(no_of_out):
            ret = gp_est.weight_var(w_i, y_dat[w_index, o_idx], cl_i, c_dict,
                                    weights=w_t,
                                    bootstrap=c_dict['se_boot_iate'])
            pot_y_i[t_idx, o_idx] = ret[0]
            pot_y_var_i[t_idx, o_idx] = ret[1]
            if c_dict['cluster_std']:
                ret2 = gp_est.aggregate_cluster_pos_w(
                    cl_dat, w_all_i, y_dat[:, o_idx], sweights=w_dat)
                if o_idx == 0:
                    w_add[t_idx, :] = np.copy(ret2[0])
                    if w_ate is None:
                        w_diff = w_all_i_unc  # Dummy if no w_ate
                    else:
                        w_diff = w_all_i_unc - w_ate[t_idx, :]
                ret = gp_est.weight_var(
                    w_diff, y_dat[:, o_idx], cl_dat, c_dict, norm=False,
                    weights=w_dat, bootstrap=c_dict['se_boot_iate'])
            else:
                if o_idx == 0:
                    w_add[t_idx, w_index] = ret[2]
                    w_i_unc_sum = np.sum(w_i_unc)
                    if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                        w_add_unc[t_idx, w_index] = w_i_unc / w_i_unc_sum
                    else:
                        w_add_unc[t_idx, w_index] = w_i_unc
                    if w_ate is None:
                        w_diff = w_add_unc[t_idx, :]
                    else:
                        w_diff = w_add_unc[t_idx, :] - w_ate[t_idx, :]
                ret = gp_est.weight_var(
                    w_diff, y_dat[:, o_idx], None, c_dict, norm=False,
                    weights=w_dat, bootstrap=c_dict['se_boot_iate'])
            pot_y_m_ate_i[t_idx, o_idx] = ret[0]
            pot_y_m_ate_var_i[t_idx, o_idx] = ret[1]
    l1_to_9 = mcf_ate.analyse_weights_ate(w_add, None, c_dict, False)
    return (idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
            l1_to_9, share_i)


def print_iate(iate, iate_se, iate_p, effect_list, v_dict):
    """Print statistics for the two types of IATEs.

    Parameters
    ----------
    iate : 4D Numpy array. Effects. (obs x outcome x effects x type_of_effect)
    iate_se : 4D Numpy array. Standard errors.
    iate_t : 4D Numpy array.
    iate_p : 4D Numpy array.
    effect_list : List. Names of effects.
    v : Dict. Variables.

    Returns
    -------
    None.

    """
    no_outcomes = np.size(iate, axis=1)
    n_obs = len(iate)
    str_f = '=' * 80
    str_m = '-' * 80
    str_l = '- ' * 40
    print('\n')
    print(str_f, '\nDescriptives for IATE estimation', '\n' + str_m)
    for types in range(2):
        if types == 0:
            print('IATE with corresponding statistics', '\n' + str_l)
        else:
            print('IATE minus ATE with corresponding statistics ',
                  '(weights not censored)', '\n' + str_l)
        for o_idx in range(no_outcomes):
            print('\nOutcome variable: ', v_dict['y_name'][o_idx])
            print(str_l)
            print('Comparison     Mean      Median      Std   Effect > 0',
                  'mean(SE)  sig 10% sig 5%  sig 1%')
            for jdx, effects in enumerate(effect_list):
                print('{:<3} vs {:>3}'.format(effects[0],
                                              effects[1]), end=' ')
                est = iate[:, o_idx, jdx, types].reshape(-1)
                stderr = iate_se[:, o_idx, jdx, types].reshape(-1)
                p_val = iate_p[:, o_idx, jdx, types].reshape(-1)
                print('{:10.5f} {:10.5f} {:10.5f}'.format(
                    np.mean(est), np.median(est), np.std(est)), end=' ')
                print('{:6.2f}% {:10.5f} {:6.2f}% {:6.2f}% {:6.2f}%'.format(
                    np.count_nonzero(est > 1e-15) / n_obs * 100,
                    np.mean(stderr),
                    np.count_nonzero(p_val < 0.1) / n_obs * 100,
                    np.count_nonzero(p_val < 0.05) / n_obs * 100,
                    np.count_nonzero(p_val < 0.01) / n_obs * 100))
        print(str_m, '\n')


def post_estimation_iate(file_name, iate_pot_all_name, ate_all, ate_all_se,
                         effect_list, v_dict, c_dict, v_x_type):
    """Do post-estimation analysis: correlations, k-means, sorted effects.

    Parameters
    ----------
    file_name : String. Name of file with potential outcomes and effects.
    iate_pot_all_name : Dict. Name of potential outcomes and effects.
    ate_all : 3D Numpy array. ATEs.
    ate_all_se : 3D Numpy array. Std.errors of ATEs.
    effect_list : List of list. Explanation of effects related to ATEs.
    v : Dict. Variables.
    c : Dict. Parameters.

    Returns
    -------
    None.

    """
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nPost estimation analysis')
    if c_dict['relative_to_first_group_only']:
        iate_pot_name = iate_pot_all_name[1]
        dim_all = (len(ate_all), c_dict['no_of_treat']-1)
        ate = np.empty(dim_all)
        ate_se = np.empty(dim_all)
        jdx = 0
        for idx, i_lab in enumerate(effect_list):
            if i_lab[1] == c_dict['d_values'][0]:  # compare to 1st treat only
                ate[:, jdx] = ate_all[:, 0, idx]
                ate_se[:, jdx] = ate_all_se[:, 0, idx]
                jdx += 1
    else:
        iate_pot_name = iate_pot_all_name[0]
        dim_all = (np.size(ate_all, axis=0), np.size(ate_all, axis=2))
        ate = np.empty(dim_all)
        ate_se = np.empty(dim_all)
        ate = ate_all[:, 0, :]
        ate_se = ate_all_se[:, 0, :]
    ate = ate.reshape(-1)
    ate_se = ate_se.reshape(-1)
    data = pd.read_csv(file_name)
    pot_y = data[iate_pot_name['names_pot_y']]      # deep copies
    iate = data[iate_pot_name['names_iate']]
    x_name = delete_x_with_catv(v_x_type.keys())
    x_dat = data[x_name]
    cint = sct.norm.ppf(c_dict['fig_ci_level'] +
                        0.5 * (1 - c_dict['fig_ci_level']))
    if c_dict['bin_corr_yes']:
        print('\n' + ('=' * 80), '\nCorrelations of effects with ... in %')
        print('-' * 80)
    label_ci = str(c_dict['fig_ci_level'] * 100) + '%-CI'
    for idx in range(len(iate_pot_name['names_iate'])):
        for imate in range(2):
            if imate == 0:
                name_eff = 'names_iate'
                ate_t = ate[idx].copy()
                ate_se_t = ate_se[idx].copy()
            else:
                name_eff = 'names_iate_mate'
                ate_t = 0
            name_se = name_eff + '_se'
            name_iate_t = iate_pot_name[name_eff][idx]
            name_iate_se_t = iate_pot_name[name_se][idx]
            titel = 'Sorted' + name_iate_t
            # Add correlation analyis of IATEs
            if c_dict['bin_corr_yes'] and (imate == 0):
                print('Effect:', name_iate_t, '\n' + ('-' * 80))
                corr = iate.corrwith(data[name_iate_t])
                for jdx in corr.keys():
                    print('{:<20} {:>8.2f}'.format(jdx, corr[jdx] * 100))
                print('-' * 80)
                corr = pot_y.corrwith(data[name_iate_t])
                for jdx in corr.keys():
                    print('{:<20} {:>8.2f}'.format(jdx, corr[jdx] * 100))
                print('-' * 80)
                corr = x_dat.corrwith(data[name_iate_t])
                corr = corr.sort_values()
                for jdx in corr.keys():
                    if np.abs(corr[jdx].item()) > c_dict['bin_corr_thresh']:
                        print('{:<20} {:>8.2f}'.format(jdx, corr[jdx] * 100))
                print('-' * 80)
            iate_temp = data[name_iate_t].to_numpy()
            iate_se_temp = data[name_iate_se_t].to_numpy()
            sorted_ind = np.argsort(iate_temp)
            iate_temp = iate_temp[sorted_ind]
            iate_se_temp = iate_se_temp[sorted_ind]
            x_values = np.arange(len(iate_temp)) + 1
            k = np.round(c_dict['knn_const'] * np.sqrt(len(iate_temp)) * 2)
            iate_temp = gp_est.moving_avg_mean_var(iate_temp, k, False)[0]
            iate_se_temp = gp_est.moving_avg_mean_var(
                iate_se_temp, k, False)[0]
            file_name_jpeg = c_dict['fig_pfad_jpeg'] + '/' + titel + '.jpeg'
            file_name_pdf = c_dict['fig_pfad_pdf'] + '/' + titel + '.pdf'
            file_name_csv = c_dict['fig_pfad_csv'] + '/' + titel + '.csv'
            upper = iate_temp + iate_se_temp * cint
            lower = iate_temp - iate_se_temp * cint
            ate_t = ate_t * np.ones(len(upper))
            if imate == 0:
                ate_upper = ate_t + (ate_se_t * cint * np.ones(len(upper)))
                ate_lower = ate_t - (ate_se_t * cint * np.ones(len(upper)))
            line_ate = '_-r'
            line_iate = '-b'
            fig, axe = plt.subplots()
            if imate == 0:
                label_t = 'IATE'
                label_r = 'ATE'
            else:
                label_t = 'IATE-ATE'
                label_r = '_nolegend_'
            axe.plot(x_values, iate_temp, line_iate, label=label_t)
            axe.set_ylabel(label_t)
            axe.plot(x_values, ate_t, line_ate, label=label_r)
            if imate == 0:
                axe.fill_between(x_values, ate_upper, ate_lower,
                                 alpha=0.3, color='r', label=label_ci)
            axe.set_title(titel)
            axe.set_xlabel('Ordered observations')
            axe.fill_between(x_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
            axe.legend(loc='lower right', shadow=True,
                       fontsize=c_dict['fig_fontsize'])
            if c_dict['post_plots']:
                gp.delete_file_if_exists(file_name_jpeg)
                gp.delete_file_if_exists(file_name_pdf)
                fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
                fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
            if c_dict['show_plots']:
                plt.show()
            else:
                plt.close()
            iate_temp = iate_temp.reshape(-1, 1)
            upper = upper.reshape(-1, 1)
            lower = lower.reshape(-1, 1)
            ate_t = ate_t.reshape(-1, 1)
            iate_temp = iate_temp.reshape(-1, 1)
            if imate == 0:
                ate_upper = ate_upper.reshape(-1, 1)
                ate_lower = ate_lower.reshape(-1, 1)
                effects_et_al = np.concatenate((upper, iate_temp, lower, ate_t,
                                                ate_upper, ate_lower), axis=1)
                cols = ['upper', 'effects', 'lower', 'ate', 'ate_l', 'ate_u']
            else:
                effects_et_al = np.concatenate((upper, iate_temp, lower,
                                                ate_t), axis=1)
                cols = ['upper', 'effects', 'lower', 'ate']
            datasave = pd.DataFrame(data=effects_et_al, columns=cols)
            gp.delete_file_if_exists(file_name_csv)
            datasave.to_csv(file_name_csv, index=False)
            # density plots
            if imate == 0:
                titel = 'Density' + iate_pot_name['names_iate'][idx]
                file_name_jpeg = (c_dict['fig_pfad_jpeg'] + '/' + titel +
                                  '.jpeg')
                file_name_pdf = c_dict['fig_pfad_pdf'] + '/' + titel + '.pdf'
                file_name_csv = c_dict['fig_pfad_csv'] + '/' + titel + '.csv'
                iate_temp = data[name_iate_t].to_numpy()
                bandwidth = gp_est.bandwidth_silverman(iate_temp, 1)
                dist = np.abs(iate_temp.max() - iate_temp.min())
                low_b = iate_temp.min() - 0.1 * dist
                up_b = iate_temp.max() + 0.1 * dist
                grid = np.linspace(low_b, up_b, 1000)
                density = gp_est.kernel_density(iate_temp, grid, 1, bandwidth)
                fig, axe = plt.subplots()
                axe.set_title(titel)
                axe.set_ylabel('Estimated density')
                axe.plot(grid, density, '-b')
                axe.fill_between(grid, density, alpha=0.3, color='b')
                if c_dict['post_plots']:
                    gp.delete_file_if_exists(file_name_jpeg)
                    gp.delete_file_if_exists(file_name_pdf)
                    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
                    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
                if c_dict['show_plots']:
                    plt.show()
                else:
                    plt.close()
                density = density.reshape(-1, 1)
                cols = ['grid', 'density']
                grid = grid.reshape(-1, 1)
                density = density.reshape(-1, 1)
                effects_et_al = np.concatenate((grid, density), axis=1)
                datasave = pd.DataFrame(data=effects_et_al, columns=cols)
                gp.delete_file_if_exists(file_name_csv)
                datasave.to_csv(file_name_csv, index=False)
    # k-means clustering
    if c_dict['post_km']:
        pd.set_option('display.max_rows', 1000, 'display.max_columns', 100)
        iate_np = iate.to_numpy()
        silhouette_avg_prev = -1
        print('\n' + ('=' * 80), '\nK-Means++ clustering', '\n' + ('-' * 80))
        print('-' * 80)
        for cluster_no in c_dict['post_km_no_of_groups']:
            cluster_lab_tmp = KMeans(
                n_clusters=cluster_no,
                n_init=c_dict['post_km_replications'], init='k-means++',
                max_iter=c_dict['post_kmeans_max_tries'], algorithm='full',
                random_state=42, tol=1e-5, verbose=0, copy_x=True
                ).fit_predict(iate_np)
            silhouette_avg = silhouette_score(iate_np, cluster_lab_tmp)
            print('Number of clusters: ', cluster_no,
                  'Average silhouette score:', silhouette_avg)
            if silhouette_avg > silhouette_avg_prev:
                cluster_lab_np = np.copy(cluster_lab_tmp)
                silhouette_avg_prev = np.copy(silhouette_avg)
        print('Best value of average silhouette score:', silhouette_avg_prev)
        print('-' * 80)
        del iate_np
        # Reorder labels for better visible inspection of results
        iate_name = iate_pot_name['names_iate']
        namesfirsty = iate_name[0:round(len(iate_name)/len(v_dict['y_name']))]
        cl_means = iate[namesfirsty].groupby(by=cluster_lab_np).mean()
        cl_means_np = cl_means.to_numpy()
        cl_means_np = np.mean(cl_means_np, axis=1)
        sort_ind = np.argsort(cl_means_np)
        cl_group = cluster_lab_np.copy()
        for cl_j, cl_old in enumerate(sort_ind):
            cl_group[cluster_lab_np == cl_old] = cl_j
        print('Effects are ordered w.r.t. to size of the effects for the',
              ' first outcome.')
        print('Effects', '\n' + ('-' * 80))
        daten_neu = data.copy()
        daten_neu['IATE_Cluster'] = cl_group
        gp.delete_file_if_exists(file_name)
        daten_neu.to_csv(file_name)
        del daten_neu
        cl_means = iate.groupby(by=cl_group).mean()
        print(cl_means.transpose())
        print('-' * 80, '\nPotential outcomes', '\n' + ('-' * 80))
        cl_means = pot_y.groupby(by=cl_group).mean()
        print(cl_means.transpose())
        print('-' * 80, '\nCovariates', '\n' + ('-' * 80))
        names_unordered = []
        for x_name in v_x_type.keys():
            if v_x_type[x_name] > 0:
                names_unordered.append(x_name)
        if names_unordered:  # List is not empty
            x_dummies = pd.get_dummies(x_dat, columns=names_unordered)
            x_km = pd.concat([x_dat, x_dummies], axis=1)
        else:
            x_km = x_dat
        cl_means = x_km.groupby(by=cl_group).mean()
        print(cl_means.transpose())
        print('-' * 80)
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
    if c_dict['post_random_forest_vi'] and c_dict['with_output']:
        names_unordered = []
        for x_name in v_x_type.keys():
            if v_x_type[x_name] > 0:
                names_unordered.append(x_name)
        x_name = x_dat.columns.tolist()
        dummy_group_names = []
        if names_unordered:  # List is not empty
            dummy_names = []
            replace_dict = dict(zip(gp.primes_list(1000), list(range(1000))))
            for name in names_unordered:
                x_t_d = x_dat[name].replace(replace_dict)
                x_t_d = pd.get_dummies(x_t_d, prefix=name)
                this_dummy_names = x_t_d.columns.tolist()
                dummy_names.extend(this_dummy_names[:])
                this_dummy_names.append(name)
                dummy_group_names.append(this_dummy_names[:])
                x_dat = pd.concat([x_dat, x_t_d], axis=1)
            x_name.extend(dummy_names)
            if c_dict['with_output'] and c_dict['verbose']:
                print('The following dummy variables have been created',
                      dummy_names)
        x_train = x_dat.to_numpy(copy=True)
        if c_dict['with_output'] and c_dict['verbose']:
            print('Features used to build random forest')
            print(x_dat.describe())
            print()
        for _, y_name in enumerate(iate_pot_name['names_iate']):
            print('Computing post estimation random forests for ', y_name)
            y_train = iate[y_name].to_numpy(copy=True)
            gp_est.RandomForest_scikit(
                x_train, y_train, None, x_name=x_name, y_name=y_name,
                boot=c_dict['boot'], n_min=2, no_features='sqrt',
                max_depth=None, workers=c_dict['no_parallel'], alpha=0,
                var_im_groups=dummy_group_names,
                max_leaf_nodes=None, pred_p_flag=False, pred_t_flag=True,
                pred_oob_flag=True, with_output=True, variable_importance=True,
                pred_uncertainty=False, pu_ci_level=0.9, pu_skew_sym=0.5,
                var_im_with_output=True)


def delete_x_with_catv(names_with_catv):
    """
    Delete variables which end with CATV.

    Parameters
    ----------
    names_with_catv : List of str.

    Returns
    -------
    x_names : List of str.

    """
    x_names = []
    for x_name in names_with_catv:
        if x_name[-4:] != 'CATV':
            x_names.append(x_name)
    return x_names
