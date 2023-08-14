"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the IATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd
import ray

from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def iate_est_mp(mcf_, weights_dic, w_ate, reg_round=True):
    """
    Estimate IATE and their standard errors, MP version.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame. Prediction data.
    weights_dic : Dict.
              Contains weights and numpy data.
    balancing_test: Boolean.
              Default is False.
    reg_round : Boolean.
              First round of estimation

    Returns
    -------
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.

    """
    def warn_text_to_console():
        print('If prediction file is large, this step may take long. If '
              'nothing seems to happen, it may be worth to try do the '
              'estimation without sparse weight matrix. This needs more '
              'memory, but could be substantially faster (int_weight_as_sparse'
              ' = False).')
    p_dic, int_dic, gen_dic = mcf_.p_dict, mcf_.int_dict, mcf_.gen_dict
    var_dic, ct_dic = mcf_.var_dict, mcf_.ct_dict
    if reg_round:
        iate_se_flag, se_boot_iate = p_dic['iate_se'], p_dic['se_boot_iate']
        iate_m_ate_flag = p_dic['iate_m_ate']
    else:
        iate_se_flag = se_boot_iate = iate_m_ate_flag = False
    if int_dic['with_output'] and int_dic['verbose']:
        print('\nComputing IATEs 1/2 (potential outcomes)')
    weights, y_dat = weights_dic['weights'], weights_dic['y_dat_np']
    w_dat = weights_dic['w_dat_np'] if gen_dic['weighted'] else None
    if p_dic['cluster_std'] and iate_se_flag:
        cl_dat = weights_dic['cl_dat_np']
        no_of_cluster = len(np.unique(cl_dat))
    else:
        no_of_cluster = cl_dat = None
    n_x = weights[0].shape[0] if int_dic['weight_as_sparse'] else len(weights)
    n_y, no_of_out = len(y_dat), len(var_dic['y_name'])
    if gen_dic['d_type'] == 'continuous':
        no_of_treat, d_values = ct_dic['grid_w'], ct_dic['grid_w_val']
        d_values_dr = ct_dic['d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    larger_0 = np.zeros(no_of_treat_dr)
    equal_0, mean_pos = np.zeros_like(larger_0), np.zeros_like(larger_0)
    std_pos, gini_all = np.zeros_like(larger_0), np.zeros_like(larger_0)
    gini_pos, share_censored = np.zeros_like(larger_0), np.zeros_like(larger_0)
    share_largest_q = np.zeros((no_of_treat_dr, 3))
    sum_larger = np.zeros((no_of_treat_dr, len(p_dic['q_w'])))
    obs_larger = np.zeros_like(sum_larger)
    pot_y = np.empty((n_x, no_of_treat_dr, no_of_out))
    pot_y_m_ate = np.empty_like(pot_y) if iate_m_ate_flag else None
    pot_y_var = np.empty_like(pot_y) if iate_se_flag else None
    pot_y_m_ate_var = np.empty_like(pot_y) if (iate_se_flag and iate_m_ate_flag
                                               ) else None
    if w_ate is not None:
        w_ate = w_ate[0, :, :]
    if not p_dic['iate_m_ate']:
        w_ate = None
    l1_to_9 = [None] * n_x
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])
        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and int_dic['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if int_dic['weight_as_sparse']:
        iterator = len(weights)
    if maxworkers == 1:
        for idx in range(n_x):
            if int_dic['weight_as_sparse']:
                weights_idx = [weights[t_idx].getrow(idx) for
                               t_idx in range(iterator)]
            else:
                weights_idx = weights[idx]
            ret_all_i = iate_func1_for_mp(
                idx, weights_idx, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
                no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic,
                iate_se_flag, se_boot_iate, iate_m_ate_flag)
            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
             share_censored) = assign_ret_all_i(
                 pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                 share_censored, ret_all_i, n_x, idx)
            if int_dic['with_output'] and int_dic['verbose']:
                mcf_gp.share_completed(idx+1, n_x)
    else:
        rows_per_split = 1e9    # Just a large number; feature is not used
        no_of_splits = round(n_x / rows_per_split)
        no_of_splits = min(max(no_of_splits, maxworkers), n_x)
        if int_dic['with_output'] and int_dic['verbose']:
            print('IATE-1: Avg. number of obs per split:',
                  f'{n_x / no_of_splits:5.2f}.',
                  ' Number of splits: ', no_of_splits)
        obs_idx_list = np.array_split(np.arange(n_x), no_of_splits)
        if int_dic['ray_or_dask'] == 'ray':
            if int_dic['mem_object_store_3'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=maxworkers, include_dashboard=False,
                        object_store_memory=int_dic['mem_object_store_3'])
                if int_dic['with_output'] and int_dic['verbose']:
                    print("Size of Ray Object Store: ", round(
                        int_dic['mem_object_store_3']/(1024*1024)), ' MB')
            if int_dic['weight_as_sparse']:
                still_running = [ray_iate_func1_for_mp_many_obs.remote(
                    idx, [weights[t_idx][idx, :] for t_idx in
                          range(iterator)], cl_dat, no_of_cluster,
                    w_dat, w_ate, y_dat, no_of_out, n_y, ct_dic, int_dic,
                    gen_dic, p_dic, iate_se_flag, se_boot_iate,
                    iate_m_ate_flag) for idx in obs_idx_list]
                warn_text_to_console()
            else:
                still_running = [ray_iate_func1_for_mp_many_obs.remote(
                    idx, [weights[idxx] for idxx in idx], cl_dat,
                    no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                    ct_dic, int_dic, gen_dic, p_dic, iate_se_flag,
                    se_boot_iate, iate_m_ate_flag) for idx in obs_idx_list]
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
                    if int_dic['with_output'] and int_dic['verbose']:
                        mcf_gp.share_completed(jdx+1, no_of_splits)
                    jdx += 1
            if 'rest' in int_dic['mp_ray_del']:
                del finished_res, finished
            if int_dic['mp_ray_shutdown']:
                ray.shutdown()
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
        if int_dic['with_output']:
            txt = '\n' + '=' * 100
            txt += ('\nAnalysis of weights (normalised to add to 1) of IATE'
                    '(stats are averaged over all effects)')
            txt += ps.txt_weight_stat(
                larger_0 / n_x, equal_0 / n_x, mean_pos / n_x, std_pos / n_x,
                gini_all / n_x, gini_pos / n_x, share_largest_q / n_x,
                sum_larger / n_x, obs_larger / n_x, gen_dic, p_dic,
                share_censored, continuous=gen_dic['d_type'] == 'continuous',
                d_values_cont=d_values_dr)
        else:
            txt = ''
    else:
        txt = ''
    return pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, txt


def iate_effects_print(mcf_, effect_dic, effect_m_ate_dic, effect_eff_dic,
                       y_pred_x_df):
    """Compute, print effects, add potential outcomes to prediction data."""
    p_dic, int_dic, gen_dic = mcf_.p_dict, mcf_.int_dict, mcf_.gen_dict
    var_dic, ct_dic, lc_dic = mcf_.var_dict, mcf_.ct_dict, mcf_.lc_dict
    if gen_dic['d_type'] == 'continuous':
        no_of_treat, d_values = ct_dic['grid_w'], ct_dic['grid_w_val']
        d_values_dr = ct_dic['d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    if int_dic['with_output'] and int_dic['verbose']:
        print('\nComputing IATEs 2/2 (effects)')
    if gen_dic['d_type'] == 'continuous':
        dim_3 = round(no_of_treat_dr - 1)
    else:
        dim_3 = round(no_of_treat * (no_of_treat - 1) / 2)
    y_pot, y_pot_var = effect_dic['y_pot'], effect_dic['y_pot_var']
    if effect_m_ate_dic is None:
        y_pot_m_ate_var = y_pot_m_ate = None
    else:
        y_pot_m_ate = effect_m_ate_dic['y_pot']
        y_pot_m_ate_var = effect_m_ate_dic['y_pot_var']
    iate_eff_yes = effect_eff_dic is not None
    n_x, no_of_out = y_pot.shape[0], y_pot.shape[2]
    iate = np.empty((n_x, no_of_out, dim_3, 2))    # iate, iate_m_ate
    if iate_eff_yes:
        y_pot_eff = effect_eff_dic['y_pot']
        iate_eff = np.empty((n_x, no_of_out, dim_3, 2))
    else:
        y_pot_eff = iate_eff = None
    if p_dic['iate_se']:
        iate_se, iate_p = np.empty_like(iate), np.empty_like(iate)
    else:
        iate_se = iate_p = None
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])
        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and int_dic['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(n_x):
            if p_dic['iate_se'] and p_dic['iate_m_ate']:
                y_pot_v = y_pot_var[idx]
                y_pot_m = y_pot_m_ate[idx]
                y_pot_m_v = y_pot_m_ate_var[idx]
            elif p_dic['iate_se'] and not p_dic['iate_m_ate']:
                y_pot_v = y_pot_var[idx]
                y_pot_m = y_pot_m_v = None
            elif not p_dic['iate_se'] and p_dic['iate_m_ate']:
                y_pot_m = y_pot_m_ate[idx]
                y_pot_v = y_pot_m_v = None
            else:
                y_pot_v = y_pot_m = y_pot_m_v = None
            ret_all_idx = iate_func2_for_mp(
                idx, no_of_out, y_pot[idx], y_pot_v, y_pot_m, y_pot_m_v,
                gen_dic['d_type'], d_values_dr, no_of_treat_dr,
                p_dic['iate_se'], p_dic['iate_m_ate'])
            if int_dic['with_output'] and int_dic['verbose']:
                mcf_gp.share_completed(idx+1, n_x)
            iate[idx, :, :, :] = ret_all_idx[1]
            if p_dic['iate_se']:
                iate_se[idx, :, :, :] = ret_all_idx[2]
                iate_p[idx, :, :, :] = ret_all_idx[3]
            if idx == n_x - 1:
                effect_list = ret_all_idx[4]
            if iate_eff_yes:
                ret_all_idx = iate_func2_for_mp(
                    idx, no_of_out, y_pot_eff[idx], None, None, None,
                    gen_dic['d_type'], d_values_dr, no_of_treat_dr, False,
                    False)
                if int_dic['with_output'] and int_dic['verbose']:
                    mcf_gp.share_completed(idx+1, n_x)
                iate_eff[idx, :, :, :] = ret_all_idx[1]
    else:
        if int_dic['ray_or_dask'] == 'ray':
            if int_dic['mem_object_store_3'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=int_dic['mem_object_store_3'])
                if int_dic['with_output'] and int_dic['verbose']:
                    print("Size of Ray Object Store: ", round(
                        int_dic['mem_object_store_3']/(1024*1024)), " MB")
            if p_dic['iate_se'] and p_dic['iate_m_ate']:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, y_pot[idx], y_pot_var[idx],
                    y_pot_m_ate[idx], y_pot_m_ate_var[idx], gen_dic['d_type'],
                    d_values_dr, no_of_treat_dr, True, True)
                    for idx in range(n_x)]
            elif p_dic['iate_se'] and not p_dic['iate_m_ate']:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, y_pot[idx], y_pot_var[idx], None, None,
                    gen_dic['d_type'], d_values_dr, no_of_treat_dr, True,
                    False) for idx in range(n_x)]
            elif not p_dic['iate_se'] and p_dic['iate_m_ate']:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, y_pot[idx], None, y_pot_m_ate[idx], None,
                    gen_dic['d_type'], d_values_dr, no_of_treat_dr, False,
                    True) for idx in range(n_x)]
            else:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, y_pot[idx], None, None, None,
                    gen_dic['d_type'], d_values_dr, no_of_treat_dr, False,
                    False) for idx in range(n_x)]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i2 in finished_res:
                    iix = ret_all_i2[0]
                    iate[iix, :, :, :] = ret_all_i2[1]
                    if p_dic['iate_se']:
                        iate_se[iix, :, :, :] = ret_all_i2[2]
                        iate_p[iix, :, :, :] = ret_all_i2[3]
                    if jdx == n_x-1:
                        effect_list = ret_all_i2[4]
                    if int_dic['with_output'] and int_dic['verbose']:
                        mcf_gp.share_completed(jdx+1, n_x)
                    jdx += 1
            if iate_eff_yes:
                still_running = [ray_iate_func2_for_mp.remote(
                    idx, no_of_out, y_pot_eff[idx], None, None, None,
                    gen_dic['d_type'], d_values_dr, no_of_treat_dr, False,
                    False) for idx in range(n_x)]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i3 in finished_res:
                        iix = ret_all_i3[0]
                        iate_eff[iix, :, :, :] = ret_all_i3[1]
                        if int_dic['with_output'] and int_dic['verbose']:
                            mcf_gp.share_completed(jdx+1, n_x)
                        jdx += 1
            if 'rest' in int_dic['mp_ray_del']:
                del finished_res, finished
            if int_dic['mp_ray_shutdown']:
                ray.shutdown()
    if int_dic['with_output']:
        txt = ps.print_iate(iate, iate_se, iate_p, effect_list, gen_dic, p_dic,
                            var_dic)
        ps.print_mcf(gen_dic, txt, summary=True, non_summary=False)
        ps.print_mcf(gen_dic, effect_dic['txt_weights'] + txt, summary=False)

    # Add results to data file
    y_pot_np = np.empty((n_x, no_of_out * no_of_treat_dr))
    if p_dic['iate_se']:
        y_pot_se_np = np.empty_like(y_pot_np)
    if gen_dic['d_type'] == 'continuous':
        dim = round(no_of_out * (no_of_treat_dr - 1))
    else:
        dim = round(no_of_out * no_of_treat * (no_of_treat - 1) / 2)
    iate_np = np.empty((n_x, dim))
    if p_dic['iate_m_ate']:
        iate_mate_np = np.empty_like(iate_np)
    if p_dic['iate_se']:
        iate_se_np = np.empty_like(iate_np)
        if p_dic['iate_m_ate']:
            iate_mate_se_np = np.empty_like(iate_np)
    if iate_eff_yes:
        y_pot_eff_np = np.empty((n_x, no_of_out * no_of_treat_dr))
        iate_eff_np = np.empty((n_x, dim))
    else:
        y_pot_eff_np = iate_eff_np = None
    jdx = j2dx = jdx_unlc = 0
    name_pot, name_eff, name_eff0 = [], [], []
    if lc_dic['uncenter_po'] and isinstance(y_pred_x_df,
                                            (pd.Series, pd.DataFrame)):
        name_pot_unlc, y_pot_unlc_np = [], np.empty((n_x, no_of_treat_dr))
        if iate_eff_yes:
            name_pot_eff_unlc = []
            y_pot_eff_unlc_np = np.empty_like(y_pot_unlc_np)
        else:
            name_pot_eff_unlc = y_pot_eff_unlc_np = None
        if isinstance(var_dic['y_tree_name'], list):
            y_tree_name = var_dic['y_tree_name'][0]
        else:
            y_tree_name = var_dic['y_tree_name']
        y_pred_x_np = y_pred_x_df.to_numpy()
    else:
        name_pot_unlc = y_tree_name = name_y_pot_unlc = None
        name_pot_eff_unlc = y_pot_eff_unlc_np = None
    for o_idx, o_name in enumerate(var_dic['y_name']):
        for t_idx, t_name in enumerate(d_values_dr):
            name_pot += [o_name + str(t_name)]
            y_pot_np[:, jdx] = y_pot[:, t_idx, o_idx]
            if iate_eff_yes:
                y_pot_eff_np[:, jdx] = y_pot_eff[:, t_idx, o_idx]
            if o_name == y_tree_name and lc_dic['uncenter_po']:
                name_pot_unlc += [o_name + str(t_name) + '_un_lc']
                y_pot_unlc_np[:, jdx_unlc] = (y_pot_np[:, jdx]
                                              + y_pred_x_np[:, o_idx])
                if iate_eff_yes:
                    name_pot_eff_unlc += [o_name + str(t_name) + '_eff_un_lc']
                    y_pot_eff_unlc_np[:, jdx_unlc] = (
                        y_pot_eff_np[:, jdx] + y_pred_x_np[:, o_idx])
                jdx_unlc += 1
            if p_dic['iate_se']:
                y_pot_se_np[:, jdx] = np.sqrt(y_pot_var[:, t_idx, o_idx])
            jdx += 1
        for t2_idx, t2_name in enumerate(effect_list):
            name_eff += [o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            if t2_name[1] == d_values_dr[0]:   # Usually, control
                name_eff0 += [
                    o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            iate_np[:, j2dx] = iate[:, o_idx, t2_idx, 0]
            if p_dic['iate_m_ate']:
                iate_mate_np[:, j2dx] = iate[:, o_idx, t2_idx, 1]
            if iate_eff_yes:
                iate_eff_np[:, j2dx] = iate_eff[:, o_idx, t2_idx, 0]
            if p_dic['iate_se']:
                iate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 0]
                if p_dic['iate_m_ate']:
                    iate_mate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 1]
            j2dx += 1
    name_y_pot = [s + '_pot' for s in name_pot]
    uncenter = lc_dic['uncenter_po'] and isinstance(y_pred_x_df,
                                                    (pd.Series, pd.DataFrame))
    if uncenter:
        name_y_pot_unlc = [s + '_pot' for s in name_pot_unlc]
    name_iate = [s + '_iate' for s in name_eff]
    name_iate0 = [s + '_iate' for s in name_eff0]
    name_y_pot_eff = name_iate_eff = name_iate0_eff = None
    name_y_pot_eff_unlc = name_y_pot_se = name_iate_se = None
    name_iate_mate = name_iate_mate0 = name_iate_mate_se = None
    name_iate_se0 = name_iate_mate_se0 = None
    if p_dic['iate_m_ate']:
        name_iate_mate = [s + '_iatemate' for s in name_eff]
        name_iate_mate0 = [s + '_iatemate' for s in name_eff0]
    if iate_eff_yes:
        name_y_pot_eff = [s + '_pot_eff' for s in name_pot]
        if uncenter:
            name_y_pot_eff_unlc = [s + '_pot_eff' for s in name_pot_unlc]
        name_iate_eff = [s + '_iate_eff' for s in name_eff]
        name_iate0_eff = [s + '_iate_eff' for s in name_eff0]
    if p_dic['iate_se']:
        name_y_pot_se = [s + '_pot_se' for s in name_pot]
        name_iate_se = [s + '_iate_se' for s in name_eff]
        name_iate_se0 = [s + '_iate_se' for s in name_eff0]
        if p_dic['iate_m_ate']:
            name_iate_mate_se = [s + '_iatemate_se' for s in name_eff]
            name_iate_mate_se0 = [s + '_iatemate_se' for s in name_eff0]
    if int_dic['with_output'] or int_dic['return_iate_sp']:
        y_pot_df = pd.DataFrame(data=y_pot_np, columns=name_y_pot)
        iate_df = pd.DataFrame(data=iate_np, columns=name_iate)
        if p_dic['iate_m_ate']:
            iate_mate_df = pd.DataFrame(data=iate_mate_np,
                                        columns=name_iate_mate)
        if p_dic['iate_se']:
            y_pot_se_df = pd.DataFrame(data=y_pot_se_np, columns=name_y_pot_se)
            iate_se_df = pd.DataFrame(data=iate_se_np, columns=name_iate_se)
            if p_dic['iate_m_ate']:
                iate_mate_se_df = pd.DataFrame(data=iate_mate_se_np,
                                               columns=name_iate_mate_se)
        if iate_eff_yes:
            y_pot_eff_df = pd.DataFrame(data=y_pot_eff_np,
                                        columns=name_y_pot_eff)
            iate_eff_df = pd.DataFrame(data=iate_eff_np, columns=name_iate_eff)

        if p_dic['iate_se'] and p_dic['iate_m_ate']:
            df_list = [y_pot_df, y_pot_se_df, iate_df, iate_se_df,
                       iate_mate_df, iate_mate_se_df]
        elif p_dic['iate_se'] and not p_dic['iate_m_ate']:
            df_list = [y_pot_df, y_pot_se_df, iate_df, iate_se_df]
        elif not p_dic['iate_se'] and p_dic['iate_m_ate']:
            df_list = [y_pot_df, iate_df, iate_mate_df,]
        else:
            df_list = [y_pot_df, iate_df]
        if uncenter:
            pot_y_unlc_df = pd.DataFrame(data=y_pot_unlc_np,
                                         columns=name_y_pot_unlc)
            df_list.append(pot_y_unlc_df)
        if iate_eff_yes:
            df_list.append(y_pot_eff_df)
            df_list.append(iate_eff_df)
            if uncenter:
                y_pot_eff_unlc_df = pd.DataFrame(data=y_pot_eff_unlc_np,
                                                 columns=name_y_pot_eff_unlc)
                df_list.append(y_pot_eff_unlc_df)
        results_df = pd.concat(df_list, axis=1)
        if int_dic['with_output']:
            ps.print_mcf(gen_dic, '\nIndividualized ATE', summary=True)
            ps.print_descriptive_df(gen_dic, results_df, varnames='all',
                                    summary=True)
    names_pot_iate = {
        'names_y_pot': name_y_pot, 'names_y_pot_se': name_y_pot_se,
        'names_iate': name_iate, 'names_iate_se': name_iate_se,
        'names_iate_mate': name_iate_mate, 'names_iate_mate_se':
            name_iate_mate_se}
    names_pot_iate0 = {
        'names_y_pot': name_y_pot, 'names_y_pot_se': name_y_pot_se,
        'names_iate': name_iate0, 'names_iate_se': name_iate_se0,
        'names_iate_mate': name_iate_mate0, 'names_iate_mate_se':
            name_iate_mate_se0}
    if iate_eff_yes:
        names_pot_iate['names_y_pot_eff'] = name_y_pot_eff
        names_pot_iate0['names_y_pot_eff'] = name_y_pot_eff
        names_pot_iate['name_iate_eff'] = name_iate_eff
        names_pot_iate0['name_iate_eff'] = name_iate0_eff
    if uncenter:
        names_pot_iate['names_y_pot_uncenter'] = name_y_pot_unlc
        names_pot_iate0['names_y_pot_uncenter'] = name_y_pot_unlc
        if iate_eff_yes:
            names_pot_iate['names_y_pot_eff_uncenter'] = name_y_pot_eff_unlc
            names_pot_iate0['names_y_pot_eff_uncenter'] = name_y_pot_eff_unlc
    if not int_dic['return_iate_sp']:
        results_df = None
    return (iate, iate_se, iate_eff, (names_pot_iate, names_pot_iate0),
            results_df)


@ray.remote
def ray_iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat, w_ate,
                          y_dat, no_of_out, n_y, ct_dic, int_dic, gen_dic,
                          p_dic, iate_se_flag, se_boot_iate):
    """Make function useful for Ray."""
    return iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat,
                             w_ate, y_dat, no_of_out, n_y, ct_dic, int_dic,
                             gen_dic, p_dic, iate_se_flag, se_boot_iate)


def iate_func1_for_mp(idx, weights_i, cl_dat, no_of_cluster, w_dat, w_ate,
                      y_dat, no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic,
                      iate_se_flag, se_boot_iate, iate_m_ate_flag):
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
    ct_dic, int_dic, gen_dic, p_dic : Dict. Parameters.
    iate_se_flag : Boolean. Compute standard errors.
    se_boot_iate : Boolean. Compute bootstrap standard errors.
    iate_m_ate_flag : Boolean. Compute difference to average potential outcome.

    Returns
    -------
    idx: Int. Counter.
    pot_y_i: Numpy array.
    pot_y_var_i: Numpy array.
    pot_y_m_ate_i: Numpy array.
    pot_y_m_ate_var_i: Numpy array.
    l1_to_9: Tuple of lists.
    """
    def get_walli(w_index, n_y, w_i, w_i_unc):
        w_all_i = np.zeros(n_y)
        w_all_i[w_index] = w_i
        w_all_i_unc = np.zeros_like(w_all_i)
        w_all_i_unc[w_index] = w_i_unc
        return w_all_i, w_all_i_unc

    if (gen_dic['with_output'] and (idx == 0)
            and int_dic['ray_or_dask'] != 'ray' and gen_dic['verbose']):
        print('Starting to compute IATE - procedure 1', flush=True)
    if gen_dic['d_type'] == 'continuous':
        continuous, d_values_dr = True, ct_dic['d_values_dr_np']
        no_of_treat = ct_dic['grid_w']
        i_w01 = ct_dic['w_to_dr_int_w01']
        i_w10 = ct_dic['w_to_dr_int_w10']
        index_full = ct_dic['w_to_dr_index_full']
        no_of_treat_dr = len(d_values_dr)
    else:
        continuous, d_values_dr = False, gen_dic['d_values']
        no_of_treat = no_of_treat_dr = gen_dic['no_of_treat']
    pot_y_i = np.empty((no_of_treat_dr, no_of_out))
    pot_y_m_ate_i = np.empty_like(pot_y_i) if iate_m_ate_flag else None
    share_i = np.zeros(no_of_treat_dr)
    if iate_se_flag:
        pot_y_var_i = np.empty_like(pot_y_i)
        pot_y_m_ate_var_i = np.empty_like(pot_y_i) if iate_m_ate_flag else None
        cluster_std = p_dic['cluster_std']
    else:
        pot_y_var_i = pot_y_m_ate_var_i = None
        cluster_std = False
    w_add = (np.zeros((no_of_treat_dr, no_of_cluster)) if cluster_std
             else np.zeros((no_of_treat_dr, n_y)))
    w_add_unc = np.zeros((no_of_treat_dr, n_y))
    for t_idx in range(no_of_treat):
        extra_weight_p1 = continuous and t_idx < no_of_treat-1
        if int_dic['weight_as_sparse']:
            w_index = weights_i[t_idx].indices
            w_i_t = weights_i[t_idx].data
        else:
            w_index = weights_i[t_idx][0]    # Indices of non-zero weights
            w_i_t = weights_i[t_idx][1].copy()
        if extra_weight_p1:
            if int_dic['weight_as_sparse']:
                w_index_p1 = weights_i[t_idx+1].indices
            else:
                w_index_p1 = weights_i[t_idx+1][0]
            w_index_both = np.unique(np.concatenate((w_index, w_index_p1)))
            w_i = np.zeros(n_y)
            w_i[w_index] = w_i_t
            w_i_p1 = np.zeros_like(w_i)
            if int_dic['weight_as_sparse']:
                w_i_p1[w_index_p1] = weights_i[t_idx+1].data
            else:
                w_i_p1[w_index_p1] = weights_i[t_idx+1][1].copy()
            w_i = w_i[w_index_both]
            w_i_p1 = w_i_p1[w_index_both]
        else:
            w_index_both = w_index
            w_i = w_i_t
        if gen_dic['weighted']:
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
        if p_dic['max_weight_share'] < 1 and not continuous:
            w_i, _, share_i[t_idx] = mcf_gp.bound_norm_weights(
                w_i, p_dic['max_weight_share'])
        if extra_weight_p1:
            w_i_unc_p1 = np.copy(w_i_p1)
        if cluster_std:
            w_all_i, w_all_i_unc = get_walli(w_index, n_y, w_i, w_i_unc)
            cl_i = cl_dat[w_index]
            if extra_weight_p1:
                w_all_i_p1, w_all_i_unc_p1 = get_walli(w_index_p1, n_y, w_i_p1,
                                                       w_i_unc_p1)
                cl_i_both = cl_dat[w_index_both]
        else:
            cl_i = cl_i_both = None
        for o_idx in range(no_of_out):
            if continuous:
                y_dat_cont = y_dat[w_index_both, o_idx]
                for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
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
                    if p_dic['max_weight_share'] < 1:
                        w_i_cont, _, share_cont = mcf_gp.bound_norm_weights(
                            w_i_cont, p_dic['max_weight_share'])
                        if i == 0:
                            share_i[t_idx] = share_cont
                    ret = mcf_est.weight_var(
                        w_i_cont, y_dat_cont, cl_i_cont, gen_dic, p_dic,
                        weights=w_t_cont, se_yes=iate_se_flag,
                        bootstrap=se_boot_iate)
                    ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                    pot_y_i[ti_idx, o_idx] = ret[0]
                    if iate_se_flag:
                        pot_y_var_i[ti_idx, o_idx] = ret[1]
                    if cluster_std:
                        w_cont = (w10 * w_all_i + w01 * w_all_i_p1
                                  if extra_weight_p1 else w_all_i)
                        ret2 = mcf_est.aggregate_cluster_pos_w(
                            cl_dat, w_cont, y_dat[:, o_idx], sweights=w_dat)
                        if o_idx == 0:
                            w_add[ti_idx, :] = np.copy(ret2[0])
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
                        if iate_m_ate_flag:
                            ret = mcf_est.weight_var(
                                w_diff, y_dat[:, o_idx], cl_dat, gen_dic,
                                p_dic, norm=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag)
                    else:
                        if o_idx == 0:
                            w_add[ti_idx, w_index_both] = ret[2]
                            w_i_unc_sum = np.sum(w_i_unc_cont)
                            if not (1-1e-10) < w_i_unc_sum < (1+1e-10):
                                w_add_unc[ti_idx, w_index_both] = (
                                    w_i_unc_cont / w_i_unc_sum)
                            else:
                                w_add_unc[ti_idx, w_index_both] = w_i_unc_cont
                            if w_ate is None:
                                w_diff = w_add_unc[ti_idx, :]
                            else:
                                if extra_weight_p1:
                                    w_ate_cont = (w10 * w_ate[t_idx, :] +
                                                  w01 * w_ate[t_idx+1, :])
                                    w_ate_cont /= np.sum(w_ate_cont)
                                    w_diff = w_add_unc[ti_idx, :] - w_ate_cont
                                else:
                                    w_diff = (w_add_unc[ti_idx, :]
                                              - w_ate[t_idx, :])
                        if iate_m_ate_flag:
                            ret = mcf_est.weight_var(
                                w_diff, y_dat[:, o_idx], None, gen_dic, p_dic,
                                norm=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag)
                    if iate_m_ate_flag:
                        pot_y_m_ate_i[ti_idx, o_idx] = ret[0]
                    if iate_se_flag and iate_m_ate_flag:
                        pot_y_m_ate_var_i[ti_idx, o_idx] = ret[1]
                    if not extra_weight_p1:
                        break
            else:  # discrete treatment
                ret = mcf_est.weight_var(
                    w_i, y_dat[w_index, o_idx], cl_i, gen_dic, p_dic,
                    weights=w_t, se_yes=iate_se_flag, bootstrap=se_boot_iate)
                pot_y_i[t_idx, o_idx] = ret[0]
                if iate_se_flag:
                    pot_y_var_i[t_idx, o_idx] = ret[1]
                if cluster_std:
                    ret2 = mcf_est.aggregate_cluster_pos_w(
                        cl_dat, w_all_i, y_dat[:, o_idx], sweights=w_dat)
                    if o_idx == 0:
                        w_add[t_idx, :] = np.copy(ret2[0])
                        if w_ate is None:
                            w_diff = w_all_i_unc  # Dummy if no w_ate
                        else:
                            w_diff = w_all_i_unc - w_ate[t_idx, :]
                    if iate_m_ate_flag:
                        ret = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat, gen_dic, p_dic,
                            norm=False, weights=w_dat, bootstrap=se_boot_iate,
                            se_yes=iate_se_flag)
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
                    if iate_m_ate_flag:
                        ret = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], None, gen_dic, p_dic,
                            norm=False, weights=w_dat, bootstrap=se_boot_iate,
                            se_yes=iate_se_flag)
                if iate_m_ate_flag:
                    pot_y_m_ate_i[t_idx, o_idx] = ret[0]
                if iate_m_ate_flag and iate_se_flag:
                    pot_y_m_ate_var_i[t_idx, o_idx] = ret[1]
    l1_to_9 = mcf_est.analyse_weights(
        w_add, None, gen_dic, p_dic, ate=False, continuous=continuous,
        no_of_treat_cont=no_of_treat_dr, d_values_cont=d_values_dr)
    return (idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
            l1_to_9, share_i)


def assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                     share_censored, ret_all_i, n_x, idx=None):
    """Use to avoid duplicate code."""
    if idx is None:
        idx = ret_all_i[0]
    pot_y[idx, :, :] = ret_all_i[1]
    if pot_y_m_ate is not None:
        pot_y_m_ate[idx, :, :] = ret_all_i[3]
    if pot_y_var is not None:
        pot_y_var[idx, :, :] = ret_all_i[2]
    if pot_y_m_ate_var is not None:
        pot_y_m_ate_var[idx, :, :] = ret_all_i[4]
    l1_to_9[idx] = ret_all_i[5]
    share_censored += ret_all_i[6] / n_x
    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
            share_censored)


@ray.remote
def ray_iate_func1_for_mp_many_obs(
        idx_list, weights_list, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
        no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic, iate_se_flag,
        se_boot_iate, iate_m_ate_flag):
    """Compute IATE for several obs in one loop (MP)."""
    return iate_func1_for_mp_many_obs(
        idx_list, weights_list, cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
        no_of_out, n_y, ct_dic, int_dic, gen_dic, p_dic, iate_se_flag,
        se_boot_iate, iate_m_ate_flag)


def iate_func1_for_mp_many_obs(idx_list, weights_list, cl_dat, no_of_cluster,
                               w_dat, w_ate, y_dat, no_of_out, n_y, ct_dic,
                               int_dic, gen_dic, p_dic, iate_se_flag,
                               se_boot_iate, iate_m_ate_flag):
    """Compute IATE for several obs in one loop (MP)."""
    ret_all = []
    if int_dic['weight_as_sparse']:
        iterator = len(weights_list)
    for i, idx_org in enumerate(idx_list):
        if int_dic['weight_as_sparse']:
            weights_i = [weights_list[t_idx].getrow(i)
                         for t_idx in range(iterator)]
        else:
            weights_i = weights_list[i]
        ret = iate_func1_for_mp(idx_org, weights_i, cl_dat, no_of_cluster,
                                w_dat, w_ate, y_dat, no_of_out, n_y, ct_dic,
                                int_dic, gen_dic, p_dic, iate_se_flag,
                                se_boot_iate, iate_m_ate_flag)
        ret_all.append(ret)
    return ret_all


@ray.remote
def ray_iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                          pot_y_m_ate_var_i, d_type, d_values, no_of_treat,
                          iate_se_flag, iate_m_ate_flag):
    """Make function compatible with Ray."""
    return iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i,
                             pot_y_m_ate_i, pot_y_m_ate_var_i, d_type,
                             d_values, no_of_treat, iate_se_flag,
                             iate_m_ate_flag)


def iate_func2_for_mp(idx, no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                      pot_y_m_ate_var_i, d_type, d_values, no_of_treat,
                      iate_se_flag, iate_m_ate_flag):
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
    if d_type == 'continuous':
        dim = (no_of_out, no_of_treat - 1, 2)
    else:
        dim = (no_of_out, round(no_of_treat * (no_of_treat - 1) / 2), 2)
    iate_i = np.empty(dim)
    if iate_se_flag:
        iate_se_i = np.empty(dim)  # obs x outcome x effects x type_of_effect
        iate_p_i = np.empty_like(iate_se_i)
    else:
        iate_se_i = iate_p_i = None
    iterator = 2 if iate_m_ate_flag else 1
    for o_i in range(no_of_out):
        for jdx in range(iterator):
            if jdx == 0:
                pot_y_ao = pot_y_i[:, o_i]
                pot_y_var_ao = pot_y_var_i[:, o_i] if iate_se_flag else None
            else:
                pot_y_ao = pot_y_m_ate_i[:, o_i]
                pot_y_var_ao = (pot_y_m_ate_var_i[:, o_i]
                                if iate_se_flag else None)
            ret = mcf_est.effect_from_potential(
                pot_y_ao, pot_y_var_ao, d_values,
                se_yes=iate_se_flag, continuous=d_type == 'continuous')
            if iate_se_flag:
                (iate_i[o_i, :, jdx], iate_se_i[o_i, :, jdx], _,
                 iate_p_i[o_i, :, jdx], effect_list) = ret
            else:
                (iate_i[o_i, :, jdx], _, _, _, effect_list) = ret
    return idx, iate_i, iate_se_i, iate_p_i, effect_list
