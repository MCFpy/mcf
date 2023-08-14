"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from copy import deepcopy
from itertools import chain, compress, repeat
import os

import numpy as np
import ray

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_weight_functions as mcf_w


def gate_est(mcf_, data_df, weights_dic, w_atemain, gate_type='GATE',
             z_name_amgate=None, with_output=True):
    """Estimate GATE(T)s and AMGAT(T) and their standard errors."""
    if gate_type not in ('GATE', 'AMGATE', 'BGATE'):
        raise ValueError('Wrong GATE specifified ({gate_type}). gate_type must'
                         'be on of the following: GATE, AMGATE, BGATE')
    gen_dic, ct_dic, int_dic = mcf_.gen_dict, mcf_.ct_dict, mcf_.int_dict
    p_dic, var_dic = mcf_.p_dict, deepcopy(mcf_.var_dict)
    if gate_type != 'GATE':
        var_dic['z_name'] = z_name_amgate
    var_x_type = deepcopy(mcf_.var_x_type)
    var_x_values = deepcopy(mcf_.var_x_values)
    w_ate = deepcopy(w_atemain)
    txt = ''
    if int_dic['with_output'] and int_dic['verbose'] and with_output:
        print('\nComputing', gate_type)
    y_dat = weights_dic['y_dat_np']
    weights_all = weights_dic['weights']
    w_dat = weights_dic['w_dat_np'] if gen_dic['weighted'] else None
    cl_dat = weights_dic['cl_dat_np'] if p_dic['cluster_std'] else None
    n_y, no_of_out = len(y_dat), len(var_dic['y_name'])
    if p_dic['gates_smooth']:
        var_dic, var_x_values, smooth_yes, z_name_smooth = addsmoothvars(
            data_df, var_dic, var_x_values, p_dic)
    else:
        smooth_yes, z_name_smooth = False, None
    d_p, z_p, w_p, _ = mcf_ate.get_data_for_final_ate_estimation(
        data_df, gen_dic, p_dic, var_dic, ate=False, need_count=False)
    z_type_l = [None] * len(var_dic['z_name'])
    z_values_l = z_type_l[:]
    z_smooth_l = [False] * len(var_dic['z_name'])

    if gen_dic['d_type'] == 'continuous':
        continuous = True
        p_dic['atet'] = p_dic['gatet'] = False
        no_of_treat, d_values = ct_dic['grid_w'], ct_dic['grid_w_val']
        d_values_dr = ct_dic['d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
        treat_comp_label = [None] * round(no_of_treat_dr - 1)
    else:
        continuous = False
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
        treat_comp_label = [None] * round(no_of_treat * (no_of_treat - 1) / 2)
    i_d_val = np.arange(no_of_treat)
    ref_pop_lab = ['All']
    if p_dic['gatet']:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab += str(lab)
    for zj_idx, z_name in enumerate(var_dic['z_name']):
        z_type_l[zj_idx] = var_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = var_x_values[z_name]
        if smooth_yes:
            z_smooth_l[zj_idx] = z_name in z_name_smooth
    if (d_p is not None) and p_dic['gatet']:
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        p_dic['gatet'], no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = p_dic['choice_based_probs']

    jdx = 0
    for t1_idx, t1_lab in enumerate(d_values):
        for t2_idx in range(t1_idx+1, no_of_treat):
            treat_comp_label[jdx] = str(d_values[t2_idx]) + 'vs' + str(t1_lab)
            jdx += 1
        if continuous:
            break
    if p_dic['gates_minus_previous']:
        w_ate = None
    else:
        w_ate_sum = np.sum(w_ate, axis=2)
        for a_idx in range(no_of_tgates):  # Weights for ATE are normalized
            for t_idx in range(no_of_treat):
                if not ((1-1e-10) < w_ate_sum[a_idx, t_idx] < (1+1e-10)):
                    w_ate[a_idx, t_idx, :] = (w_ate[a_idx, t_idx, :]
                                              / w_ate_sum[a_idx, t_idx])
    files_to_delete, save_w_file = set(), None
    if gen_dic['mp_parallel'] > 1 and int_dic['ray_or_dask'] != 'ray':
        memory_weights = mcf_sys.total_size(weights_all)
        if int_dic['weight_as_sparse']:
            for d_idx in range(no_of_treat):
                memory_weights += (weights_all[d_idx].data.nbytes
                                   + weights_all[d_idx].indices.nbytes
                                   + weights_all[d_idx].indptr.nbytes)
        if memory_weights > 2e+9:  # Two Gigabytes (2e+9)
            if int_dic['with_output'] and int_dic['verbose']:
                txt += ('Weights need ', memory_weights/1e+9, 'GB RAM'
                        '==> Weights are passed as file to MP processes')
            save_w_file = 'w_all.pickle'
            mcf_sys.save_load(save_w_file, weights_all, save=True,
                              output=int_dic['with_output'])
            files_to_delete.add(save_w_file)
            weights_all2 = None
        else:
            weights_all2 = weights_all
    else:
        weights_all2 = weights_all
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share']/2)
        else:
            maxworkers = gen_dic['no_parallel']
        if weights_all2 is None:
            maxworkers = round(maxworkers / 2)
        if not maxworkers > 0:
            maxworkers = 1
    if int_dic['with_output'] and int_dic['verbose'] and with_output:
        print('Number of parallel processes: ', maxworkers, flush=True)
    if int_dic['ray_or_dask'] == 'ray':
        if int_dic['mem_object_store_3'] is None:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False)
        else:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False,
                         object_store_memory=int_dic['mem_object_store_3'])
            if int_dic['with_output'] and int_dic['verbose']:
                print('Size of Ray Object Store: ',
                      round(int_dic['mem_object_store_3']/(1024*1024)), " MB")
        weights_all_ref = ray.put(weights_all)
    y_pot_all, y_pot_var_all, y_pot_mate_all = [], [], []
    y_pot_mate_var_all, txt_all = [], []
    for z_name_j, z_name in enumerate(var_dic['z_name']):
        txt_z_name = txt + ' '
        if int_dic['with_output'] and int_dic['verbose'] and with_output:
            print(z_name_j+1, '(', len(var_dic['z_name']), ')', z_name,
                  flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        if z_smooth:
            kernel = 1  # Epanechikov
            bandw_z = mcf_est.bandwidth_nw_rule_of_thumb(z_p[:, z_name_j])
            bandw_z = bandw_z * p_dic['gates_smooth_bandwidth']
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
        if (maxworkers == 1) or p_dic['gates_minus_previous']:
            for zj_idx in range(no_of_zval):
                if p_dic['gates_minus_previous']:
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
                    no_of_tgates, no_of_out, ct_dic, gen_dic, int_dic, p_dic,
                    bandw_z, kernel, z_smooth, continuous)
                y_pot, y_pot_var, y_pot_mate, y_pot_mate_var = assign_pot(
                     y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                     results_fut_zj, zj_idx)
                w_gate, w_gate_unc, w_censored = assign_w(
                     w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx)
        else:
            if int_dic['ray_or_dask'] == 'ray':
                still_running = [ray_gate_zj_mp.remote(
                         z_values[zj_idx], zj_idx, y_dat, cl_dat,
                         w_dat, z_p, d_p, w_p, z_name_j, weights_all_ref,
                         w_gate0_dim, w_ate, i_d_val, t_probs, no_of_tgates,
                         no_of_out, ct_dic, gen_dic, int_dic, p_dic, n_y,
                         bandw_z, kernel, save_w_file,
                         z_smooth, continuous)
                    for zj_idx in range(no_of_zval)]
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        (y_pot, y_pot_var, y_pot_mate, y_pot_mate_var
                         ) = assign_pot(
                             y_pot, y_pot_var, y_pot_mate, y_pot_mate_var,
                             results_fut_idx, results_fut_idx[6])
                        w_gate, w_gate_unc, w_censored = assign_w(
                            w_gate, w_gate_unc, w_censored, results_fut_idx,
                            results_fut_idx[6])
        if int_dic['with_output']:
            # Describe weights
            for a_idx in range(no_of_tgates):
                w_st = np.zeros((6, no_of_treat))
                share_largest_q = np.zeros((no_of_treat, 3))
                sum_larger = np.zeros((no_of_treat, len(p_dic['q_w'])))
                obs_larger = np.zeros_like(sum_larger)
                w_censored_all = np.zeros(no_of_treat)
                for zj_idx in range(no_of_zval):
                    ret = mcf_est.analyse_weights(
                        w_gate[zj_idx, a_idx, :, :], None, gen_dic, p_dic,
                        ate=False, continuous=continuous,
                        no_of_treat_cont=no_of_treat, d_values_cont=d_values)
                    for idx in range(6):
                        w_st[idx] += ret[idx] / no_of_zval
                    share_largest_q += ret[6] / no_of_zval
                    sum_larger += ret[7] / no_of_zval
                    obs_larger += ret[8] / no_of_zval
                    w_censored_all += w_censored[zj_idx, a_idx, :]
                if gate_type == 'GATE':
                    txt_z_name += '\n' + '=' * 100
                    txt_z_name += (
                        '\nAnalysis of weights (normalised to add to 1): '
                        f'{gate_type} for {z_name} '
                        f'(stats are averaged over {no_of_zval} groups).')
                    if p_dic['gatet']:
                        txt += f'\nTarget population: {ref_pop_lab[a_idx]:<4}'
                    txt_z_name += ps.txt_weight_stat(
                        w_st[0], w_st[1], w_st[2], w_st[3], w_st[4], w_st[5],
                        share_largest_q, sum_larger, obs_larger, gen_dic,
                        p_dic, w_censored_all, continuous=continuous,
                        d_values_cont=d_values)  # Discretized weights if cont
            txt_z_name += '\n'
        y_pot_all.append(y_pot)
        y_pot_var_all.append(y_pot_var)
        y_pot_mate_all.append(y_pot_mate)
        y_pot_mate_var_all.append(y_pot_mate_var)
        txt_all.append(txt_z_name)
        if files_to_delete:  # delete temporary files
            for file in files_to_delete:
                os.remove(file)
        gate_est_dic = {
            'continuous': continuous, 'd_values': d_values,
            'd_values_dr': d_values_dr, 'treat_comp_label': treat_comp_label,
            'no_of_out': no_of_out, 'var_dic': var_dic,
            'var_x_values': var_x_values, 'smooth_yes': smooth_yes,
            'z_name_smooth': z_name_smooth, 'ref_pop_lab': ref_pop_lab,
            'z_p': z_p, 'no_of_tgates': no_of_tgates, 'p_dic': p_dic,
            }
    if int_dic['ray_or_dask'] == 'ray':
        if 'refs' in int_dic['mp_ray_del']:
            del weights_all_ref
        if 'rest' in int_dic['mp_ray_del']:
            del finished_res, finished
        if int_dic['mp_ray_shutdown']:
            ray.shutdown()
    return (y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all,
            gate_est_dic, txt_all)


def bamgate_est(mcf_, data_df, weights_dic, w_ate, forest_dic,
                gate_type='AMGATE'):
    """Compute AMGATE & BGATE for single variables keeping others constant."""
    int_dic, var_dic = mcf_.int_dict, mcf_.var_dict
    var_x_type = deepcopy(mcf_.var_x_type)
    var_x_values = deepcopy(mcf_.var_x_values)
    p_dic, var_dic = deepcopy(mcf_.p_dict), deepcopy(mcf_.var_dict)
    x_name_mcf = mcf_.cf_dict['x_name_mcf']
    bgate = gate_type == 'BGATE'
    if var_dic['z_name'] is None or var_dic['z_name'] == []:
        raise ValueError(f'Something wrong with {var_dic["z_name"]}')
    if bgate:
        if var_dic['bgate_name'] is None or var_dic['bgate_name'] == []:
            raise ValueError(f'Something wrong with {var_dic["bgate_name"]}')
    txt = ''
    if int_dic['with_output']:
        txt_1 = '\n' + '=' * 100 + f'\nComputing {gate_type}'
        print(txt)
    if p_dic['gatet']:
        p_dic['gatet'] = p_dic['atet'] = False
        if int_dic['with_output']:
            txt_1 += f'\nNo treatment specific effects for {gate_type}.'
    eva_values = ref_vals_amgate(data_df, var_x_type, var_x_values,
                                 no_eva_values=p_dic['gmate_no_evalu_points'])
    if int_dic['with_output'] and int_dic['verbose']:
        print(f'\n{gate_type} variable under investigation: ', end=' ')
    y_pot_all, y_pot_var_all, y_pot_mate_all, txt_all = [], [], [], []
    y_pot_mate_var_all = []
    first_run = True
    for vname in var_dic['z_name']:
        txt = txt_1[:]
        if vname not in x_name_mcf:
            raise ValueError(f'Heterogeneity variable for {gate_type} NOT used'
                             ' for splitting. Perhaps turn off {type_txt}.')
        if int_dic['with_output'] and int_dic['verbose']:
            print(vname, end=' ')
        if bgate:
            data_df_new, z_values, txt_sim = ref_data_bgate(
                data_df.copy(), vname, int_dic, p_dic, eva_values,
                var_dic['bgate_name'][:])
        else:
            data_df_new, z_values, txt_sim = ref_data_amgate(
                data_df.copy(), vname, int_dic, p_dic, eva_values)
        txt += txt_sim
        var_x_values[vname] = z_values[:]
        weights_dic = mcf_w.get_weights_mp(mcf_, data_df_new, forest_dic,
                                           'regular', with_output=False)
        (w_ate, _, _, _) = mcf_ate.ate_est(
            mcf_, data_df_new, weights_dic, with_output=False)
        (y_pot_gate_z, y_pot_var_gate_z, y_pot_mate_gate_z,
         y_pot_mate_var_gate_z, gate_est_dic_z, txt_p) = gate_est(
             mcf_, data_df_new, weights_dic, w_ate, gate_type=gate_type,
             z_name_amgate=[vname], with_output=False)
        txt += txt_p[0]
        txt_all.append(txt)
        y_pot_all.append(y_pot_gate_z[0])
        y_pot_var_all.append(y_pot_var_gate_z[0])
        y_pot_mate_all.append(y_pot_mate_gate_z[0])
        y_pot_mate_var_all.append(y_pot_mate_var_gate_z[0])
        if first_run:
            gate_est_dic_all = deepcopy(gate_est_dic_z)
            gate_est_dic_all['z_p'] = []
            gate_est_dic_all['var_dic']['z_name'] = []
        gate_est_dic_all['z_p'].append(gate_est_dic_z['z_p'])
        gate_est_dic_all['var_dic']['z_name'].append(vname)
        first_run = False
    return (y_pot_all, y_pot_var_all, y_pot_mate_all, y_pot_mate_var_all,
            gate_est_dic_all, txt_all, txt_sim)


def ref_data_amgate(data_df, z_name, int_dic, p_dic, eva_values):
    """Create reference samples for covariates (AMGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = p_dic['gmate_sample_share'] / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if int_dic['with_output'] and int_dic['verbose']:
                txt += f'\nAMGATE: {share:5.2%} random sample drawn'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_all_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_all_df.loc[:, z_name] = new_values_z
    if int_dic['with_output'] and int_dic['verbose']:
        txt += ('\nAMGATEs minus ATE are evaluated at fixed z-feature values'
                '(equally weighted).')
    return data_all_df, eva_values, txt


def ref_data_bgate(data_df, z_name, int_dic, p_dic, eva_values, bgate_name):
    """Create reference samples for covariates (BGATE)."""
    eva_values = eva_values[z_name]
    no_eval, obs, txt = len(eva_values), len(data_df), ''
    if z_name in bgate_name:
        bgate_name.remove(z_name)
        if not bgate_name:
            raise ValueError('BGATE {z_name}: No variables left for '
                             'balancing.')
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = p_dic['gmate_sample_share'] / no_eval
        if 0 < share < 1:
            rng = np.random.default_rng(seed=9324561)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if int_dic['with_output'] and int_dic['verbose']:
                txt += f'\nBGATE: {share:5.2%} random sample drawn.'
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(chain.from_iterable(repeat(idx, no_eval)))
    data_new_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in eva_values]))
    data_new_df.loc[:, z_name] = new_values_z
    if int_dic['with_output'] and int_dic['verbose']:
        txt += (f'\nBGATEs are balanced with respect to {" ".join(bgate_name)}'
                '\nBGATEs minus ATE are evaluated at fixed z-feature values'
                '(equally weighted).')
    # Until here these are the identical steps to AMGATE. Next, observations in
    # data_all_df get substituted by their nearest neighbours in terms of
    # bgate_name conditional on the z_name variables.
    data_new_b_np = data_new_df[bgate_name].to_numpy(copy=True)
    data_new_z_np = data_new_df[z_name].to_numpy(copy=True)
    data_org_b_np = data_df[bgate_name].to_numpy()
    data_org_z_np = data_df[z_name].to_numpy()
    data_org_np = data_df.to_numpy()
    if data_org_b_np.shape[1] > 1:
        bz_cov_inv = invcovariancematrix(data_org_b_np)
        bz_cov_inv[-1, -1] *= 10
    else:
        bz_cov_inv = 1
    # Give much more additional weight to the z-related component in matching
    for idx, z_value in enumerate(data_new_z_np):
        z_true = data_org_z_np == z_value
        data_org_np_condz = data_org_np[z_true]
        data_org_b_np_condz = data_org_b_np[z_true]
        diff = data_org_b_np_condz - data_new_b_np[idx, :]
        dist = np.sum(np.dot(diff, bz_cov_inv) * diff, axis=1)
        match_neigbour_idx = np.argmin(dist)
        data_new_df.iloc[idx] = data_org_np_condz[match_neigbour_idx, :]
    return data_new_df, eva_values, txt


def invcovariancematrix(data_np):
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


def ref_vals_amgate(data_df, var_x_type, var_x_values, no_eva_values=50):
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


def addsmoothvars(data_df, var_dic, var_x_values, p_dic):
    """
    Find variables for which to smooth gates and evaluation points.

    Parameters
    ----------
    data_df: DataFrame. Prediction data.
    var_dic : Dict. Variables.
    var_x_values : Dict. Variables
    p_dic : Dict. Controls.

    Returns
    -------
    var_dic_new : Dict. Updated variables.
    var_x_values_new : Dict. Updated with evaluation points.
    smooth_yes : Bool. Indicator if smoothing will happen.

    """
    smooth_yes, z_name = False, var_dic['z_name']
    z_name_add = [name[:-4] for name in z_name if (name.endswith('CATV')
                                                   and (len(name) > 4))]
    if z_name_add:
        smooth_yes = True
        var_dic_new = deepcopy(var_dic)
        var_x_values_new = deepcopy(var_x_values)
        data_np = data_df[z_name_add].to_numpy()
        for idx, name in enumerate(z_name_add):
            var_x_values_new[name] = smooth_gate_eva_values(
                data_np[:, idx], p_dic['gates_smooth_no_evalu_points'])
            var_dic_new['z_name'].append(name)
    else:
        var_dic_new, var_x_values_new = var_dic, var_x_values
    return var_dic_new, var_x_values_new, smooth_yes, z_name_add


def smooth_gate_eva_values(z_dat, no_eva_values):
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
def ray_gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                   z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
                   no_of_tgates, no_of_out, ct_dic, gen_dic, int_dic, p_dic,
                   n_y, bandw_z, kernel, save_w_file=None, smooth_it=False,
                   continuous=False):
    """Make function compatible with Ray."""
    return gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                      z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val,
                      t_probs, no_of_tgates, no_of_out, ct_dic, gen_dic,
                      int_dic, p_dic, n_y, bandw_z, kernel, save_w_file,
                      smooth_it, continuous)


def gate_zj(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p, z_name_j,
            weights_all, w_gate0_dim, w_gate_zj, w_gate_unc_zj, w_censored_zj,
            w_ate, y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj,
            i_d_val, t_probs, no_of_tgates, no_of_out, ct_dic, gen_dic,
            int_dic, p_dic, bandw_z, kernel, smooth_it=False,
            continuous=False):
    """Compute Gates and their variances for MP."""
    if continuous:
        no_of_treat, d_values = ct_dic['ct_grid_w'], ct_dic['ct_grid_w_val']
        i_w01 = ct_dic['ct_w_to_dr_int_w01']
        i_w10 = ct_dic['ct_w_to_dr_int_w10']
        index_full = ct_dic['ct_w_to_dr_index_full']
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
    weights, relevant_z,  w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=int_dic['weight_as_sparse'])
    if p_dic['gatet']:
        d_p_z = d_p[relevant_z]
    if gen_dic['weighted']:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if int_dic['weight_as_sparse'] else len(weights)
    # Step 1: Aggregate weights
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_dic['weight_as_sparse']:
                weight_i = weights[t_idx].getrow(n_idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_dic['weighted']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not 1-1e-10 < w_i_sum < 1+1e-10:
                w_i = w_i / w_i_sum
            if gen_dic['weighted']:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_dic['choice_based_sampling']:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if p_dic['gatet']:
            t_pos_i = i_d_val[d_p_z[n_idx] == d_values]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd
    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx], w_gate_zj,
                    w_censored_zj, w_gate_unc_zj, w_ate, int_dic, p_dic)
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             int_dic['max_weight_share'])
                        ret = mcf_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, gen_dic,
                            p_dic, weights=w_dat,
                            bootstrap=p_dic['se_boot_gate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if int_dic['with_output']:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = mcf_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, gen_dic,
                                p_dic, norm=False, weights=w_dat,
                                bootstrap=p_dic['se_boot_gate'])
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_dic, p_dic, weights=w_dat,
                        bootstrap=p_dic['se_boot_gate'])
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if int_dic['with_output']:
                        ret2 = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat,
                            gen_dic, p_dic, norm=False, weights=w_dat,
                            bootstrap=p_dic['se_boot_gate'])
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj)


def gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
               z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
               no_of_tgates, no_of_out, ct_dic, gen_dic, int_dic, p_dic, n_y,
               bandw_z, kernel, save_w_file=None, smooth_it=False,
               continuous=False):
    """Compute Gates and their variances for MP."""
    if continuous:
        no_of_treat, d_values = ct_dic['grid_w'], ct_dic['grid_w_val']
        d_values_dr = ct_dic['d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
        i_w01 = ct_dic['w_to_dr_int_w01']
        i_w10 = ct_dic['w_to_dr_int_w10']
        index_full = ct_dic['w_to_dr_index_full']
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    if save_w_file is not None:
        weights_all = mcf_sys.save_load(save_w_file, save=False,
                                        output=int_dic['with_output'])
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
        kernel=kernel, w_is_csr=int_dic['weight_as_sparse'])
    if p_dic['gatet']:
        d_p_z = d_p[relevant_z]
    if gen_dic['weighted']:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if int_dic['weight_as_sparse'] else len(weights)
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if int_dic['weight_as_sparse']:
                weight_i = weights[t_idx].getrow(n_idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if gen_dic['weighted']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not (1-1e-10) < w_i_sum < (1+1e-10):
                w_i = w_i / w_i_sum
            if gen_dic['weighted']:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if p_dic['choice_based_sampling']:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if p_dic['gatet']:
            t_pos_i = i_d_val[d_p_z[n_idx] == d_values]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd
    # Step 2: Get potential outcomes for particular z_value
    if not continuous:
        sum_wgate = np.sum(w_gate_zj, axis=2)
    for a_idx in range(no_of_tgates):
        for t_idx in range(no_of_treat):
            if not continuous:
                w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj = w_gate_func(
                    a_idx, t_idx, sum_wgate[a_idx, t_idx], w_gate_zj,
                    w_censored_zj, w_gate_unc_zj, w_ate, int_dic, p_dic)
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             p_dic['max_weight_share'])
                        ret = mcf_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, gen_dic,
                            p_dic, weights=w_dat,
                            bootstrap=p_dic['se_boot_gate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        y_pot_zj[a_idx, ti_idx, o_idx] = ret[0]
                        y_pot_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if int_dic['with_output']:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = mcf_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, gen_dic,
                                p_dic, norm=False, weights=w_dat,
                                bootstrap=p_dic['se_boot_gate'])
                            y_pot_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            y_pot_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_dic, p_dic, weights=w_dat,
                        bootstrap=p_dic['se_boot_gate'])
                    y_pot_zj[a_idx, t_idx, o_idx] = ret[0]
                    y_pot_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if int_dic['with_output']:
                        ret2 = mcf_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat, gen_dic,
                            p_dic, norm=False, weights=w_dat,
                            bootstrap=p_dic['se_boot_gate'])
                        y_pot_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        y_pot_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    if w_gate_zj.nbytes > 1e+9 and int_dic['ray_or_dask'] != 'ray':
        # otherwise tuple gets too large for MP
        save_name_w = 'wtemp' + str(zj_idx) + '.npy'
        save_name_wunc = 'wunctemp' + str(zj_idx) + '.npy'
        np.save(save_name_w, w_gate_zj, fix_imports=False)
        np.save(save_name_wunc, w_gate_unc_zj, fix_imports=False)
        w_gate_zj = w_gate_unc_zj = None
    else:
        save_name_w = save_name_wunc = None
    return (y_pot_zj, y_pot_var_zj, y_pot_mate_zj, y_pot_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj, save_name_w,
            save_name_wunc)


def get_w_rel_z(z_dat, z_val, weights_all, smooth_it, bandwidth=1, kernel=1,
                w_is_csr=False):
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
        w_z_val = mcf_est.kernel_proc((z_dat - z_val) / bandwidth, kernel)
        relevant_data_points = w_z_val > 1e-10
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


def w_gate_func(a_idx, t_idx, sum_wgate, w_gate_zj, w_censored_zj,
                w_gate_unc_zj, w_ate, int_dic, p_dic):
    """Compute weights for discrete case."""
    if (not 1-1e-10 < sum_wgate < 1+1e-10) and (sum_wgate > 1e-10):
        w_gate_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :] / sum_wgate
    w_gate_unc_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :]
    if p_dic['max_weight_share'] < 1:
        (w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx]
         ) = mcf_gp.bound_norm_weights(w_gate_zj[a_idx, t_idx, :],
                                       p_dic['max_weight_share'])
    if int_dic['with_output']:
        w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx, t_idx, :]
    else:
        w_diff = None
    return w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj


def w_gate_cont_funct(t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01,  i,
                      w_gate_unc_zj, w_censored_zj, max_weight_share):
    """Approximate weights for continuous treatments."""
    if t_idx == (no_of_treat - 1):  # last element,no inter
        w_gate_cont = w_gate_zj[a_idx, t_idx, :]
    else:
        w_gate_cont = (w10 * w_gate_zj[a_idx, t_idx, :]
                       + w01 * w_gate_zj[a_idx, t_idx+1, :])
    sum_wgate = np.sum(w_gate_cont)
    if not ((-1e-15 < sum_wgate < 1e-15) or (1-1e-10 < sum_wgate < 1+1e-10)):
        w_gate_cont = w_gate_cont / sum_wgate
    if i == 0:
        w_gate_unc_zj[a_idx, t_idx, :] = w_gate_cont
    w_gate_cont_unc = w_gate_cont.copy()
    if max_weight_share < 1:
        w_gate_cont, _, w_censored = mcf_gp.bound_norm_weights(
            w_gate_cont, max_weight_share)
        if i == 0:
            w_censored_zj[a_idx, t_idx] = w_censored
    return w_gate_cont, w_gate_cont_unc, w_gate_unc_zj, w_censored_zj


def w_diff_cont_func(t_idx, a_idx, no_of_treat, w_gate_cont, w_ate, w10, w01):
    """Compute weights for difference in continuous case."""
    w_ate_cont = w_ate[a_idx, t_idx, :] if t_idx == no_of_treat - 1 else (
        w10 * w_ate[a_idx, t_idx, :] + w01 * w_ate[a_idx, t_idx+1, :])
    w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
    w_diff_cont = w_gate_cont - w_ate_cont
    return w_diff_cont


def assign_pot(y_pot, y_pot_var, y_pot_mate, y_pot_mate_var, results_fut_zj,
               zj_idx):
    """Reduce repetetive code."""
    y_pot[zj_idx, :, :, :] = results_fut_zj[0]
    y_pot_var[zj_idx, :, :, :] = results_fut_zj[1]
    y_pot_mate[zj_idx, :, :, :] = results_fut_zj[2]
    y_pot_mate_var[zj_idx, :, :, :] = results_fut_zj[3]
    return y_pot, y_pot_var, y_pot_mate, y_pot_mate_var


def assign_w(w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx):
    """Reduce repetetive code."""
    w_gate[zj_idx, :, :, :] = results_fut_zj[4]
    w_gate_unc[zj_idx, :, :, :] = results_fut_zj[5]
    w_censored[zj_idx, :, :] = results_fut_zj[7]
    return w_gate, w_gate_unc, w_censored
