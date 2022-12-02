"""
Procedures needed for GATEs estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import os
from concurrent import futures
from dask.distributed import Client, as_completed
import itertools

import numpy as np
import pandas as pd

import ray

from mcf import mcf_weight_functions as mcf_w
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_gate_out_functions as mcf_gateout
from mcf import mcf_general_purpose as mcf_gp
from mcf import mcf_hf
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_system_files as gp_sys


def marg_gates_est(forest, fill_y_sample, pred_sample, v_dict, c_dict,
                   x_name_mcf, var_x_type, var_x_values, w_ate=None,
                   regrf=False):
    """Compute MGATE and AMGATE for single variables keeping others constant.

    Parameters
    ----------
    forest : List of list.
    fill_y_sample : String. Name of sample used to fill tree.
    pred_sample : String. Name of prediction sample.
    v_dict : Dict.
    c_dict : Dict.
    x_name_mcf : Names from MCF procedure.
    var_x_type : List of int. Type of feature.
    var_x_values : List of List of float or int. Values of features.
    w_ate : Numpy array. Weights for ATE computation. Default is None.
    regrf: Boolean. False if MCF (default).

    Returns
    -------
    mgate: List of Numpy Array. Marginal Gates.
    mgate_se: List of Numpy Array. Standard errors of marginal Gates.
    mgate_diff: List of Numpy Array. Differnce of Marginal Gates.
    mgate_se_diff: List of Numpy Array. Standard errors of diff. mgates
    amgate: List of Numpy Array. Marginal Average Gates.
    amgate_se: List of Numpy Array. Standard errors of AMGates.
    amgate_diff: List of Numpy Array. Differnce of AMGates.
    amgate_se_diff: List of Numpy Array. Standard errors of diff. of AMgates

    """
    any_plots_mgate = any_plots_amgate = False
    mgate = mgate_se = mgate_diff = mgate_se_diff = None
    amgate = amgate_se = amgate_diff = amgate_se_diff = None
    c_dict_mgate = copy.deepcopy(c_dict)
    c_dict_mgate['with_output'] = False       # reduce unnecessary infos
    c_dict_mgate['iate_se_flag'] = True
    if v_dict['z_name_mgate'] and c_dict['with_output'] and (
            not c_dict['gates_minus_previous']):
        if c_dict['with_output']:
            print_str = '=' * 80
            if regrf:
                print_str += '\nMarginal variable predictive plots'
            else:
                print_str += '\nMarginale GATEs evaluated at median (MGATES)'
            if c_dict_mgate['choice_based_yes']:
                print_str += '\nChoice based sampling deactivated for MGATES.'
            print_str += '\n' + '-' * 80
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
        (any_plots_mgate, mgate, mgate_se, mgate_diff, mgate_se_diff
         ) = mgate_function(forest, fill_y_sample, pred_sample, v_dict, c_dict,
                            x_name_mcf, var_x_type, var_x_values, regrf,
                            c_dict_mgate, w_ate)
        if not any_plots_mgate:
            if regrf:
                print("No variables for marginal plots left.")
            else:
                print("No variables for MGATE left.")
        else:
            print('\n')
    if v_dict['z_name_amgate'] and c_dict['with_output']:
        if c_dict['with_output']:
            print_str = ('\n' + '=' * 80 + '\nOther-X-fixed-GATEs averaged'
                         + ' over sample (AMGATEs)')
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
        (any_plots_amgate, amgate, amgate_se, amgate_diff, amgate_se_diff
         ) = amgate_function(forest, fill_y_sample, pred_sample, v_dict,
                             c_dict, x_name_mcf, var_x_type, var_x_values,
                             c_dict_mgate)
        if not any_plots_amgate:
            print("No variables for AMGATE left.")
        else:
            print('\n')
    return (mgate, mgate_se, mgate_diff, mgate_se_diff, amgate, amgate_se,
            amgate_diff, amgate_se_diff)


def amgate_function(forest, fill_y_sample, pred_sample, v_dict, c_dict,
                    x_name_mcf, var_x_type, var_x_values, c_dict_mgate):
    """Compute AMGATE for single variables keeping others constant.

    For each value of z
        create data with this value
        collect all data and write it to file --> new prediction file
    compute standard GATE

    this needs some adjustment for continous variables

    Parameters
    ----------
    forest : List of list.
    fill_y_sample : String. Name of sample used to fill tree.
    pred_sample: String. Name of sample to predict effects for.
    v_dict : Dict.
    c_dict : Dict.
    x_name_mcf : Names from MCF procedure.
    var_x_type : List of int. Type of feature.
    var_x_values : List of List of float or int. Values of features.
    c_dict_mgate: Dict. Differs only for 'with_output' (t) from c_dict.

    Returns
    -------
    any_plots_done : Bool.

    """
    if c_dict['gatet_flag']:
        c_dict_mgate['gatet_flag'] = c_dict_mgate['atet_flag'] = False
        if c_dict['with_output']:
            print('No treatment specific effects for MGATE and AMGATE.')
    amgate = amgate_se = amgate_diff = amgate_se_diff = None
    any_plots_done = False
    _, eva_values = mcf_gateout.ref_vals_margplot(
        pred_sample, var_x_type, var_x_values,
        with_output=c_dict['with_output'], ref_values_needed=False,
        no_eva_values=c_dict['gmate_no_evaluation_points'])
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nVariable under investigation: ', end=' ')
    z_name_old = v_dict['z_name']
    for vname in v_dict['z_name_amgate']:
        if vname in x_name_mcf:
            if c_dict['with_output'] and c_dict['verbose']:
                print(vname, end=' ')
            any_plots_done = True
            new_predict_file, z_values = mcf_gateout.ref_file_marg_plot_amgate(
                pred_sample, vname, c_dict, eva_values)
            v_dict['z_name'] = [vname]
            var_x_values[vname] = z_values[:]
            weights, y_f, _, cl_f, w_f = mcf_w.get_weights_mp(
                forest, new_predict_file, fill_y_sample, v_dict,
                c_dict_mgate, x_name_mcf, regrf=False)
            w_ate_iate, _, _, ate_z, ate_se_z, _ = mcf_ate.ate_est(
                    weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                    c_dict_mgate, print_output=False)
            c_dict_mgate['with_output'] = c_dict['with_output']
            amgate, amgate_se, amgate_diff, amgate_se_diff = gate_est(
                weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                c_dict_mgate, var_x_type, var_x_values, w_ate_iate, ate_z,
                ate_se_z, amgate_flag=True)
            os.remove(new_predict_file)  # Delete new file
    v_dict['z_name'] = z_name_old
    return any_plots_done, amgate, amgate_se, amgate_diff, amgate_se_diff


def mgate_function(
        forest, fill_y_sample, pred_sample, v_dict, c_dict, x_name_mcf,
        var_x_type, var_x_values, regrf, c_dict_mgate, w_ate):
    """Compute MGATE for single variables keeping others constant.

    Parameters
    ----------
    forest : List of list.
    fill_y_sample : String. Name of sample used to fill tree.
    v_dict : Dict.
    c_dict : Dict.
    x_name_mcf : Names from MCF procedure.
    var_x_type : List of int. Type of feature.
    var_x_values : List of List of float or int. Values of features.
    regrf: Boolean. False if MCF (default).
    c_dict_mgate: Dict. Differs only for 'with_output' from c_dict.

    Returns
    -------
    any_plots_done : Bool.

    """
    def mgate_corrections(eff, eff_se, counter):
        for i in range(eff.shape[0]):
            if np.abs(eff[i, -1]) > 10 * np.abs(eff[i, -2]):
                eff[i, -1], eff_se[i, -1] = eff[i, -2], eff_se[i, -2]
                counter += 1
        return eff, eff_se, counter

    any_plots_done, print_str = False, ''
    ref_values, eva_values = mcf_gateout.ref_vals_margplot(
        pred_sample, var_x_type, var_x_values,
        with_output=c_dict['with_output'], ref_values_needed=True,
        no_eva_values=c_dict['gmate_no_evaluation_points'])
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nVariable under investigation: ', end=' ')
    w_yes_old = c_dict_mgate['w_yes']
    c_dict_mgate['w_yes'] = False   # Weighting not needed here
    with_output_old = c_dict_mgate['with_output']
    c_dict_mgate['with_output'] = False
    choice_based_yes_old = c_dict_mgate['choice_based_yes']
    c_dict_mgate['choice_based_yes'] = False
    correct_m_gate_cont = 0
    z_name_mgate = [z for z in v_dict['z_name_mgate'] if z in x_name_mcf]
    mgate = [None] * len(z_name_mgate)
    mgate_se, mgate_diff,  mgate_se_diff = mgate[:], mgate[:], mgate[:]
    for z_name_j, z_name in enumerate(z_name_mgate):
        if c_dict['with_output'] and c_dict['verbose']:
            print(z_name, end=' ')
        any_plots_done = True
        new_predict_file = mcf_gateout.ref_file_marg_plot(
            z_name, c_dict_mgate, var_x_type, ref_values, eva_values)
        weights, y_f, _, cl_f, w_f = mcf_w.get_weights_mp(
            forest, new_predict_file, fill_y_sample, v_dict, c_dict_mgate,
            x_name_mcf, regrf=regrf)
        if regrf:
            _, y_pred, y_pred_se, name_pred, _ = mcf_hf.predict_hf(
                weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                c_dict_mgate)
        else:
            w_ate_iate, _, _, _, _, _ = mcf_ate.ate_est(
                weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                c_dict_mgate, w_ate_only=True, print_output=False)
            (_, _, _, iate, iate_se, namesiate, _) = mcf_iate.iate_est_mp(
                weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                c_dict_mgate, save_predictions=False, w_ate=w_ate_iate)
            names_iate = namesiate[0]
            name_pred = names_iate['names_iate']
            shape = np.shape(iate[:, :, :, 0])
            y_pred = iate[:, :, :, 0].reshape(shape[0], shape[1]*shape[2])
            y_pred_se = iate_se[:, :, :, 0].reshape(
                shape[0], shape[1]*shape[2])
            if w_ate is not None:
                names_iate_mate = namesiate[0]
                name_mate_pred = names_iate_mate['names_iate_mate']
                y_pred_mate = iate[:, :, :, 1].reshape(
                    shape[0], shape[1]*shape[2])
                y_pred_mate_se = iate_se[:, :, :, 1].reshape(
                    shape[0], shape[1]*shape[2])
                if y_pred_mate.shape[1] > 1:
                    (y_pred_mate, y_pred_mate_se, correct_m_gate_cont
                     ) = mgate_corrections(y_pred_mate, y_pred_mate_se,
                                           correct_m_gate_cont)
        mgate[z_name_j], mgate_se[z_name_j] = y_pred, y_pred_se
        mgate_diff[z_name_j] = y_pred_mate
        mgate_se_diff[z_name_j] = y_pred_mate_se
        if c_dict['with_output']:
            if c_dict['d_type'] == 'continuous':
                mcf_gateout.plot_marginal_cont(
                    y_pred, y_pred_se, z_name, eva_values[z_name],
                    var_x_type[z_name], c_dict, minus_ate=False)
            else:
                mcf_gateout.plot_marginal(
                    y_pred, y_pred_se, name_pred, z_name, eva_values[z_name],
                    var_x_type[z_name], c_dict, regrf, minus_ate=False)
            if not regrf and (w_ate is not None):
                if c_dict['d_type'] == 'continuous':
                    mcf_gateout.plot_marginal_cont(
                        y_pred_mate, y_pred_mate_se, z_name,
                        eva_values[z_name], var_x_type[z_name], c_dict,
                        minus_ate=True)
                else:
                    mcf_gateout.plot_marginal(
                        y_pred_mate, y_pred_mate_se, name_mate_pred, z_name,
                        eva_values[z_name], var_x_type[z_name], c_dict,
                        regrf, minus_ate=True)
    if not regrf and (w_ate is not None):
        if c_dict['with_output'] and c_dict['verbose']:
            print_str += ('\nMGATEs minus ATE are evaluated at fixed feature'
                          + ' values equally weighted).')
            if correct_m_gate_cont > 0:
                print_str += (
                    f'MGATE-ATE {correct_m_gate_cont} times corrected for '
                    + 'excessive last value. This is not good. Probably,'
                    + ' some bug when computing difference of MATE with ate.')
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
    c_dict_mgate['w_yes'] = w_yes_old
    c_dict_mgate['with_output'] = with_output_old
    c_dict_mgate['choice_based_yes'] = choice_based_yes_old
    return any_plots_done, mgate, mgate_se, mgate_diff, mgate_se_diff


def gate_est(weights_all, pred_data, y_dat, cl_dat, w_dat, v_dict, c_dict,
             v_x_type, v_x_values, w_ate, ate, ate_se, amgate_flag=False):
    """Estimate GATE(T)s and AMGAT(T) and their standard errors.

    Parameters
    ----------
    weights_all : List of lists. For every obs, positive weights are saved.
    pred_data : String. csv-file with data to make predictions for.
    y_dat : Numpy array.
    cl_dat : Numpy array.
    w_dat : Numpy array.
    v_dictin : Dict. Variables.
    c_dict : Dict. Parameters.
    w_ate: Weights of ATE estimation
    amgate_flag : Bool. Average marginal effect title. Default is False.

    Returns
    -------
    gate: Lists of Numpy arrays.
    gate_se: Lists of Numpy arrays.
    gate_diff: like gate but for the difference of gates or to ATE
    gate_se_diff: like gate_se but for the difference of gates or to ATE
    """
    gate_str = 'AMGATE' if amgate_flag else 'GATE'
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nComputing', gate_str)
    n_y, no_of_out = len(y_dat), len(v_dict['y_name'])
    if c_dict['smooth_gates']:
        v_dict, v_x_values, smooth_yes, z_name_smooth = addsmoothvars(
            pred_data, v_dict, v_x_values, c_dict)
    else:
        smooth_yes = False
    d_p, z_p, w_p, _ = mcf_ate.get_data_for_final_estimation(
        pred_data, v_dict, c_dict, ate=False, need_count=False)
    z_type_l = [None] * len(v_dict['z_name'])
    z_values_l = z_type_l[:]
    z_smooth_l = [False] * len(v_dict['z_name'])
    gate = [None] * len(v_dict['z_name'])
    gate_se, gate_diff,  gate_se_diff = gate[:], gate[:], gate[:]
    if c_dict['d_type'] == 'continuous':
        continuous = True
        c_dict['atet_flag'] = c_dict['gatet_flag'] = False
        no_of_treat, d_values = c_dict['ct_grid_w'], c_dict['ct_grid_w_val']
        d_values_dr = c_dict['ct_d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
        treat_comp_label = [None] * round(no_of_treat_dr - 1)
    else:
        continuous = False
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
        treat_comp_label = [None] * round(no_of_treat * (no_of_treat - 1) / 2)
    no_of_comp = len(treat_comp_label)
    i_d_val = np.arange(no_of_treat)
    if not c_dict['w_yes']:
        w_dat = None
    ref_pop_lab = ['All']
    if c_dict['gatet_flag']:    # Always False for continuous treatments
        for lab in d_values:
            ref_pop_lab += str(lab)
    for zj_idx, z_name in enumerate(v_dict['z_name']):
        z_type_l[zj_idx] = v_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = v_x_values[z_name]
        if smooth_yes:
            z_smooth_l[zj_idx] = z_name in z_name_smooth
    if (d_p is not None) and c_dict['gatet_flag']:
        no_of_tgates = no_of_treat + 1  # Compute GATEs, GATET, ...
    else:
        c_dict['gatet_flag'], no_of_tgates = 0, 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = c_dict['choice_based_probs']
    if c_dict['gates_minus_previous']:
        effect_type_label = (gate_str, gate_str + '(change)')
    else:
        effect_type_label = (gate_str, gate_str + ' - ATE')
    jdx = 0
    for t1_idx, t1_lab in enumerate(d_values):
        for t2_idx in range(t1_idx+1, no_of_treat):
            treat_comp_label[jdx] = str(d_values[t2_idx]) + 'vs' + str(t1_lab)
            jdx += 1
        if continuous:
            break
    if c_dict['gates_minus_previous']:
        w_ate = None
    else:
        w_ate_sum = np.sum(w_ate, axis=2)
        for a_idx in range(no_of_tgates):  # Weights for ATE are normalized
            for t_idx in range(no_of_treat):
                if not ((1-1e-10) < w_ate_sum[a_idx, t_idx] < (1+1e-10)):
                    w_ate[a_idx, t_idx, :] = (w_ate[a_idx, t_idx, :]
                                              / w_ate_sum[a_idx, t_idx])
    files_to_delete = set()
    save_w_file = None
    if c_dict['no_parallel'] > 1 and c_dict['_ray_or_dask'] != 'ray':
        memory_weights = gp_sys.total_size(weights_all)
        if c_dict['weight_as_sparse']:
            for d_idx in range(no_of_treat):
                memory_weights += (weights_all[d_idx].data.nbytes
                                   + weights_all[d_idx].indices.nbytes
                                   + weights_all[d_idx].indptr.nbytes)
        if memory_weights > 2e+9:  # Two Gigabytes (2e+9)
            if c_dict['with_output'] and c_dict['verbose']:
                print('Weights need ', memory_weights/1e+9, 'GB RAM',
                      '==> Weights are passed as file to MP processes')
            save_w_file = 'w_all.pickle'
            gp_sys.save_load(save_w_file, weights_all, save=True,
                             output=c_dict['with_output'])
            files_to_delete.add(save_w_file)
            weights_all2 = None
        else:
            weights_all2 = weights_all
    else:
        weights_all2 = weights_all
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share']/2)
        else:
            maxworkers = c_dict['no_parallel']
        if weights_all2 is None:
            maxworkers = round(maxworkers / 2)
        if not maxworkers > 0:
            maxworkers = 1
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers, flush=True)
    if c_dict['_ray_or_dask'] == 'ray':
        if c_dict['mem_object_store_3'] is None:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False)
        else:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False,
                         object_store_memory=c_dict['mem_object_store_3'])
            if c_dict['with_output'] and c_dict['verbose']:
                print("Size of Ray Object Store: ",
                      round(c_dict['mem_object_store_3']/(1024*1024)), " MB")
        weights_all_ref = ray.put(weights_all)
    for z_name_j, z_name in enumerate(v_dict['z_name']):
        if c_dict['with_output'] and c_dict['verbose']:
            print(z_name_j+1, '(', len(v_dict['z_name']), ')', z_name,
                  flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        if z_smooth:
            kernel = 1  # Epanechikov
            bandw_z = gp_est.bandwidth_nw_rule_of_thumb(z_p[:, z_name_j])
            bandw_z = bandw_z * c_dict['sgates_bandwidth']
        else:
            kernel = bandw_z = None
        no_of_zval = len(z_values)
        gate_z = np.empty((no_of_zval, no_of_out, no_of_tgates, no_of_comp))
        gate_z_se, gate_z_mate = np.empty_like(gate_z), np.empty_like(gate_z)
        gate_z_mate_se = np.empty_like(gate_z)
        w_gate = np.zeros((no_of_zval, no_of_tgates, no_of_treat, n_y))
        w_gate_unc = np.zeros_like(w_gate)
        w_censored = np.zeros((no_of_zval, no_of_tgates, no_of_treat))
        w_gate0_dim = (no_of_treat, n_y)
        pot_y = np.empty((no_of_zval, no_of_tgates, no_of_treat_dr, no_of_out))
        pot_y_var = np.empty_like(pot_y)
        pot_y_mate, pot_y_mate_var = np.empty_like(pot_y), np.empty_like(pot_y)
        if (maxworkers == 1) or c_dict['gates_minus_previous']:
            for zj_idx in range(no_of_zval):
                if c_dict['gates_minus_previous']:
                    if zj_idx > 0:
                        w_ate = w_gate_unc[zj_idx-1, :, :, :]
                    else:
                        w_ate = w_gate_unc[zj_idx, :, :, :]
                results_fut_zj = gate_zj(
                    z_values[zj_idx], zj_idx, y_dat, cl_dat, w_dat, z_p, d_p,
                    w_p, z_name_j, weights_all, w_gate0_dim,
                    w_gate[zj_idx, :, :, :], w_gate_unc[zj_idx, :, :, :],
                    w_censored[zj_idx, :, :], w_ate, pot_y[zj_idx, :, :, :],
                    pot_y_var[zj_idx, :, :, :], pot_y_mate[zj_idx, :, :, :],
                    pot_y_mate_var[zj_idx, :, :, :], i_d_val, t_probs,
                    no_of_tgates, no_of_out, c_dict, bandw_z, kernel, z_smooth,
                    continuous)
                pot_y, pot_y_var, pot_y_mate, pot_y_mate_var = assign_pot(
                     pot_y, pot_y_var, pot_y_mate, pot_y_mate_var,
                     results_fut_zj, zj_idx)
                w_gate, w_gate_unc, w_censored = assign_w(
                     w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx)
        else:
            if c_dict['_ray_or_dask'] == 'ray':
                still_running = [ray_gate_zj_mp.remote(
                         z_values[zj_idx], zj_idx, y_dat, cl_dat,
                         w_dat, z_p, d_p, w_p, z_name_j, weights_all_ref,
                         w_gate0_dim, w_ate, i_d_val, t_probs, no_of_tgates,
                         no_of_out, c_dict, n_y, bandw_z, kernel, save_w_file,
                         z_smooth, continuous)
                    for zj_idx in range(no_of_zval)]
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        (pot_y, pot_y_var, pot_y_mate, pot_y_mate_var
                         ) = assign_pot(
                             pot_y, pot_y_var, pot_y_mate, pot_y_mate_var,
                             results_fut_idx, results_fut_idx[6])
                        w_gate, w_gate_unc, w_censored = assign_w(
                            w_gate, w_gate_unc, w_censored, results_fut_idx,
                            results_fut_idx[6])
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as clt:
                    y_dat_ref = clt.scatter(y_dat)
                    weights_all2_ref = clt.scatter(weights_all2)
                    ret_fut = [clt.submit(
                        gate_zj_mp, z_values[zj_idx], zj_idx, y_dat_ref,
                        cl_dat, w_dat, z_p, d_p, w_p, z_name_j,
                        weights_all2_ref, w_gate0_dim, w_ate, i_d_val, t_probs,
                        no_of_tgates, no_of_out, c_dict, n_y, bandw_z, kernel,
                        save_w_file, z_smooth, continuous)
                        for zj_idx in range(no_of_zval)]
                    jdx = 0
                    for _, res in as_completed(ret_fut, with_results=True):
                        zjj = res[6]
                        (pot_y, pot_y_var, pot_y_mate, pot_y_mate_var
                         ) = assign_pot(pot_y, pot_y_var, pot_y_mate,
                                        pot_y_mate_var, res, zjj)
                        if res[8] is not None:
                            w_gate[zjj, :, :, :] = np.load(res[8])
                            w_gate_unc[zjj, :, :, :] = np.load(res[9])
                            files_to_delete.add(res[8])
                            files_to_delete.add(res[9])
                        else:
                            w_gate[zjj, :, :, :] = res[4]
                            w_gate_unc[zjj, :, :, :] = res[5]
                        w_censored[zjj, :, :] = res[7]
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        gate_zj_mp, z_values[zj_idx], zj_idx, y_dat, cl_dat,
                        w_dat, z_p, d_p, w_p, z_name_j, weights_all2,
                        w_gate0_dim, w_ate, i_d_val, t_probs, no_of_tgates,
                        no_of_out, c_dict, n_y, bandw_z, kernel, save_w_file,
                        z_smooth, continuous):
                            zj_idx for zj_idx in range(no_of_zval)}
                    for frv in futures.as_completed(ret_fut):
                        results_fut_idx = frv.result()
                        del ret_fut[frv]                  # Saves memory
                        zjj = results_fut_idx[6]
                        (pot_y, pot_y_var, pot_y_mate, pot_y_mate_var
                         ) = assign_pot(pot_y, pot_y_var, pot_y_mate,
                                        pot_y_mate_var, results_fut_idx, zjj)
                        if results_fut_idx[8] is not None:
                            w_gate[zjj, :, :, :] = np.load(results_fut_idx[8])
                            w_gate_unc[zjj, :, :, :] = np.load(
                                results_fut_idx[9])
                            files_to_delete.add(results_fut_idx[8])
                            files_to_delete.add(results_fut_idx[9])
                        else:
                            w_gate[zjj, :, :, :] = results_fut_idx[4]
                            w_gate_unc[zjj, :, :, :] = results_fut_idx[5]
                        w_censored[zjj, :, :] = results_fut_idx[7]
        if c_dict['with_output']:
            # Describe weights
            for a_idx in range(no_of_tgates):
                w_st = np.zeros((6, no_of_treat))
                share_largest_q = np.zeros((no_of_treat, 3))
                sum_larger = np.zeros((no_of_treat, len(c_dict['q_w'])))
                obs_larger = np.zeros_like(sum_larger)
                w_censored_all = np.zeros(no_of_treat)
                for zj_idx in range(no_of_zval):
                    ret = mcf_ate.analyse_weights_ate(
                        w_gate[zj_idx, a_idx, :, :], None, c_dict, ate=False,
                        continuous=continuous, no_of_treat_cont=no_of_treat,
                        d_values_cont=d_values)
                    for idx in range(6):
                        w_st[idx] += ret[idx] / no_of_zval
                    share_largest_q += ret[6] / no_of_zval
                    sum_larger += ret[7] / no_of_zval
                    obs_larger += ret[8] / no_of_zval
                    w_censored_all += w_censored[zj_idx, a_idx, :]
                if not amgate_flag:
                    print('\n')
                    print('=' * 80)
                    print('Analysis of weights (normalised to add to 1): ',
                          gate_str, 'for ', z_name,
                          f'(stats are averaged over {no_of_zval:<4} groups).')
                    if c_dict['gatet_flag']:
                        print(f'\nTarget population: {ref_pop_lab[a_idx]:<4}')
                    mcf_ate.print_weight_stat(
                        w_st[0], w_st[1], w_st[2], w_st[3], w_st[4], w_st[5],
                        share_largest_q, sum_larger, obs_larger, c_dict,
                        w_censored_all, continuous=continuous,
                        d_values_cont=d_values)  # Discretized weights if cont
            print('\n')
        for o_idx in range(no_of_out):
            if c_dict['with_output']:
                print_str = ('-' * 80 + '\nOutcome variable: '
                             + f'{v_dict["y_name"][o_idx]} \n' + '- ' * 40)
                print(print_str)
                gp.print_f(c_dict['outfilesummary'], print_str)
            for a_idx in range(no_of_tgates):
                if c_dict['with_output']:
                    print_str = f'Reference population: {ref_pop_lab[a_idx]}'
                    print_str += '\n' + '- ' * 40
                    if not continuous:
                        print_str += mcf_gateout.wald_test(
                            z_name, no_of_zval, w_gate, y_dat, w_dat, cl_dat,
                            a_idx, o_idx, w_ate, c_dict, gate_str, no_of_treat,
                            d_values, print_output=False)
                ret_gate = [None] * no_of_zval
                ret_gate_mate = [None] * no_of_zval
                for zj_idx, _ in enumerate(z_values):
                    ret = mcf_gp.effect_from_potential(
                        pot_y[zj_idx, a_idx, :, o_idx].reshape(-1),
                        pot_y_var[zj_idx, a_idx, :, o_idx].reshape(-1),
                        d_values_dr, continuous=continuous)
                    ret_gate[zj_idx] = np.array(ret, dtype=object, copy=True)
                    gate_z[zj_idx, o_idx, a_idx, :] = ret[0]
                    gate_z_se[zj_idx, o_idx, a_idx, :] = ret[1]
                    ret = mcf_gp.effect_from_potential(
                        pot_y_mate[zj_idx, a_idx, :, o_idx].reshape(-1),
                        pot_y_mate_var[zj_idx, a_idx, :, o_idx].reshape(
                            -1), d_values_dr, continuous=continuous)
                    gate_z_mate[zj_idx, o_idx, a_idx, :] = ret[0]
                    gate_z_mate_se[zj_idx, o_idx, a_idx, :] = ret[1]
                    ret_gate_mate[zj_idx] = np.array(ret, dtype=object,
                                                     copy=True)
                if c_dict['with_output']:
                    print_str += ('\nGroup Average Treatment effects '
                                  + f'({gate_str})' + '\n' + '- ' * 40)
                    print_str += (f'\nHeterogeneity: {z_name} Outcome: '
                                  + f'{v_dict["y_name"][o_idx]} Ref. pop.: '
                                  + f'{ref_pop_lab[a_idx]}' + '\n')
                    print_str += mcf_gp.print_effect_z(
                        ret_gate, ret_gate_mate, z_values, gate_str,
                        print_output=False,
                        gates_minus_previous=c_dict['gates_minus_previous'])
                    print_str += '\n' + gp_est.print_se_info(
                        c_dict['cluster_std'], c_dict['se_boot_gate'])
                    if not c_dict['gates_minus_previous']:
                        print_str += gp_est.print_minus_ate_info(
                            c_dict['w_yes'])
                    print(print_str)
                    gp.print_f(c_dict['outfilesummary'], print_str)
        if c_dict['with_output'] and not c_dict['gates_minus_previous']:
            primes = gp.primes_list()                 # figures
            for a_idx, a_lab in enumerate(ref_pop_lab):
                gatet_yes = not a_idx == 0
                for o_idx, o_lab in enumerate(v_dict['y_name']):
                    for t_idx, t_lab in enumerate(treat_comp_label):
                        for e_idx, e_lab in enumerate(effect_type_label):
                            if e_idx == 0:
                                effects = gate_z[:, o_idx, a_idx, t_idx]
                                ste = gate_z_se[:, o_idx, a_idx, t_idx]
                                ate_f = ate[o_idx, a_idx, t_idx]
                                ate_f_se = ate_se[o_idx, a_idx, t_idx]
                            else:
                                effects = gate_z_mate[:, o_idx, a_idx, t_idx]
                                ste = gate_z_mate_se[:, o_idx, a_idx, t_idx]
                                ate_f = 0
                                ate_f_se = None
                            z_values_f = v_x_values[z_name].copy()
                            if v_x_type[z_name] > 0:
                                for zjj, zjjlab in enumerate(z_values_f):
                                    for jdx, j_lab in enumerate(primes):
                                        if j_lab == zjjlab:
                                            z_values_f[zjj] = jdx
                            if not continuous:
                                mcf_gateout.make_gate_figures_discr(
                                    e_lab + ' ' + z_name + ' ' + a_lab +
                                    ' ' + o_lab + ' ' + t_lab, z_name,
                                    z_values_f, z_type_l, effects, ste,
                                    c_dict, ate_f, ate_f_se, amgate_flag,
                                    z_smooth, gatet_yes=gatet_yes)
                            if continuous and t_idx == len(treat_comp_label)-1:
                                if e_idx == 0:
                                    ate_f = ate[o_idx, a_idx, :]
                                    effects = gate_z[:, o_idx, a_idx, :]
                                else:
                                    ate_f = None
                                    effects = gate_z_mate[:, o_idx, a_idx, :]
                                mcf_gateout.make_gate_figures_cont(
                                    e_lab + ' ' + z_name + ' ' + a_lab +
                                    ' ' + o_lab, z_name, z_values_f,
                                    effects, c_dict, ate_f, amgate_flag,
                                    d_values=d_values_dr)
        if c_dict['with_output']:
            print('-' * 80)
        gate[z_name_j], gate_se[z_name_j] = gate_z, gate_z_se
        gate_diff[z_name_j] = gate_z_mate
        gate_se_diff[z_name_j] = gate_z_mate_se
    if c_dict['_ray_or_dask'] == 'ray':
        if 'refs' in c_dict['_mp_ray_del']:
            del weights_all_ref
        if 'rest' in c_dict['_mp_ray_del']:
            del finished_res, finished
        if c_dict['_mp_ray_shutdown']:
            ray.shutdown()
    if files_to_delete:  # delete temporary files
        for file in files_to_delete:
            os.remove(file)
    return gate, gate_se, gate_diff, gate_se_diff


def assign_pot(pot_y, pot_y_var, pot_y_mate, pot_y_mate_var, results_fut_zj,
               zj_idx):
    """Reduce repetetive code."""
    pot_y[zj_idx, :, :, :] = results_fut_zj[0]
    pot_y_var[zj_idx, :, :, :] = results_fut_zj[1]
    pot_y_mate[zj_idx, :, :, :] = results_fut_zj[2]
    pot_y_mate_var[zj_idx, :, :, :] = results_fut_zj[3]
    return pot_y, pot_y_var, pot_y_mate, pot_y_mate_var


def assign_w(w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx):
    """Reduce repetetive code."""
    w_gate[zj_idx, :, :, :] = results_fut_zj[4]
    w_gate_unc[zj_idx, :, :, :] = results_fut_zj[5]
    w_censored[zj_idx, :, :] = results_fut_zj[7]
    return w_gate, w_gate_unc, w_censored


@ray.remote
def ray_gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                   z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
                   no_of_tgates, no_of_out, c_dict, n_y, bandw_z, kernel,
                   save_w_file=None, smooth_it=False, continuous=False):
    """Make function compatible with Ray."""
    return gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                      z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val,
                      t_probs, no_of_tgates, no_of_out, c_dict, n_y, bandw_z,
                      kernel, save_w_file, smooth_it, continuous)


def gate_zj(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p, z_name_j,
            weights_all, w_gate0_dim, w_gate_zj, w_gate_unc_zj, w_censored_zj,
            w_ate, pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            i_d_val, t_probs, no_of_tgates, no_of_out, c_dict, bandw_z, kernel,
            smooth_it=False, continuous=False):
    """Compute Gates and their variances for MP."""
    if continuous:
        no_of_treat, d_values = c_dict['ct_grid_w'], c_dict['ct_grid_w_val']
        i_w01 = c_dict['ct_w_to_dr_int_w01']
        i_w10 = c_dict['ct_w_to_dr_int_w10']
        index_full = c_dict['ct_w_to_dr_index_full']
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
    weights, relevant_z,  w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=c_dict['weight_as_sparse'])
    if c_dict['gatet_flag']:
        d_p_z = d_p[relevant_z]
    if c_dict['w_yes']:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if c_dict['weight_as_sparse'] else len(weights)
    # Step 1: Aggregate weights
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if c_dict['weight_as_sparse']:
                weight_i = weights[t_idx].getrow(n_idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if c_dict['w_yes']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not 1-1e-10 < w_i_sum < 1+1e-10:
                w_i = w_i / w_i_sum
            if c_dict['w_yes']:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if c_dict['choice_based_yes']:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if c_dict['gatet_flag']:
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
                    w_censored_zj, w_gate_unc_zj, w_ate, c_dict)
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             c_dict['max_weight_share'])
                        ret = gp_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, c_dict,
                            weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y_zj[a_idx, ti_idx, o_idx] = ret[0]
                        pot_y_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if c_dict['with_output']:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = gp_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, c_dict,
                                norm=False, weights=w_dat,
                                bootstrap=c_dict['se_boot_gate'])
                            pot_y_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            pot_y_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = gp_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        c_dict, weights=w_dat,
                        bootstrap=c_dict['se_boot_gate'])
                    pot_y_zj[a_idx, t_idx, o_idx] = ret[0]
                    pot_y_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if c_dict['with_output']:
                        ret2 = gp_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat,
                            c_dict, norm=False, weights=w_dat,
                            bootstrap=c_dict['se_boot_gate'])
                        pot_y_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        pot_y_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    return (pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj)


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


def w_gate_func(a_idx, t_idx, sum_wgate, w_gate_zj, w_censored_zj,
                w_gate_unc_zj, w_ate, c_dict):
    """Compute weights for discrete case."""
    if (not 1-1e-10 < sum_wgate < 1+1e-10) and (sum_wgate > 1e-10):
        w_gate_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :] / sum_wgate
    w_gate_unc_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :]
    if c_dict['max_weight_share'] < 1:
        (w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx]
         ) = mcf_gp.bound_norm_weights(w_gate_zj[a_idx, t_idx, :],
                                       c_dict['max_weight_share'])
    if c_dict['with_output']:
        w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx, t_idx, :]
    else:
        w_diff = None
    return w_gate_zj, w_diff, w_censored_zj, w_gate_unc_zj


def w_diff_cont_func(t_idx, a_idx, no_of_treat, w_gate_cont, w_ate, w10, w01):
    """Compute weights for difference in continuous case."""
    w_ate_cont = w_ate[a_idx, t_idx, :] if t_idx == no_of_treat - 1 else (
        w10 * w_ate[a_idx, t_idx, :] + w01 * w_ate[a_idx, t_idx+1, :])
    w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
    w_diff_cont = w_gate_cont - w_ate_cont
    return w_diff_cont


def gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
               z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
               no_of_tgates, no_of_out, c_dict, n_y, bandw_z, kernel,
               save_w_file=None, smooth_it=False, continuous=False):
    """Compute Gates and their variances for MP."""
    if continuous:
        no_of_treat, d_values = c_dict['ct_grid_w'], c_dict['ct_grid_w_val']
        d_values_dr = c_dict['ct_d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
        i_w01 = c_dict['ct_w_to_dr_int_w01']
        i_w10 = c_dict['ct_w_to_dr_int_w10']
        index_full = c_dict['ct_w_to_dr_index_full']
    else:
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
    if save_w_file is not None:
        weights_all = gp_sys.save_load(save_w_file, save=False,
                                       output=c_dict['with_output'])
    w_gate_zj = np.zeros((no_of_tgates, no_of_treat, n_y))
    w_gate_unc_zj = np.zeros_like(w_gate_zj)
    w_censored_zj = np.zeros((no_of_tgates, no_of_treat))
    pot_y_zj = np.empty((no_of_tgates, no_of_treat_dr, no_of_out))
    pot_y_var_zj = np.empty_like(pot_y_zj)
    pot_y_mate_zj = np.empty_like(pot_y_zj)
    pot_y_mate_var_zj = np.empty_like(pot_y_zj)
    # Step 1: Aggregate weights
    weights, relevant_z, w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=c_dict['weight_as_sparse'])
    if c_dict['gatet_flag']:
        d_p_z = d_p[relevant_z]
    if c_dict['w_yes']:
        w_p_z = w_p[relevant_z]
    n_x = weights[0].shape[0] if c_dict['weight_as_sparse'] else len(weights)
    for n_idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_idx, _ in enumerate(d_values):
            if c_dict['weight_as_sparse']:
                weight_i = weights[t_idx].getrow(n_idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                w_index = weights[n_idx][t_idx][0].copy()  # Ind weights>0
                w_i = weights[n_idx][t_idx][1].copy()
            if c_dict['w_yes']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not (1-1e-10) < w_i_sum < (1+1e-10):
                w_i = w_i / w_i_sum
            if c_dict['w_yes']:
                w_i = w_i * w_p_z[n_idx]
            if smooth_it:
                w_i = w_i * w_z_val[n_idx]
            if c_dict['choice_based_yes']:
                i_pos = i_d_val[d_p[n_idx] == d_values]
                w_gadd[t_idx, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_idx, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if c_dict['gatet_flag']:
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
                    w_censored_zj, w_gate_unc_zj, w_ate, c_dict)
            for o_idx in range(no_of_out):
                if continuous:
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        (w_gate_cont, w_gate_cont_unc, w_gate_unc_zj,
                         w_censored_zj) = w_gate_cont_funct(
                             t_idx, a_idx, no_of_treat, w_gate_zj, w10, w01, i,
                             w_gate_unc_zj, w_censored_zj,
                             c_dict['max_weight_share'])
                        ret = gp_est.weight_var(
                            w_gate_cont, y_dat[:, o_idx], cl_dat, c_dict,
                            weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y_zj[a_idx, ti_idx, o_idx] = ret[0]
                        pot_y_var_zj[a_idx, ti_idx, o_idx] = ret[1]
                        if c_dict['with_output']:
                            w_diff_cont = w_diff_cont_func(
                                t_idx, a_idx, no_of_treat, w_gate_cont_unc,
                                w_ate, w10, w01)
                            ret2 = gp_est.weight_var(
                                w_diff_cont, y_dat[:, o_idx], cl_dat, c_dict,
                                norm=False, weights=w_dat,
                                bootstrap=c_dict['se_boot_gate'])
                            pot_y_mate_zj[a_idx, ti_idx, o_idx] = ret2[0]
                            pot_y_mate_var_zj[a_idx, ti_idx, o_idx] = ret2[1]
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = gp_est.weight_var(
                        w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        c_dict, weights=w_dat,
                        bootstrap=c_dict['se_boot_gate'])
                    pot_y_zj[a_idx, t_idx, o_idx] = ret[0]
                    pot_y_var_zj[a_idx, t_idx, o_idx] = ret[1]
                    if c_dict['with_output']:
                        ret2 = gp_est.weight_var(
                            w_diff, y_dat[:, o_idx], cl_dat, c_dict,
                            norm=False, weights=w_dat,
                            bootstrap=c_dict['se_boot_gate'])
                        pot_y_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                        pot_y_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    if w_gate_zj.nbytes > 1e+9 and c_dict['_ray_or_dask'] != 'ray':
        # otherwise tuple gets too large for MP
        save_name_w = 'wtemp' + str(zj_idx) + '.npy'
        save_name_wunc = 'wunctemp' + str(zj_idx) + '.npy'
        np.save(save_name_w, w_gate_zj, fix_imports=False)
        np.save(save_name_wunc, w_gate_unc_zj, fix_imports=False)
        w_gate_zj = w_gate_unc_zj = None
    else:
        save_name_w = save_name_wunc = None
    return (pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj, save_name_w,
            save_name_wunc)


def addsmoothvars(in_csv_file, v_dict, v_x_values, c_dict):
    """
    Find variables for which to smooth gates and evaluation points.

    Parameters
    ----------
    in_csv_file: Str. Data file.
    v_dict : Dict. Variables.
    v_x_values : Dict. Variables
    c_dict : Dict. Controls.

    Returns
    -------
    v_dict_new : Dict. Updated variables.
    v_x_values_new : Dict. Updated with evaluation points.
    smooth_yes : Bool. Indicator if smoothing will happen.

    """
    smooth_yes = False
    z_name = v_dict['z_name']
    z_name_add = [name[:-4] for name in z_name if (name.endswith('CATV')
                                                   and (len(name) > 4))]
    if z_name_add:
        smooth_yes = True
        v_dict_new = copy.deepcopy(v_dict)
        v_x_values_new = copy.deepcopy(v_x_values)
        data_df = pd.read_csv(in_csv_file)
        data_np = data_df[z_name_add].to_numpy()
        for idx, name in enumerate(z_name_add):
            v_x_values_new[name] = smooth_gate_eva_values(
                data_np[:, idx], c_dict['sgates_no_evaluation_points'])
            v_dict_new['z_name'].append(name)
    else:
        v_dict_new = v_dict
        v_x_values_new = v_x_values
    return v_dict_new, v_x_values_new, smooth_yes, z_name_add


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
        eva_values = unique_vals
    else:
        quas = np.linspace(0.01, 0.99, no_eva_values)
        eva_values = np.unique(np.quantile(z_dat, quas))
    return eva_values.tolist()


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
        w_z_val = gp_est.kernel_proc((z_dat - z_val) / bandwidth, kernel)
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
        weights = list(itertools.compress(weights_all, relevant_data_points))
    return weights, relevant_data_points, w_z_val
