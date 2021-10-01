"""
Procedures needed for GATEs estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import os
from concurrent import futures
import itertools
import numpy as np
import pandas as pd
import scipy.stats as sct
import matplotlib.pyplot as plt
import ray
from mcf import mcf_weight_functions as mcf_w
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_hf
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_system_files as gp_sys
from mcf import general_purpose_mcf as gp_mcf


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
    regrf: Boolean. False if MCF (default).

    Returns
    -------
    None.

    """
    any_plots_mgate = False
    any_plots_amgate = False
    c_dict_mgate = copy.deepcopy(c_dict)
    c_dict_mgate['with_output'] = False       # reduce unnecessary infos
    if v_dict['z_name_mgate'] and c_dict['with_output']:
        any_plots_mgate = mgate_function(
            forest, fill_y_sample, pred_sample, v_dict, c_dict, x_name_mcf,
            var_x_type, var_x_values, regrf, c_dict_mgate, w_ate)
        if not any_plots_mgate:
            if regrf:
                print("No variables for marginal plots left.")
            else:
                print("No variables for MGATE left.")
        else:
            print('\n')
    if v_dict['z_name_amgate'] and c_dict['with_output']:
        any_plots_amgate = amgate_function(
            forest, fill_y_sample, pred_sample, v_dict, c_dict, x_name_mcf,
            var_x_type, var_x_values, c_dict_mgate)
        if not any_plots_amgate:
            print("No variables for AMGATE left.")
        else:
            print('\n')


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
        c_dict_mgate['gatet_flag'] = False
        c_dict_mgate['atet_flag'] = False
        if c_dict['with_output']:
            print('No treatment specific effects for MGATE and AMGATE.')
    any_plots_done = False
    if c_dict['with_output']:
        print()
        print('Marginale GATEs averaged over sample (AMGATEs)', '\n')
    _, eva_values = ref_vals_margplot(
        pred_sample, var_x_type, var_x_values,
        with_output=c_dict['with_output'], ref_values_needed=False,
        no_eva_values=c_dict['gmate_no_evaluation_points'])
    if c_dict['with_output'] and c_dict['verbose']:
        print()
        print('Variable under investigation: ', end=' ')
    z_name_old = v_dict['z_name']
    for vname in v_dict['z_name_amgate']:
        if vname in x_name_mcf:
            if c_dict['with_output'] and c_dict['verbose']:
                print(vname, end=' ')
            any_plots_done = True
            new_predict_file, z_values = ref_file_marg_plot_amgate(
                pred_sample, vname, c_dict, eva_values)
            v_dict['z_name'] = [vname]
            var_x_values[vname] = z_values[:]
            weights, y_f, _, cl_f, w_f = mcf_w.get_weights_mp(
                forest, new_predict_file, fill_y_sample, v_dict,
                c_dict_mgate, x_name_mcf, regrf=False)
            w_ate_iate, _, _, ate_z, ate_se_z, _ = mcf_ate.ate_est(
                    weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                    c_dict_mgate)
            c_dict_mgate['with_output'] = c_dict['with_output']
            gate_est(weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                     c_dict_mgate, var_x_type, var_x_values, w_ate_iate, ate_z,
                     ate_se_z, amgate_flag=True)
            os.remove(new_predict_file)  # Delete new file
    v_dict['z_name'] = z_name_old
    return any_plots_done


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
    any_plots_done = False
    if c_dict['with_output'] and c_dict['verbose']:
        print()
        if regrf:
            print('Marginal variable predictive plots')
        else:
            print('Marginale GATEs evaluated at median (MGATES)', '\n')
        if c_dict_mgate['choice_based_yes']:
            print('Choice based sampling deactivated for MGATES.')
    ref_values, eva_values = ref_vals_margplot(
        pred_sample, var_x_type, var_x_values,
        with_output=c_dict['with_output'], ref_values_needed=True,
        no_eva_values=c_dict['gmate_no_evaluation_points'])
    if c_dict['with_output'] and c_dict['verbose']:
        print()
        print('Variable under investigation: ', end=' ')
    w_yes_old = c_dict_mgate['w_yes']
    c_dict_mgate['w_yes'] = False   # Weighting not needed here
    with_output_old = c_dict_mgate['with_output']
    c_dict_mgate['with_output'] = False
    choice_based_yes_old = c_dict_mgate['choice_based_yes']
    c_dict_mgate['choice_based_yes'] = False
    for vname in v_dict['z_name_mgate']:
        if vname in x_name_mcf:
            if c_dict['with_output'] and c_dict['verbose']:
                print(vname, end=' ')
            any_plots_done = True
            new_predict_file = ref_file_marg_plot(
                vname, c_dict_mgate, var_x_type, ref_values, eva_values)
            weights, y_f, _, cl_f, w_f = mcf_w.get_weights_mp(
                forest, new_predict_file, fill_y_sample, v_dict,
                c_dict_mgate, x_name_mcf, regrf=regrf)
            if regrf:
                _, y_pred, y_pred_se, name_pred, _ = mcf_hf.predict_hf(
                    weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                    c_dict_mgate)
            else:
                w_ate_iate, _, _, _, _, _ = mcf_ate.ate_est(
                    weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                    c_dict_mgate, w_ate_only=True)
                _, _, _, iate, iate_se, namesiate = mcf_iate.iate_est_mp(
                    weights, new_predict_file, y_f, cl_f, w_f, v_dict,
                    c_dict_mgate, save_predictions=False, w_ate=w_ate_iate)
                names_iate = namesiate[0]
                name_pred = names_iate['names_iate']
                shape = np.shape(iate[:, :, :, 0])
                y_pred = iate[:, :, :, 0].reshape(shape[0], shape[1]*shape[2])
                y_pred_se = iate_se[:, :, :, 0].reshape(shape[0],
                                                        shape[1]*shape[2])
                if w_ate is not None:
                    names_iate_mate = namesiate[0]
                    name_mate_pred = names_iate_mate['names_iate_mate']
                    y_pred_mate = iate[:, :, :, 1].reshape(
                        shape[0], shape[1]*shape[2])
                    y_pred_mate_se = iate_se[:, :, :, 1].reshape(
                        shape[0], shape[1]*shape[2])
            if c_dict['with_output']:
                plot_marginal(
                    y_pred, y_pred_se, name_pred, vname, eva_values[vname],
                    var_x_type[vname], c_dict, regrf)
            if not regrf and (w_ate is not None):
                plot_marginal(
                    y_pred_mate, y_pred_mate_se, name_mate_pred, vname,
                    eva_values[vname], var_x_type[vname], c_dict, regrf,
                    minus_ate=True)
    if not regrf and (w_ate is not None):
        if c_dict['with_output'] and c_dict['verbose']:
            print()
            print('MGATEs minus ATE are evaluated at fixed feature values',
                  '(equally weighted).')
    c_dict_mgate['w_yes'] = w_yes_old
    c_dict_mgate['with_output'] = with_output_old
    c_dict_mgate['choice_based_yes'] = choice_based_yes_old
    return any_plots_done


def plot_marginal(pred, pred_se, names_pred, x_name, x_values_in, x_type,
                  c_dict, regrf, minus_ate=False):
    """Show the plots, similar to GATE and IATE.

    Parameters
    ----------
    pred : Numpy array. Predictions.
    pred_se : Numpy array. Standard errors.
    names_pred : List of strings.
    v_name : Str.
    x_values : List of float or int. Values to evaluate the predictions.
    x_type : Type of variable.
    c_dict : Dict. Parameters.
    regrf : Boolean. Regression forest.

    Returns
    -------
    None.

    """
    conf_int = sct.norm.ppf(c_dict['fig_ci_level'] +
                            0.5 * (1 - c_dict['fig_ci_level']))
    if x_type > 0:  # categorical variables
        x_values = gp.primeposition(x_values_in, start_with_1=True)
    else:
        x_values = x_values_in
    for idx, _ in enumerate(names_pred):
        if regrf:
            titel = 'Marginal predictive plot ' + x_name + names_pred[idx]
        else:
            if minus_ate:
                titel = 'MGATEMATE ' + x_name + names_pred[idx][:-9]
            else:
                titel = 'MGATE ' + x_name + names_pred[idx][:-5]
        pred_temp = pred[:, idx]
        pred_se_temp = pred_se[:, idx]
        if x_type == 0 and len(x_values) > c_dict['no_filled_plot']:
            pred_temp = gp_est.moving_avg_mean_var(pred_temp, 3, False)[0]
            pred_se_temp = gp_est.moving_avg_mean_var(
                pred_se_temp, 3, False)[0]
        file_titel = titel.replace(" ", "")
        file_name_jpeg = c_dict['fig_pfad_jpeg'] + '/' + file_titel + '.jpeg'
        file_name_pdf = c_dict['fig_pfad_pdf'] + '/' + file_titel + '.pdf'
        file_name_csv = c_dict['fig_pfad_csv'] + '/' + file_titel + '.csv'
        upper = pred_temp + pred_se_temp * conf_int
        lower = pred_temp - pred_se_temp * conf_int
        fig, axs = plt.subplots()
        label_u = 'Upper ' + str(round(c_dict['fig_ci_level'] * 100)) + '%-CI'
        label_l = 'Lower ' + str(round(c_dict['fig_ci_level'] * 100)) + '%-CI'
        label_ci = str(c_dict['fig_ci_level'] * 100) + '%-CI'
        if regrf:
            label_m = 'Conditional prediction'
        else:
            if minus_ate:
                label_m = 'MGATE-ATE'
                label_0 = '_nolegend_'
                line_0 = '_-k'
                zeros = np.zeros_like(pred_temp)
            else:
                label_m = 'MGATE'
        if x_type == 0 and len(x_values) > c_dict['no_filled_plot']:
            axs.plot(x_values, pred_temp, label=label_m, color='b')
            axs.fill_between(x_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
        else:
            u_line = 'v'
            l_line = '^'
            middle = 'o'
            connect_y = np.empty(2)
            connect_x = np.empty(2)
            for ind, i_lab in enumerate(x_values):
                connect_y[0] = upper[ind].copy()
                connect_y[1] = lower[ind].copy()
                connect_x[0] = i_lab
                connect_x[1] = i_lab
                axs.plot(connect_x, connect_y, 'b-', linewidth=0.7)
            axs.plot(x_values, pred_temp, middle +'b', label=label_m)
            axs.plot(x_values, upper, u_line + 'b', label=label_u)
            axs.plot(x_values, lower, l_line + 'b', label=label_l)
            if minus_ate:
                axs.plot(x_values, zeros, line_0, label=label_0)
        if c_dict['with_output']:
            print_mgate(pred_temp, pred_se_temp, titel, x_values)
        axs.set_ylabel(label_m)
        axs.legend(loc='lower right', shadow=True,
                   fontsize=c_dict['fig_fontsize'])
        axs.set_title(titel)
        axs.set_xlabel('Values of ' + x_name)
        gp.delete_file_if_exists(file_name_jpeg)
        gp.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
        fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
        if c_dict['show_plots']:
            plt.show()
        else:
            plt.close()
        upper = upper.reshape(-1, 1)
        lower = lower.reshape(-1, 1)
        pred_temp = pred_temp.reshape(-1, 1)
        x_values_np = np.array(x_values, copy=True).reshape(-1, 1)
        effects_et_al = np.concatenate((upper, pred_temp, lower, x_values_np),
                                        axis=1)
        cols = ['upper', 'effects', 'lower', 'x_values']
        datasave = pd.DataFrame(data=effects_et_al, columns=cols)
        gp.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)


def print_mgate(est, stderr, titel, z_values):
    """
    Print various GATEs (MGATE, AMGATE, ...).

    Parameters
    ----------
    est : 1-dim Numpy array. Effects.
    stderr : 1-dim Numpy array. Standard errors.
    titel : Str. Titel of Table
    z_values : List of floats/int. Values of z.

    Returns
    -------
    None.

    """
    t_val = np.abs(est / stderr)
    p_val = sct.t.sf(np.abs(t_val), 1000000) * 2
    if isinstance(z_values[0], (float, np.float32, np.float64)):
        z_values = np.around(z_values, 2)
        z_is_float = True
    else:
        z_is_float = False
    print()
    print('- ' * 40)
    print(titel)
    print('Value of Z       Est        SE    t-val   p-val')
    print('- ' * 40)
    for z_ind, z_val in enumerate(z_values):
        if z_is_float:
            print('{:>6.2f}        '.format(z_val), end=' ')
        else:
            print('{:>6.0f}        '.format(z_val), end=' ')
        print('{:>8.5f}  {:>8.5f}'.format(est[z_ind], stderr[z_ind]), end=' ')
        print('{:>5.2f}  {:>5.2f}%'.format(t_val[z_ind], p_val[z_ind]*100),
              end=' ')
        if p_val[z_ind] < 0.001:
            print('****')
        elif p_val[z_ind] < 0.01:
            print(' ***')
        elif p_val[z_ind] < 0.05:
            print('  **')
        elif p_val[z_ind] < 0.1:
            print('   *')
        else:
            print('    ')
    print('-' * 80)
    print('Values of Z may have been recoded into primes.')
    print('-' * 80)


def ref_file_marg_plot_amgate(in_csv_file, z_name, c_dict, eva_values):
    """
    Create reference samples for covariates (AMGATE).

    Revised procedure by Hugo, 5.1.2021. Should be faster using less memory.

    Parameters
    ----------
    in_csv_file : String. File for which predictions will be obtained
    z_name : String. Name of variable that is varying.
    c_dict : Dict. Parameters.
    eva_values : List. Values at which z is evaluated.

    Returns
    -------
    out_csv_file : String. csv-file with reference values.
    z_values : List of float/Int. List of values

    """
    eva_values = eva_values[z_name]
    data_df = pd.read_csv(in_csv_file)
    no_eval = len(eva_values)
    obs = len(data_df)
    if obs/no_eval > 10:  # Save computation time by using random samples
        share = c_dict['gmate_sample_share'] / no_eval
        if 0 < share < 1:
            seed = 9324561
            rng = np.random.default_rng(seed)
            idx = rng.choice(obs, int(np.floor(obs * share)), replace=False)
            obs = len(idx)
            if c_dict['with_output'] and c_dict['verbose']:
                print('{:5.2f}% '.format(share * 100),
                      'random sample drawn')
        else:
            idx = np.arange(obs)
    else:
        idx = np.arange(obs)
    new_idx_dataframe = list(itertools.chain.from_iterable(
        itertools.repeat(idx, no_eval)))
    data_all_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(itertools.chain.from_iterable(
        [[i] * obs for i in eva_values]))
    data_all_df.loc[:, z_name] = new_values_z
    out_csv_file = c_dict['temporary_file']
    if c_dict['with_output'] and c_dict['verbose']:
        print()
        print('AMGATEs minus ATE are evaluated at fixed z-feature values',
              '(equally weighted).')
    gp.delete_file_if_exists(out_csv_file)
    data_all_df.to_csv(out_csv_file, index=False)
    return out_csv_file, eva_values


def ref_file_marg_plot(plotx_name, c_dict, var_x_type, reference_values,
                       eva_values):
    """
    Create reference values for covariates (MAGE).

    Parameters
    ----------
    plotx_name : String. Name of variable that is varying.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    var_x_type : Dict. Name and type of variable.

    Returns
    -------
    out_csv_file : String. csv-file with reference values.

    """
    names = []
    eva_values = eva_values[plotx_name]
    no_rows = len(eva_values)
    x_vals = np.empty((no_rows, len(var_x_type.keys())))
    for c_idx, vname in enumerate(var_x_type.keys()):
        names.append(vname)
        for r_idx in range(no_rows):
            if vname == plotx_name:
                x_vals[r_idx, c_idx] = eva_values[r_idx]
            else:
                x_vals[r_idx, c_idx] = reference_values[vname]
    out_csv_file = c_dict['temporary_file']
    datasave = pd.DataFrame(data=x_vals, columns=names)
    datasave.to_csv(out_csv_file, index=False)
    return out_csv_file


def ref_vals_margplot(in_csv_file, var_x_type, var_x_values,
                      with_output=False, ref_values_needed=True,
                      no_eva_values=50):
    """Compute reference values for marginal plots.

    Parameters
    ----------
    in_csv_file : String. Data.
    var_x_type : Dict. Variable names and types (0,1,2)
    var_x_values: Dict. Variable names and values.
    with_output : Bool. Print output. Default is False.
    ref_values_needed : Bool. Compute reference values. Default is True.
    no_eva_values : Bool. Number of evaluation points (cont. var).
                          Default is 50.

    Returns
    -------
    reference_values : Dict. Variable names and reference values

    """
    if with_output and ref_values_needed:
        print('Effects are evaluated at these values for other features:')
        print("Variable                  Type     Reference value")
    reference_values = {}
    evaluation_values = {}
    data_df = pd.read_csv(in_csv_file)
    obs = len(data_df)
    for vname in var_x_type.keys():
        ddf = data_df[vname]
        if var_x_type[vname] == 0:
            if ref_values_needed:
                ref_val = ddf.median()
            type_str = '  ordered'
        else:
            type_str = 'unordered'
            if ref_values_needed:
                ref_val = ddf.mode()
                ref_val = ref_val.to_numpy()
                if np.size(ref_val) > 1:
                    ref_val = ref_val[0]
                ref_val = int(ref_val)
        if not var_x_values[vname]:
            no_eva_values = min(no_eva_values, obs)
            quas = np.linspace(0.01, 0.99, no_eva_values)
            eva_val = ddf.quantile(quas)
            eva_val = eva_val.to_list()
        else:
            eva_val = var_x_values[vname].copy()
        if ref_values_needed:
            reference_values.update({vname: ref_val})
        evaluation_values.update({vname: eva_val})
        if with_output and ref_values_needed:
            print('{:20}: '.format(vname), type_str, end=' ')
            if isinstance(ref_val, float):
                print(' {:8.4f}'.format(ref_val))
            else:
                print(' {:8.0f}'.format(float(ref_val)))
    return reference_values, evaluation_values


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

    """
    if amgate_flag:
        gate_str = 'AMGATE'
    else:
        gate_str = 'GATE'
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nComputing', gate_str)
    n_y = len(y_dat)
    no_of_out = len(v_dict['y_name'])
    if c_dict['smooth_gates']:
        v_dict, v_x_values, smooth_yes, z_name_smooth = addsmoothvars(
            pred_data, v_dict, v_x_values, c_dict)
    else:
        smooth_yes = False
    d_p, z_p, w_p, _ = mcf_ate.get_data_for_final_estimation(
        pred_data, v_dict, c_dict, ate=False, need_count=False)
    z_type_l = [None] * len(v_dict['z_name'])
    z_values_l = [None] * len(v_dict['z_name'])
    z_smooth_l = [False] * len(v_dict['z_name'])
    gate = [None] * len(v_dict['z_name'])
    gate_se = [None] * len(v_dict['z_name'])
    if not c_dict['w_yes']:
        w_dat = None
    if c_dict['gatet_flag']:
        ref_pop_lab = ['All']
        for lab in c_dict['d_values']:
            ref_pop_lab += str(lab)
    else:
        ref_pop_lab = ['All']
    for zj_idx, z_name in enumerate(v_dict['z_name']):
        z_type_l[zj_idx] = v_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = v_x_values[z_name]
        if smooth_yes:
            z_smooth_l[zj_idx] = z_name in z_name_smooth
    if (d_p is not None) and c_dict['gatet_flag']:
        no_of_tgates = c_dict['no_of_treat'] + 1  # Compute GATEs, GATET, ...
    else:
        c_dict['gatet_flag'] = 0
        no_of_tgates = 1
        ref_pop_lab = [ref_pop_lab[0]]
    t_probs = c_dict['choice_based_probs']
    i_d_val = np.arange(c_dict['no_of_treat'])
    treat_comp_label = [None] * round(c_dict['no_of_treat'] *
                                      (c_dict['no_of_treat']-1)/2)
    effect_type_label = (gate_str, gate_str + 'MATE')
    jdx = 0
    for t1_idx, t1_lab in enumerate(c_dict['d_values']):
        for t2_idx in range(t1_idx+1, c_dict['no_of_treat']):
            treat_comp_label[jdx] = str(
                c_dict['d_values'][t2_idx]) + 'vs' + str(t1_lab)
            jdx += 1
    w_ate_sum = np.sum(w_ate, axis=2)
    for a_idx in range(no_of_tgates):  # Weights for ATE are normalized
        for t_idx in range(c_dict['no_of_treat']):
            if not ((1-1e-10) < w_ate_sum[a_idx, t_idx] < (1+1e-10)):
                w_ate[a_idx, t_idx, :] = w_ate[a_idx, t_idx, :] / w_ate_sum[
                    a_idx, t_idx]
    files_to_delete = set()
    save_w_file = None
    if c_dict['no_parallel'] > 1 and not c_dict['mp_with_ray']:
        memory_weights = gp_sys.total_size(weights_all)
        if c_dict['weight_as_sparse']:
            for d_idx in range(c_dict['no_of_treat']):
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
            maxworkers = gp_mcf.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share']/2)
        else:
            maxworkers = c_dict['no_parallel']
        if weights_all2 is None:
            maxworkers = round(maxworkers / 2)
        if not maxworkers > 0:
            maxworkers = 1
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers, flush=True)
    if c_dict['mp_with_ray']:
        if c_dict['mem_object_store_3'] is None:
            ray.init(num_cpus=maxworkers, include_dashboard=False)
        else:
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
        z_values = z_values_l[z_name_j]
        z_smooth = z_smooth_l[z_name_j]
        if z_smooth:
            kernel = 1  # Epanechikov
            bandw_z = gp_est.bandwidth_nw_rule_of_thumb(z_p[:, z_name_j])
            bandw_z = bandw_z * c_dict['sgates_bandwidth']
        else:
            kernel = None
            bandw_z = None
        no_of_zval = len(z_values)
        dim_all = (no_of_zval, no_of_out, no_of_tgates, round(
                     c_dict['no_of_treat'] * (c_dict['no_of_treat']-1)/2))
        gate_z = np.empty(dim_all)
        gate_z_se = np.empty(dim_all)
        gate_z_mate = np.empty(dim_all)
        gate_z_mate_se = np.empty(dim_all)
        dim_all = (no_of_zval, no_of_tgates, c_dict['no_of_treat'], n_y)
        w_gate = np.zeros(dim_all)
        w_gate_unc = np.zeros(dim_all)
        w_censored = np.zeros((no_of_zval, no_of_tgates,
                               c_dict['no_of_treat']))
        w_gate0_dim = (c_dict['no_of_treat'], n_y)
        dim_all = (no_of_zval, no_of_tgates, c_dict['no_of_treat'], no_of_out)
        pot_y = np.empty(dim_all)
        pot_y_var = np.empty(dim_all)
        pot_y_mate = np.empty(dim_all)
        pot_y_mate_var = np.empty(dim_all)
        if maxworkers == 1:
            for zj_idx in range(no_of_zval):
                results_fut_zj = gate_zj(
                    z_values[zj_idx], zj_idx, y_dat, cl_dat, w_dat, z_p, d_p,
                    w_p, z_name_j, weights_all, w_gate0_dim,
                    w_gate[zj_idx, :, :, :], w_gate_unc[zj_idx, :, :, :],
                    w_censored[zj_idx, :, :], w_ate, pot_y[zj_idx, :, :, :],
                    pot_y_var[zj_idx, :, :, :], pot_y_mate[zj_idx, :, :, :],
                    pot_y_mate_var[zj_idx, :, :, :], i_d_val, t_probs,
                    no_of_tgates, no_of_out, c_dict, bandw_z, kernel, z_smooth)
                pot_y, pot_y_var, pot_y_mate, pot_y_mate_var = assign_pot(
                     pot_y, pot_y_var, pot_y_mate, pot_y_mate_var,
                     results_fut_zj, zj_idx)
                w_gate, w_gate_unc, w_censored = assign_w(
                     w_gate, w_gate_unc, w_censored, results_fut_zj, zj_idx)
        else:
            if c_dict['mp_with_ray']:
                # ray.init(num_cpus=maxworkers, include_dashboard=False,
                #          object_store_memory=c_dict['mem_object_store'])
                # weights_all_ref = ray.put(weights_all)
                tasks = [ray_gate_zj_mp.remote(
                         z_values[zj_idx], zj_idx, y_dat, cl_dat,
                         w_dat, z_p, d_p, w_p, z_name_j, weights_all_ref,
                         w_gate0_dim, w_ate, i_d_val, t_probs, no_of_tgates,
                         no_of_out, c_dict, n_y, bandw_z, kernel, save_w_file,
                         z_smooth) for zj_idx in range(no_of_zval)]
                still_running = list(tasks)
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
                del finished, still_running, tasks
                # del finished, still_running, tasks, weights_all_ref
                # ray.shutdown()
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        gate_zj_mp, z_values[zj_idx], zj_idx, y_dat, cl_dat,
                        w_dat, z_p, d_p, w_p, z_name_j, weights_all2,
                        w_gate0_dim, w_ate, i_d_val, t_probs, no_of_tgates,
                        no_of_out, c_dict, n_y, bandw_z, kernel, save_w_file,
                        z_smooth): zj_idx for zj_idx in range(no_of_zval)}
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
                w_st = np.zeros((6, c_dict['no_of_treat']))
                share_largest_q = np.zeros((c_dict['no_of_treat'], 3))
                dim_all = (c_dict['no_of_treat'], len(c_dict['q_w']))
                sum_larger = np.zeros(dim_all)
                obs_larger = np.zeros(dim_all)
                w_censored_all = np.zeros(c_dict['no_of_treat'])
                for zj_idx in range(no_of_zval):
                    ret = mcf_ate.analyse_weights_ate(
                        w_gate[zj_idx, a_idx, :, :], None, c_dict, False)
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
                          '(stats are averaged over {:<4}'.format(no_of_zval),
                          'groups).')
                    if c_dict['gatet_flag']:
                        print('\nTarget population: {:<4}'.format(
                            ref_pop_lab[a_idx]))
                    mcf_ate.print_weight_stat(
                        w_st[0], w_st[1], w_st[2], w_st[3], w_st[4], w_st[5],
                        share_largest_q, sum_larger, obs_larger, c_dict,
                        w_censored_all)
            print('\n')
        for o_idx in range(no_of_out):
            if c_dict['with_output']:
                print('-' * 80)
                print('Outcome variable: ', v_dict['y_name'][o_idx])
                print('-' * 80)
            for a_idx in range(no_of_tgates):
                if c_dict['with_output']:
                    print('Reference population:', ref_pop_lab[a_idx])
                    print('- ' * 40)
                    wald_test(z_name, no_of_zval, w_gate, y_dat, w_dat,
                              cl_dat, a_idx, o_idx, w_ate, c_dict, gate_str)
                ret_gate = [None] * no_of_zval
                ret_gate_mate = [None] * no_of_zval
                for zj_idx, _ in enumerate(z_values):
                    ret = gp_mcf.effect_from_potential(
                        pot_y[zj_idx, a_idx, :, o_idx].reshape(-1),
                        pot_y_var[zj_idx, a_idx, :, o_idx].reshape(-1),
                        c_dict['d_values'])
                    ret_gate[zj_idx] = np.array(ret, dtype=object, copy=True)
                    gate_z[zj_idx, o_idx, a_idx, :] = ret[0]
                    gate_z_se[zj_idx, o_idx, a_idx, :] = ret[1]
                    if c_dict['with_output']:
                        ret = gp_mcf.effect_from_potential(
                            pot_y_mate[zj_idx, a_idx, :, o_idx].reshape(-1),
                            pot_y_mate_var[zj_idx, a_idx, :, o_idx].reshape(
                                -1), c_dict['d_values'])
                        gate_z_mate[zj_idx, o_idx, a_idx, :] = ret[0]
                        gate_z_mate_se[zj_idx, o_idx, a_idx, :] = ret[1]
                        ret_gate_mate[zj_idx] = np.array(ret, dtype=object,
                                                         copy=True)
                if c_dict['with_output']:
                    print('- ' * 40)
                    print('Group Average Treatment effects (', gate_str, ')')
                    print('- ' * 40)
                    print('Heterogeneity: ', z_name, 'Outcome: ',
                          v_dict['y_name'][o_idx], 'Ref. pop.: ',
                          ref_pop_lab[a_idx])
                    gp_mcf.print_effect_z(ret_gate, ret_gate_mate, z_values,
                                          gate_str)
        if c_dict['with_output']:   # figures
            primes = gp.primes_list()
            for a_idx, a_lab in enumerate(ref_pop_lab):
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
                            make_gate_figures(
                                e_lab + z_name + a_lab + o_lab + t_lab, z_name,
                                z_values_f, z_type_l, effects, ste, c_dict,
                                ate_f, ate_f_se, amgate_flag, z_smooth)
        if c_dict['with_output']:
            print('-' * 80)
        gate[z_name_j] = gate_z
        gate_se[z_name_j] = gate_z_se
    if c_dict['mp_with_ray']:
        del weights_all_ref
        ray.shutdown()
    if files_to_delete:  # delete temporary files
        for file in files_to_delete:
            os.remove(file)
    return gate, gate_se


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
                   save_w_file=None, smooth_it=False):
    """Make function compatible with Ray."""
    return gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
                      z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val,
                      t_probs, no_of_tgates, no_of_out, c_dict, n_y, bandw_z,
                      kernel, save_w_file, smooth_it)


def gate_zj(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p, z_name_j,
            weights_all, w_gate0_dim, w_gate_zj, w_gate_unc_zj, w_censored_zj,
            w_ate, pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            i_d_val, t_probs, no_of_tgates, no_of_out, c_dict, bandw_z, kernel,
            smooth_it=False):
    """Compute Gates and their variances for MP."""
    # Step 1: Aggregate weights
    weights, relevant_z,  w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=c_dict['weight_as_sparse'])
    if c_dict['gatet_flag']:
        d_p_z = d_p[relevant_z]
    if c_dict['w_yes']:
        w_p_z = w_p[relevant_z]
    if c_dict['weight_as_sparse']:
        n_x = weights[0].shape[0]
    else:
        n_x = len(weights)
    # for idx, weights_i in enumerate(weights):
    for idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_ind, _ in enumerate(c_dict['d_values']):
            if c_dict['weight_as_sparse']:
                weight_i = weights[t_ind].getrow(idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                # w_index = weights_i[t_ind][0].copy()  # Ind weights>0
                # w_i = weights_i[t_ind][1].copy()
                w_index = weights[idx][t_ind][0].copy()  # Ind weights>0
                w_i = weights[idx][t_ind][1].copy()
            if c_dict['w_yes']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not 1-1e-10 < w_i_sum < 1+1e-10:
                w_i = w_i / w_i_sum
            if c_dict['w_yes']:
                w_i = w_i * w_p_z[idx]
            if smooth_it:
                w_i = w_i * w_z_val[idx]
            if c_dict['choice_based_yes']:
                i_pos = i_d_val[d_p[idx] == c_dict['d_values']]
                w_gadd[t_ind, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_ind, w_index] = w_i.copy()
        w_gate_zj[0, :, :] = w_gate_zj[0, :, :] + w_gadd
        if c_dict['gatet_flag']:
            t_pos_i = i_d_val[d_p_z[idx] == c_dict['d_values']]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd
    # Step 2: Get potential outcomes for particular z_value
    sum_wgate = np.sum(w_gate_zj, axis=2)
    for a_idx in range(no_of_tgates):
        for t_idx in range(c_dict['no_of_treat']):
            if not ((-1e-15 < sum_wgate[a_idx, t_idx] < 1e-15)
                    or (1-1e-10 < sum_wgate[a_idx, t_idx] < 1+1e-10)):
                w_gate_zj[a_idx, t_idx, :] = w_gate_zj[
                    a_idx, t_idx, :] / sum_wgate[a_idx, t_idx]
            w_gate_unc_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :]
            if c_dict['max_weight_share'] < 1:
                w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx] = (
                    gp_mcf.bound_norm_weights(w_gate_zj[a_idx, t_idx, :],
                                              c_dict['max_weight_share']))
            if c_dict['with_output']:
                w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx,
                                                                t_idx, :]
            for o_idx in range(no_of_out):
                ret = gp_est.weight_var(
                    w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                    c_dict, weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                pot_y_zj[a_idx, t_idx, o_idx] = ret[0]
                pot_y_var_zj[a_idx, t_idx, o_idx] = ret[1]
                if c_dict['with_output']:
                    ret2 = gp_est.weight_var(
                        w_diff, y_dat[:, o_idx], cl_dat, c_dict, norm=False,
                        weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                    pot_y_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                    pot_y_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    return (pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj)


def gate_zj_mp(z_val, zj_idx, y_dat, cl_dat, w_dat, z_p, d_p, w_p,
               z_name_j, weights_all, w_gate0_dim, w_ate, i_d_val, t_probs,
               no_of_tgates, no_of_out, c_dict, n_y, bandw_z, kernel,
               save_w_file=None, smooth_it=False):
    """Compute Gates and their variances for MP."""
    if save_w_file is not None:
        weights_all = gp_sys.save_load(save_w_file, save=False,
                                       output=c_dict['with_output'])
    w_gate_zj = np.zeros((no_of_tgates, c_dict['no_of_treat'], n_y))
    w_gate_unc_zj = np.zeros((no_of_tgates, c_dict['no_of_treat'], n_y))
    w_censored_zj = np.zeros((no_of_tgates, c_dict['no_of_treat']))
    dim_all = (no_of_tgates, c_dict['no_of_treat'], no_of_out)
    pot_y_zj = np.empty(dim_all)
    pot_y_var_zj = np.empty(dim_all)
    pot_y_mate_zj = np.empty(dim_all)
    pot_y_mate_var_zj = np.empty(dim_all)
    # Step 1: Aggregate weights
    weights, relevant_z, w_z_val = get_w_rel_z(
        z_p[:, z_name_j], z_val, weights_all, smooth_it, bandwidth=bandw_z,
        kernel=kernel, w_is_csr=c_dict['weight_as_sparse'])
    if c_dict['gatet_flag']:
        d_p_z = d_p[relevant_z]
    if c_dict['w_yes']:
        w_p_z = w_p[relevant_z]
    if c_dict['weight_as_sparse']:
        n_x = weights[0].shape[0]
    else:
        n_x = len(weights)
    # for idx, weights_i in enumerate(weights):
    for idx in range(n_x):
        w_gadd = np.zeros(w_gate0_dim)
        for t_ind, _ in enumerate(c_dict['d_values']):
            if c_dict['weight_as_sparse']:
                weight_i = weights[t_ind].getrow(idx)
                w_index = weight_i.indices
                w_i = weight_i.data.copy()
            else:
                # w_index = weights_i[t_ind][0].copy()  # Ind weights>0
                # w_i = weights_i[t_ind][1].copy()
                w_index = weights[idx][t_ind][0].copy()  # Ind weights>0
                w_i = weights[idx][t_ind][1].copy()
            if c_dict['w_yes']:
                w_i = w_i * w_dat[w_index].reshape(-1)
            w_i_sum = np.sum(w_i)
            if not (1-1e-10) < w_i_sum < (1+1e-10):
                w_i = w_i / w_i_sum
            if c_dict['w_yes']:
                w_i = w_i * w_p_z[idx]
            if smooth_it:
                w_i = w_i * w_z_val[idx]
            if c_dict['choice_based_yes']:
                i_pos = i_d_val[d_p[idx] == c_dict['d_values']]
                w_gadd[t_ind, w_index] = w_i * t_probs[int(i_pos)]
            else:
                w_gadd[t_ind, w_index] = w_i.copy()
        w_gate_zj[0, :, :] += w_gadd
        if c_dict['gatet_flag']:
            t_pos_i = i_d_val[d_p_z[idx] == c_dict['d_values']]
            w_gate_zj[t_pos_i+1, :, :] += w_gadd
    # Step 2: Get potential outcomes for particular z_value
    sum_wgate = np.sum(w_gate_zj, axis=2)
    for a_idx in range(no_of_tgates):
        for t_idx in range(c_dict['no_of_treat']):
            if (not (1-1e-10 < sum_wgate[a_idx, t_idx] < 1+1e-10)) and (
                    sum_wgate[a_idx, t_idx] > 1e-10):
                w_gate_zj[a_idx, t_idx, :] = w_gate_zj[
                    a_idx, t_idx, :] / sum_wgate[a_idx, t_idx]
            w_gate_unc_zj[a_idx, t_idx, :] = w_gate_zj[a_idx, t_idx, :]
            if c_dict['max_weight_share'] < 1:
                w_gate_zj[a_idx, t_idx, :], _, w_censored_zj[a_idx, t_idx] = (
                    gp_mcf.bound_norm_weights(w_gate_zj[a_idx, t_idx, :],
                                              c_dict['max_weight_share']))
            if c_dict['with_output']:
                w_diff = w_gate_unc_zj[a_idx, t_idx, :] - w_ate[a_idx,
                                                                t_idx, :]
                w_diff = w_gate_zj[a_idx, t_idx, :] - w_ate[a_idx, t_idx, :]
            for o_idx in range(no_of_out):
                ret = gp_est.weight_var(
                    w_gate_zj[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                    c_dict, weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                pot_y_zj[a_idx, t_idx, o_idx] = ret[0]
                pot_y_var_zj[a_idx, t_idx, o_idx] = ret[1]
                if c_dict['with_output']:
                    ret2 = gp_est.weight_var(
                        w_diff, y_dat[:, o_idx], cl_dat, c_dict, norm=False,
                        weights=w_dat, bootstrap=c_dict['se_boot_gate'])
                    pot_y_mate_zj[a_idx, t_idx, o_idx] = ret2[0]
                    pot_y_mate_var_zj[a_idx, t_idx, o_idx] = ret2[1]
    if w_gate_zj.nbytes > 1e+9 and not c_dict['mp_with_ray']:
        # otherwise tuple gets too large for MP
        save_name_w = 'wtemp' + str(zj_idx) + '.npy'
        save_name_wunc = 'wunctemp' + str(zj_idx) + '.npy'
        np.save(save_name_w, w_gate_zj, fix_imports=False)
        np.save(save_name_wunc, w_gate_unc_zj, fix_imports=False)
        w_gate_zj = None
        w_gate_unc_zj = None
    else:
        save_name_w = None
        save_name_wunc = None
    return (pot_y_zj, pot_y_var_zj, pot_y_mate_zj, pot_y_mate_var_zj,
            w_gate_zj, w_gate_unc_zj, zj_idx, w_censored_zj, save_name_w,
            save_name_wunc)


def wald_test(z_name, no_of_zval, w_gate, y_dat, w_dat, cl_dat, a_idx, o_idx,
              w_ate, c_dict, gate_str='GATE'):
    """Compute Wald tests for GATE(T).

    Parameters
    ----------
    z_name : String.
    no_of_zval : Int.
    w_gate : Numpy array. Weights.
    y : Numpy array. Outcomes.
    w : Numpy array. Sampling weights (if needed).
    cl : Numpy array. Cluster indicator.
    a_idx : Int. Reference population indicator.
    o_idx : Int. Outcome indicator.
    w_ate : Numpy array. Weights.
    c : Dict. Parameters.
    gate_str : Str. Str for label of test. Default is 'GATE'.

    Returns
    -------
    None.

    """
    print('Wald tests: ', gate_str, 'for ', z_name)
    print('Comparison     Chi2 stat   df   p-value (%)')
    w_gate_sum = np.sum(w_gate, axis=3)
    w_ate_sum = np.sum(w_ate, axis=2)
    for t1_idx in range(c_dict['no_of_treat']):
        if not (1-1e-10 < w_ate_sum[a_idx, t1_idx] < 1+1e-10):
            w_ate[a_idx, t1_idx, :] = w_ate[
                a_idx, t1_idx, :] / w_ate_sum[a_idx, t1_idx]
        for zj1 in range(no_of_zval):
            if not ((-1e-15 < w_gate_sum[zj1, a_idx, t1_idx] < 1e-15)
                    or (1-1e-10 < w_gate_sum[zj1, a_idx, t1_idx] < 1+1e-10)):
                w_gate[zj1, a_idx, t1_idx, :] = w_gate[
                    zj1, a_idx, t1_idx, :] / w_gate_sum[zj1, a_idx, t1_idx]
    for t1_idx in range(c_dict['no_of_treat']):
        for t2_idx in range(t1_idx+1, c_dict['no_of_treat']):
            diff_w = np.empty(no_of_zval-1)
            var_w = np.empty((no_of_zval-1, no_of_zval-1))
            for zj1 in range(no_of_zval-1):
                ret1 = gp_est.weight_var(
                    w_gate[zj1, a_idx, t1_idx, :] - w_ate[a_idx, t1_idx, :],
                    y_dat[:, o_idx], cl_dat, c_dict, False, weights=w_dat,
                    bootstrap=c_dict['se_boot_gate'])
                ret2 = gp_est.weight_var(
                    w_gate[zj1, a_idx, t2_idx, :] - w_ate[a_idx, t2_idx, :],
                    y_dat[:, o_idx], cl_dat, c_dict, False, weights=w_dat,
                    bootstrap=c_dict['se_boot_gate'])
                diff_w[zj1] = ret2[0] - ret1[0]
                var_w[zj1, zj1] = ret1[1] + ret2[1]
                if no_of_zval > 2:
                    for zj2 in range(zj1+1, no_of_zval-1):
                        if c_dict['cluster_std']:
                            ret1 = gp_est.aggregate_cluster_pos_w(
                                cl_dat, w_gate[zj1, a_idx, t1_idx, :]
                                - w_ate[a_idx, t1_idx, :], y_dat[:, o_idx],
                                False, w2_dat=w_gate[zj2, a_idx, t1_idx, :]
                                - w_ate[a_idx, t1_idx, :], sweights=w_dat,
                                y2_compute=True)
                            ret2 = gp_est.aggregate_cluster_pos_w(
                                cl_dat, w_gate[zj1, a_idx, t2_idx, :]
                                - w_ate[a_idx, t2_idx, :], y_dat[:, o_idx],
                                False, w2_dat=w_gate[zj2, a_idx, t2_idx, :]
                                - w_ate[a_idx, t2_idx, :], sweights=w_dat,
                                y2_compute=True)
                            w11 = ret1[0]
                            w12 = ret1[2]
                            w21 = ret2[0]
                            w22 = ret2[2]
                            y_t11 = ret1[1]
                            y_t12 = ret1[3]
                            y_t21 = ret2[1]
                            y_t22 = ret2[3]
                        else:
                            w11 = (w_gate[zj1, a_idx, t1_idx, :]
                                   - w_ate[a_idx, t1_idx, :])
                            w12 = (w_gate[zj2, a_idx, t1_idx, :]
                                   - w_ate[a_idx, t1_idx, :])
                            w21 = (w_gate[zj1, a_idx, t2_idx, :]
                                   - w_ate[a_idx, t2_idx, :])
                            w22 = (w_gate[zj2, a_idx, t2_idx, :]
                                   - w_ate[a_idx, t2_idx, :])
                            # y_t11 = y_dat[:, o_idx].copy()
                            y_t11 = y_dat[:, o_idx]
                            y_t12 = y_t11
                            y_t21 = y_t11
                            y_t22 = y_t11
                        cv1 = (np.mean(w11 * w12 * (y_t11 * y_t12)) -
                               (np.mean(w11 * y_t11) * np.mean(w12 * y_t12)))
                        cv2 = (np.mean(w21 * w22 * y_t21 * y_t22) -
                               (np.mean(w21 * y_t21) * np.mean(w22 * y_t22)))
                        # cv12 = np.size(w11) * cv1 + np.size(w21) * cv2
                        cv12 = len(w11) * cv1 + len(w21) * cv2
                        # var_w[zj1, zj2] = cv12.copy()
                        var_w[zj1, zj2] = cv12
                        var_w[zj2, zj1] = var_w[zj1, zj2]
            stat, dfr, pval = gp.waldtest(diff_w, var_w)
            print('{:<3} vs. {:<3}:'.format(c_dict['d_values'][t2_idx],
                                            c_dict['d_values'][t1_idx]),
                  ' {:>8.3f}  {:>4}   {:>7.3f}%'.format(stat, dfr, pval * 100))


def make_gate_figures(titel, z_name, z_values, z_type, effects, stderr,
                      c_dict, ate=0, ate_se=None, am_gate=False,
                      z_smooth=False):
    """Generate the figures for GATE results.

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_values : List. Values of z-variables.
    z_type : Int. Type of variable (ordered or unordered)
    effects : 1D Numpy array. Effects for all z-values.
    stderr : 1D Numpy array. Standard errors for all effects.
    c : Dict. Parameters.

    Returns
    -------
    None.

    """
    file_name_jpeg = c_dict['fig_pfad_jpeg'] + '/' + titel + '.jpeg'
    file_name_pdf = c_dict['fig_pfad_pdf'] + '/' + titel + '.pdf'    # pic pdf
    file_name_csv = c_dict['fig_pfad_csv'] + '/' + titel + '.csv'    # Data
    cint = sct.norm.ppf(
        c_dict['fig_ci_level'] + 0.5 * (1 - c_dict['fig_ci_level']))
    upper = effects + stderr * cint
    lower = effects - stderr * cint
    ate = ate * np.ones((len(z_values), 1))
    if ate_se is not None:
        ate_upper = ate + ate_se * cint
        ate_lower = ate - ate_se * cint
    label_ci = str(c_dict['fig_ci_level'] * 100) + '%-CI'
    if isinstance(z_type, (list, tuple, np.ndarray)):
        z_type = z_type[0]
    if (z_type == 0) and (len(z_values) > c_dict['no_filled_plot']):
        if am_gate or z_smooth:
            file_name_f_jpeg = (c_dict['fig_pfad_jpeg'] + '/' + titel
                                + 'fill.jpeg')
            file_name_f_pdf = c_dict['fig_pfad_pdf'] + '/' + titel + 'fill.pdf'
            figs, axs = plt.subplots()
            if ate_se is None:
                if am_gate:
                    label_m = 'AMGATE-ATE'
                else:
                    label_m = 'GATE-ATE'
            else:
                if am_gate:
                    label_m = 'AMGATE'
                else:
                    label_m = 'GATE'
            axs.plot(z_values, effects, label=label_m, color='b')
            axs.fill_between(z_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
            line_ate = '_-r'
            if ate_se is not None:
                axs.fill_between(
                    z_values, ate_upper.reshape(-1), ate_lower.reshape(-1),
                    alpha=0.3, color='r', label=label_ci)
                label_ate = 'ATE'
            else:
                label_ate = '_nolegend_'
            axs.plot(z_values, ate, line_ate, label=label_ate)
            axs.set_ylabel(label_m)
            axs.legend(loc='lower right', shadow=True,
                       fontsize=c_dict['fig_fontsize'])
            axs.set_title(titel)
            axs.set_xlabel('Values of ' + z_name)
            gp.delete_file_if_exists(file_name_f_jpeg)
            gp.delete_file_if_exists(file_name_f_pdf)
            figs.savefig(file_name_f_jpeg, dpi=c_dict['fig_dpi'])
            figs.savefig(file_name_f_pdf, dpi=c_dict['fig_dpi'])
            if c_dict['show_plots']:
                plt.show()
            else:
                plt.close()
        e_line = '_-'
        u_line = 'v-'
        l_line = '^-'
    else:
        e_line = 'o'
        u_line = 'v'
        l_line = '^'
    connect_y = np.empty(2)
    connect_x = np.empty(2)
    fig, axe = plt.subplots()
    for idx, i_lab in enumerate(z_values):
        connect_y[0] = upper[idx]
        connect_y[1] = lower[idx]
        connect_x[0] = i_lab
        connect_x[1] = i_lab
        axe.plot(connect_x, connect_y, 'b-', linewidth=0.7)
    if ate_se is not None:
        if am_gate:
            gate_str = 'AMGATE'
        else:
            gate_str = 'GATE'
        ate_label = 'ATE'
    else:
        if am_gate:
            gate_str = 'AMGATE-ATE'
        else:
            gate_str = 'GATE-ATE'
        ate_label = '_nolegend_'
    axe.plot(z_values, effects, e_line + 'b', label=gate_str)
    axe.set_ylabel(gate_str)
    label_u = 'Upper ' + str(round(c_dict['fig_ci_level'] * 100)) + '%-CI'
    label_l = 'Lower ' + str(round(c_dict['fig_ci_level'] * 100)) + '%-CI'
    axe.plot(z_values, upper, u_line + 'b', label=label_u)
    axe.plot(z_values, lower, l_line + 'b', label=label_l)
    axe.plot(z_values, ate, '-' + 'k', label=ate_label)
    if ate_se is not None:
        axe.plot(z_values, ate_upper, '--' + 'k', label=label_u)
        axe.plot(z_values, ate_lower, '--' + 'k', label=label_l)
    axe.legend(loc='lower right', shadow=True, fontsize=c_dict['fig_fontsize'])
    axe.set_title(titel)
    axe.set_xlabel('Values of ' + z_name)
    gp.delete_file_if_exists(file_name_jpeg)
    gp.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
    fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
    if c_dict['show_plots']:
        plt.show()
    else:
        plt.close()
    effects = effects.reshape(-1, 1)
    upper = upper.reshape(-1, 1)
    lower = lower.reshape(-1, 1)
    z_values_np = np.array(z_values, copy=True).reshape(-1, 1)
    if ate_se is not None:
        ate_upper = ate_upper.reshape(-1, 1)
        ate_lower = ate_lower.reshape(-1, 1)
        effects_et_al = np.concatenate((upper, effects, lower, ate,
                                        ate_upper, ate_lower, z_values_np),
                                       axis=1)
        cols = ['upper', 'effects', 'lower', 'ate', 'ate_upper', 'ate_lower',
                'z_values']
    else:
        cols = ['upper', 'effects', 'lower', 'ate', 'z_values']
        effects_et_al = np.concatenate((upper, effects, lower, ate,
                                        z_values_np), axis=1)
    datasave = pd.DataFrame(data=effects_et_al, columns=cols)
    gp.delete_file_if_exists(file_name_csv)
    datasave.to_csv(file_name_csv, index=False)


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
    z_name_add = []
    for name in z_name:
        if (name[-4:] == 'CATV') and (len(name) > 4):
            z_name_add.append(name[:-4])
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
        eva_values = unique_vals.tolist()
    else:
        quas = np.linspace(0.01, 0.99, no_eva_values)
        eva_values = np.quantile(z_dat, quas)
        eva_values = eva_values.tolist()
    return eva_values


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
