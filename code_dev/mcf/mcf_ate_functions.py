"""
Procedures needed for ATE estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy

import numpy as np
import pandas as pd
import scipy.stats as sct
import matplotlib.pyplot as plt

from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import mcf_general_purpose as mcf_gp


def ate_est(weights, pred_data, y_dat, cl_dat, w_dat, var, con,
            balancing_test=False, w_ate_only=False, print_output=True):
    """Estimate ATE and their standard errors.

    Parameters
    ----------
    weights : List of lists. For every obs, positive weights are saved.
                     Alternative: Sparse (csr) matrix.
    pred_data : String. csv-file with data to make predictions for.
    y_dat : Numpy array.
    cl_dat : Numpy array.
    w_dat : Numpy array.
    v_dict : Dict.
              Variables.
    c_dict : Dict.
              Parameters.
    balancing_test: Boolean.
              Default is False.
    w_ate_only : Boolean.
              Only weights are needede as output. Default is False.
    print_output : Boolean.
              Prints output. Default is True.

    Returns
    -------
    w_ate_1dim : Numpy array. Weights used for ATE computation.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.
    ate : Numpy array. ATE.
    ate_se : Numpy array. ATE.
    effect_list : List of strings with name of effects (same order as ATE)

    """
    if balancing_test:
        v_dict, c_dict = copy.deepcopy(var), copy.deepcopy(con)
        v_dict['y_name'] = v_dict['x_balance_name']
        c_dict['atet_flag'], c_dict['gatet_flag'] = 0, 0
    else:
        v_dict, c_dict = var, con
    if c_dict['d_type'] == 'continuous':
        continuous = True
        c_dict['atet_flag'] = c_dict['gatet_flag'] = False
        no_of_treat, d_values = c_dict['ct_grid_w'], c_dict['ct_grid_w_val']
    else:
        continuous = False
        no_of_treat, d_values = c_dict['no_of_treat'], c_dict['d_values']
    if not c_dict['w_yes']:
        w_dat = None
    n_y, no_of_out = len(y_dat), len(v_dict['y_name'])
    d_p, _, w_p, n_p = get_data_for_final_estimation(
        pred_data, v_dict, c_dict, ate=True,
        need_count=c_dict['weight_as_sparse'])
    if (d_p is not None) and (c_dict['atet_flag'] or c_dict['gatet_flag']):
        no_of_ates = no_of_treat + 1  # Compute ATEs, ATET, ATENT
    else:
        c_dict['atet_flag'], no_of_ates = False, 1
    t_probs = c_dict['choice_based_probs']
    # Step 1: Aggregate weights
    if c_dict['with_output'] and c_dict['verbose'] and print_output:
        if balancing_test:
            print('\nComputing balancing tests (in ATE-like fashion)')
        else:
            print('\nComputing ATEs')
    w_ate = np.zeros((no_of_ates, no_of_treat, n_y))
    w_ate_export = np.zeros_like(w_ate)
    ind_d_val = np.arange(no_of_treat)
    if c_dict['weight_as_sparse']:
        for i in range(n_p):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i_csr = weights[t_ind].getrow(i)    # copy
                if c_dict['w_yes']:
                    w_i_csr = w_i_csr.multiply(w_dat.reshape(-1))
                sum_wi = w_i_csr.sum()
                if sum_wi <= 1e-15:
                    print('Observation: ', i)
                    raise Exception('Empty leaf.')
                if not (1-1e-10) < sum_wi < (1+1e-10):
                    w_i_csr = w_i_csr.multiply(1/sum_wi)
                if c_dict['w_yes']:
                    w_i_csr = w_i_csr.multiply(w_p[i])
                if c_dict['choice_based_yes']:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_i_csr = w_i_csr.multiply(t_probs[int(i_pos)])
                w_add[t_ind, :] = w_i_csr.todense()
            w_ate[0, :, :] += w_add
            if c_dict['atet_flag']:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    else:
        for i, weight_i in enumerate(weights):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i = weight_i[t_ind][1].copy()
                if c_dict['w_yes']:
                    w_i = w_i * w_dat[weight_i[t_ind][0]].reshape(-1)
                sum_wi = np.sum(w_i)
                if sum_wi <= 1e-15:
                    print('Index: ', weight_i[t_ind][0], 'd_value: ', t_ind,
                          '\nWeights: ', w_i)
                    raise Exception('Zero weight')
                if not (1-1e-10) < sum_wi < (1+1e-10):
                    w_i = w_i / sum_wi
                if c_dict['w_yes']:
                    w_i = w_i * w_p[i]
                if c_dict['choice_based_yes']:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_add[t_ind, weight_i[t_ind][0]] = w_i * t_probs[
                        int(i_pos)]
                else:
                    w_add[t_ind, weight_i[t_ind][0]] = w_i
            w_ate[0, :, :] += w_add
            if c_dict['atet_flag']:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    # Step 2: Get potential outcomes
    sumw = np.sum(w_ate, axis=2)
    for a_idx in range(no_of_ates):
        for ta_idx in range(no_of_treat):
            if -1e-15 < sumw[a_idx, ta_idx] < 1e-15:
                if c_dict['with_output'] and print_output:
                    print('Treatment:', ta_idx, 'ATE number: ', a_idx)
                    print('ATE weights:', w_ate[a_idx, ta_idx, :], flush=True)
                if w_ate_only:
                    sumw[a_idx, ta_idx] = 1
                    if c_dict['with_output']:
                        print('ATE weights are all zero.')
                else:
                    print('ATE weights:', w_ate[a_idx, ta_idx, :], flush=True)
                    raise Exception('ATE weights are all zero. Not good.' +
                                    'Redo statistic without this variable. ' +
                                    'Or try to use more bootstraps. ' +
                                    'Sample may be too small. ' +
                                    'Problem may be with AMGATE only.')
            if not continuous:
                w_ate[a_idx, ta_idx, :] /= sumw[a_idx, ta_idx]
            w_ate_export[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :]
            if (c_dict['max_weight_share'] < 1) and not continuous:
                w_ate[a_idx, ta_idx, :], _, share = mcf_gp.bound_norm_weights(
                    w_ate[a_idx, ta_idx, :], c_dict['max_weight_share'])
                if c_dict['with_output']:
                    print('Share of weights censored at',
                          f'{c_dict["max_weight_share"]*100:8.3f}%: ',
                          f'{share*100:8.3f}%  ATE type: {a_idx:2}',
                          f'Treatment: {ta_idx:2}')
    if w_ate_only:
        return w_ate_export, None, None, None, None, None
    if continuous:
        # Use the larger grid for estimation. This means that the 'missing
        # weights have to generated from existing weights (linear interpol.)
        i_w01 = c_dict['ct_w_to_dr_int_w01']
        i_w10 = c_dict['ct_w_to_dr_int_w10']
        index_full = c_dict['ct_w_to_dr_index_full']
        d_values_dr = c_dict['ct_d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
        pot_y = np.empty((no_of_ates, no_of_treat_dr, no_of_out))
    else:
        pot_y = np.empty((no_of_ates, no_of_treat, no_of_out))
    pot_y_var = np.empty_like(pot_y)
    no_of_treat_1dim = no_of_treat_dr if continuous else no_of_treat
    if c_dict['cluster_std'] and not c_dict['se_boot_ate']:
        w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim,
                               len(np.unique(cl_dat))))
    else:
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
                                          + w01 * w_ate[a_idx, t_idx+1, :])
                        w_ate_cont = w_ate_cont / np.sum(w_ate_cont)
                        if c_dict['max_weight_share'] < 1:
                            w_ate_cont, _, share = mcf_gp.bound_norm_weights(
                                w_ate_cont, c_dict['max_weight_share'])
                        ret = gp_est.weight_var(
                            w_ate_cont, y_dat[:, o_idx], cl_dat, c_dict,
                            weights=w_dat, bootstrap=c_dict['se_boot_ate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y[a_idx, ti_idx, o_idx] = ret[0]
                        pot_y_var[a_idx, ti_idx, o_idx] = ret[1]
                        if o_idx == 0:
                            w_ate_1dim[a_idx, ti_idx, :] = (
                                ret[2] if c_dict['cluster_std']
                                else w_ate_cont)
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = gp_est.weight_var(
                        w_ate[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        c_dict, weights=w_dat, bootstrap=c_dict['se_boot_ate'])
                    pot_y[a_idx, t_idx, o_idx] = ret[0]
                    pot_y_var[a_idx, t_idx, o_idx] = ret[1]
                    if o_idx == 0:
                        w_ate_1dim[a_idx, t_idx, :] = (
                            ret[2] if c_dict['cluster_std']
                            else w_ate[a_idx, t_idx, :])
    if continuous:
        no_of_treat, d_values = no_of_treat_dr, d_values_dr
    if c_dict['with_output'] and print_output:
        if not balancing_test:
            analyse_weights_ate(
                w_ate_1dim[0, :, :], 'Weights to compute ATE', c_dict,
                ate=True, continuous=continuous, no_of_treat_cont=no_of_treat,
                d_values_cont=d_values)
        print('\n')
        print('=' * 80)
        if balancing_test:
            print('Balancing Tests')
        else:
            print_str = ('\n' + '=' * 80
                         + '\nAverage Treatment Effects Estimation' + '\n'
                         + '-' * 80)
            print(print_str)
            gp.print_f(c_dict['outfilesummary'], print_str)
        print('-' * 80)
        print('Potential outcomes')
        print('-' * 80)
        if c_dict['se_boot_ate'] > 1:
            print(f'Bootstrap standard errors with {c_dict["se_boot_ate"]:<6}',
                  ' replications')
        for o_idx in range(no_of_out):
            print('\nOutcome variable: ', v_dict['y_name'][o_idx])
            for a_idx in range(no_of_ates):
                if a_idx == 0:
                    print('Reference population: All')
                else:
                    print('Reference population: Treatment group:',
                          d_values[a_idx-1])
                print('Treatment Potential outcome   SE of PO')
                for t_idx in range(no_of_treat):
                    fdstring = (f'{d_values[t_idx]:>9.5f} '
                                if continuous else f'{d_values[t_idx]:<9} ')
                    print(fdstring, f' {pot_y[a_idx, t_idx, o_idx]:>12.6f} ',
                          f' {np.sqrt(pot_y_var[a_idx, t_idx, o_idx]):>12.6f}')
        print('-' * 80)
        print('Treatment effects (ATE, ATETs):')
        print('-' * 80)
    if continuous:  # only comparison to zero
        ate = np.empty((no_of_out, no_of_ates, no_of_treat - 1))
    else:
        ate = np.empty((no_of_out, no_of_ates,
                        round(no_of_treat * (no_of_treat - 1) / 2)))
    ate_se = np.empty_like(ate)
    for o_idx, out_name in enumerate(v_dict['y_name']):
        if c_dict['with_output'] and print_output:
            print_str = f'Outcome variable: {out_name}'
            print('\n' + print_str)
            if not balancing_test:
                gp.print_f(c_dict['outfilesummary'], print_str)
        for a_idx in range(no_of_ates):
            if c_dict['with_output'] and print_output:
                if a_idx == 0:
                    print_str = 'Reference population: All'
                    print(print_str)
                    if not balancing_test:
                        gp.print_f(c_dict['outfilesummary'], print_str)
                    label_ate = 'ATE'
                else:
                    print_str = ('Reference population: Treatment group '
                                 + f'{d_values[a_idx-1]}')
                    print(print_str)
                    if not balancing_test:
                        gp.print_f(c_dict['outfilesummary'], print_str)
                    label_ate = 'ATET' + str(d_values[a_idx-1])
                print('- ' * 40)
            pot_y_ao = pot_y[a_idx, :, o_idx]
            pot_y_var_ao = pot_y_var[a_idx, :, o_idx]
            (est, stderr, t_val, p_val, effect_list
             ) = mcf_gp.effect_from_potential(
                pot_y_ao, pot_y_var_ao, d_values, continuous=continuous)
            ate[o_idx, a_idx], ate_se[o_idx, a_idx] = est, stderr
            if c_dict['with_output'] and print_output:
                print_str = gp.print_effect(est, stderr, t_val, p_val,
                                            effect_list, continuous=continuous)
                gp.effect_to_csv(est, stderr, t_val, p_val, effect_list,
                                 path=c_dict['cs_ate_iate_fig_pfad_csv'],
                                 label=label_ate+out_name)
                print_str2 = gp_est.print_se_info(
                    c_dict['cluster_std'], c_dict['se_boot_ate'])
                if not balancing_test:
                    gp.print_f(c_dict['outfilesummary'],
                               print_str + '\n' + print_str2)
                if continuous and not balancing_test and a_idx == 0:
                    dose_response_figure(out_name, v_dict['d_name'][0], est,
                                         stderr, d_values[1:], c_dict)
    return w_ate_export, pot_y, pot_y_var, ate, ate_se, effect_list


def dose_response_figure(y_name, d_name, effects, stderr, d_values, c_dict):
    """Plot the average dose response curve."""
    titel = 'Dose response relative to non-treated: ' + y_name + ' ' + d_name
    file_title = 'DR_rel_treat0' + y_name + d_name
    cint = sct.norm.ppf(c_dict['fig_ci_level'] +
                        0.5 * (1 - c_dict['fig_ci_level']))
    upper = effects + stderr * cint
    lower = effects - stderr * cint
    label_ci = f'{c_dict["fig_ci_level"]:2.0%}-CI'
    label_m, label_0, line_0 = 'ADR', '_nolegend_', '_-k'
    zeros = np.zeros_like(effects)
    file_name_jpeg = (c_dict['cs_ate_iate_fig_pfad_jpeg'] + '/' + file_title
                      + '.jpeg')
    file_name_pdf = (c_dict['cs_ate_iate_fig_pfad_pdf'] + '/' + file_title
                     + '.pdf')
    file_name_csv = (c_dict['cs_ate_iate_fig_pfad_csv'] + '/' + file_title
                     + 'plotdat.csv')
    fig, axs = plt.subplots()
    axs.set_title(titel.replace('vs', ' vs '))
    axs.set_ylabel("Average dose response (relative to 0)")
    axs.set_xlabel('Treatment level')
    axs.plot(d_values, effects, label=label_m, color='b')
    axs.plot(d_values, zeros, line_0, label=label_0)
    axs.fill_between(d_values, upper, lower, alpha=0.3, color='b',
                     label=label_ci)
    axs.legend(loc=c_dict['fig_legend_loc'], shadow=True,
               fontsize=c_dict['fig_fontsize'])
    if c_dict['with_output']:
        gp.delete_file_if_exists(file_name_jpeg)
        gp.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=c_dict['fig_dpi'])
        fig.savefig(file_name_pdf, dpi=c_dict['fig_dpi'])
        if c_dict['show_plots']:
            plt.show()
        else:
            plt.close()
        upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
        effects = effects.reshape(-1, 1)
        d_values_np = np.array(d_values, copy=True).reshape(-1, 1)
        effects_et_al = np.concatenate((upper, effects, lower, d_values_np),
                                       axis=1)
        cols = ['upper', 'effects', 'lower', 'd_values']
        datasave = pd.DataFrame(data=effects_et_al, columns=cols)
        gp.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)


def analyse_weights_ate(weights, title, c_dict, ate=True, continuous=False,
                        no_of_treat_cont=None, d_values_cont=None):
    """Describe the weights.

    Parameters
    ----------
    weights : Numyp array. Weights.
    title : String. Title for output.
    c : Dict. Parameters.
    ate: Boolean. True if Ate is estimated. Default is True.
    continuous : Boolean. Continuous treatment. Default is False.
    no_of_treat_cont : Int. Number of discretized treatments of continuous
                            treatments used for weights. Default is None.
    d_values_cont : Numpy array. Values of discretized treatments of continuous
                                 treatments used for weights. Default is None.

    Returns
    -------
    larger_0 : Numpy array.
    equal_0 : Numpy array.
    mean_pos : Numpy array.
    std_pos : Numpy array.
    gini_all : Numpy array.
    gini_pos : Numpy array.
    share_largest_q : Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.

    """
    if ate:
        print('\n')
        print('=' * 80)
        print('Analysis of weights (normalised to add to 1): ', title)
    no_of_treat = no_of_treat_cont if continuous else c_dict['no_of_treat']
    larger_0 = np.empty(no_of_treat, dtype=np.uint32)
    equal_0 = np.empty_like(larger_0)
    mean_pos = np.empty(no_of_treat)
    std_pos, gini_all = np.empty_like(mean_pos), np.empty_like(mean_pos)
    gini_pos = np.empty_like(mean_pos)
    share_largest_q = np.empty((no_of_treat, 3))
    sum_larger = np.empty((no_of_treat, len(c_dict['q_w'])))
    obs_larger = np.empty_like(sum_larger)
    sum_weights = np.sum(weights, axis=1)
    for j in range(no_of_treat):
        if not (((1 - 1e-10) < sum_weights[j] < (1 + 1e-10))
                or (-1e-15 < sum_weights[j] < 1e-15)):
            w_j = weights[j] / sum_weights[j]
        else:
            w_j = weights[j]
        w_pos = w_j[w_j > 1e-15]
        n_pos = len(w_pos)
        larger_0[j] = n_pos
        n_all = len(w_j)
        equal_0[j] = n_all - n_pos
        mean_pos[j], std_pos[j] = np.mean(w_pos), np.std(w_pos)
        gini_all[j] = gp_est.gini_coeff_pos(w_j, n_all) * 100
        gini_pos[j] = gp_est.gini_coeff_pos(w_pos, n_pos) * 100
        if n_pos > 5:
            qqq = np.quantile(w_pos, (0.99, 0.95, 0.9))
            for i in range(3):
                share_largest_q[j, i] = np.sum(w_pos[w_pos >=
                                                     (qqq[i] - 1e-15)]) * 100
            for idx, val in enumerate(c_dict['q_w']):
                sum_larger[j, idx] = np.sum(
                    w_pos[w_pos >= (val - 1e-15)]) * 100
                obs_larger[j, idx] = len(
                    w_pos[w_pos >= (val - 1e-15)]) / n_pos * 100
        else:
            share_largest_q = np.empty((no_of_treat, 3))
            sum_larger = np.zeros((no_of_treat, len(c_dict['q_w'])))
            obs_larger = np.zeros_like(sum_larger)
            if c_dict['with_output']:
                print('Less than 5 observations in some groups.')
    if ate:
        print_weight_stat(larger_0, equal_0, mean_pos, std_pos, gini_all,
                          gini_pos, share_largest_q, sum_larger, obs_larger,
                          c_dict, continuous=continuous,
                          d_values_cont=d_values_cont)
    return (larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger)


def print_weight_stat(larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
                      share_largest_q, sum_larger, obs_larger, c_dict,
                      share_censored=0, continuous=False, d_values_cont=None):
    """Print the weight statistics.

    Parameters
    ----------
    larger_0 : Numpy array.
    equal_0 : Numpy array.
    mean_pos : Numpy array.
    std_pos : Numpy array.
    gini_all : Numpy array.
    gini_pos : Numpy array.
    share_largest_q :Numpy array.
    sum_larger : Numpy array.
    obs_larger : Numpy array.
    share_censored: Numpy array.
    q : List.
    c : Dict. Parameters.
    share_censored : Float.
             Default is 0.

    Returns
    -------
    None.

    """
    d_values = d_values_cont if continuous else c_dict['d_values']
    for j, d_value in enumerate(d_values):
        if continuous:
            if j != round(len(d_values)/2):  # print only for 1 value in middle
                continue
        print(f'\nTreatment group: {d_value:<4}', '-' * 64)
        print(f'# of weights > 0: {round(larger_0[j], 2):<6}, ',
              f'# of weights = 0: {round(equal_0[j], 2):<6}, ',
              f'Mean of positive weights: {mean_pos[j]:7.4f}, ',
              f'Std of positive weights: {std_pos[j]:7.4f}')
        print('Gini coefficient (incl. weights=0):                        ',
              f'{gini_all[j]:7.4f}%')
        print('Gini coefficient (weights > 0):                            ',
              f'{gini_pos[j]:7.4f}%')
        print('Share of 1% / 5% / 10% largest weights of all weights > 0: ',
              f'{share_largest_q[j, 0]:7.4f}% {share_largest_q[j, 1]:7.4f}%',
              f' {share_largest_q[j, 2]:7.4f}%')
        print('Share of weights > 0.5,0.25,0.1,0.05,...,0.01 (among w>0): ',
              end=' ')
        for i in range(len(c_dict['q_w'])):
            print(f'{sum_larger[j, i]:7.4}%', end=' ')
        print('\nShare of obs. with weights > 0.5, ..., 0.01   (among w>0): ',
              end=' ')
        for i in range(len(c_dict['q_w'])):
            print(f'{obs_larger[j, i]:7.4}%', end=' ')
        print('\n')
        if np.size(share_censored) > 1:
            print('Share of weights censored at',
                  f' {c_dict["max_weight_share"]*100:8.2f}%: ',
                  f'{share_censored[j]*100:8.4f}% ')
    print('=' * 80)


def get_data_for_final_estimation(data_file, v_dict, c_dict, ate=True,
                                  need_count=False):
    """Get data needed to compute final weight based estimates.

    Parameters
    ----------
    data_file : String. Contains prediction data
    v : Dict. Variables.
    c : Dict. Parameters.
    ate : Boolean, optional. ATE or GATE estimation. Default is True (ATE).
    need_count: Boolean. Need to count number of observations.

    Returns
    -------
    d_at : Numpy array. Treatment.
    z_dat : Numpy array. Heterogeneity variables.
    w_dat : Numpy array. External sampling weights.
    obs: Int. Number of observations.

    """
    data = pd.read_csv(data_file)
    obs = len(data.index) if need_count else None
    w_dat = (data[v_dict['w_name']].to_numpy()  # pylint: disable=E1136
             if c_dict['w_yes'] else None)
    if (v_dict['d_name'][0] in data.columns
        ) and (c_dict['atet_flag'] or c_dict['gatet_flag']
               or c_dict['choice_based_yes']):
        d_dat = np.int16(np.round(
            data[v_dict['d_name']].to_numpy()))  # pylint: disable=E1136
    else:
        d_dat = None
    if (ate is False) and (not v_dict['z_name'] == []):
        z_dat = data[v_dict['z_name']].to_numpy()  # pylint: disable=E1136
    else:
        z_dat = None
    return d_dat, z_dat, w_dat, obs
