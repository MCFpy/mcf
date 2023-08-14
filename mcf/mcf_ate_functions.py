"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the ATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sct

from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def ate_est(mcf_, data_df, weights_dic, balancing_test=False,
            w_ate_only=False, with_output=True):
    """Estimate ATE and their standard errors.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame. Prediction data.
    weights_dic : Dict.
              Contains weights and numpy data.
    balancing_test: Boolean.
              Default is False.
    w_ate_only : Boolean.
              Only weights are needed as output. Default is False.

    Returns
    -------
    w_ate_1dim : Numpy array. Weights used for ATE computation.
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.
    effect_list : List of strings with name of effects (same order as ATE)

    """
    gen_dic, ct_dic, int_dic = mcf_.gen_dict, mcf_.ct_dict, mcf_.int_dict
    txt = ''
    print_output = not w_ate_only and with_output
    if balancing_test:
        var_dic, p_dic = deepcopy(mcf_.var_dict), deepcopy(mcf_.p_dict)
        var_dic['y_name'] = var_dic['x_balance_name']
        p_dic['atet'], p_dic['gatet'] = 0, 0
        y_dat = weights_dic['x_bala_np']
    else:
        var_dic, p_dic = mcf_.var_dict, mcf_.p_dict
        y_dat = weights_dic['y_dat_np']
    if gen_dic['d_type'] == 'continuous':
        continuous = True
        p_dic['atet'] = p_dic['gatet'] = False
        no_of_treat, d_values = ct_dic['grid_w'], ct_dic['grid_w_val']
    else:
        continuous = False
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
    w_dat = weights_dic['w_dat_np'] if gen_dic['weighted'] else None

    d_p, _, w_p, n_p = get_data_for_final_ate_estimation(
        data_df, gen_dic, p_dic, var_dic, ate=True,
        need_count=int_dic['weight_as_sparse'])
    n_y, no_of_out = len(y_dat), len(var_dic['y_name'])
    if (d_p is not None) and (p_dic['atet'] or p_dic['gatet']):
        no_of_ates = no_of_treat + 1      # Compute ATEs, ATET, ATENT
    else:
        p_dic['atet'], no_of_ates = False, 1
    t_probs = p_dic['choice_based_probs']
    # Step 1: Aggregate weights
    if int_dic['with_output'] and int_dic['verbose'] and print_output:
        if balancing_test:
            print('\n\nComputing balancing tests (in ATE-like fashion)')
        else:
            print('\n\nComputing ATEs')
    w_ate = np.zeros((no_of_ates, no_of_treat, n_y))
    w_ate_export, ind_d_val = np.zeros_like(w_ate), np.arange(no_of_treat)
    weights = weights_dic['weights']
    if int_dic['weight_as_sparse']:
        for i in range(n_p):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i_csr = weights[t_ind].getrow(i)    # copy
                if gen_dic['weighted']:
                    w_i_csr = w_i_csr.multiply(w_dat.reshape(-1))
                sum_wi = w_i_csr.sum()
                if sum_wi <= 1e-15:
                    txt = f'\nEmpty leaf. Observation: {i}'
                    ps.print_mcf(gen_dic, txt, summary=True)
                    raise RuntimeError(txt)
                if not (1-1e-10) < sum_wi < (1+1e-10):
                    w_i_csr = w_i_csr.multiply(1 / sum_wi)
                if gen_dic['weighted']:
                    w_i_csr = w_i_csr.multiply(w_p[i])
                if p_dic['choice_based_sampling']:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_i_csr = w_i_csr.multiply(t_probs[int(i_pos)])
                w_add[t_ind, :] = w_i_csr.todense()
            w_ate[0, :, :] += w_add
            if p_dic['atet']:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    else:
        for i, weight_i in enumerate(weights):
            w_add = np.zeros((no_of_treat, n_y))
            for t_ind, _ in enumerate(d_values):
                w_i = weight_i[t_ind][1].copy()
                if gen_dic['weighted']:
                    w_i = w_i * w_dat[weight_i[t_ind][0]].reshape(-1)
                sum_wi = np.sum(w_i)
                if sum_wi <= 1e-15:
                    txt = (f'\nZero weight. Index: {weight_i[t_ind][0]}'
                           f'd_value: {t_ind}\nWeights: {w_i}')
                    ps.print_mcf(gen_dic, txt, summary=True)
                    raise RuntimeError(txt)
                if not (1-1e-10) < sum_wi < (1+1e-10):
                    w_i = w_i / sum_wi
                if gen_dic['weighted']:
                    w_i = w_i * w_p[i]
                if p_dic['choice_based_sampling']:
                    i_pos = ind_d_val[d_p[i] == d_values]
                    w_add[t_ind, weight_i[t_ind][0]] = w_i * t_probs[
                        int(i_pos)]
                else:
                    w_add[t_ind, weight_i[t_ind][0]] = w_i
            w_ate[0, :, :] += w_add
            if p_dic['atet']:
                w_ate[ind_d_val[d_p[i] == d_values]+1, :, :] += w_add
    # Step 2: Get potential outcomes
    sumw = np.sum(w_ate, axis=2)
    for a_idx in range(no_of_ates):
        for ta_idx in range(no_of_treat):
            if -1e-15 < sumw[a_idx, ta_idx] < 1e-15:
                if int_dic['with_output'] and print_output:
                    txt += (f'\nTreatment: {ta_idx}, ATE number: {a_idx})'
                            f'\nATE weights: {w_ate[a_idx, ta_idx, :]}')
                if w_ate_only:
                    sumw[a_idx, ta_idx] = 1
                    if int_dic['with_output']:
                        txt += 'ATE weights are all zero.'
                    ps.print_mcf(gen_dic, txt, summary=True)
                else:
                    txt += (f'\nATE weights: {w_ate[a_idx, ta_idx, :]}'
                            'ATE weights are all zero. Not good.'
                            ' Redo statistic without this variable.'
                            ' \nOr try to use more bootstraps.'
                            ' \nOr Sample may be too small.'
                            '\nOr Problem may be with AMGATE only.')
                    ps.print_mcf(gen_dic, txt, summary=True)
                    raise RuntimeError(txt)
            if not continuous:
                w_ate[a_idx, ta_idx, :] /= sumw[a_idx, ta_idx]
            w_ate_export[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :]
            if (p_dic['max_weight_share'] < 1) and not continuous:
                w_ate[a_idx, ta_idx, :], _, share = mcf_gp.bound_norm_weights(
                    w_ate[a_idx, ta_idx, :], p_dic['max_weight_share'])
                if int_dic['with_output']:
                    txt += ('\nShare of weights censored at'
                            f'{p_dic["max_weight_share"]*100:8.3f}%: '
                            f'{share*100:8.3f}%  ATE type: {a_idx:2} '
                            f'Treatment: {ta_idx:2}')
    if w_ate_only:
        return w_ate_export, None, None, None
    if continuous:
        # Use the larger grid for estimation. This means that the 'missing
        # weights have to generated from existing weights (linear interpol.)
        i_w01, i_w10 = ct_dic['w_to_dr_int_w01'], ct_dic['w_to_dr_int_w10']
        index_full = ct_dic['w_to_dr_index_full']
        d_values_dr = ct_dic['d_values_dr_np']
        no_of_treat_dr = len(d_values_dr)
    no_of_treat_1dim = no_of_treat_dr if continuous else no_of_treat
    pot_y = np.empty((no_of_ates, no_of_treat_1dim, no_of_out))
    pot_y_var = np.empty_like(pot_y)
    if p_dic['cluster_std']:
        cl_dat = weights_dic['cl_dat_np']
        if p_dic['se_boot_ate'] < 1:
            w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim,
                                   len(np.unique(cl_dat))))
        else:
            w_ate_1dim = np.zeros((no_of_ates, no_of_treat_1dim, n_y))
    else:
        cl_dat = None
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
                        if p_dic['max_weight_share'] < 1:
                            w_ate_cont, _, share = mcf_gp.bound_norm_weights(
                                w_ate_cont, p_dic['max_weight_share'])
                        ret = mcf_est.weight_var(
                            w_ate_cont, y_dat[:, o_idx], cl_dat, gen_dic,
                            p_dic, weights=w_dat,
                            bootstrap=p_dic['se_boot_ate'])
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y[a_idx, ti_idx, o_idx] = ret[0]
                        pot_y_var[a_idx, ti_idx, o_idx] = ret[1]
                        if o_idx == 0:
                            w_ate_1dim[a_idx, ti_idx, :] = (
                                ret[2] if p_dic['cluster_std']
                                else w_ate_cont)
                        if t_idx == (no_of_treat - 1):  # last element,no inter
                            break
                else:
                    ret = mcf_est.weight_var(
                        w_ate[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat,
                        gen_dic, p_dic, weights=w_dat,
                        bootstrap=p_dic['se_boot_ate'])
                    pot_y[a_idx, t_idx, o_idx] = ret[0]
                    pot_y_var[a_idx, t_idx, o_idx] = ret[1]
                    if o_idx == 0:
                        w_ate_1dim[a_idx, t_idx, :] = (
                            ret[2] if p_dic['cluster_std']
                            else w_ate[a_idx, t_idx, :])
    if int_dic['with_output'] and print_output:
        if continuous:
            no_of_treat, d_values = no_of_treat_dr, d_values_dr
        if not balancing_test:
            (_, _, _, _, _, _, _, _, _, txt_weight) = mcf_est.analyse_weights(
                w_ate_1dim[0, :, :], 'Weights to compute ATE', gen_dic, p_dic,
                ate=True, continuous=continuous, no_of_treat_cont=no_of_treat,
                d_values_cont=d_values)
            txt += txt_weight
    return w_ate_export, pot_y, pot_y_var, txt


def ate_effects_print(mcf_, effect_dic, y_pred_lc, balancing_test=False):
    """Compute ate's from potential outcomes and print them."""
    gen_dic, ct_dic, p_dic = mcf_.gen_dict, mcf_.ct_dict, mcf_.p_dict
    var_dic, int_dic = mcf_.var_dict, mcf_.int_dict
    y_pot, y_pot_var = effect_dic['y_pot'], effect_dic['y_pot_var']
    no_of_ates = y_pot.shape[0]
    if isinstance(y_pred_lc, (pd.Series, pd.DataFrame)):
        lc_yes = True
        y_pred_lc_ate = np.mean(y_pred_lc, axis=0)
    else:
        lc_yes = False
    continuous = gen_dic['d_type'] == 'continuous'
    d_values = ct_dic['d_values_dr_np'] if continuous else gen_dic['d_values']
    no_of_treat = len(d_values)
    y_name = var_dic['x_balance_name'] if balancing_test else var_dic['y_name']
    if gen_dic['with_output']:
        txt = '\n' * 2 + '=' * 100
        if balancing_test:
            txt += '\nBalancing Tests\n' + '-' * 100
        else:
            txt += '\nAverage Treatment Effects Estimation\n' + '-' * 100
        txt += '\nPotential outcomes\n' + '-' * 100
        if p_dic['se_boot_ate'] > 1:
            txt += ('\nBootstrap standard errors with '
                    '{p_dic["se_boot_ate"]:<6} replications')
        for o_idx, out_name in enumerate(y_name):
            txt += '\nOutcome variable: ' + out_name
            for a_idx in range(no_of_ates):
                if a_idx == 0:
                    txt += '    Reference population: All'
                else:
                    txt += ('\n   Reference population: Treatment group:'
                            f' {d_values[a_idx-1]}')
                txt += '\nTreatment  Potential Outcome  SE of PO'
                if lc_yes:
                    txt += '       Uncentered Outcome'
                for t_idx in range(no_of_treat):
                    txt += '\n' + (f'{d_values[t_idx]:>10.5f} '
                                   if continuous else f'{d_values[t_idx]:<9} ')
                    sqrt_var = np.sqrt(y_pot_var[a_idx, t_idx, o_idx])
                    txt += (f' {y_pot[a_idx, t_idx, o_idx]:>12.6f} '
                            f' {sqrt_var:>12.6f}')
                    if lc_yes:
                        y_adjust = (y_pot[a_idx, t_idx, o_idx]
                                    + y_pred_lc_ate[o_idx])
                        txt += f'      {y_adjust:>12.6f}'
        txt += '\n' + '-' * 100 + '\nTreatment effects (ATE, ATETs)'
        txt += '\n' + '-' * 100
    if continuous:  # only comparison to zero
        ate = np.empty((len(y_name), no_of_ates, no_of_treat - 1))
    else:
        ate = np.empty((len(y_name), no_of_ates,
                        round(no_of_treat * (no_of_treat - 1) / 2)))
    ate_se = np.empty_like(ate)
    if balancing_test and gen_dic['with_output']:
        ate_t = np.empty_like(ate)
    for o_idx, out_name in enumerate(y_name):
        if gen_dic['with_output']:
            txt += f'\nOutcome variable: {out_name}'
        for a_idx in range(no_of_ates):
            if gen_dic['with_output']:
                if a_idx == 0:
                    txt += '   Reference population: All'
                    label_ate = 'ATE'
                else:
                    txt += ('   Reference population: Treatment group '
                            f'{d_values[a_idx-1]}')
                    label_ate = 'ATET' + str(d_values[a_idx-1])
                txt += '\n' + '- ' * 50
            pot_y_ao = y_pot[a_idx, :, o_idx]
            pot_y_var_ao = y_pot_var[a_idx, :, o_idx]
            (est, stderr, t_val, p_val, effect_list
             ) = mcf_est.effect_from_potential(
                pot_y_ao, pot_y_var_ao, d_values, continuous=continuous)
            ate[o_idx, a_idx], ate_se[o_idx, a_idx] = est, stderr
            if balancing_test and gen_dic['with_output']:
                ate_t[o_idx, a_idx] = t_val
            if gen_dic['with_output']:
                txt += ps.print_effect(est, stderr, t_val, p_val, effect_list,
                                       continuous=continuous)
                ps.effect_to_csv(est, stderr, t_val, p_val, effect_list,
                                 path=p_dic['ate_iate_fig_pfad_csv'],
                                 label=label_ate+out_name)
                txt += ps.print_se_info(p_dic['cluster_std'],
                                        p_dic['se_boot_ate'])
                if continuous and not balancing_test and a_idx == 0:
                    dose_response_figure(out_name, var_dic['d_name'][0], est,
                                         stderr, d_values[1:], int_dic, p_dic)
    if balancing_test and gen_dic['with_output']:
        average_t = np.mean(ate_t)
        txt += '\nVariables investigated for balancing test:'
        for name in var_dic['x_balance_name']:
            txt += f' {name}'
        txt += '\n' + '- ' * 50 + '\n' + 'Balancing test summary measure'
        txt += ' (average t-value of ATEs):'
        txt += f' {average_t:6.2f}' + '\n' + '-' * 100
    if gen_dic['with_output']:
        ps.print_mcf(gen_dic, txt, summary=True, non_summary=False)
        ps.print_mcf(gen_dic, effect_dic['txt_weights'] + txt,
                     summary=False)
    return ate, ate_se, effect_list


def get_data_for_final_ate_estimation(data_df, gen_dic, p_dic, var_dic,
                                      ate=True, need_count=False):
    """Get data needed to compute final weight based estimates.

    Parameters
    ----------
    data_df : DataFrame. Contains prediction data.
    var_dic : Dict. Variables.
    p_dic : Dict. Parameters.
    ate : Boolean, optional. ATE or GATE estimation. Default is True (ATE).
    need_count: Boolean. Need to count number of observations.

    Returns
    -------
    d_at : Numpy array. Treatment.
    z_dat : Numpy array. Heterogeneity variables.
    w_dat : Numpy array. External sampling weights.
    obs: Int. Number of observations.

    """
    obs = len(data_df.index) if need_count else None
    w_dat = (data_df[var_dic['w_name']].to_numpy()     # pylint: disable=E1136
             if gen_dic['weighted'] else None)
    if (var_dic['d_name'][0] in data_df.columns
        ) and (p_dic['atet'] or p_dic['gatet']
               or p_dic['choice_based_sampling']):
        d_dat = np.int16(np.round(
            data_df[var_dic['d_name']].to_numpy()))  # pylint: disable=E1136
    else:
        d_dat = None
    if (ate is False) and (not var_dic['z_name'] == []):
        z_dat = data_df[var_dic['z_name']].to_numpy()  # pylint: disable=E1136
    else:
        z_dat = None
    return d_dat, z_dat, w_dat, obs


def dose_response_figure(y_name, d_name, effects, stderr, d_values, int_dic,
                         p_dic):
    """Plot the average dose response curve."""
    titel = 'Dose response relative to non-treated: ' + y_name + ' ' + d_name
    file_title = 'DR_rel_treat0' + y_name + d_name
    cint = sct.norm.ppf(p_dic['ci_level'] +
                        0.5 * (1 - p_dic['ci_level']))
    upper = effects + stderr * cint
    lower = effects - stderr * cint
    label_ci = f'{p_dic["ci_level"]:2.0%}-CI'
    label_m, label_0, line_0 = 'ADR', '_nolegend_', '_-k'
    zeros = np.zeros_like(effects)
    file_name_jpeg = (p_dic['ate_iate_fig_pfad_jpeg'] + '/' + file_title
                      + '.jpeg')
    file_name_pdf = (p_dic['ate_iate_fig_pfad_pdf'] + '/' + file_title
                     + '.pdf')
    file_name_csv = (p_dic['ate_iate_fig_pfad_csv'] + '/' + file_title
                     + 'plotdat.csv')
    fig, axs = plt.subplots()
    axs.set_title(titel.replace('vs', ' vs '))
    axs.set_ylabel("Average dose response (relative to 0)")
    axs.set_xlabel('Treatment level')
    axs.plot(d_values, effects, label=label_m, color='b')
    axs.plot(d_values, zeros, line_0, label=label_0)
    axs.fill_between(d_values, upper, lower, alpha=0.3, color='b',
                     label=label_ci)
    axs.legend(loc=int_dic['legend_loc'], shadow=True,
               fontsize=int_dic['fontsize'])
    if int_dic['with_output']:
        mcf_sys.delete_file_if_exists(file_name_jpeg)
        mcf_sys.delete_file_if_exists(file_name_pdf)
        fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
        fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
        if int_dic['show_plots']:
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
        mcf_sys.delete_file_if_exists(file_name_csv)
        datasave.to_csv(file_name_csv, index=False)
