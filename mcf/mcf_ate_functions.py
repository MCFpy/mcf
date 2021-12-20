"""
Procedures needed for ATE estimation.

Created on Thu Dec  8 15:48:57 2020.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import copy
import numpy as np
import pandas as pd
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_mcf as gp_mcf


def analyse_weights_ate(weights, title, c_dict, ate=True):
    """Describe the weights.

    Parameters
    ----------
    weights : Numyp array. Weights.
    title : String. Title for output.
    c : Dict. Parameters.
    ate: Boolean. True if Ate is estimated. Default is True.

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
    larger_0 = np.empty(c_dict['no_of_treat'], dtype=np.uint32)
    equal_0 = np.empty(c_dict['no_of_treat'], dtype=np.uint32)
    mean_pos = np.empty(c_dict['no_of_treat'])
    std_pos = np.empty(c_dict['no_of_treat'])
    gini_all = np.empty(c_dict['no_of_treat'])
    gini_pos = np.empty(c_dict['no_of_treat'])
    share_largest_q = np.empty((c_dict['no_of_treat'], 3))
    sum_larger = np.empty((c_dict['no_of_treat'], len(c_dict['q_w'])))
    obs_larger = np.empty((c_dict['no_of_treat'], len(c_dict['q_w'])))
    sum_weights = np.sum(weights, axis=1)
    for j in range(c_dict['no_of_treat']):
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
        mean_pos[j] = np.mean(w_pos)
        std_pos[j] = np.std(w_pos)
        # gini_all[j] = gp_est.gini_coefficient(w_j) * 100
        # gini_pos[j] = gp_est.gini_coefficient(w_pos) * 100
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
            share_largest_q = np.empty((c_dict['no_of_treat'], 3))
            sum_larger = np.zeros((c_dict['no_of_treat'], len(c_dict['q_w'])))
            obs_larger = np.zeros((c_dict['no_of_treat'], len(c_dict['q_w'])))
            if c_dict['with_output']:
                print('Less than 5 observations in some groups.')
    if ate:
        print_weight_stat(larger_0, equal_0, mean_pos, std_pos, gini_all,
                          gini_pos, share_largest_q, sum_larger, obs_larger,
                          c_dict)
    return (larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
            share_largest_q, sum_larger, obs_larger)


def print_weight_stat(larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
                      share_largest_q, sum_larger, obs_larger, c_dict,
                      share_censored=0):
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

    Returns
    -------
    None.

    """
    for j, d_value in enumerate(c_dict['d_values']):
        print('\nTreatment group: {0:<4}'.format(d_value),
              '-' * 64)
        print('# of weights > 0: {0:<6}'.format(round(larger_0[j], 2)), ', ',
              '# of weights = 0: {0:<6}'.format(round(equal_0[j], 2)), ', ',
              'Mean of positive weights: {:7.4f}'.format(mean_pos[j]), ', ',
              'Std of positive weights: {:7.4f}'.format(std_pos[j]))
        print('Gini coefficient (incl. weights=0):                        ',
              '{:7.4f}%'.format(gini_all[j]))
        print('Gini coefficient (weights > 0):                            ',
              '{:7.4f}%'.format(gini_pos[j]))
        print('Share of 1% / 5% / 10% largest weights of all weights > 0: ',
              '{:7.4f}% {:7.4f}% {:7.4f}%'.format(share_largest_q[j, 0],
                                                  share_largest_q[j, 1],
                                                  share_largest_q[j, 2]))
        print('Share of weights > 0.5,0.25,0.1,0.05,...,0.01 (among w>0): ',
              end=' ')
        for i in range(len(c_dict['q_w'])):
            print('{:7.4}%'.format(sum_larger[j, i]), end=' ')
        print('\nShare of obs. with weights > 0.5, ..., 0.01   (among w>0): ',
              end=' ')
        for i in range(len(c_dict['q_w'])):
            print('{:7.4}%'.format(obs_larger[j, i]), end=' ')
        print('\n')
        if np.size(share_censored) > 1:
            # if share_censored[j] > 1e-10:
            print('Share of weights censored at {:8.2f}%: '.format(
                c_dict['max_weight_share']*100), '{:8.4f}% '.format(
                share_censored[j]*100))
            # else:
            #     print(' ')
    print('=' * 80)


def ate_est(weights, pred_data, y_dat, cl_dat, w_dat, var, con,
            balancing_test=False, w_ate_only=False):
    """Estimate ATE and their standard errors.

    Parameters
    ----------
    weights : List of lists. For every obs, positive weights are saved.
                     Alternative: Sparse (csr) matrix.
    pred_data : String. csv-file with data to make predictions for.
    y_dat : Numpy array.
    cl_dat : Numpy array.
    w_dat : Numpy array.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    balancing_test: Boolean. Default is False.
    w_ate_only : only weights are needede as output. Default is False.

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
        v_dict = copy.deepcopy(var)
        c_dict = copy.deepcopy(con)
        v_dict['y_name'] = v_dict['x_balance_name']
        c_dict['atet_flag'] = 0
        c_dict['gatet_flag'] = 0
    else:
        v_dict = var
        c_dict = con
    if not c_dict['w_yes']:
        w_dat = None
    n_y = len(y_dat)
    no_of_out = len(v_dict['y_name'])
    d_p, _, w_p, n_p = get_data_for_final_estimation(
        pred_data, v_dict, c_dict, ate=True,
        need_count=c_dict['weight_as_sparse'])
    if (d_p is not None) and (c_dict['atet_flag'] or c_dict['gatet_flag']):
        no_of_ates = c_dict['no_of_treat'] + 1  # Compute ATEs, ATET, ATENT
    else:
        c_dict['atet_flag'] = 0
        no_of_ates = 1
    t_probs = c_dict['choice_based_probs']
    # Step 1: Aggregate weights
    if c_dict['with_output'] and c_dict['verbose']:
        if balancing_test:
            print('\nComputing balancing tests (in ATE-like fashion)')
        else:
            print('\nComputing ATEs')
    w_ate = np.zeros((no_of_ates, c_dict['no_of_treat'], n_y))
    w_ate_export = np.zeros((no_of_ates, c_dict['no_of_treat'], n_y))
    ind_d_val = np.arange(c_dict['no_of_treat'])
    if c_dict['weight_as_sparse']:
        for i in range(n_p):
            w_add = np.zeros((c_dict['no_of_treat'], n_y))
            for t_ind, _ in enumerate(c_dict['d_values']):
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
                    i_pos = ind_d_val[d_p[i] == c_dict['d_values']]
                    w_i_csr = w_i_csr.multiply(t_probs[int(i_pos)])
                w_add[t_ind, :] = w_i_csr.todense()
            w_ate[0, :, :] += w_add
            if c_dict['atet_flag']:
                w_ate[ind_d_val[d_p[i] == c_dict['d_values']]+1, :, :] += w_add
    else:
        for i, weight_i in enumerate(weights):
            w_add = np.zeros((c_dict['no_of_treat'], n_y))
            for t_ind, _ in enumerate(c_dict['d_values']):
                w_i = weight_i[t_ind][1].copy()
                if c_dict['w_yes']:
                    w_i = w_i * w_dat[weight_i[t_ind][0]].reshape(-1)
                sum_wi = np.sum(w_i)
                if sum_wi <= 1e-15:
                    print('Index: ', weight_i[t_ind][0], '\nWeights: ', w_i)
                    raise Exception('Empty leaf.')
                if not (1-1e-10) < sum_wi < (1+1e-10):
                    w_i = w_i / sum_wi
                if c_dict['w_yes']:
                    w_i = w_i * w_p[i]
                if c_dict['choice_based_yes']:
                    i_pos = ind_d_val[d_p[i] == c_dict['d_values']]
                    w_add[t_ind, weight_i[t_ind][0]] = w_i * t_probs[
                        int(i_pos)]
                else:
                    w_add[t_ind, weight_i[t_ind][0]] = w_i
            w_ate[0, :, :] += w_add
            if c_dict['atet_flag']:
                w_ate[ind_d_val[d_p[i] == c_dict['d_values']]+1, :, :] += w_add
    # Step 2: Get potential outcomes
    pot_y = np.empty((no_of_ates, c_dict['no_of_treat'], no_of_out))
    pot_y_var = np.empty_like(pot_y)
    # Normalize weights
    sumw = np.sum(w_ate, axis=2)
    for a_idx in range(no_of_ates):
        for ta_idx in range(c_dict['no_of_treat']):
            if -1e-15 < sumw[a_idx, ta_idx] < 1e-15:
                if c_dict['with_output']:
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
            w_ate[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :] / sumw[a_idx,
                                                                     ta_idx]
            w_ate_export[a_idx, ta_idx, :] = w_ate[a_idx, ta_idx, :]
            if c_dict['max_weight_share'] < 1:
                w_ate[a_idx, ta_idx, :], _, share = gp_mcf.bound_norm_weights(
                    w_ate[a_idx, ta_idx, :], c_dict['max_weight_share'])
                if c_dict['with_output']:
                    print('Share of weights censored at',
                          ' {:8.3f}%: '.format(c_dict['max_weight_share']*100),
                          '{:8.3f}% '.format(share*100),
                          'ATE type: {:2}'.format(a_idx),
                          'Treatment: {:2}'.format(ta_idx))
    if w_ate_only:
        return w_ate_export, None, None, None, None, None
    if c_dict['cluster_std'] and not c_dict['se_boot_ate']:
        w_ate_1dim = np.zeros((no_of_ates, c_dict['no_of_treat'],
                               len(np.unique(cl_dat))))
    else:
        w_ate_1dim = np.zeros((no_of_ates, c_dict['no_of_treat'], n_y))
    for a_idx in range(no_of_ates):
        for t_idx in range(c_dict['no_of_treat']):
            for o_idx in range(no_of_out):
                ret = gp_est.weight_var(
                    w_ate[a_idx, t_idx, :], y_dat[:, o_idx], cl_dat, c_dict,
                    weights=w_dat, bootstrap=c_dict['se_boot_ate'])
                pot_y[a_idx, t_idx, o_idx] = ret[0]
                pot_y_var[a_idx, t_idx, o_idx] = ret[1]
                if o_idx == 0:
                    if c_dict['cluster_std']:
                        w_ate_1dim[a_idx, t_idx, :] = ret[2]
                    else:
                        w_ate_1dim[a_idx, t_idx, :] = w_ate[a_idx, t_idx, :]
    if c_dict['with_output']:
        if not balancing_test:
            analyse_weights_ate(w_ate_1dim[0, :, :], 'Weights to compute ATE',
                                c_dict)
        print('\n')
        print('=' * 80)
        if balancing_test:
            print('Balancing Tests')
        else:
            print('Average Treatment Effects Estimation')
        print('-' * 80)
        print('Potential outcomes')
        print('-' * 80)
        if c_dict['se_boot_ate'] > 1:
            print('Bootstrap standard errors with {:<6} replications'.format(
                c_dict['se_boot_ate']))
        for o_idx in range(no_of_out):
            print('\nOutcome variable: ', v_dict['y_name'][o_idx])
            for a_idx in range(no_of_ates):
                if a_idx == 0:
                    print('Reference population: All')
                else:
                    print('Reference population: Treatment group:',
                          c_dict['d_values'][a_idx-1])
                print('Treatment Potential outcome   SE of PO')
                for t_idx in range(c_dict['no_of_treat']):
                    print('{:<9}  {:>12.6f}  {:>12.6f} '.format(
                        c_dict['d_values'][t_idx], pot_y[a_idx, t_idx, o_idx],
                        np.sqrt(pot_y_var[a_idx, t_idx, o_idx])))
        print('-' * 80)
        print('Treatment effects (ATE, ATETs):')
        print('-' * 80)
    ate = np.empty((no_of_out, no_of_ates, round(c_dict['no_of_treat'] * (
                   c_dict['no_of_treat'] - 1) / 2)))
    ate_se = np.empty_like(ate)
    for o_idx in range(no_of_out):
        if c_dict['with_output']:
            print('\nOutcome variable: ', v_dict['y_name'][o_idx])
        for a_idx in range(no_of_ates):
            if c_dict['with_output']:
                if a_idx == 0:
                    print('Reference population: All')
                else:
                    print('Reference population: Treatment group:',
                          c_dict['d_values'][a_idx-1])
                print('- ' * 40)
            pot_y_ao = pot_y[a_idx, :, o_idx]
            pot_y_var_ao = pot_y_var[a_idx, :, o_idx]
            (est, stderr, t_val, p_val, effect_list
             ) = gp_mcf.effect_from_potential(
                pot_y_ao, pot_y_var_ao, c_dict['d_values'])
            ate[o_idx, a_idx] = est
            ate_se[o_idx, a_idx] = stderr
            if c_dict['with_output']:
                gp.print_effect(est, stderr, t_val, p_val, effect_list)
                gp_est.print_se_info(c_dict['cluster_std'],
                                     c_dict['se_boot_ate'])
    return w_ate_export, pot_y, pot_y_var, ate, ate_se, effect_list


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
    if need_count:
        obs = len(data.index)
    else:
        obs = None
    if c_dict['w_yes']:
        w_dat = data[v_dict['w_name']].to_numpy()
    else:
        w_dat = None
    if (v_dict['d_name'][0] in data.columns
        ) and (c_dict['atet_flag'] or c_dict['gatet_flag']
               or c_dict['choice_based_yes']):
        d_dat = data[v_dict['d_name']].to_numpy()
        d_dat = np.int16(np.round(d_dat))
    else:
        d_dat = None
    if (ate is False) and (not v_dict['z_name'] == []):
        z_dat = data[v_dict['z_name']].to_numpy()
    else:
        z_dat = None
    return d_dat, z_dat, w_dat, obs
