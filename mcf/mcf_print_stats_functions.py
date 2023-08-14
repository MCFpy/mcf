"""
Contains general functions for printing and descriptive stats.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from datetime import datetime, timedelta

import scipy.stats as sct
import pandas as pd
import numpy as np

from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys


def print_dic_values_all(mcf_, summary_top=True, summary_dic=True, train=True):
    """Print the dictionaries."""
    txt = '=' * 100 + '\nModified Causal Forest: '
    txt += 'Training ' if train else 'Prediction'
    txt += '\n' + '-' * 100
    print_mcf(mcf_.gen_dict, txt, summary=summary_top)
    print_dic_values(mcf_, summary=summary_dic, train=train)


def print_dic_values(mcf_, summary=False, train=None):
    """Print values of dictionaries that determine module."""
    dic_list = [mcf_.int_dict, mcf_.gen_dict, mcf_.dc_dict, mcf_.ct_dict,
                mcf_.fs_dict, mcf_.cs_dict, mcf_.lc_dict, mcf_.cf_dict,
                mcf_.p_dict, mcf_.post_dict, mcf_.var_dict]
    dic_name_list = ['int_dict', 'gen_dict', 'dc_dict', 'ct_dict',
                     'fs_dict', 'cs_dict', 'lc_dict', 'cf_dict',
                     'p_dict', 'post_dict', 'var_dict']
    if not train:
        dic_list.extend([mcf_.var_x_type, mcf_.var_x_values])
        dic_name_list.extend(['var_x_type', 'var_x_values'])
    for dic, dic_name in zip(dic_list, dic_name_list):
        print_dic(dic, dic_name, mcf_.gen_dict, summary=summary)
    print_mcf(mcf_.gen_dict, '\n', summary=summary)


def desc_by_treatment(mcf_, data_df, summary=False, stage=1):
    """Descripe stats by treatment in different versions."""
    if stage == 1:
        variables_to_desc = [*mcf_.var_dict['y_name'],
                             *mcf_.var_dict['x_balance_name']]
    elif stage == 2:
        variables_to_desc = [*mcf_.var_dict['y_name'],
                             *mcf_.var_dict['x_name']]
    else:
        variables_to_desc = [*mcf_.var_dict['x_name']]
    d_name = (mcf_.var_dict['grid_nn_name']
              if mcf_.gen_dict['d_type'] == 'continuous'
              else mcf_.var_dict['d_name'])
    statistics_by_treatment(
        mcf_.gen_dict, data_df, d_name, variables_to_desc,
        only_next=mcf_.gen_dict['d_type'] == 'continuous',
        summary=summary)


def print_mcf(gen_dic, *strings, summary=False, non_summary=True):
    """Print output to different files and terminal."""
    if gen_dic['print_to_terminal']:
        print(*strings)
    if gen_dic['print_to_file']:
        if non_summary:
            print_f(gen_dic['outfiletext'], *strings)
        if summary:
            print_f(gen_dic['outfilesummary'], *strings)


def print_f(file_to_print_to, *strings):
    """
    Print strings into file (substitute print function).

    Parameters
    ----------
    file_to_print : String.
        Name of file to print to.
    *strings : Non-keyword arguments.

    Returns
    -------
    None.

    """
    with open(file_to_print_to, mode="a", encoding="utf-8") as file:
        file.write('\n')
        for text in strings:
            if not isinstance(text, str):
                file.write(str(text))
            else:
                file.write(text)


def print_dic(dic, dic_name, gen_dic, summary=False):
    """Print dictionary in a simple way."""
    print_mcf(gen_dic, '\n' + dic_name, '\n' + '- ' * 50, summary=summary)
    for keys, values in dic.items():
        if isinstance(values, (list, tuple)):
            sss = [str(x) for x in values]
            print_mcf(gen_dic, keys, ':  ', ' '.join(sss), summary=summary)
        else:
            print_mcf(gen_dic, keys, ':  ', values, summary=summary)


def print_timing(gen_dic, title, text, time_diff, summary=False):
    """Show date and duration of a programme and its different parts."""
    print_str = '\n' + '-' * 100 + '\n'
    print_str += f'{title} executed at: {datetime.now()}\n' + '- ' * 50
    for i in range(0, len(text), 1):
        print_str += '\n' + f'{text[i]} {timedelta(seconds=time_diff[i])}'
    print_str += '\n' + '-' * 100
    print_mcf(gen_dic, print_str, summary=summary)
    return print_str


def print_descriptive_df(gen_dic, data_df, varnames='all', summary=False):
    """Print descriptive statistics of a DataFrame."""
    data_sel = data_df[varnames] if varnames != 'all' else data_df
    desc_stat = data_sel.describe()
    if (varnames == 'all') or len(varnames) > 10:
        to_print = desc_stat.transpose()
    else:
        to_print = desc_stat
    with pd.option_context(
            'display.max_rows', 500, 'display.max_columns', 500,
            'display.expand_frame_repr', True, 'display.width', 150,
            'chop_threshold', 1e-13):
        print_mcf(gen_dic, to_print, summary=summary)


def statistics_by_treatment(gen_dic, data_df, treat_name, var_name,
                            only_next=False, summary=False, median_yes=True,
                            std_yes=True, balancing_yes=True):
    """Descriptive statistics by treatment status."""
    txt = '\n------------- Statistics by treatment status ------------------'
    print_mcf(gen_dic, txt, summary=summary)
    data = data_df[treat_name+var_name]
    mean = data.groupby(treat_name).mean(numeric_only=True)
    std = data.groupby(treat_name).std()
    count = data.groupby(treat_name).count()
    count2 = data[treat_name+[var_name[0]]].groupby(treat_name).count()
    with pd.option_context(
            'display.max_rows', 500, 'display.max_columns', 500,
            'display.expand_frame_repr', True, 'display.width', 150,
            'chop_threshold', 1e-13):
        print_mcf(gen_dic, 'Number of observations:', summary=summary)
        print_mcf(gen_dic, count2.transpose(), summary=summary)
        print_mcf(gen_dic, '\nMean', summary=summary)
        print_mcf(gen_dic, mean.transpose(), summary=summary)
        if median_yes:
            print_mcf(gen_dic, '\nMedian', summary=summary)
            print_mcf(gen_dic, data.groupby(treat_name).median().transpose(),
                      summary=summary)
        if std_yes:
            print_mcf(gen_dic, '\nStandard deviation', summary=summary)
            print_mcf(gen_dic, std.transpose(), summary=summary)
        if balancing_yes:
            balancing_tests(gen_dic, mean, std, count, only_next,
                            summary=summary, subtitle='(descriptive)')


def balancing_tests(gen_dic, mean, std, count, only_next=False, summary=False,
                    subtitle=''):
    """Compute balancing tests."""
    std = std.replace(to_replace=0, value=-1)
    value_of_treat = list(reversed(mean.index))
    value_of_treat2 = value_of_treat[:]
    print_mcf(gen_dic, '\n' + '=' * 100 + '\nBalancing tests ' + subtitle
              + '\n' + '- ' * 50, summary=summary)
    for i in value_of_treat:
        if i >= value_of_treat[-1]:
            value_of_treat2.remove(i)
            for j in value_of_treat2:
                mean_diff = mean.loc[i, :] - mean.loc[j, :]
                std_diff = np.sqrt((std.loc[i, :]**2) / count.loc[i]
                                   + (std.loc[j, :]**2) / count.loc[j])
                t_diff = mean_diff.div(std_diff).abs()
                p_diff = 2 * sct.norm.sf(t_diff) * 100
                stand_diff = (mean_diff / np.sqrt(
                    (std.loc[i, :]**2 + std.loc[j, :]**2) / 2) * 100)
                stand_diff.abs()
                txt = f'\nComparing treatments {i:>2.0f} and {j:>2.0f}'
                txt += '\nVariable                          Mean       Std'
                txt += '        t-val   p-val (%) Stand.Difference (%)'
                for jdx, val in enumerate(mean_diff):
                    txt += (
                        f'\n{mean_diff.index[jdx]:30} {val:10.5f}'
                        f'{std_diff[jdx]:10.5f} {t_diff[jdx]:9.2f}'
                        f'{p_diff[jdx]:9.2f} {stand_diff[jdx]:9.2f}')
                print_mcf(gen_dic, txt, summary=summary)
                if only_next:
                    break


def variable_features(mcf_, summary=False):
    """
    Show variables and their key features.

    Parameters
    ----------
    gen_dic : Dict. General parameters.
    var_x_type : Dict. Name and type of variable.
    var_x_values : Dict. Name and values of variables.

    Returns
    -------
    None.

    """
    var_x_type, var_x_values = mcf_.var_x_type, mcf_.var_x_values
    txt = '\n' + '=' * 100 + '\nFeatures used to build causal forest\n'
    txt += '-' * 100
    for name in var_x_type.keys():
        txt += f'\n{name:20}   '
        if var_x_type[name] == 0:
            txt += 'Ordered   '
            if var_x_values[name]:
                if isinstance(var_x_values[name], (list, tuple, set, dict)):
                    for val in var_x_values[name]:
                        if isinstance(val, float):
                            txt += f'{val:>6.2f} '
                        else:
                            txt += f'{val:>6} '
                else:
                    txt += var_x_values[name]
            else:
                txt += 'Continuous'
        else:
            txt += f'Unordered {len(var_x_values[name])} different values'
    txt += '\n' + '-' * 100
    print_mcf(mcf_.gen_dict, txt, summary=summary)


def print_effect(est, stderr, t_val, p_val, effect_list, add_title=None,
                 continuous=False):
    """Print treatment effects.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    stderr : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    add_title : None or string. Additional title.
    add_info : None or Int. Additional information about parameter.
    continuous : Boolean. True if treatment is continuous.

    Returns
    -------
    print_str : String. String version of output.
    """
    print_str = ''
    if add_title is None:
        print_str += ('\nComparison                Estimate   Standard error'
                      ' t-value   p-value')
    else:
        print_str += (f'Comparison {add_title}              Estimate'
                      '   Standard error t-value   p-value')
    print_str += '\n' + '- ' * 50 + '\n' + ''
    for j in range(np.size(est)):
        if continuous:
            compf = f'{effect_list[j][0]:<9.5f} vs {effect_list[j][1]:>9.5f}'
        else:
            compf = f'{effect_list[j][0]:<9} vs {effect_list[j][1]:>9}'
        print_str += compf + ' '
        if add_title is not None:
            print_str += 'f{add_info:6.2f} '
        print_str += f'{est[j]:12.6f}  {stderr[j]:12.6f} '
        print_str += f'{t_val[j]:8.2f}  {p_val[j]:8.3%} '
        if p_val[j] < 0.001:
            print_str += '****'
        elif p_val[j] < 0.01:
            print_str += ' ***'
        elif p_val[j] < 0.05:
            print_str += '  **'
        elif p_val[j] < 0.1:
            print_str += '   *'
        print_str += '\n'
    print_str += '- ' * 50
    return print_str


def effect_to_csv(est, stderr, t_val, p_val, effect_list, path=None,
                  label=None):
    """Save effects to csv files.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    stderr : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    pfad: String. Path to which the save csv-File is saved to. Default is None.
    label: String. Label for filename. Default is None.

    Returns
    -------
    None.
    """
    file_tmp = label + '.csv' if isinstance(label, str) else 'effect.csv'
    file = path + '/' + file_tmp if isinstance(path, str) else file_tmp
    mcf_sys.delete_file_if_exists(file)
    names_est, names_se, names_tval, names_pval = [], [], [], []
    for j in range(np.size(est)):
        if isinstance(effect_list[j][0], str):
            effect_name = effect_list[j][0].join(effect_list[j][1])
        else:
            effect_name = str(effect_list[j][0]) + str(effect_list[j][1])
        names_est.append('ATE_' + effect_name)
        names_se.append('ATE_SE_' + effect_name)
        names_tval.append('ATE_t_' + effect_name)
        names_pval.append('ATE_p_' + effect_name)
    names = names_est + names_se + names_tval + names_pval
    if isinstance(est, (list, tuple)):
        data = est + stderr + t_val + p_val
    elif isinstance(est, (float, int)):
        data = [est, stderr, t_val, p_val]
    elif isinstance(est, np.ndarray):
        data = np.concatenate((est, stderr, t_val, p_val))
        data = np.reshape(data, (1, -1))
    else:
        raise TypeError('Unknown data type for saving effects to file.')
    data_df = pd.DataFrame(data, columns=names)
    data_df.to_csv(file, index=False)


def print_se_info(cluster_std, se_boot):
    """Print some info on computation of standard errors."""
    print_str = ''
    if cluster_std:
        if se_boot > 0:
            print_str += '\nClustered standard errors by bootstrap.'
        else:
            print_str += '\nClustered standard errors by group aggregation.'
        print_str += '\n' + '-' * 100 + '\n'
    if se_boot > 0:
        print_str += f'Bootstrap replications: {se_boot:d}' + '\n' + '-' * 100
    print(print_str)
    return print_str


def txt_weight_stat(larger_0, equal_0, mean_pos, std_pos, gini_all, gini_pos,
                    share_largest_q, sum_larger, obs_larger, gen_dic, p_dic,
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
    gen_dic, p_dic : Dict. Parameters.
    share_censored: Numpy array. Default is 0.
    continuous : Boolean. Default is False.
    d_values_cont : Boolean. Default is None.

    Returns
    -------
    None.

    """
    d_values = d_values_cont if continuous else gen_dic['d_values']
    txt = ''
    for j, d_value in enumerate(d_values):
        if continuous:
            if j != round(len(d_values)/2):  # print only for 1 value in middle
                continue
        txt += '\n' + '- ' * 50 + f'\nTreatment group: {d_value:<4}\n'
        txt += '- ' * 50
        txt += (f'\n# of weights > 0: {round(larger_0[j], 2):<6} '
                f'# of weights = 0: {round(equal_0[j], 2):<6} '
                f'Mean of positive weights: {mean_pos[j]:7.4f} '
                f'Std of positive weights: {std_pos[j]:7.4f}')
        txt += ('\nGini coefficient (incl. weights=0):                        '
                f'{gini_all[j]:7.2f}%')
        txt += ('\nGini coefficient (weights > 0):                            '
                f'{gini_pos[j]:7.2f}%')
        txt += ('\nShare of 1% / 5% / 10% largest weights of all weights > 0: '
                f'{share_largest_q[j, 0]:7.2f}% {share_largest_q[j, 1]:7.2f}%'
                f' {share_largest_q[j, 2]:7.2f}%')
        txt += '\nShare of weights > 0.5,0.25,0.1,0.05,...,0.01 (among w>0): '
        for i in range(len(p_dic['q_w'])):
            txt += f'{sum_larger[j, i]:7.2f}%'
        txt += '\nShare of obs. with weights > 0.5, ..., 0.01   (among w>0): '
        for i in range(len(p_dic['q_w'])):
            txt += f'{obs_larger[j, i]:7.2f}%'
        if np.size(share_censored) > 1:
            txt += ('\nShare of weights censored at'
                    f' {p_dic["max_weight_share"]:8.2%}: '
                    f'{share_censored[j]:8.4%}')
    txt += '\n' + '=' * 100
    return txt


def print_iate(iate, iate_se, iate_p, effect_list, gen_dic, p_dic, var_dic):
    """Print statistics for the two types of IATEs.

    Parameters
    ----------
    iate : 4D Numpy array. Effects. (obs x outcome x effects x type_of_effect)
    iate_se : 4D Numpy array. Standard errors.
    iate_t : 4D Numpy array.
    iate_p : 4D Numpy array.
    effect_list : List. Names of effects.
    gen_dic, p_dic : Dict. Control paramaters.
    var_dic : Dict. Variables.
    reg_round : Boolean. True if regular estimation round. Default is True.

    Returns
    -------
    txt : String. Text to print.

    """
    no_outcomes = np.size(iate, axis=1)
    n_obs = len(iate)
    str_f, str_m, str_l = '=' * 100, '-' * 100, '- ' * 50
    print_str = '\n' + str_f + '\nDescriptives for IATE estimation\n' + str_m
    iterator = 2 if p_dic['iate_m_ate'] else 1
    for types in range(iterator):
        if types == 0:
            print_str += '\nIATE with corresponding statistics\n' + str_l
        else:
            print_str += ('IATE minus ATE with corresponding statistics '
                          + '(weights not censored)\n' + str_l)
        for o_idx in range(no_outcomes):
            print_str += (f'\nOutcome variable: {var_dic["y_name"][o_idx]}\n'
                          + str_l)
            str1 = '\n        Comparison          Mean       Median      Std'
            if p_dic['iate_se']:
                print_str += (str1 + '  Effect > 0 mean(SE)  sig 10%'
                              '  sig 5%   sig 1%')
            else:
                print_str += str1 + '  Effect > 0'
            for jdx, effects in enumerate(effect_list):
                fdstring = (f'{effects[0]:<9.5f} vs {effects[1]:>9.5f}'
                            if gen_dic['d_type'] == 'continuous' else
                            f'{effects[0]:<9} vs {effects[1]:>9} ')
                print_str += '\n' + fdstring
                est = iate[:, o_idx, jdx, types].reshape(-1)
                if p_dic['iate_se']:
                    stderr = iate_se[:, o_idx, jdx, types].reshape(-1)
                    p_val = iate_p[:, o_idx, jdx, types].reshape(-1)
                print_str += (f'{np.mean(est):10.5f} {np.median(est):10.5f}'
                              f' {np.std(est):10.5f} '
                              f'{np.count_nonzero(est > 1e-15) / n_obs:7.2%}')
                if p_dic['iate_se']:
                    print_str += (
                        f' {np.mean(stderr):10.5f}'
                        f' {np.count_nonzero(p_val < 0.1)/n_obs:8.2%}'
                        f' {np.count_nonzero(p_val < 0.05)/n_obs:8.2%}'
                        f' {np.count_nonzero(p_val < 0.01)/n_obs:8.2%}')
        print_str += '\n' + str_m + '\n'
    print_str += '\n' + str_m
    if p_dic['iate_se']:
        print_str += ('\n' + print_se_info(p_dic['cluster_std'],
                                           p_dic['se_boot_iate']))
        print_str += print_minus_ate_info(gen_dic['weighted'], print_it=False,
                                          gate_or_iate='IATE')
    return print_str


def print_minus_ate_info(weighted, print_it=True, gate_or_iate='GATE'):
    """Print info about effects minus ATE."""
    print_str = ('Weights used for comparison with ATE are not truncated. '
                 f'Therefore, {gate_or_iate}s - ATE may not aggregate to 0.'
                 + '\n' + '-' * 100)
    if weighted:
        print_str += ('Such differences may be particulary pronounced when '
                      'sampling weights are used.')
    if print_it:
        print(print_str)
    return print_str


def print_effect_z(g_r, gm_r, z_values, gate_str, print_output=True,
                   gates_minus_previous=False):
    """Print treatment effects."""
    no_of_effect_per_z = np.size(g_r[0][0])
    if gates_minus_previous:
        print_str = ('- ' * 50 + f'\n                   {gate_str}'
                     + f'                                {gate_str}(change)')
    else:
        print_str = ('- ' * 50 + f'\n                   {gate_str}'
                     + f'                                {gate_str} - ATE')
    print_str += ('\nComparison      Z      Est         SE  t-val   p-val'
                  + '         Est        SE  t-val  p-val\n' + '- ' * 50
                  + '\n')
    prec = find_precision(z_values)
    if prec == 0:
        z_values, _ = mcf_gp.recode_if_all_prime(z_values.copy(), None)
    for j in range(no_of_effect_per_z):
        for zind, z_val in enumerate(z_values):
            treat_s = f'{g_r[zind][4][j][0]:<3} vs {g_r[zind][4][j][1]:>3}'
            val_s = f'{z_val:>7.{prec}f}'
            estse_s = f'{g_r[zind][0][j]:>9.5f}  {g_r[zind][1][j]:>9.5f}'
            t_p_s = f'{g_r[zind][2][j]:>6.2f}  {g_r[zind][3][j]:>6.2%}'
            s_s = stars(g_r[zind][3][j])
            if gm_r is not None:
                estsem_s = (
                    f'{gm_r[zind][0][j]:>9.5f}  {gm_r[zind][1][j]:>9.5f}')
                tm_p_s = f'{gm_r[zind][2][j]:>6.2f}  {gm_r[zind][3][j]:>6.2%}'
                sm_s = stars(gm_r[zind][3][j])
            else:
                estsem_s = tm_p_s = sm_s = ' '
            print_str += (treat_s + val_s + estse_s + t_p_s + s_s + estsem_s
                          + tm_p_s + sm_s + '\n')
        if j < no_of_effect_per_z-1:
            print_str += '- ' * 50 + '\n'
    print_str += '-' * 100
    print_str += ('\nShown values of Z may represent the order of the values'
                  + ' (starting with 0) instead of the original values.')
    print_str += '\n' + '-' * 100
    if print_output:
        print(print_str)
    return print_str


def find_precision(values):
    """Find precision so that all values can be differentiated in printing."""
    len_v = len(np.unique(values))
    precision = 20
    for prec in range(20):
        rounded = np.around(values, decimals=prec)
        if len(set(rounded)) == len_v:  # all unique
            precision = prec            # + 2
            break
    return precision


def stars(pval):
    """Create string with stars for p-values."""
    if pval < 0.001:
        return '****'
    if pval < 0.01:
        return ' ***'
    if pval < 0.05:
        return '  **'
    if pval < 0.1:
        return '   *'
    return '    '
