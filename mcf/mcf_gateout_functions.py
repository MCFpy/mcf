"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from copy import copy, deepcopy
import itertools
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sct

from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def make_gate_figures_discr(
        titel, z_name, z_vals, z_type, effects, stderr, int_dic, p_dic, ate=0,
        ate_se=None, gate_type='GATE', z_smooth=False, gatet_yes=False):
    """Generate the figures for GATE results (discrete outcomes).

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_values : List. Values of z-variables.
    z_type : Int. Type of variable (ordered or unordered)
    effects : 1D Numpy array. Effects for all z-values.
    stderr : 1D Numpy array. Standard errors for all effects.
    int_dic, p_dic : Dict. Parameters.
    Additional keyword parameters.
    """
    z_values = z_vals.copy()
    if ps.find_precision(z_values) == 0:  # usually adjusted
        z_values, z_name = mcf_gp.recode_if_all_prime(z_values.copy(), z_name)
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')
    if gate_type == 'GATE':
        file_name_jpeg = p_dic['gate_fig_pfad_jpeg'] + '/' + titel_f + '.jpeg'
        file_name_pdf = p_dic['gate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        file_name_csv = (p_dic['gate_fig_pfad_csv'] + '/' + titel_f
                         + 'plotdat.csv')
    elif gate_type == 'AMGATE':
        file_name_jpeg = (p_dic['amgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = p_dic['amgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        file_name_csv = (p_dic['amgate_fig_pfad_csv'] + '/' + titel_f
                         + 'plotdat.csv')
    elif gate_type == 'BGATE':
        file_name_jpeg = (p_dic['bgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = p_dic['bgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        file_name_csv = (p_dic['bgate_fig_pfad_csv'] + '/' + titel_f
                         + 'plotdat.csv')
    if ate_se is None:
        gate_str_y = 'Effect - average'
        if gate_type == 'GATE':
            gate_str = 'GATE-ATE'
            label_m = 'GATET-ATET' if gatet_yes else 'GATE-ATE'
        elif gate_type == 'AMGATE':
            gate_str, label_m = 'AMGATE-avg.(AMGATE)', 'AMGATE-avg.(AMGATE)'
        elif gate_type == 'BGATE':
            gate_str, label_m = 'BGATE-avg.(BGATE)', 'BGATE-avg.(BGATE)'
        label_y = 'Effect - average'
        ate_label = '_nolegend_'
    else:
        gate_str_y = 'Effect'
        if gate_type == 'GATE':
            label_m = 'GATET' if gatet_yes else 'GATE'
            ate_label, gate_str = 'ATE', 'GATE'
        elif gate_type == 'AMGATE':
            label_m, ate_label, gate_str = 'AMGATE', 'avg.(AMGATE)', 'AMGATE'
        elif gate_type == 'BGATE':
            label_m, ate_label, gate_str = 'BGATE', 'avg.(BGATE)', 'BGATE'
        label_y = 'Effect'
    ate = ate * np.ones((len(z_values), 1))
    if isinstance(z_type, (list, tuple, np.ndarray)):
        z_type = z_type[0]
    cint = sct.norm.ppf(
        p_dic['ci_level'] + 0.5 * (1 - p_dic['ci_level']))
    upper, lower = effects + stderr * cint, effects - stderr * cint
    if ate_se is not None:
        ate_upper, ate_lower = ate + ate_se * cint, ate - ate_se * cint
    label_ci = f'{p_dic["ci_level"]:2.0%}-CI'
    if (z_type == 0) and (len(z_values) > int_dic['no_filled_plot']):
        if gate_type in ('AMGATE', 'BGATE',) or z_smooth:
            if gate_type == 'GATE':
                file_name_f_jpeg = (p_dic['gate_fig_pfad_jpeg']
                                    + '/' + titel_f + 'fill.jpeg')
                file_name_f_pdf = (p_dic['gate_fig_pfad_pdf']
                                   + '/' + titel_f + 'fill.pdf')
            elif gate_type == 'AMGATE':
                file_name_f_jpeg = (p_dic['amgate_fig_pfad_jpeg']
                                    + '/' + titel_f + 'fill.jpeg')
                file_name_f_pdf = (p_dic['amgate_fig_pfad_pdf']
                                   + '/' + titel_f + 'fill.pdf')
            elif gate_type == 'BGATE':
                file_name_f_jpeg = (p_dic['bgate_fig_pfad_jpeg']
                                    + '/' + titel_f + 'fill.jpeg')
                file_name_f_pdf = (p_dic['bgate_fig_pfad_pdf']
                                   + '/' + titel_f + 'fill.pdf')
            figs, axs = plt.subplots()
            axs.plot(z_values, effects, label=label_m, color='b')
            axs.fill_between(z_values, upper, lower, alpha=0.3, color='b',
                             label=label_ci)
            line_ate = '-r'
            if ate_se is not None:
                axs.fill_between(
                    z_values, ate_upper.reshape(-1), ate_lower.reshape(-1),
                    alpha=0.3, color='r', label=label_ci)
                label_ate = 'ATE'
            else:
                label_ate = '_nolegend_'
            axs.plot(z_values, ate, line_ate, label=label_ate)
            axs.set_ylabel(label_y)
            axs.legend(loc=int_dic['legend_loc'], shadow=True,
                       fontsize=int_dic['fontsize'])
            titel_tmp = titel[:-4] + ' ' + titel[-4:]
            titel_tmp = titel_tmp.replace('vs', ' vs ')
            axs.set_title(titel_tmp)
            axs.set_xlabel(z_name)
            mcf_sys.delete_file_if_exists(file_name_f_jpeg)
            mcf_sys.delete_file_if_exists(file_name_f_pdf)
            figs.savefig(file_name_f_jpeg, dpi=int_dic['dpi'])
            figs.savefig(file_name_f_pdf, dpi=int_dic['dpi'])
            if int_dic['show_plots']:
                plt.show()
            else:
                plt.close()
        e_line, u_line, l_line = '_-', 'v-', '^-'
    else:
        e_line, u_line, l_line = 'o', 'v', '^'
    connect_y, connect_x = np.empty(2), np.empty(2)
    fig, axe = plt.subplots()
    for idx, i_lab in enumerate(z_values):
        connect_y[0], connect_y[1] = upper[idx], lower[idx]
        connect_x[0], connect_x[1] = i_lab, i_lab
        axe.plot(connect_x, connect_y, 'b-', linewidth=0.7)
    axe.plot(z_values, effects, e_line + 'b', label=gate_str)
    axe.set_ylabel(gate_str_y)
    label_u = f'Upper {p_dic["ci_level"]:2.0%}-CI'
    label_l = f'Lower {p_dic["ci_level"]:2.0%}-CI'
    axe.plot(z_values, upper, u_line + 'b', label=label_u)
    axe.plot(z_values, lower, l_line + 'b', label=label_l)
    axe.plot(z_values, ate, '-' + 'r', label=ate_label)
    if ate_se is not None:
        axe.plot(z_values, ate_upper, '--' + 'r', label=label_u)
        axe.plot(z_values, ate_lower, '--' + 'r', label=label_l)
    axe.legend(loc=int_dic['legend_loc'], shadow=True,
               fontsize=int_dic['fontsize'])
    titel_tmp = titel[:-4] + ' ' + titel[-4:]
    titel_tmp = titel_tmp.replace('vs', ' vs ')
    axe.set_title(titel_tmp)
    axe.set_xlabel(z_name)
    mcf_sys.delete_file_if_exists(file_name_jpeg)
    mcf_sys.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
    fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
    if int_dic['show_plots']:
        plt.show()
    else:
        plt.close()
    effects = effects.reshape(-1, 1)
    upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
    z_values_np = np.array(z_values, copy=True).reshape(-1, 1)
    if ate_se is not None:
        ate_upper = ate_upper.reshape(-1, 1)
        ate_lower = ate_lower.reshape(-1, 1)
        effects_et_al = np.concatenate(
            (upper, effects, lower, ate, ate_upper, ate_lower, z_values_np),
            axis=1)
        cols = ['upper', 'effects', 'lower', 'ate', 'ate_upper', 'ate_lower',
                'z_values']
    else:
        cols = ['upper', 'effects', 'lower', 'ate', 'z_values']
        effects_et_al = np.concatenate(
            (upper, effects, lower, ate, z_values_np), axis=1)
    datasave = pd.DataFrame(data=effects_et_al, columns=cols)
    mcf_sys.delete_file_if_exists(file_name_csv)
    datasave.to_csv(file_name_csv, index=False)


def make_gate_figures_cont(titel, z_name, z_values, effects, int_dic, p_dic,
                           ate=None, gate_type='GATE', d_values=None):
    """Generate the figures for GATE results.

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_values : List. Values of z-variables.
    effects : 1D Numpy array. Effects for all z-values.
    int_dic, p_dic : Dict. Parameters.
    Additional keyword parameters.
    """
    titel = 'Dose response ' + titel
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')
    if gate_type == 'GATE':
        file_name_jpeg = p_dic['gate_fig_pfad_jpeg'] + '/' + titel_f + '.jpeg'
        file_name_pdf = p_dic['gate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        gate_str = 'GATE-ATE' if ate is not None else 'GATE'
    elif gate_type == 'AMGATE':
        file_name_jpeg = (p_dic['amgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = p_dic['amgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        gate_str = 'AMGATE-avg(AMGATE)' if ate is not None else 'AMGATE'
    elif gate_type == 'BGATE':
        file_name_jpeg = (p_dic['bgate_fig_pfad_jpeg'] + '/' + titel_f
                          + '.jpeg')
        file_name_pdf = p_dic['bgate_fig_pfad_pdf'] + '/' + titel_f + '.pdf'
        gate_str = 'BGATE-avg(BGATE)' if ate is not None else 'BGATE'
    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
    z_plt = np.transpose(effects)
    x_plt, y_plt = np.meshgrid(z_values, d_values[1:])
    surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap='coolwarm_r',
                            linewidth=0, antialiased=False)
    plt.title(titel)
    axe.set_ylabel('Treatment levels')
    axe.set_zlabel(gate_str)
    axe.set_xlabel(z_name)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    mcf_sys.delete_file_if_exists(file_name_jpeg)
    mcf_sys.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
    fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
    if int_dic['show_plots']:
        plt.show()
    else:
        plt.close()


def gate_tables_nice(p_dic, d_values, gate_type='GATE'):
    """
    Show the nice tables.

    Parameters
    ----------
    c_dict : Dict.
        Parameters of mcf.
    gate : Boolean. Optional.
        Define is gate (True) or mgate/amgate (False). The default is True.

    Returns
    -------
    None.

    """
    if gate_type == 'GATE':
        folder = p_dic['gate_fig_pfad_csv']
    elif gate_type == 'BGATE':
        folder = p_dic['bgate_fig_pfad_csv']
    elif gate_type == 'AMGATE':
        folder = p_dic['amgate_fig_pfad_csv']
    else:
        raise ValueError('Wrong gate_type variable.')
    try:
        file_list = os.listdir(folder)
    except OSError:
        print(f'Folder {folder} not found.',
              f' No nice printing of {gate_type}')
        return None
    gate_tables_nice_by_hugo_bodory(folder, file_list, d_values)
    return None


def count_digits(x_tuple):
    """
    Count digits of numbers in tuple including minus signs.

    Parameters
    ----------
    x_tuple : TUPLE
        Integers of smallest treatment comparison group.

    Returns
    -------
    digits : Integers
        Digits of numbers in tuple.

    """
    digits = 0
    for i in range(2):
        if x_tuple[i] > 0:
            digits += int(math.log10(x_tuple[i])) + 1
        elif x_tuple[i] == 0:
            digits += 1
        else:
            digits += int(math.log10(-x_tuple[i])) + 2  # +1 without minus sign
    return digits


def gate_tables_nice_by_hugo_bodory(path, filenames, treatments):
    """
    Generate a wrapper for tables.py.

    Parameters
    ----------
    path : STRING
        Directory storing CSV files for GATEs or MGATEs or AMGATEs
        For example: r'D:/uni/wia0/amgate/csv
    filenames : LIST
         CSV file names including the extension '.csv'
    treatments : Sorted LIST of integers
        All possible treatment values.

    Returns
    -------
        None

    """
    filenames = [f_name for f_name in filenames if f_name[0] != '_']
    filenames.sort()
    number_of_treatments = len(treatments)
    treatments.sort()
    combi = list(itertools.combinations(treatments, 2))
    number_of_combi = len(combi)

    params = {}  # Parameter dictionary
    params['path'] = path
    params['number_of_treatments'] = number_of_treatments
    params['number_of_stats'] = 3  # Effects and lower/upper confidence bounds.
    params['number_of_decimal_places'] = 3  # Decimals for statistics.
    params['multiplier_rows'] = 2  # Multiplier to print confidence intervals.
    params['combi'] = combi
    params['number_of_combi'] = number_of_combi

    # Length of file ending of first treatment combination
    len_end = 13 + count_digits(combi[0])

    for i_name in range(0, len(filenames), number_of_combi):
        params['effect_name'] = filenames[i_name][:-len_end]
        tables(params)


def generate_treatment_names(p_dict):
    """
    Generate names for comparisons of treatment levels.

    Parameters
    ----------
    p_dict : DICTIONARY
        combi = LIST of tuples. Treatment combinations.

    Returns
    -------
    columns : LIST
        List of strings with all possible combinations of treatment levels.

    """
    columns = [str(i_c[1]) + " vs. " + str(i_c[0]) for i_c in p_dict['combi']]
    return columns


def generate_gate_table(p_dict, label_row=False):
    """
    Generate a dataframe.

    The dataframe includes the effects and confidence bounds for all
    possible combinations of treatment levels.

    Parameters
    ----------
    p_dict : DICTIONARY
        combi = LIST of tuples. Treatment combinations.
        number_of_stats = INTEGER. Number of statisticss.
        number_of_decimal_places = INTEGER. Decimal points.
        directory_effect : STRING. Directory including CVS filename for table.
        treatment_names : STRING. Names for comparisons of treatment groups.
    label_row : BOOLEAN, optional
        User-defined index for new dataframe. The default is False.

    Returns
    -------
    df_new : PANDAS DATAFRAME
        New dataframe.

    """
    name = p_dict['directory_effect']

    for i_co, j_co in enumerate(p_dict['combi']):
        if i_co == 0:
            data = pd.DataFrame(pd.read_csv(name + str(j_co[1]) + "vs" +
                                str(j_co[0]) + "plotdat.csv"))
            if 'x_values' in data.columns:
                data = data.rename(columns={'x_values': 'z_values'})
            data['d'] = i_co
        else:
            dat = pd.DataFrame(pd.read_csv(name + str(j_co[1]) + "vs" +
                               str(j_co[0]) + "plotdat.csv"))
            if 'x_values' in dat.columns:
                dat = dat.rename(columns={'x_values': 'z_values'})
            dat['d'] = i_co
            data = pd.concat((data, dat))

    data.drop_duplicates(subset=['z_values'])

    data_0 = np.array(data.pivot(index='z_values', columns="d",
                                 values="effects"))
    data_lower = np.array(data.pivot(index='z_values', columns="d",
                                     values="lower"))
    data_upper = np.array(data.pivot(index='z_values', columns="d",
                                     values="upper"))

    results = np.concatenate((data_0, data_lower, data_upper), axis=1)
    df_new = pd.DataFrame(np.round(results,
                                   p_dict['number_of_decimal_places']))
    df_new.columns = p_dict['number_of_stats'] * p_dict['treatment_names']
    if not label_row:
        if len(p_dict['combi']) > 1:
            dat.drop_duplicates(subset=['z_values'])
            df_new.index = dat.z_values
        else:
            df_new.index = data.z_values
    else:
        df_new.index = label_row
    return df_new


def create_dataframe_for_results(data, n_1=2, n_2=3):
    """
    Create an empty dataframe.

    Parameters
    ----------
    data : PANDAS DATAFRAME
        Dateframe with treatment effects and confidence bounds.
    n_1 : INTEGER, optional
        Multiplier to increase the number of rows. The default is 2.
    n_2 : INTEGER, optional
        Constant by which the number of columns has to be divided to obtain
        the number of columns with treatment effects only . Default: 3.

    Returns
    -------
    df_empty : PANDAS DATAFRAME
        Empty DataFrame with index and column names.

    """
    nrows = n_1 * data.shape[0]  # Number of rows for new dataframe.
    ncols = int(data.shape[1] / n_2)  # Number of cols for new dataframe.
    matrix = np.empty([nrows, ncols])
    df_empty = pd.DataFrame(matrix)
    df_empty.columns = data.columns[:ncols]
    df_empty['idx'] = ''
    for i in range(0, len(df_empty), n_1):
        df_empty.loc[i, 'idx'] = data.index[int(i / n_1)]
    df_empty.set_index('idx', inplace=True)
    return df_empty


def create_confidence_interval(x_var, y_var):
    """
    Create confidence interval as string in squared brackets.

    Parameters
    ----------
    x_var : FLOAT
        Lower bound of a confidence interval.
    y_var : FLOAT
        Upper bound of a confidence interval.

    Returns
    -------
    STRING
        Confidence interval as string in squared brackets.

    """
    return '[' + str(x_var) + ', ' + str(y_var) + ']'


def tables(params):
    """
    Generate CSV files with point estimates and inference.

    Create a dataframe indicating effects (upper rows) and confidence
    intervals (lower rows) for each treatment combination and evaluation
    point. Then transform the dataframe to a CSV file.

    Parameters
    ----------
    params : DICTIONARY
        path: STRING. Directory for CSV files.
        effect_name : STRING. Effect name based on CSV file name.
        low : INTEGER. Lowest treatment level (>=0)
        high : INTEGER Highest treatment level.
        number_of_treatments = number of treatments
        number_of_stats = number of statisticss
        number_of_decimal_places = decimal points
        multiplier_rows = multiplier rows
        number_of_combi = number of treatment combinations

    Returns
    -------
        None

    """
    params['treatment_names'] = generate_treatment_names(params)
    params['directory_effect'] = params['path'] + '/' + params['effect_name']
    stats_table = generate_gate_table(params)
    d_f = create_dataframe_for_results(stats_table,
                                       n_1=params['multiplier_rows'],
                                       n_2=params['number_of_stats'])
    for k_int in range(0, d_f.shape[0], params['multiplier_rows']):
        for l_int in range(d_f.shape[1]):
            d_f.iloc[k_int, l_int] = stats_table.iloc[
                int(k_int / params['multiplier_rows']), l_int]
            c_int = create_confidence_interval(
                stats_table.iloc[int(k_int / params['multiplier_rows']),
                                 l_int + params['number_of_combi']],
                stats_table.iloc[int(k_int / params['multiplier_rows']),
                                 l_int + 2 * params['number_of_combi']])
            d_f.iloc[k_int + 1, l_int] = c_int
        if d_f.index[k_int] == int(d_f.index[k_int]):  # No decimal if int.
            idx_list = d_f.index.tolist()
            d_f.index = idx_list[:k_int] + [int(d_f.index[k_int])] + \
                idx_list[k_int + 1:]
    directory_table = \
        params['path'] + '/' + '_' + params['effect_name'] + '_table.csv'
    d_f.to_csv(directory_table)


def gate_effects_print(mcf_, effect_dic, effect_m_ate_dic, gate_est_dic,
                       ate, ate_se, gate_type='GATE', special_txt=None):
    """Compute effects and print them."""
    p_dic, int_dic, gen_dic = mcf_.p_dict, mcf_.int_dict, mcf_.gen_dict
    var_x_type = copy(mcf_.var_x_type)
    y_pot_all, y_pot_var_all = effect_dic['y_pot'], effect_dic['y_pot_var']
    txt_all = effect_dic['txt_weights']
    m_ate_yes = effect_m_ate_dic is not None
    if m_ate_yes:
        y_pot_m_ate_all = effect_m_ate_dic['y_pot']
        y_pot_m_ate_var_all = effect_m_ate_dic['y_pot_var']
    else:
        y_pot_m_ate_var_all = y_pot_m_ate_all = None
    if special_txt is not None:
        ps.print_mcf(mcf_.gen_dict, '\n' + '=' * 100 + special_txt,
                     summary=True, non_summary=False)
    # Get parameters and info computed in 'gate_est'
    continuous = gate_est_dic['continuous']
    d_values_dr = gate_est_dic['d_values_dr']
    treat_comp_label = gate_est_dic['treat_comp_label']
    no_of_out, var_dic = gate_est_dic['no_of_out'], gate_est_dic['var_dic']
    var_x_values, p_dic = gate_est_dic['var_x_values'], gate_est_dic['p_dic']
    ref_pop_lab = gate_est_dic['ref_pop_lab']
    if p_dic['gates_minus_previous']:
        effect_type_label = (gate_type, gate_type + '(change)')
    else:
        effect_type_label = (gate_type, gate_type + ' - ATE')
    gate = [None] * len(var_dic['z_name'])
    gate_se, gate_diff,  gate_se_diff = gate[:], gate[:], gate[:]
    z_type_l = [None] * len(var_dic['z_name'])
    z_values_l = z_type_l[:]
    z_smooth_l = [False] * len(var_dic['z_name'])
    for zj_idx, z_name in enumerate(var_dic['z_name']):
        z_type_l[zj_idx] = var_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = var_x_values[z_name]
        if gate_est_dic['smooth_yes']:
            z_smooth_l[zj_idx] = z_name in gate_est_dic['z_name_smooth']
    for z_name_j, z_name in enumerate(var_dic['z_name']):
        y_pot = deepcopy(y_pot_all[z_name_j])
        y_pot_var = deepcopy(y_pot_var_all[z_name_j])
        if gate_type == 'GATE':
            txt_weight = txt_all[z_name_j]
        txt = ''
        if m_ate_yes:
            y_pot_m_ate = deepcopy(y_pot_m_ate_all[z_name_j])
            y_pot_m_ate_var = deepcopy(y_pot_m_ate_var_all[z_name_j])
        if int_dic['with_output'] and int_dic['verbose']:
            print(z_name_j+1, '(', len(var_dic['z_name']), ')', z_name,
                  flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        if z_smooth:
            bandw_z = mcf_est.bandwidth_nw_rule_of_thumb(
                gate_est_dic['z_p'][:, z_name_j])
            bandw_z = bandw_z * p_dic['gates_smooth_bandwidth']
        no_of_zval = len(z_values)
        gate_z = np.empty((no_of_zval, no_of_out, gate_est_dic['no_of_tgates'],
                           len(treat_comp_label)))
        gate_z_se, gate_z_mate = np.empty_like(gate_z), np.empty_like(gate_z)
        gate_z_mate_se = np.empty_like(gate_z)
        for o_idx in range(no_of_out):
            if int_dic['with_output']:
                txt += ('\n' + '-' * 100 + '\nOutcome variable: '
                        f'{var_dic["y_name"][o_idx]}      ')
            for a_idx in range(gate_est_dic['no_of_tgates']):
                if int_dic['with_output']:
                    txt += f'Reference population: {ref_pop_lab[a_idx]}'
                    txt += '\n' + '- ' * 50
                ret_gate = [None] * no_of_zval
                ret_gate_mate = [None] * no_of_zval
                for zj_idx, _ in enumerate(z_values):
                    ret = mcf_est.effect_from_potential(
                        y_pot[zj_idx, a_idx, :, o_idx].reshape(-1),
                        y_pot_var[zj_idx, a_idx, :, o_idx].reshape(-1),
                        d_values_dr, continuous=continuous)
                    ret_gate[zj_idx] = np.array(ret, dtype=object, copy=True)
                    gate_z[zj_idx, o_idx, a_idx, :] = ret[0]
                    gate_z_se[zj_idx, o_idx, a_idx, :] = ret[1]
                    if m_ate_yes:
                        ret = mcf_est.effect_from_potential(
                            y_pot_m_ate[zj_idx, a_idx, :, o_idx].reshape(-1),
                            y_pot_m_ate_var[zj_idx, a_idx, :, o_idx].reshape(
                                -1), d_values_dr, continuous=continuous)
                        gate_z_mate[zj_idx, o_idx, a_idx, :] = ret[0]
                        gate_z_mate_se[zj_idx, o_idx, a_idx, :] = ret[1]
                        ret_gate_mate[zj_idx] = np.array(ret, dtype=object,
                                                         copy=True)
                    else:
                        gate_z_mate = gate_z_mate_se = ret_gate_mate = None
                if int_dic['with_output']:
                    txt += ('\nGroup Average Treatment Effects '
                            + f'({gate_type})' + '\n' + '- ' * 50)
                    txt += (f'\nHeterogeneity: {z_name} Outcome: '
                            + f'{var_dic["y_name"][o_idx]} Ref. pop.: '
                            + f'{ref_pop_lab[a_idx]}\n')
                    txt += ps.print_effect_z(
                        ret_gate, ret_gate_mate, z_values, gate_type,
                        print_output=False,
                        gates_minus_previous=p_dic['gates_minus_previous'])
                    txt += '\n' + ps.print_se_info(
                        p_dic['cluster_std'], p_dic['se_boot_gate'])
                    if not p_dic['gates_minus_previous']:
                        txt += ps.print_minus_ate_info(gen_dic['weighted'],
                                                       print_it=False)
        if int_dic['with_output']:
            primes = mcf_gp.primes_list()                 # figures
            for a_idx, a_lab in enumerate(ref_pop_lab):
                gatet_yes = not a_idx == 0
                for o_idx, o_lab in enumerate(var_dic['y_name']):
                    for t_idx, t_lab in enumerate(treat_comp_label):
                        for e_idx, e_lab in enumerate(effect_type_label):
                            if e_idx == 0:
                                effects = gate_z[:, o_idx, a_idx, t_idx]
                                ste = gate_z_se[:, o_idx, a_idx, t_idx]
                                ate_f = ate[o_idx, a_idx, t_idx]
                                ate_f_se = ate_se[o_idx, a_idx, t_idx]
                            else:
                                ate_f, ate_f_se = 0, None
                                if (m_ate_yes
                                        and not p_dic['gates_minus_previous']):
                                    effects = gate_z_mate[:, o_idx, a_idx,
                                                          t_idx]
                                    ste = gate_z_mate_se[:, o_idx, a_idx,
                                                         t_idx]
                                else:
                                    effects = ste = None
                            z_values_f = var_x_values[z_name].copy()
                            if var_x_type[z_name] > 0:
                                for zjj, zjjlab in enumerate(z_values_f):
                                    for jdx, j_lab in enumerate(primes):
                                        if j_lab == zjjlab:
                                            z_values_f[zjj] = jdx
                            if not continuous and effects is not None:
                                make_gate_figures_discr(
                                    e_lab + ' ' + z_name + ' ' + a_lab +
                                    ' ' + o_lab + ' ' + t_lab, z_name,
                                    z_values_f, z_type_l, effects, ste,
                                    int_dic, p_dic, ate_f, ate_f_se,
                                    gate_type, z_smooth, gatet_yes=gatet_yes)
                            if continuous and t_idx == len(treat_comp_label)-1:
                                if e_idx == 0:
                                    ate_f = ate[o_idx, a_idx, :]
                                    effects = gate_z[:, o_idx, a_idx, :]
                                else:
                                    ate_f = None
                                    effects = gate_z_mate[:, o_idx, a_idx, :]
                                make_gate_figures_cont(
                                    e_lab + ' ' + z_name + ' ' + a_lab +
                                    ' ' + o_lab, z_name, z_values_f,
                                    effects, int_dic, p_dic, ate_f,
                                    gate_type, d_values=d_values_dr)
        gate[z_name_j], gate_se[z_name_j] = gate_z, gate_z_se
        if m_ate_yes:
            gate_diff[z_name_j] = gate_z_mate
            gate_se_diff[z_name_j] = gate_z_mate_se
        else:
            gate_diff = gate_se_diff = None
        if int_dic['with_output']:
            if gate_type in ('AMGATE', 'BGATE'):
                ps.print_mcf(gen_dic, txt, summary=True)
            else:
                ps.print_mcf(gen_dic, txt_weight + txt, summary=False)
                ps.print_mcf(gen_dic, txt, summary=True, non_summary=False)
        if int_dic['with_output']:
            txt += '-' * 100
    if int_dic['with_output']:
        gate_tables_nice(p_dic, gate_est_dic['d_values'], gate_type=gate_type)
    return gate, gate_se, gate_diff, gate_se_diff
