"""Creates nice looking tables for the Gate, MGate, and AMGate.

Created on Thu Sep  1 17:47:06 2022
# -*- coding: utf-8 -*-

@author: MLechner, orginal code from hbodory
"""
import os

import pandas as pd
import numpy as np


def gate_tables_nice(c_dict, gate=True):
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
    if gate:
        folders = [c_dict['gate_fig_pfad_csv'], c_dict['amgate_fig_pfad_csv']]
        gate_type = ['GATE', 'AMGATE']
    else:
        folders = (c_dict['mgate_fig_pfad_csv'], )
        gate_type = ['MGATE']
    for fdx, folder in enumerate(folders):
        try:
            file_list = os.listdir(folder)
        except OSError:
            print(f'Folder {folder} not found.',
                  f' No nice printing of {gate_type[fdx]}')
            return None
        gate_tables_nice_by_hugo_bodory(folder, file_list, c_dict['d_values'])
    return None


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
    treatments : Sorted list of integers
        All possible treatment values.

    Returns
    -------
        None

    """
    filenames = [f_name for f_name in filenames if f_name[0] != '_']
    filenames.sort()
    number_of_treatments = len(treatments)
    number_of_combi = 0  # Number of possible treatment combinations.
    for i_int in range(2, number_of_treatments + 1):
        number_of_combi = number_of_combi + i_int - 1

    idx_min = 0  # Index of filename with minimum length.
    names_eff = filenames[:number_of_combi]  # File names of first feature.
    len_of_names_eff = [len(i_len) for i_len in names_eff]
    if not min(len_of_names_eff) == max(len_of_names_eff):
        idx_min = len_of_names_eff.index(min(len_of_names_eff))

    params = {}  # Parameter dictionary
    params['path'] = path
    params['low'] = treatments[0]
    params['high'] = treatments[-1]
    params['number_of_treatments'] = number_of_treatments
    params['number_of_stats'] = 3  # Effects and lower/upper confidence bounds.
    params['number_of_decimal_places'] = 3  # Decimals for statistics
    params['multiplier_rows'] = 2  # Multiplier to print confidence intervals.
    params['number_of_combi'] = number_of_combi

    for i_name in range(0, len(filenames), number_of_combi):
        params['effect_name'] = filenames[i_name + idx_min][:-15]
        # params['effect_name'] = filenames[i_name][:-15]
        tables(params)


def generate_treatment_names(x_int, y_int):
    """
    Generate names for comparisons of treatment levels.

    Parameters
    ----------
    x_int : INTEGER
        Smallest level of treatment variable.
    y_int : INTEGER
        Number of treatment and control levels.

    Returns
    -------
    columns : LIST
        List of strings with all possible combinations of treatment levels.

    """
    columns = []
    for jj_int in range(x_int, y_int):
        for ii_int in range(x_int + 1, y_int + 1):
            if ii_int <= jj_int:
                pass
            else:
                columns.append(str(ii_int) + " vs. " + str(jj_int))
    return columns


def generate_gate_table(p_dict, label_row=False):
    """
    Generate a dataframe.

    The dataframe includes the effects and confidence bounds for all
    possible combinations of treatment levels.

    Parameters
    ----------
    p_dict : DICTIONARY
        low : INTEGER. Lowest treatment level (>=0).
        high : INTEGER Highest treatment level.
        number_of_stats = INTEGER. Number of statisticss.
        number_of_decimal_places = INTEGER. Decimal points.
        directory_effect : STRING- Directory including CVS filename for table.
        treatment_names : STRING. Names for comparisons of treatment groups.
    label_row : BOOLEAN, optional
        User-defined index for new dataframe. The default is False.

    Returns
    -------
    df_new : PANDAS DATAFRAME
        New dataframe.

    """
    d_count = 1 + p_dict['high']
    name = p_dict['directory_effect']
    d_start = p_dict['low']

    for j in range(1 + d_start, d_count):
        for i in range(j, d_count):
            if i == j:
                data = pd.DataFrame(pd.read_csv(name + str(i) + "vs" +
                                    str(j-1) + "plotdat.csv"))
                if 'x_values' in data.columns:
                    data = data.rename(columns={'x_values': 'z_values'})
                data['d'] = i
            else:
                dat = pd.DataFrame(pd.read_csv(name + str(i) + "vs" +
                                   str(j-1) + "plotdat.csv"))
                if 'x_values' in dat.columns:
                    dat = dat.rename(columns={'x_values': 'z_values'})
                dat['d'] = i
                data = pd.concat((data, dat))
        if j == 1 + d_start:
            data_0 = np.array(data.pivot(index='z_values', columns="d",
                                         values="effects"))
            data_lower = np.array(data.pivot(index='z_values', columns="d",
                                             values="lower"))
            data_upper = np.array(data.pivot(index='z_values', columns="d",
                                             values="upper"))
        else:
            data_0 = np.concatenate([data_0,
                                     data.pivot(index='z_values',
                                                columns="d",
                                                values="effects")],
                                    axis=1)
            data_lower = np.concatenate([data_lower,
                                         data.pivot(index='z_values',
                                                    columns="d",
                                                    values="lower")],
                                        axis=1)
            data_upper = np.concatenate([data_upper,
                                         data.pivot(index='z_values',
                                                    columns="d",
                                                    values="upper")],
                                        axis=1)

    results = np.concatenate((data_0, data_lower, data_upper), axis=1)
    df_new = pd.DataFrame(np.round(results,
                                   p_dict['number_of_decimal_places']))
    df_new.columns = p_dict['number_of_stats'] * p_dict['treatment_names']
    if not label_row:
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
    params['treatment_names'] = generate_treatment_names(
        params['low'], params['number_of_treatments'] + params['low']
        - 1)
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
