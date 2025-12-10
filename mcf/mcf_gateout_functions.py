"""
Created on Fri Jun 23 10:03:35 2023.

Contains the functions needed for computing the GATEs.

@author: MLechner
-*- coding: utf-8 -*-

"""
from copy import copy, deepcopy
from itertools import combinations
from math import log10, isnan
from os import listdir
from pathlib import Path
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from mcf import mcf_estimation_functions as mcf_est
from mcf.mcf_estimation_generic_functions import bandwidth_nw_rule_of_thumb
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps

type FloatScalar = float | np.floating[Any]
type ListTupleNdArray = list[Any] | tuple[Any, ...] | NDArray[Any]

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def make_gate_figures_discr(
        titel: str,
        z_name: str,
        z_vals: list,
        z_type: int,
        effects: NDArray[Any],
        stderr: NDArray[Any],
        int_cfg: Any,
        p_cfg: Any,
        ate: FloatScalar = 0,
        ate_se: FloatScalar | None = None,
        gate_type: str = 'GATE',
        z_smooth: bool = False,
        gatet_yes: bool = False,
        x_values_unord_org: dict | None = None,
        ) -> Path:
    """Generate the figures for GATE results (discrete treatments).

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_name : String. Name of variable.
    z_vals : List. Values of z-variables.
    z_type : Int. Type of variable (ordered or unordered)
    effects : 1D Numpy array. Effects for all z-values.
    stderr : 1D Numpy array. Standard errors for all effects.
    int_cfg : IntCfg Dataclass. Parameters.
    p_cfg : PCfg dataclass. Parameters.
    Additional keyword parameters.
    """
    z_values = z_vals.copy()
    if mcf_ps.find_precision(z_values) == 0:  # usually adjusted
        z_values, z_name = mcf_gp.recode_if_all_prime(z_values.copy(),
                                                      z_name,
                                                      x_values_unord_org
                                                      )
    z_name_ = mcf_ps.del_added_chars(z_name, prime=True)
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')
    label_m, gate_str, ate_label = '', '', ''

    file_name_jpeg, file_name_pdf, file_name_csv = gate_fig_file_helper(
        p_cfg, gate_type, titel_f
        )

    if ate_se is None:
        gate_str_y = 'Effect - average'
        match gate_type:
            case 'GATE':
                gate_str = 'GATE-ATE'
                label_m = 'GATET-ATET' if gatet_yes else 'GATE-ATE'
            case 'CBGATE':
                gate_str = 'CBGATE-avg.(CBGATE)'
                label_m = 'CBGATE-avg.(CBGATE)'
            case 'BGATE':
                gate_str = 'BGATE-avg.(BGATE)'
                label_m = 'BGATE-avg.(BGATE)'
            case _:
                raise ValueError(f'unknown gate_type={gate_type!r}')
        label_y = 'Effect - average'
        ate_label = '_nolegend_'
    else:
        gate_str_y = 'Effect'
        match gate_type:
            case 'GATE':
                label_m = 'GATET' if gatet_yes else 'GATE'
                ate_label = 'ATE'
                gate_str = 'GATE'
            case 'CBGATE':
                label_m = 'CBGATE'
                ate_label = 'avg.(CBGATE)'
                gate_str = 'CBGATE'
            case 'BGATE':
                label_m = 'BGATE'
                ate_label = 'avg.(BGATE)'
                gate_str = 'BGATE'
            case _:
                raise ValueError(f'unknown gate_type={gate_type!r}')
        label_y = 'Effect'

    ate *= np.ones((len(z_values), 1))
    if isinstance(z_type, (list, tuple, np.ndarray)):
        z_type = z_type[0]
    cint = norm.ppf(p_cfg.ci_level + 0.5 * (1 - p_cfg.ci_level))
    upper, lower = effects + stderr * cint, effects - stderr * cint
    if ate_se is not None:
        ate_upper, ate_lower = ate + ate_se * cint, ate - ate_se * cint
    label_ci = f'{p_cfg.ci_level:2.0%}-CI'
    file_name_f_jpeg = None
    if (z_type == 0) and (len(z_values) > int_cfg.no_filled_plot):
        if z_smooth:
            (file_name_f_jpeg, file_name_f_pdf, _
             ) = gate_fig_file_helper(p_cfg, gate_type, titel_f + 'fill')

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
            axs.legend(loc=int_cfg.legend_loc, shadow=True,
                       fontsize=int_cfg.fontsize)
            titel_tmp = titel[:-4] + ' ' + titel[-4:]
            titel_tmp = titel_tmp.replace('vs', ' vs ')
            axs.set_title(titel_tmp)
            axs.set_xlabel(z_name_)
            mcf_sys.delete_file_if_exists(file_name_f_jpeg)
            mcf_sys.delete_file_if_exists(file_name_f_pdf)
            figs.savefig(file_name_f_jpeg, dpi=int_cfg.dpi)
            figs.savefig(file_name_f_pdf, dpi=int_cfg.dpi)
            if int_cfg.show_plots:
                plt.show()
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
    label_u = f'Upper {p_cfg.ci_level:2.0%}-CI'
    label_l = f'Lower {p_cfg.ci_level:2.0%}-CI'
    axe.plot(z_values, upper, u_line + 'b', label=label_u)
    axe.plot(z_values, lower, l_line + 'b', label=label_l)
    axe.plot(z_values, ate, '-' + 'r', label=ate_label)
    if ate_se is not None:
        axe.plot(z_values, ate_upper, '--' + 'r', label=label_u)
        axe.plot(z_values, ate_lower, '--' + 'r', label=label_l)
    axe.legend(loc=int_cfg.legend_loc, shadow=True,
               fontsize=int_cfg.fontsize)
    titel_tmp = titel[:-4] + ' ' + titel[-4:]
    titel_tmp = titel_tmp.replace('vs', ' vs ')
    axe.set_title(titel_tmp)
    axe.set_xlabel(z_name_)
    mcf_sys.delete_file_if_exists(file_name_jpeg)
    mcf_sys.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=int_cfg.dpi)
    fig.savefig(file_name_pdf, dpi=int_cfg.dpi)
    if int_cfg.show_plots:
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

    file_name = file_name_jpeg if file_name_f_jpeg is None else file_name_f_jpeg

    return file_name


def make_gate_figures_cont(titel: str, z_name: str, z_vals: list,
                           effects: NDArray[Any],
                           int_cfg: Any,
                           p_cfg: Any,
                           ate: FloatScalar | None = None,
                           gate_type: str = 'GATE',
                           d_values: list | None = None,
                           x_values_unord_org: dict | None = None
                           ) -> Path:
    """Generate the figures for GATE results (continuous treatments).

    Parameters
    ----------
    titel : String. (Messy) title of plot and basis for files.
    z_vals : List. Values of z-variables.
    effects : 1D Numpy array. Effects for all z-values.
    int_cfg : IntCfg Dataclass instance. Parameters.
    p_cfg : PCfg dataclass. Parameters.
    Additional keyword parameters.
    """
    z_values = z_vals.copy()
    if mcf_ps.find_precision(z_values) == 0:  # usually adjusted
        _, z_name = mcf_gp.recode_if_all_prime(z_values.copy(),
                                               z_name,
                                               x_values_unord_org
                                               )
    z_name_ = mcf_ps.del_added_chars(z_name, prime=True)
    titel = 'Dose response ' + titel
    titel_f = titel.replace(' ', '')
    titel_f = titel_f.replace('-', 'M')
    titel_f = titel_f.replace('.', '')

    match gate_type:
        case 'GATE' | 'CBGATE' | 'BGATE' as gt:
            prefix = gt.lower()
            file_name_jpeg = (p_cfg.paths[f'{prefix}_fig_pfad_jpeg']
                              / f'{titel_f}.jpeg'
                              )
            file_name_pdf = (p_cfg.paths[f'{prefix}_fig_pfad_pdf']
                             / f'{titel_f}.pdf'
                             )
            suffix_map = {'GATE': 'ATE',
                          'CBGATE': 'avg(CBGATE)',
                          'BGATE': 'avg(BGATE)'
                          }
            gate_str = gt if ate is None else f'{gt}-{suffix_map[gt]}'
        case _:
            file_name_jpeg = file_name_pdf = gate_str = None

    fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
    z_plt = np.transpose(effects)
    x_plt, y_plt = np.meshgrid(z_values, d_values[1:])
    surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap='coolwarm_r',
                            linewidth=0, antialiased=False)
    plt.title(titel)
    axe.set_ylabel('Treatment levels')
    axe.set_zlabel(gate_str)
    axe.set_xlabel(z_name_)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    mcf_sys.delete_file_if_exists(file_name_jpeg)
    mcf_sys.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=int_cfg.dpi)
    fig.savefig(file_name_pdf, dpi=int_cfg.dpi)
    if int_cfg.show_plots:
        plt.show()
    plt.close()

    return file_name_jpeg


def gate_tables_nice(p_cfg: Any,
                     d_values: ListTupleNdArray,
                     gate_type: bool = 'GATE'
                     ) -> None:
    """
    Show the nice tables.

    Parameters
    ----------
    p_cfg : PCfg dataclass.
        Parameters.
    gate : Boolean. Optional.
        Define is gate (True) or bgate/cbgate (False). The default is True.

    Returns
    -------
    None.

    """
    match gate_type:
        case 'GATE' | 'BGATE' | 'CBGATE' as gt:
            folder = p_cfg.paths[f'{gt.lower()}_fig_pfad_csv']
        case _:
            raise ValueError('Wrong gate_type variable.')

    try:
        file_list = listdir(folder)
    except OSError:
        print(f'Folder {folder} not found.',
              f' No nice printing of {gate_type}')
        return None

    gate_tables_nice_by_hugo_bodory(folder, file_list, d_values)


def count_digits(x_tuple: tuple[int, ...]) -> int:
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
            digits += int(log10(x_tuple[i])) + 1
        elif x_tuple[i] == 0:
            digits += 1
        else:
            digits += int(log10(-x_tuple[i])) + 2  # +1 without minus sign

    return digits


def gate_tables_nice_by_hugo_bodory(path: Path,
                                    filenames: Path,
                                    treatments: list[int]
                                    ) -> None:
    """
    Generate a wrapper for tables.py.

    Parameters
    ----------
    path : Pathlib object
        Directory storing CSV files for GATEs or CBGATEs
        For example: r'D:/uni/wia0/cbgate/csv
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
    combi = list(combinations(treatments, 2))
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


def generate_treatment_names(combi: Any) -> list[str]:
    """
    Generate names for comparisons of treatment levels.

    Parameters
    ----------
    combi = LIST of tuples. Treatment combinations.

    Returns
    -------
    columns : LIST
        List of strings with all possible combinations of treatment levels.

    """
    columns = [str(i_c[1]) + " vs. " + str(i_c[0]) for i_c in combi]

    return columns


def generate_gate_table(params: dict, label_row: bool = False) -> pd.DataFrame:
    """
    Generate a dataframe.

    The dataframe includes the effects and confidence bounds for all
    possible combinations of treatment levels.

    Parameters
    ----------
    params : dictionary
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
    name = params['directory_effect']

    for i_co, j_co in enumerate(params['combi']):
        if i_co == 0:
            file_name = f'{name}{j_co[1]}vs{j_co[0]}plotdat.csv'
            data = pd.DataFrame(pd.read_csv(file_name))
            if 'x_values' in data.columns:
                data = data.rename(columns={'x_values': 'z_values'})
            data['d'] = i_co
        else:
            file_name = f'{name}{j_co[1]}vs{j_co[0]}plotdat.csv'
            dat = pd.DataFrame(pd.read_csv(file_name))
            if 'x_values' in dat.columns:
                dat = dat.rename(columns={'x_values': 'z_values'})
            dat['d'] = i_co
            data = pd.concat((data, dat))
    has_duplicates = data['z_values'].duplicated().any()
    if has_duplicates:
        return None

    data_0 = np.array(data.pivot(index='z_values', columns="d",
                                 values="effects")
                      )
    data_lower = np.array(data.pivot(index='z_values', columns="d",
                                     values="lower")
                          )
    data_upper = np.array(data.pivot(index='z_values', columns="d",
                                     values="upper")
                          )
    results = np.concatenate((data_0, data_lower, data_upper), axis=1)
    df_new = pd.DataFrame(np.round(results,
                                   params['number_of_decimal_places']))
    df_new.columns = params['number_of_stats'] * params['treatment_names']
    if not label_row:
        if len(params['combi']) > 1:
            # dat.drop_duplicates(subset=['z_values'])
            df_new.index = dat.z_values
        else:
            df_new.index = data.z_values
    else:
        df_new.index = label_row

    return df_new


def create_dataframe_for_results(data: pd.DataFrame,
                                 n_1: int = 2,
                                 n_2: int = 3,
                                 ) -> pd.DataFrame:
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
    df_empty = pd.DataFrame(matrix, dtype=object)
    df_empty.columns = data.columns[:ncols]
    df_empty['idx'] = ''
    for i in range(0, len(df_empty), n_1):
        df_empty.loc[i, 'idx'] = data.index[int(i / n_1)]
    df_empty.set_index('idx', inplace=True)
    return df_empty


def create_confidence_interval(x_var: FloatScalar, y_var: FloatScalar) -> str:
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


def tables(params: dict) -> None:
    """
    Generate CSV files with point estimates and inference.

    Create a dataframe indicating effects (upper rows) and confidence
    intervals (lower rows) for each treatment combination and evaluation
    point. Then transform the dataframe to a CSV file.

    Parameters
    ----------
    params : DICTIONARY
        path: Pathlib object. Directory for CSV files.
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
    params['treatment_names'] = generate_treatment_names(params['combi'])
    params['directory_effect'] = params['path'] / params['effect_name']
    stats_table = generate_gate_table(params)
    if stats_table is not None:
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
                d_f.iloc[int(k_int) + 1, l_int] = c_int
            if not isnan(d_f.index[k_int]) and d_f.index[k_int] == int(
                    d_f.index[k_int]):  # No decimal if int.
                idx_list = d_f.index.tolist()
                d_f.index = idx_list[:k_int] + [int(d_f.index[k_int])] + \
                    idx_list[int(k_int) + 1:]
        directory_table = (
            params['path'] / ('_' + params['effect_name'] + '_table.csv')
            )
        d_f.to_csv(directory_table)


def gate_effects_print(
        mcf_: 'ModifiedCausalForest',
        effect_dic: dict,
        effect_m_ate_dic: dict,
        gate_est_dic: dict,
        ate: NDArray[Any],
        ate_se: NDArray[Any] | None,
        gate_type: str = 'GATE',
        iv: bool = False,
        special_txt: str | None = None
        ) -> tuple[list[FloatScalar],
                   list[FloatScalar | None], list[FloatScalar | None],
                   list[FloatScalar | None],
                   list[Path]
                   ]:
    """Compute effects and print them."""
    p_cfg, int_cfg, gen_cfg = mcf_.p_cfg, mcf_.int_cfg, mcf_.gen_cfg
    ate_type = 'LATE' if iv else 'ATE'
    var_x_type = copy(mcf_.var_x_type)
    y_pot_all, y_pot_var_all = effect_dic['y_pot'], effect_dic['y_pot_var']
    txt_all = effect_dic['txt_weights']
    m_ate_yes = effect_m_ate_dic is not None
    figure_list = []
    if m_ate_yes:
        y_pot_m_ate_all = effect_m_ate_dic['y_pot']
        y_pot_m_ate_var_all = effect_m_ate_dic['y_pot_var']
    else:
        y_pot_m_ate_var_all = y_pot_m_ate_all = None
    if special_txt is not None:
        mcf_ps.print_mcf(mcf_.gen_cfg, '\n' + '=' * 100 + special_txt,
                         summary=True, non_summary=False)
    # Get parameters and info computed in 'gate_est'
    continuous = gate_est_dic['continuous']
    d_values_dr = gate_est_dic['d_values_dr']
    treat_comp_label = gate_est_dic['treat_comp_label']
    no_of_out, var_cfg = gate_est_dic['no_of_out'], gate_est_dic['var_cfg']
    var_x_values, p_cfg = gate_est_dic['var_x_values'], gate_est_dic['p_cfg']
    ref_pop_lab = gate_est_dic['ref_pop_lab']
    iv_lab = 'L' if iv else ''
    if p_cfg.gates_minus_previous:
        effect_type_label = (iv_lab + gate_type,
                             iv_lab + gate_type + '(change)'
                             )
    else:
        effect_type_label = (iv_lab + gate_type,
                             iv_lab + gate_type + f' - {ate_type}'
                             )
    gate = [None for _ in range(len(var_cfg.z_name))]
    gate_se, gate_diff,  gate_se_diff = gate[:], gate[:], gate[:]
    z_type_l = [None for _ in range(len(var_cfg.z_name))]
    z_values_l = z_type_l[:]
    z_smooth_l = [False] * len(var_cfg.z_name)
    for zj_idx, z_name in enumerate(var_cfg.z_name):
        z_type_l[zj_idx] = var_x_type[z_name]    # Ordered: 0, Unordered > 0
        z_values_l[zj_idx] = var_x_values[z_name]
        if gate_est_dic['smooth_yes']:
            z_smooth_l[zj_idx] = z_name in gate_est_dic['z_name_smooth']

    old_filters = warnings.filters.copy()
    warnings.filterwarnings('error', category=RuntimeWarning)
    for z_name_j, z_name in enumerate(var_cfg.z_name):
        z_name_ = mcf_ps.del_added_chars(z_name, prime=True)
        z_type = z_type_l[z_name_j]
        y_pot = deepcopy(y_pot_all[z_name_j])
        y_pot_var = deepcopy(y_pot_var_all[z_name_j])
        if gate_type == 'GATE':
            txt_weight = txt_all[z_name_j]
        else:
            txt_weight = ''
        txt = ''
        if m_ate_yes:
            y_pot_m_ate = deepcopy(y_pot_m_ate_all[z_name_j])
            y_pot_m_ate_var = deepcopy(y_pot_m_ate_var_all[z_name_j])
        if gen_cfg.with_output and gen_cfg.verbose:
            print(z_name_j+1, '(', len(var_cfg.z_name), ')', z_name_,
                  flush=True)
        z_values, z_smooth = z_values_l[z_name_j], z_smooth_l[z_name_j]
        if z_smooth:
            z_p = gate_est_dic['z_p'][:, z_name_j]
            z_p_clean = z_p[~np.isnan(z_p)]
            bandw_z = bandwidth_nw_rule_of_thumb(z_p_clean,
                                                 zero_tol=int_cfg.zero_tol,
                                                 )
            bandw_z = bandw_z * p_cfg.gates_smooth_bandwidth
        no_of_zval = len(z_values)
        gate_z = np.empty((no_of_zval, no_of_out, gate_est_dic['no_of_tgates'],
                           len(treat_comp_label)))
        gate_z_se, gate_z_mate = np.empty_like(gate_z), np.empty_like(gate_z)
        gate_z_mate_se = np.empty_like(gate_z)
        for o_idx in range(no_of_out):
            if gen_cfg.with_output:
                txt += ('\n' + '-' * 100 + '\nOutcome variable: '
                        f'{var_cfg.y_name[o_idx]}      ')
            for a_idx in range(gate_est_dic['no_of_tgates']):
                if gen_cfg.with_output:
                    txt += f'Reference population: {ref_pop_lab[a_idx]}'
                    txt += '\n' + '- ' * 50
                ret_gate = [None for _ in range(no_of_zval)]
                ret_gate_mate = [None for _ in range(no_of_zval)]
                for zj_idx, _ in enumerate(z_values):
                    ret = mcf_est.effect_from_potential(
                        y_pot[zj_idx, a_idx, :, o_idx].reshape(-1),
                        y_pot_var[zj_idx, a_idx, :, o_idx].reshape(-1),
                        d_values_dr, continuous=continuous)
                    ret_gate[zj_idx] = ret
                    gate_z[zj_idx, o_idx, a_idx, :] = ret[0]
                    gate_z_se[zj_idx, o_idx, a_idx, :] = ret[1]
                    if m_ate_yes:
                        ret = mcf_est.effect_from_potential(
                            y_pot_m_ate[zj_idx, a_idx, :, o_idx].reshape(-1),
                            y_pot_m_ate_var[zj_idx, a_idx, :, o_idx].reshape(
                                -1), d_values_dr, continuous=continuous)
                        gate_z_mate[zj_idx, o_idx, a_idx, :] = ret[0]
                        gate_z_mate_se[zj_idx, o_idx, a_idx, :] = ret[1]
                        ret_gate_mate[zj_idx] = ret
                    else:
                        gate_z_mate = gate_z_mate_se = ret_gate_mate = None
                if gen_cfg.with_output:
                    txt += (f'\n{"Local" if iv else ""}Group Average Treatment '
                            f'Effects ({iv_lab + gate_type})'
                            + '\n' + '- ' * 50
                            )
                    txt += (f'\nHeterogeneity: {z_name_} Outcome: '
                            + f'{var_cfg.y_name[o_idx]} Ref. pop.: '
                            + f'{ref_pop_lab[a_idx]}\n')
                    if z_type > 0:
                        z_values_org = (
                            gate_est_dic['var_x_values_unord_org'][z_name_]
                            )
                    else:
                        z_values_org = None
                    txt += mcf_ps.print_effect_z(
                        ret_gate, ret_gate_mate, z_values, gate_type, ate_type,
                        iv=iv, print_output=False,
                        gates_minus_previous=p_cfg.gates_minus_previous,
                        z_values_org=z_values_org
                        )
                    txt += '\n' + mcf_ps.print_se_info(
                        p_cfg.cluster_std, p_cfg.se_boot_gate)
                    if not p_cfg.gates_minus_previous:
                        txt += mcf_ps.print_minus_ate_info(gen_cfg.weighted,
                                                           print_it=False)
        if gen_cfg.with_output:
            # primes = mcf_gp.primes_list()                 # figures
            for a_idx, a_lab in enumerate(ref_pop_lab):
                gatet_yes = not a_idx == 0
                for o_idx, o_lab in enumerate(var_cfg.y_name):
                    for t_idx, t_lab in enumerate(treat_comp_label):
                        for e_idx, e_lab in enumerate(effect_type_label):
                            figure_disc = figure_cont = None
                            if e_idx == 0:
                                effects = gate_z[:, o_idx, a_idx, t_idx]
                                ste = gate_z_se[:, o_idx, a_idx, t_idx]
                                ate_f = ate[o_idx, a_idx, t_idx]
                                ate_f_se = ate_se[o_idx, a_idx, t_idx]
                            else:
                                ate_f, ate_f_se = 0, None
                                if (m_ate_yes
                                        and not p_cfg.gates_minus_previous):
                                    effects = gate_z_mate[:, o_idx, a_idx,
                                                          t_idx]
                                    ste = gate_z_mate_se[:, o_idx, a_idx,
                                                         t_idx]
                                else:
                                    effects = ste = None
                            z_values_f = var_x_values[z_name].copy()
                            # if var_x_type[z_name] > 0:
                            #     for zjj, zjjlab in enumerate(z_values_f):
                            #         for jdx, j_lab in enumerate(primes):
                            #             if j_lab == zjjlab:
                            #                 z_values_f[zjj] = jdx
                            if not continuous and effects is not None:
                                figure_disc = make_gate_figures_discr(
                                    e_lab + ' ' + z_name_ + ' ' + a_lab +
                                    ' ' + o_lab + ' ' + t_lab, z_name,
                                    z_values_f, z_type, effects, ste,
                                    int_cfg, p_cfg, ate_f, ate_f_se,
                                    gate_type, z_smooth, gatet_yes=gatet_yes,
                                    x_values_unord_org=gate_est_dic[
                                        'var_x_values_unord_org']
                                    )
                            if continuous and t_idx == len(treat_comp_label)-1:
                                if e_idx == 0:
                                    ate_f = ate[o_idx, a_idx, :]
                                    effects = gate_z[:, o_idx, a_idx, :]
                                else:
                                    ate_f = None
                                    effects = gate_z_mate[:, o_idx, a_idx, :]
                                figure_cont = make_gate_figures_cont(
                                    e_lab + ' ' + z_name_ + ' ' + a_lab +
                                    ' ' + o_lab, z_name, z_values_f,
                                    effects, int_cfg, p_cfg, ate_f,
                                    gate_type, d_values=d_values_dr,
                                    x_values_unord_org=gate_est_dic[
                                        'var_x_values_unord_org']
                                    )
                            figure_file = (figure_disc if figure_cont is None
                                           else figure_cont)
                            vs0 = int(t_lab[-1]) == int(gen_cfg.d_values[0])
                            if a_idx == 0 and vs0 and (
                                    (m_ate_yes and e_idx == 1)
                                    or (not m_ate_yes and e_idx == 0)
                                    ):
                                figure_list.append(figure_file)

        gate[z_name_j], gate_se[z_name_j] = gate_z, gate_z_se
        if m_ate_yes:
            gate_diff[z_name_j] = gate_z_mate
            gate_se_diff[z_name_j] = gate_z_mate_se
        else:
            gate_diff = gate_se_diff = None
        if gen_cfg.with_output:
            if gate_type in ('CBGATE', 'BGATE'):
                mcf_ps.print_mcf(gen_cfg, txt, summary=True)
            else:
                mcf_ps.print_mcf(gen_cfg, txt_weight + txt, summary=False)
                mcf_ps.print_mcf(gen_cfg, txt, summary=True, non_summary=False)
        if gen_cfg.with_output:
            txt += '-' * 100
    warnings.filters = old_filters
    # warnings.resetwarnings()
    if gen_cfg.with_output:
        gate_tables_nice(p_cfg, gate_est_dic['d_values'], gate_type=gate_type)

    return gate, gate_se, gate_diff, gate_se_diff, figure_list


def get_names_values(mcf_: 'ModifiedCausalForest',
                     gate_est_dic: dict,
                     bgate_est_dic: dict,
                     cbgate_est_dic: dict
                     ) -> dict:
    """Collect information about Gates for the final results dictionary."""
    if gate_est_dic is not None:
        est_dic = gate_est_dic
    elif bgate_est_dic is not None:
        est_dic = bgate_est_dic
    elif cbgate_est_dic is not None:
        est_dic = cbgate_est_dic
    else:
        est_dic = None
    gate_names_values = {'z_names_list': est_dic['var_cfg'].z_name}
    # for idx, z_name in enumerate(est_dic['var_cfg'].z_name):
    for z_name in est_dic['var_cfg'].z_name:
        if mcf_.var_x_type[z_name] > 0:
            values, _ = mcf_gp.recode_if_all_prime(
                est_dic['var_x_values'][z_name].copy(),
                z_name,
                est_dic['var_x_values_unord_org']
                )
            var = mcf_ps.del_added_chars(z_name, prime=True)
            index_to_replace = gate_names_values['z_names_list'].index(z_name)
            gate_names_values['z_names_list'][index_to_replace] = var
        else:
            var = z_name
            values = est_dic['var_x_values'][z_name]

        gate_names_values[var] = values

    return gate_names_values


def gate_fig_file_helper(p_cfg: Any,
                         gate_type: str,
                         titel: str
                         ) -> tuple[Path, Path, Path]:
    """Get the Gate file names to save the figures."""
    match gate_type:
        case 'GATE' | 'CBGATE' | 'BGATE':
            prefix = gate_type.lower()
            file_name_jpeg = (p_cfg.paths[f'{prefix}_fig_pfad_jpeg']
                              / f'{titel}.jpeg'
                              )
            file_name_pdf = (p_cfg.paths[f'{prefix}_fig_pfad_pdf']
                             / f'{titel}.pdf'
                             )
            file_name_csv = (p_cfg.paths[f'{prefix}_fig_pfad_csv']
                             / f'{titel}plotdat.csv'
                             )
        case _:
            file_name_jpeg = file_name_pdf = file_name_csv = None

    return file_name_jpeg, file_name_pdf, file_name_csv
