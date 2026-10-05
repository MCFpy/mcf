"""
Created on Mon Jan 26 16:10:24 2026.

@author: MLechner

# -*- coding: utf-8 -*-

Helpful functions for using versions of treatments with the mcf optimal policy module. See the
example file optpolicy_versions.
"""
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from mcf.mcf_general import cleaned_var_names
from mcf.mcf_general_sys import delete_path_or_file_if_exists
from mcf.mcf_print_stats import print_mcf, print_timing
from mcf.optpolicy_data import var_available
from mcf.optpolicy_methods import allocate_method, evaluate_method

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicyVersions, OptimalPolicy
    from mcf.optpolicy_init import GenCfg

def analyse_treatment_version(var_polscore_dict: dict
                              ) -> tuple[list[str], list[str], list[list[str]], str]:
    """Analye treatment version and return unique treatment variable."""
    # 1. Type of treatments per main und per version
    txt = ('\n' + '-' * 100
           +  '\nScores for the different main treatments and their versions'
           + '\n' + '- ' * 50
           )
    policy_scores_main = list(var_polscore_dict.keys())   # Main treatments
    policy_scores_all = []
    policy_scores_main_version = []
    for key, value in var_polscore_dict.items():
        txt += f'\nMain treatment: {key:10s} Versions: '
        if value is None:
            policy_scores_all.append(key)
            policy_scores_main_version.append([key])
            txt += f'{key}'
        elif isinstance(value, str):
            policy_scores_all.append(value)
            policy_scores_main_version.append([value])
            txt += f'{value}'
        elif isinstance(value, (tuple, list)):
            policy_scores_all.extend(value)
            policy_scores_main_version.append(value)
            txt += f'{" ".join(value)}'

    if len(cleaned_var_names(policy_scores_all)) != len(policy_scores_all):
        raise ValueError('Version-score names must be distinct ignoring case and must not be 0.')

    txt += '\n' + '-' * 100

    return policy_scores_main, policy_scores_all, policy_scores_main_version, txt


def split_main_treatment_pt_only(optp_: 'OptimalPolicyVersions', *,
                                 optp_main: 'OptimalPolicy',
                                 alloc_main_df: pd.DataFrame,
                                 data_df: pd.DataFrame,
                                 data_title: str,
                                 policy_scores_main_version: list[list[str]],
                                 ) -> tuple[list[dict],
                                            list[pd.DataFrame],
                                            list[str],
                                            list[bool],
                                            ]:
    """Prepare data and else for 2nd step of tree estimation within main treatments."""
    params_v, data_v, title_v, tree_yes_v = [], [], [], []
    if len(alloc_main_df) != len(data_df):
        raise ValueError('alloc_main_df and data_df must contain the same number of observations.')

    if isinstance(alloc_main_df, pd.Series):
        allocation_main = alloc_main_df.to_numpy(copy=False)
    else:
        allocation_main = alloc_main_df.iloc[:, 0].to_numpy(copy=False)

    for main_idx, versions in enumerate(policy_scores_main_version):
        no_versions = len(versions)
        if no_versions == 1:
            tree_yes_v.append(False)
            data_v.append(None)
            title_v.append(None)
            params_v.append(None)
        else:
            # Observations allocated to main treatment main_idx
            positions = np.flatnonzero(allocation_main == main_idx)
            if positions.size and no_versions > 1:
                # Estimate tree for the versions of this main treatment
                tree_yes_v.append(True)
                # Adjust the parameters to only have the scores of version for this main treatment
                params_optpol = deepcopy(optp_.version_cfg.params_optpol)
                params_optpol['var_polscore_name'] = policy_scores_main_version[main_idx]
                params_optpol['var_d_name'] = None
                # Specific values for parameters in the versions' trees (that may be different from
                # main tree)
                params_optpol['pt_depth_tree_1'] = optp_.version_cfg.depth_version_tree[main_idx]
                params_optpol['pt_depth_tree_2'] = 0   # Only single trees used for versions
                # No additional restrictions on versions given the allocation of the main treatment
                params_optpol['other_max_shares'] = [1] * no_versions
                params_optpol['other_costs_of_treat_mult'] = [1] * no_versions

                # Using cost of main treatments for all versions
                other_cost_main = optp_main.other_cfg.costs_of_treat[main_idx]
                params_optpol['other_costs_of_treat'] = [other_cost_main] * no_versions

                params_v.append(params_optpol)
                # Select only the data previously allocated to this main treatment
                data_v.append(data_df.iloc[positions])
                title_v.append(data_title + f'M: {main_idx}')
            else:
                tree_yes_v.append(False)
                data_v.append(None)
                title_v.append(None)
                params_v.append(None)

    return params_v, data_v, title_v, tree_yes_v


def combine_results_all_dic(results_all_dic_list: list[dict],
                            policy_scores_main_version,
                            *,
                            gen_cfg: 'GenCfg',
                            ) -> dict:
    """Merge the individual dictionary from main and the version estimation steps."""
    outpath = results_all_dic_list[0]['outpath']
    result_dic = []
    for results_all in results_all_dic_list:
        result_dic.append(None if results_all is None else results_all['result_dic'])

    # Create a allocations_df with unique identifiers for the treatments.
    # Create a unique numerical identifier for all treatments starting with 0
    # treatment_labels_int = []

    # iterate over main treatments, update allocation number and insert into this dataframe
    results_all_dic_main = results_all_dic_list[0]
    results_all_dic_submain = results_all_dic_list[1:]
    allocation_df = results_all_dic_main['allocation_df'].copy()
    allocation_df.iloc[:, 0] = 0   # All values set to 0, to be filled with new treatment number

    col_target = allocation_df.columns[0]
    col_main = allocation_df.columns[0]
    for sub_dic in results_all_dic_submain:
        if sub_dic is not None:
            col_submain = sub_dic['allocation_df'].columns[0]
            break

    txt_order = '\n' * 2 + '-' * 100 + '\nTreatment numbers of versions of main treatments'
    j = 0
    for main_idx, score_list in enumerate(policy_scores_main_version):
        no_versions = len(score_list)
        versions = list(range(j, j+no_versions)) # These are the relevant numbers of the treatments
        # treatment_labels_int.append(versions)
        txt_order += f'\n Main id: {main_idx}   Overall ids: {" ".join([str(s) for s in versions])}'
        j += no_versions
        if results_all_dic_submain[main_idx] is None:
            mask = results_all_dic_main['allocation_df'][col_main].eq(int(main_idx))

            if len(versions) > 1:
                if mask.any():
                    raise ValueError('No treatment-version allocation is available although '
                                     'observations were allocated to a main treatment with '
                                     'multiple versions.'
                                     )
                continue

            allocation_df.loc[mask, col_target] = versions[0]
        else:
            v_df = results_all_dic_submain[main_idx]['allocation_df']
            allocation_df.loc[v_df.index, col_target] = (v_df[col_submain].to_numpy(copy=True)
                                                         + versions[0]
                                                         )
    if gen_cfg.with_output:
        print_mcf(gen_cfg, txt_order, summary=True)

    results_all_dic = {'allocation_df': allocation_df,
                       'result_dic': result_dic,
                       'outpath': outpath
                       }
    return results_all_dic


def get_optp_for_eval_pt(optp_version_: 'OptimalPolicyVersions',
                         data_df: pd.DataFrame,
                         ) -> tuple['OptimalPolicy', pd.DataFrame]:
    """Create new instance of OptimalPolicyVersions for Policies Trees to be used in evaluate."""
    # Take a copy of the instance for main treatments to adjust to get final version of instance
    optp = deepcopy(optp_version_.optp[0])
    # checksum for allocation based on training data
    optp.report['training_alloc_chcksm'] = optp_version_.report['training_alloc_chcksm']
    # Delete treee as it is not needed (and contains only main tree anyway)
    optp.pt_cfg.policy_tree = None

    optp.var_cfg.polscore_name = [name.casefold() for name in optp_version_.policy_scores_all]
    optp.gen_cfg.no_of_treat = len(optp_version_.policy_scores_all)
    optp.gen_cfg.d_values = [int(d) for d in range(optp.gen_cfg.no_of_treat)]
    _, _, scores_main_version, _ = analyse_treatment_version(
        optp_version_.version_cfg.policyscores_dict
        )
    version_counts = [len(scores) for scores in scores_main_version]
    if any(count > 1 for count in version_counts):
        optp.var_cfg.d_name = None
    costs_main = np.asarray(optp.other_cfg.costs_of_treat, dtype=float)
    optp.other_cfg.costs_of_treat = np.repeat(costs_main, version_counts).tolist()

    # Main-treatment shares cannot be used for the expanded treatment versions.
    # Main-treatment shares cannot be used for the expanded treatment versions.
    optp.rnd_cfg.shares = None
    optp.rnd_cfg.shares_user_specified = False
    if (var_available(optp_version_.version_cfg.d_name, data_df.columns, needed='nice_to_have')
            and len(optp_version_.version_cfg.d_name) == 2
            ):
        data_df, d_agg_name = get_full_treatment_numbers_training_data(
            optp_version_.version_cfg.d_name,
            data_df,
            optp_version_.version_cfg.policyscores_dict,
            )
        optp.var_cfg.d_name = [d_agg_name]

    return optp, data_df


def approx_constant_range(x: list[float], *, atol=1e-8) -> bool:
    """Check if elements in list are approximately equal."""
    return (not x) or (max(x) - min(x) <= atol)


def get_optp_for_eval_others(optp_version_: 'OptimalPolicyVersions',
                             data_df: pd.DataFrame,
                             ) -> tuple['OptimalPolicy', pd.DataFrame]:
    """Create new instance of OptimalPolicyVersions for Policies Trees to be used in evaluate."""
    # Take a copy of the instance for main treatments to adjust to get final version of instance
    optp = deepcopy(optp_version_.optp)
    # checksum for allocation based on training data
    optp.report['training_alloc_chcksm'] = optp_version_.report['training_alloc_chcksm']

    if (var_available(optp_version_.version_cfg.d_name, data_df.columns, needed='nice_to_have')
            and len(optp_version_.version_cfg.d_name) == 2):
        data_df, d_agg_name = get_full_treatment_numbers_training_data(
            optp_version_.version_cfg.d_name,
            data_df,
            optp_version_.version_cfg.policyscores_dict,
            )
        optp.var_cfg.d_name = [d_agg_name]

    return optp, data_df


def get_full_treatment_numbers_training_data(d_name: list[str],
                                             data_df: pd.DataFrame,
                                             policyscores_dict: dict,
                                             ) -> tuple[pd.DataFrame, str]:
    """Translate main-version information into a unique treatment number."""
    _, _, policy_scores_main_version, _ = analyse_treatment_version(policyscores_dict)
    version_counts = np.asarray(
        [len(scores) for scores in policy_scores_main_version], dtype=np.int64
        )
    if np.any(version_counts < 1):
        raise ValueError('Every main treatment must contain at least one treatment version.')

    try:
        treatment_data = data_df[d_name].to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('Main-treatment and version values must be numeric.') from exc

    treatment_rounded = np.rint(treatment_data)
    if (not np.all(np.isfinite(treatment_data))
            or not np.allclose(treatment_data, treatment_rounded, rtol=0, atol=1e-8)
            ):
        raise ValueError('Main-treatment and version values must be finite integers.')

    main_values = treatment_rounded[:, 0].astype(np.int64)
    version_values = treatment_rounded[:, 1].astype(np.int64)

    invalid_main = (main_values < 0) | (main_values >= len(version_counts))
    if np.any(invalid_main):
        unknown_values = np.unique(main_values[invalid_main]).tolist()
        raise ValueError(f'Unknown main-treatment values: {unknown_values}')

    invalid_version = (
        (version_values < 0)
        | (version_values >= version_counts[main_values])
        )
    if np.any(invalid_version):
        unknown_pairs = np.column_stack(
            (main_values[invalid_version], version_values[invalid_version])
            ).tolist()
        raise ValueError('Unknown main-treatment/version combinations: '
                         f'{unknown_pairs}'
                         )

    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(version_counts[:-1]))
        )
    aggregate_treatment = offsets[main_values] + version_values

    aggregate_name = ''.join(d_name)
    existing_names = {name.casefold() for name in data_df.columns}
    while aggregate_name.casefold() in existing_names:
        aggregate_name += '_'
    data_new_df = data_df.copy(deep=False)
    data_new_df[aggregate_name] = aggregate_treatment

    return data_new_df, aggregate_name

def solve_main_para_pt(params_optpol: dict,
                       policy_scores_main: list[str]
                       ) -> dict:
    """Get the parameters for the mcf estimation of the main treatments (policy tree)."""
    # Get the parameters to initialse the instance for the main treatments
    if len(cleaned_var_names(policy_scores_main)) != len(policy_scores_main):
        raise ValueError('Main-score names must be distinct ignoring case and must not be 0.')
    params_optpol_new = deepcopy(params_optpol)
    params_optpol_new['var_polscore_name'] = policy_scores_main

    return params_optpol_new


def adjust_attributes_version(optp_: 'OptimalPolicy', gen_cfg_print: Any) -> 'OptimalPolicy':
    """Change some attributes in the instance of the OptimalPolicy class."""
    if gen_cfg_print.with_output:
        old_outpath = deepcopy(optp_.gen_cfg.outpath)
        optp_.gen_cfg.outpath = deepcopy(gen_cfg_print.outpath)   # Path object
        optp_.gen_cfg.outfiletext = deepcopy(gen_cfg_print.outfiletext)   # Path object
        optp_.gen_cfg.outfilesummary = deepcopy(gen_cfg_print.outfilesummary)   # Path object
        # Delete redundant directory
        if old_outpath.resolve() != optp_.gen_cfg.outpath.resolve():  # old and new path different
            delete_path_or_file_if_exists(old_outpath)

    return optp_


def prepare_version_for_solve(optp_: 'OptimalPolicy',
                              gen_cfg_print: Any,
                              data_v_idx: pd.DataFrame,
                              ) -> tuple['OptimalPolicy', pd.Index, pd.DataFrame, str]:
    """Prepare the instance and data for solving a treatment-version tree."""
    optp_ = adjust_attributes_version(optp_, gen_cfg_print)

    data_version = data_v_idx.copy()
    old_index = data_version.index.copy()

    row_name = '_mcf_version_row'
    while row_name in data_version.columns:
        row_name += 'x'

    data_version[row_name] = np.arange(len(data_version), dtype=np.int64)
    data_version = data_version.reset_index(drop=True)

    return optp_, old_index, data_version, row_name


def print_title_for_version_tree(idx: int,
                                 tree_yes_v: bool,
                                 score_name: list[str],
                                 gen_cfg_print: Any
                                 ) -> str:
    """Print title for the version tree."""
    if gen_cfg_print.with_output:
        txt = '\n' + '-' * 100
        if tree_yes_v:
            txt +=  '\nPolicy tree '
        else:
            txt += '\nNo treatment versions '
        txt += f'for main treatment {idx} ({" ".join(score_name)})' + '\n' + '-' * 100
        print_mcf(gen_cfg_print, txt, summary=True)

        return txt

    return ''


# The following functions are used by the allocate method$
def split_by_treatment(alloc_df: pd.DataFrame,
                       data_df: pd.DataFrame, *,
                       alloc_name: str,
                       no_of_treat: int,
                       ) -> list[tuple[pd.DataFrame, np.ndarray] | None]:
    """Split data_df according to values in alloc_df[alloc_name]."""
    if len(alloc_df) != len(data_df):
        raise ValueError('alloc_df and data_df must contain the same number of observations.')

    allocation = alloc_df[alloc_name].to_numpy(copy=False)
    split_list: list[tuple[pd.DataFrame, np.ndarray] | None] = []

    for treatment in range(no_of_treat):
        positions = np.flatnonzero(allocation == treatment)
        if positions.size:
            split_list.append((data_df.iloc[positions], positions))
        else:
            split_list.append(None)

    return split_list


def allocate_version(optp_version: 'OptimalPolicyVersions',
                     data_df: pd.DataFrame,
                     data_title: str = '',
                     ) -> dict[str, pd.DataFrame | Path | None]:
    """Allocate new observations to optimal treatment when there are treatment versions."""
    start_time = time()
    if optp_version.version_cfg.params_optpol['gen_method'] == 'policy_tree':
        (_, optp_version.policy_scores_all, policy_scores_main_version, txt_descr
         ) = analyse_treatment_version(optp_version.version_cfg.policyscores_dict)
        if optp_version.optp[0].gen_cfg.with_output:
            txt_descr += tree_main_header(line_length=100)
            print_mcf(optp_version.optp[0].gen_cfg, txt_descr, summary=True)
        # Step 1: Allocate main treatments
        if optp_version.optp[0].gen_cfg.with_output:
            print_mcf(optp_version.optp[0].gen_cfg, '\n' * 2 + 'MAIN TREATMENT', summary=True)
        allocation_df_main, outpath = allocate_method(optp_version.optp[0],
                                                      data_df,
                                                      data_title=data_title,
                                                      )
        # Split data corresponding the main treatment value
        alloc_name = allocation_df_main.columns[0]
        no_treat_main = len(policy_scores_main_version)
        data_df_split_main = split_by_treatment(allocation_df_main,
                                                data_df,
                                                alloc_name=alloc_name,
                                                no_of_treat=no_treat_main,
                                                )
        txt_order = '\n' * 2 + '-' * 100 + '\nTreatment numbers of versions of main treatments'
        j = 0
        allocation_df = allocation_df_main.copy()
        allocation_df.iloc[:, 0] = 0   # All values set to 0 -> get new treatment numbers
        allocation_col = allocation_df.columns.get_loc(alloc_name)
        for main_idx, score_list in enumerate(policy_scores_main_version):
            # Extract part of data_df that corresponds to particular main treatment value
            if optp_version.optp[0].gen_cfg.with_output:
                print_mcf(optp_version.optp[0].gen_cfg,
                          '\n' * 2 + f'Versions of MAIN TREATMENT {main_idx}',
                          summary=True
                          )
            no_versions = len(score_list)
            versions = list(range(j, j+no_versions)) # Relevant numbers of the treatments
            txt_order += (f'\n Main id: {main_idx}   Overall ids: '
                          f'{" ".join([str(s) for s in versions])}'
                          )
            split_main = data_df_split_main[main_idx]
            if split_main is None:                    # No allocations to that main treatment
                j += no_versions
                continue

            data_versions_df, positions = split_main
            if no_versions == 1:                      # No treatment versions
                allocation_df.iloc[positions, allocation_col] = j
                j += 1
                continue

            data_title_version = data_title + f'Main{main_idx}'
            allocation_df_ver, _ = allocate_method(optp_version.optp[main_idx+1],
                                                   data_versions_df.reset_index(drop=True),
                                                   data_title=data_title_version,
                                                   )
            alloc_v_name = allocation_df_ver.columns[0]
            allocation_df.iloc[positions, allocation_col] = (
                allocation_df_ver[alloc_v_name].to_numpy(copy=False) + versions[0]
                )
            j += no_versions

        if optp_version.optp[0].gen_cfg.with_output:
            print_mcf(optp_version.optp[0].gen_cfg, txt_order, summary=True)
    else:
        allocation_df, outpath = allocate_method(optp_version.optp,
                                                 data_df,
                                                 data_title=data_title,
                                                 )
    results_dic = {'allocation_df': allocation_df,
                   'outpath': outpath,
                   }
    # Timing
    key, time_str = timestr_version(optp_version.gen_cfg_print,
                                    start_time,
                                    title='Allocation of new observations',
                                    data_title=data_title
                                    )
    optp_version.time_strings[key] = time_str

    return results_dic


def evaluate_version(optp_version: 'OptimalPolicyVersions',
                     allocation_df: pd.DataFrame,
                     data_df: pd.DataFrame,
                     data_title: str = '',
                     seed: int = 1234
                     ) -> dict:
    """Evaluate allocations with treatment versions."""
    start_time = time()
    data_df = data_df.copy(deep=False)
    data_df.columns = [name.casefold() for name in data_df.columns]
    if optp_version.version_cfg.params_optpol['gen_method'] == 'policy_tree':
        optp_eval, data_df = get_optp_for_eval_pt(optp_version, data_df)
    else:
        optp_eval, data_df = get_optp_for_eval_others(optp_version, data_df)

    results_dic, outpath = evaluate_method(optp_eval,
                                           allocation_df,
                                           data_df,
                                           data_title=data_title,
                                           seed=seed,
                                           optp_version=optp_version,
                                           )
    results_all_dic = {'results_dic': results_dic,
                       'outpath': outpath,
                       }
    # Timing
    key, time_str = timestr_version(optp_version.gen_cfg_print,
                                    start_time,
                                    title='Evaluation',
                                    data_title=data_title
                                    )
    optp_version.time_strings[key] = time_str

    return results_all_dic


def tree_main_header(line_length: int = 100) -> str:
    """Create header for display of main treatment policy tree."""
    return ('\n' + '-' * line_length + '\nPolicy tree for main treatments'
            + '\n' + '- ' * int(line_length/2)
            )


def timestr_version(gen_cfg_print: Any,
                    time_start: float,
                    title: str='',
                    data_title: str = ''
                    ) -> tuple[str, str]:
    """Get the time string formatted and perhaps printed."""
    total_str_length = 50
    time_str = f'Time for {title}:'
    time_name = [time_str + ' ' * (total_str_length - len(time_str))]

    time_difference = [time() - time_start]
    if gen_cfg_print.with_output:
        time_str = print_timing(gen_cfg_print, title, time_name, time_difference, summary=True)
    else:
        time_str = ''
    key = title + data_title

    return key, time_str
