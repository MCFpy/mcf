"""
Created on Mon Nov 24 13:53:14 2025.

@author: MLechner

 -*- coding: utf-8 -*-
"""
from numbers import Real
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from mcf import mcf_init_values_cfg_functions as mcf_initvals
from mcf.mcf_print_stats_functions import print_dic
from mcf.mcf_general_sys import define_outpath

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def change_keywords(mcf_: 'ModifiedCausalForest',
                    estimator: str,
                    predict_analyse: str,
                    new_keywords: dict | Any,
                    ) -> None:
    """Change attributes of *_cfg in mcf class for predict(s) & analyse."""
    iv = estimator == 'iv'
    analysis = predict_analyse == 'analyse'
    prediction = predict_analyse == 'predict'

    check_if_good_instance(mcf_.instance_used_for_training,
                           mcf_.instance_used_for_prediction,
                           prediction, analysis,
                           )
    check_if_dictionary(new_keywords)

    # Iterate in the user provided dictionary
    for key, value in new_keywords.items():
        mcf_ = change_method_for_prediction_and_analysis(mcf_, key, value,
                                                         iv=iv)
    if mcf_.gen_cfg.with_output:
        (mcf_.gen_cfg.outfiletext, mcf_.gen_cfg.outfilesummary,
         mcf_.gen_cfg.outpath, mcf_.p_cfg.paths
         ) = change_output_locations(mcf_.gen_cfg.outfiletext,
                                     mcf_.gen_cfg.outfilesummary,
                                     mcf_.gen_cfg.outpath,
                                     # mcf_.p_cfg.paths,
                                     qiate=mcf_.p_cfg.qiate,
                                     cbgate=mcf_.p_cfg.cbgate,
                                     bgate=mcf_.p_cfg.bgate,
                                     )
        print_dic(new_keywords, 'new_keywords', mcf_.gen_cfg, summary=True)


def change_output_locations(outfile: Path,
                            outfilesummary: Path,
                            outpath: Path,
                            # paths: dict[str, Path],
                            qiate: bool = False,
                            cbgate: bool = False,
                            bgate: bool = False,
                            ) -> tuple[Path, Path, Path, dict[str, Path]]:
    """Change the output location to avoid overwriting old results."""
    outpath_n = define_outpath(outpath, new_outpath=True)
    outfilesummary_n = outpath_n / outfilesummary.name
    outfile_n = outpath_n / outfile.name

    paths_n = mcf_initvals.get_directories_for_output(outpath_n,
                                                      True,
                                                      qiate=qiate,
                                                      cbgate=cbgate,
                                                      bgate=bgate,
                                                      )
    # for key, values in paths.items():
    #     # paths_n[key] = outpath_n / values.name
    #     paths_n[key] = outpath_n / values.relative_to(outpath)
    return outfile_n, outfilesummary_n, outpath_n, paths_n


def change_method_for_prediction_and_analysis(mcf_: 'ModifiedCausalForest',
                                              key: Any,
                                              value: Any,
                                              iv: bool,
                                              ):
    """Change configuration files for prediction and analysis."""
    check_if_string(key)
    check_if_value_is_none(key, value)
    x_ord_names = mcf_.var_cfg.x_name_ord
    x_unord_names = mcf_.var_cfg.x_name_unord

    match key:
        # predict(), predict_iv()
        # Variable names (they must already be used in training)
        case 'var_x_name_balance_test_ord':
            mcf_.var_cfg.x_name_balance_test_ord = check_valid_names(
                key, value, valid_names_ord=x_ord_names,
                )
        case 'var_x_name_balance_test_unord':
            mcf_.var_cfg.x_name_balance_test_unord = check_valid_names(
                key, value,
                valid_names_ord=None,
                valid_names_unord=x_unord_names,
                )
        case 'var_x_name_balance_bgate':
            mcf_.var_cfg.x_name_balance_bgate = check_valid_names(
                key, value,
                valid_names_ord=x_ord_names,
                valid_names_unord=x_unord_names,
                )
        case 'var_x_name_ba':
            mcf_.var_cfg.x_name_ba = check_valid_names(
                key, value,
                valid_names_ord=x_ord_names,
                valid_names_unord=x_unord_names,
                )
        case 'var_z_name_ord':
            mcf_.var_cfg.z_name_ord = check_valid_names(
                key, value,
                valid_names_ord=x_ord_names,
                valid_names_unord=None,
                )
        case 'var_z_name_unord':
            mcf_.var_cfg.z_name_unord = check_valid_names(
                key, value,
                valid_names_ord=None,
                valid_names_unord=x_unord_names,
                )
        # Limited flexibility for bias adjustment scores and variable may be
        # needed (and additionally checked) otherwise.
        case 'p_ba':
            mcf_.p_ba_cfg.yes = change_bool_one_way(key, value, False)
        case 'p_ba_adj_method':
            mcf_.p_ba_cfg.adj_method = mcf_initvals.p_ba_adjust_method(value)
        case 'p_ba_pos_weights_only':
            mcf_.p_ba_cfg.pos_weights_only = change_bool_both_ways(key, value)
        case 'p_ba_use_x':
            mcf_.p_ba_cfg.use_x = change_bool_both_ways(key, value)
        case 'p_ba_use_prop_score':
            mcf_.p_ba_cfg.use_prop_score = change_bool_one_way(key, value, False
                                                               )
        case 'p_ba_use_prog_score':
            mcf_.p_ba_cfg.use_prog_score = change_bool_one_way(key, value, False
                                                               )
        # Parameters to predict (and their standard errors)
        case 'p_ate_no_se_only':
            mcf_.p_cfg.ate_no_se_only = change_bool_both_ways(key, value)
        case 'p_atet':
            mcf_.p_cfg.atet = change_bool_both_ways(key, value)
        case 'p_gatet':
            mcf_.p_cfg.gatet = change_bool_both_ways(key, value)
        case 'p_bgate':
            mcf_.p_cfg.bgate = change_bool_both_ways(key, value)
        case 'p_cbgate':
            mcf_.p_cfg.cbgate = change_bool_both_ways(key, value)
        case 'p_iate':
            mcf_.p_cfg.iate = change_bool_both_ways(key, value)
        case 'p_iate_se':
            mcf_.p_cfg.iate_se = change_bool_both_ways(key, value)
        case 'p_iate_m_ate':
            mcf_.p_cfg.iate_m_ate = change_bool_both_ways(key, value)

        case 'p_bgate_sample_share':
            mcf_.p_cfg.bgate_sample_share = value_pos_max(key, value, maxv=1)
        case 'p_gates_minus_previous':
            mcf_.p_cfg.gates_minus_previous = change_bool_both_ways(key, value)
        case 'p_gates_smooth_bandwidth':
            mcf_.p_cfg.gates_smooth_bandwidth = value_pos_max(key, value,
                                                              maxv=float('inf'))
        case 'p_gates_smooth':
            mcf_.p_cfg.gates_smooth = change_bool_both_ways(key, value)
        case 'p_gates_smooth_no_evalu_points':
            mcf_.p_cfg.gates_smooth_no_evalu_points = value_integer(
                key, value, minv=2, maxv=100000
                )
        case 'p_gates_no_evalu_points':
            mcf_.p_cfg.gate_no_evalu_points = value_integer(
                key, value, minv=2, maxv=100000
                )

        case 'p_qiate':
            mcf_.p_cfg.qiate = change_bool_one_way(key, value, False)
        case 'p_qiate_se':
            mcf_.p_cfg.qiate_se = change_bool_one_way(key, value, False)
        case 'p_qiate_m_mqiate':
            mcf_.p_cfg.qiate_m_mqiate = change_bool_both_ways(key, value)
        case 'p_qiate_m_opp':
            mcf_.p_cfg.qiate_m_mqiate = change_bool_both_ways(key, value)
        case 'p_qiate_no_of_quantiles':
            mcf_.p_cfg.qiate_no_of_quantiles = value_integer(
                key, value, minv=10, maxv=float('inf')
                )
        case 'p_qiate_smooth':
            mcf_.p_cfg.qiate_smooth = change_bool_both_ways(key, value)
        case 'p_qiate_smooth_bandwidth':
            mcf_.p_cfg.qiate_smooth_bandwidth = value_pos_max(key, value,
                                                              maxv=float('inf'))
        case 'p_qiate_bias_adjust':
            mcf_.p_cfg.qiate_bias_adjust = change_bool_both_ways(key, value)

        case 'p_bt_yes':
            mcf_.p_cfg.bt_yes = change_bool_one_way(key, value, False)

        case 'p_choice_based_sampling':
            mcf_.p_cfg.choice_based_sampling = change_bool_both_ways(key, value)
        case 'p_choice_based_probs':
            mcf_.p_cfg.choice_based_probs = check_cb_probs(
                key, value, mcf_.gen_cfg.no_of_treat
                )
        case 'p_cond_var':
            mcf_.p_cfg.cond_var = change_bool_both_ways(key, value)
        case 'p_knn':
            mcf_.p_cfg.knn = change_bool_both_ways(key, value)
        case 'p_knn_const':
            mcf_.p_cfg.knn_const = value_pos_max(key, value, maxv=float('inf'))
        case 'p_knn_min_k':
            mcf_.p_cfg.knn_min_k = value_integer(key, value, minv=1,
                                                 maxv=float('inf'))
        case 'p_nw_bandw':
            mcf_.p_cfg.nw_bandw = value_pos_max(key, value, maxv=float('inf'))
        case 'p_nw_kern':
            mcf_.p_cfg.nw_kern = int_valid_values(key, value, (1, 2,))
        case 'p_ci_level':
            mcf_.p_cfg.ci_level = value_pos_max(key, value, maxv=0.9999999999)

        case 'p_iv_aggregation_method':
            if iv:
                mcf_.p_cfg.iv_aggregation_method = (
                    mcf_initvals.iv_agg_method_fct(value))
            else:
                raise ValueError(f'{key} requires IV estimation.')

        case 'p_se_boot_ate':
            mcf_.p_cfg.se_boot_ate = check_bootstrap_se(key, value)
        case 'p_se_boot_gate':
            mcf_.p_cfg.se_boot_gate = check_bootstrap_se(key, value)
        case 'p_se_boot_iate':
            mcf_.p_cfg.se_boot_iate = check_bootstrap_se(key, value)
        case 'p_se_boot_qiate':
            mcf_.p_cfg.se_boot_qiate = check_bootstrap_se(key, value)

        case 'gen_output_type':
            mcf_.p_cfg.nw_kern = int_valid_values(key, value, (0, 1, 2,))

        # analyse()
        case 'post_bin_corr_threshold':
            mcf_.post_cfg.bin_corr_threshold = value_pos_max(key, value, maxv=1)
        case 'post_bin_corr_yes':
            mcf_.post_cfg.bin_corr_yes = change_bool_both_ways(key, value)
        case 'post_est_stats':
            mcf_.post_cfg.est_stats = change_bool_both_ways(key, value)

        case 'post_kmeans_yes':
            mcf_.post_cfg.kmeans_yes = change_bool_both_ways(key, value)
        case 'post_kmeans_no_of_groups':
            mcf_.post_cfg.kmeans_no_of_groups = value_integer(
                key, value, minv=2, maxv=100
                )
        case 'post_kmeans_max_tries':
            mcf_.post_cfg.kmeans_max_tries = value_integer(
                key, value, minv=1, maxv=10000
                )
        case 'post_kmeans_min_size_share':
            mcf_.post_cfg.kmeans_min_size_share = value_pos_max(key, value,
                                                                maxv=33)
        case 'post_kmeans_replications':
            mcf_.post_cfg.kmeans_replications = value_integer(
                key, value, minv=1, maxv=1000
                )
        case 'post_kmeans_single':
            mcf_.post_cfg.kmeans_single = change_bool_both_ways(key, value)

        case 'post_random_forest_vi':
            mcf_.post_cfg.random_forest_vi = change_bool_both_ways(key, value)
        case 'post_relative_to_first_group_only':
            mcf_.post_cfg.relative_to_first_group_only = change_bool_both_ways(
                key, value
                )
        case 'post_plots':
            mcf_.post_cfg.plots = change_bool_both_ways(key, value)
        case 'post_tree':
            mcf_.post_cfg.tree = change_bool_both_ways(key, value)

        case _:
            not_a_valid_key(key)

    return mcf_


def check_if_bool(key: str, value: Any):
    """Check if value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f'{key} must be Boolean.')


def check_if_number(key: str, value: Any):
    """Check if value is a number."""
    if not isinstance(value, Real):
        raise ValueError(f'{key} must be a number.')


def check_if_string(key: Any):
    """Check if input is a string."""
    if not isinstance(key, str):
        raise ValueError('Key of dictionary used to change parameters for '
                         'prediction analysis must be specified as string.'
                         )


def check_if_dictionary(new_keywords: Any) -> None:
    """Check if user input is a dictionary."""
    if not isinstance(new_keywords, dict):
        raise ValueError('Keyword "new_keywords" must be passed as dictionary.'
                         )


def check_if_good_instance(used_for_training: bool,
                           used_for_prediction: bool,
                           prediction: bool,
                           analysis: bool,
                           ) -> None:
    """Check use of MCF instance in prediction & analysis before."""
    if prediction:
        if not used_for_training:
            raise ValueError('ModifiedCausalForest instance has not yet '
                             'used for training. '
                             'Train forest before using it for prediction. '
                             )
        if used_for_prediction:
            raise ValueError('ModifiedCausalForest instance has already been '
                             'used for prediction. '
                             'Use instance used in training but not yet in '
                             'prediction (deepcopy trained instance if same '
                             'forest is used in several predictions. '
                             )
    if analysis:
        if not used_for_prediction:
            raise ValueError('ModifiedCausalForest instance must be used for '
                             'prediction before the "analyse" method can be '
                             'used. '
                             )


def not_a_valid_key(key: Any) -> None:
    """Raise exception when key is not among valid keys."""
    raise ValueError(f'{key} cannot be changed '
                     'in the predict method (or is not a valid keyword). '
                     )


def check_if_value_is_none(key: str, value: None) -> None:
    """Check if value is a None, which is not allowed."""
    if value is None:
        raise ValueError(f'"None" is specified for {key}. Use a valid value '
                         'instead (other than None). '
                         )


def check_valid_names(
        key: str,
        names: Any,
        valid_names_ord: list[str] | tuple[str,...] | None = None,
        valid_names_unord: list[str] | tuple[str,...] | None = None,
        ) -> list[str]:
    """Check if value is list or tuple."""
    # List or tuple?
    if not isinstance(names, (list, tuple,)):
        raise ValueError(f'{key} must be specified as list or tuple.')
    # Strings inside?
    if any(not isinstance(name, str) for name in names):
        raise ValueError('All variable names must be specified as strings.')

    if valid_names_ord is None:
        valid_names_ord = []
    if valid_names_unord is None:
        valid_names_unord = []

    # Names already used in training?
    names = [name.casefold() for name in names]
    names_ord = [name for name in names if name in valid_names_ord]
    names_unord_prime = [name + '_prime' for name in names
                         if name + '_prime' in valid_names_unord
                         ]
    names_unord_org = [name for name in names
                       if name + '_prime' in valid_names_unord
                       ]
    names_new = [*names_ord, *names_unord_prime,]
    names_selected_org = [*names_ord, *names_unord_org,]
    invalid_names = [name for name in names if name not in names_selected_org]
    if invalid_names:
        valid_names = [*valid_names_ord, *valid_names_unord]
        raise ValueError(f'Names specified for {key} must be among '
                         f'{" ".join(valid_names)}. This is not true for '
                         f'{" ".join(invalid_names)}.'
                         )
    return names_new


def change_bool_one_way(key: str, value: Any, target_ok: bool) -> bool:
    """Change value of a boolen in only one direction."""
    check_if_bool(key, value)
    if value != target_ok:
        raise ValueError(f'{key} can only be changed to {target_ok}')

    return value

def change_bool_both_ways(key: str, value: Any) -> bool:
    """Change value of a boolen in only one direction."""
    check_if_bool(key, value)

    return value


def value_pos_max(key: str, value: Any, maxv=1) -> Real:
    """Check if we have a value between 0 and maxv."""
    check_if_number(key, value)
    if not 0 < value <= maxv:
        raise ValueError(f'Value of {key} must be positive and not larger '
                         f'than {maxv}.'
                         )
    return value


def value_integer(key: str,
                  value: Any,
                  minv: Real = 1,
                  maxv: Real = 100000
                  ) -> int:
    """Check if integer is in a certain range."""
    check_if_number(key, value)
    value = round(value)
    if not minv <= value <= maxv:
        raise ValueError(f'Value of {key} must be between {minv} and {maxv}. '
                         f'Specified value is {value}'
                         )
    return value


def check_cb_probs(key: str, probs: Any, no_of_treat: int) -> list[np.floating]:
    """Check probabilities for choice based sampling."""
    if not isinstance(probs, (list, tuple,)):
        raise ValueError(f'{key} must be specified as list or tuple.')
    if any(not isinstance(prob, (Real, np.floating)) for prob in probs):
        raise ValueError('All variable names must be specified as numbers.')
    if len(probs) != no_of_treat:
        raise ValueError(f'{key}: Number of probabilites is not equal to '
                         'number of treatments.'
                         )
    return mcf_initvals.p_cb_normalize(probs, no_of_treat)


def int_valid_values(key: str, value: Any, valid_vals: list | tuple) -> int:
    """Check if integer are within a given tuple of values."""
    if not isinstance(value, Real):
        raise ValueError(f'{key} must be specified as number.')

    value = round(value)
    if not round(value) in valid_vals:
        raise ValueError(f'{key} must have one of the following values: '
                         f'{" ".join(valid_vals)}'
                         )
    return value


def check_bootstrap_se(key: str, value: Any):
    """Set bootstrap values."""
    if not isinstance(value, (bool, Real)):
        raise ValueError(f'{key} must be boolean or a positive number.')

    if value is False:
        return value

    if value is True:
        return 99

    if value > 5:
        return round(value)

    raise ValueError(f'{key} must be larger than 5 (or False.')
