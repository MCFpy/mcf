"""Created on Mon Feb 24 10:08:06 2025.

Contains specific functions for unconfoundedness.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import copy, deepcopy
from time import time
from typing import TYPE_CHECKING

import pandas as pd

from mcf import mcf_full_sequential_weights as mcf_fsw
from mcf import mcf_data
from mcf import mcf_init_update_helper as mcf_init_update
from mcf import mcf_post
from mcf import mcf_print_stats as mcf_ps
from mcf.mcf_ate import ate_effects_print
from mcf.mcf_common_support import common_support_p_score
from mcf.mcf_effect_helpers import TimeState, inst_eff_flags, inst_effect_dicts, aggregate_effects
from mcf.mcf_feature_selection import feature_selection
from mcf.mcf_forest import train_forest
from mcf.mcf_gateout import gate_effects_print, get_names_values
from mcf.mcf_general import check_reduce_dataframe
from mcf.mcf_iate import iate_effects_print
from mcf.mcf_local_centering import local_centering
from mcf.mcf_qiate import qiate_effects_print
from mcf.mcf_versions import d_dictionary
from mcf.mcfoptp_parallel_backend_ray_classical import check_ray_shutdown

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def train_main(mcf_: 'ModifiedCausalForest',
               data_df: pd.DataFrame,
               exit_after_commonsupport: bool = False,
               ) -> dict:
    """
    Build the modified causal forest on the training data.

    Parameters
    ----------
    mcf_c: instance of ModifiedCausalForest
    data_df : DataFrame
        Data used to compute the causal forest. It must contain information about outcomes,
        treatment, and features.

    exit_after_commonsupport : Boolean, optional
        Programme exits once the common support is determined. This is useful to determine common
        support cut-offs only (that can subsequently be used by the predict method).

    Returns
    -------
    results : Dictionary.
        Contains the results. This dictionary has the following structure:

        ...  : Values and names of the effects

        'inputdata_on_support' : DataFrame or None
            Data that are inside common support.

        'common_support_probabilities': DataFrame or None
            Treatment probabilities for all treatments, the identifier of
            the observation, and a dummy variable indicating whether the observation is inside
            or outside the common support. This is for the data used to build the trees.
            None if _int_with_output is False or no common support enforcement.

        'path_output' : Pathlib object
            Location of directory in which output is saved.

    """
    time_start = time()

    if not isinstance(exit_after_commonsupport, bool):
        raise TypeError('Keyword "exit_after_commonsupport" must be Boolean.')

    # Reduce sample size to upper limit
    data_df, rnd_reduce, txt_red = check_reduce_dataframe(data_df,
                                                          title='Training',
                                                          max_obs=mcf_.int_cfg.max_obs_training,
                                                          seed=124535, ignore_index=True,
                                                          )
    if rnd_reduce and mcf_.gen_cfg.with_output:
        mcf_ps.print_mcf(mcf_.gen_cfg, txt_red, summary=True)

    # Check treatment data
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)
    # Initialise again with data information. Order of the following
    # init functions important. Change only if you know what you do.
    mcf_init_update.var_update_train(mcf_, data_df)
    mcf_init_update.gen_update_train(mcf_, data_df)
    mcf_init_update.ct_update_train(mcf_, data_df)
    mcf_init_update.cs_update_train(mcf_)
    mcf_init_update.cf_update_train(mcf_, data_df)
    mcf_init_update.lc_update_train(mcf_, data_df)
    mcf_init_update.int_update_train(mcf_)
    mcf_init_update.p_update_train(mcf_)
    if mcf_.gen_tv_cfg.yes:
        mcf_init_update.gen_tv_update_train(mcf_, data_df)
    if mcf_.low_mem_cfg.yes:
        mcf_init_update.low_mem_update_train(mcf_, len(data_df))

    if mcf_.gen_cfg.with_output:
        mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False)

    # Prepare data: Add and recode variables for GATES (Z)
    #             Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(mcf_, data_df, train=True)

    if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
        mcf_data.print_prime_value_corr(mcf_.data_train_dict, mcf_.gen_cfg, summary=False)
    # Clean data and remove missings and unncessary variables
    if mcf_.dc_cfg.clean_data:
        data_df, report = mcf_data.clean_data(mcf_, data_df, train=True)
        if mcf_.gen_cfg.with_output:
            mcf_.report['training_obs'] = report

    if mcf_.dc_cfg.screen_covariates:   # Only training
        (mcf_.gen_cfg, mcf_.var_cfg, mcf_.var_x_type, mcf_.var_x_values,report
         ) = mcf_data.screen_adjust_variables(mcf_, data_df)
        if mcf_.gen_cfg.with_output:
            mcf_.report['removed_vars'] = report
    time_1 = time()

    # Descriptives by treatment
    if mcf_.int_cfg.descriptive_stats and mcf_.gen_cfg.with_output:
        mcf_ps.desc_by_treatment(mcf_, data_df, summary=False, stage=1,
                                 subtreatments=mcf_.gen_tv_cfg.yes,
                                 )
    # Feature selection
    if mcf_.fs_cfg.yes:
        (data_df, report, mcf_.var_cfg, mcf_.var_x_type, mcf_.var_x_values
         ) = feature_selection(data_df,
                               boot=mcf_.cf_cfg.boot, fs_cfg=mcf_.fs_cfg, gen_cfg=mcf_.gen_cfg,
                               mcf_estimation=True, obs_bigdata=mcf_.int_cfg.obs_bigdata,
                               replication=mcf_.int_cfg.replication, var_cfg=mcf_.var_cfg,
                               var_x_type=copy(mcf_.var_x_type),
                               var_x_values=copy(mcf_.var_x_values), versions=mcf_.gen_tv_cfg.yes,
                               )
        if mcf_.gen_cfg.with_output:
            mcf_.report['fs_vars_deleted'] = report

    # Split sample for tree building and tree-filling-with-y
    tree_df, fill_y_df = mcf_data.split_sample_for_mcf(mcf_, data_df)

    del data_df
    time_2 = time()

    # Compute Common support
    if mcf_.cs_cfg.type_:
        (tree_df, fill_y_df, rep_sh_del, obs_remain, rep_fig, cs_tree_prob, cs_fill_y_prob, _
         ) = common_support_p_score(mcf_, tree_df, fill_y_df, train=True, p_score_only=False)
        if mcf_.gen_cfg.with_output:
            mcf_.report['cs_t_share_deleted'] = rep_sh_del
            mcf_.report['cs_t_obs_remain'] = obs_remain
            mcf_.report['cs_t_figs'] = rep_fig
    else:
        cs_fill_y_prob = cs_tree_prob = None

    # Descriptives by treatment on common support
    if mcf_.int_cfg.descriptive_stats and mcf_.gen_cfg.with_output:
        mcf_ps.desc_by_treatment(mcf_,
                                 pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1, subtreatments=mcf_.gen_tv_cfg.yes,
                                 )
    if mcf_.gen_tv_cfg.yes:
        mcf_.gen_tv_cfg.d_dict = d_dictionary(fill_y_df, mcf_.var_cfg.d_name,
                                              zero_tol=mcf_.int_cfg.zero_tol,
                                              )
    time_3 = time()

    if exit_after_commonsupport:
        return {'tree_df': tree_df,
                'fill_y_df': fill_y_df,
                'common_support_probabilities_tree': cs_tree_prob,
                'common_support_probabilities_fill_y': cs_fill_y_prob,
                'path_output': mcf_.gen_cfg.outpath,
                }
    # Local centering
    if mcf_.lc_cfg.yes:
        tree_df, fill_y_df, _, report = local_centering(mcf_, tree_df, fill_y_df=fill_y_df)
        if mcf_.gen_cfg.with_output:
            mcf_.report["lc_r2"] = report
    time_4 = time()

    # Train forest
    if mcf_.gen_cfg.with_output:
        mcf_ps.variable_features(mcf_, summary=False)
    mcf_.cf_cfg, mcf_.p_ba_cfg, mcf_.forest, time_vi, report = train_forest(mcf_,
                                                                            tree_df, fill_y_df
                                                                            )
    if mcf_.gen_cfg.with_output:
        mcf_.report['cf'] = report
    time_end = time()
    time_string = ['Data preparation and stats I:                   ',
                   'Feature preselection:                           ',
                   'Common support:                                 ',
                   'Local centering (recoding of Y):                ',
                   'Training the causal forest:                     ',
                   '  ... of which is time for variable importance: ',
                   '\nTotal time training:                            '
                   ]
    time_difference = [time_1 - time_start, time_2 - time_1, time_3 - time_2, time_4 - time_3,
                       time_end - time_4, time_vi, time_end - time_start
                       ]
    if mcf_.gen_cfg.with_output:
        time_train = mcf_ps.print_timing(mcf_.gen_cfg, 'Training', time_string, time_difference,
                                         summary=True
                                         )
        mcf_.time_strings['time_train'] = time_train

    return {'tree_df': tree_df,
            'fill_y_df': fill_y_df,
            'common_support_probabilities_tree': cs_tree_prob,
            'common_support_probabilities_fill_y': cs_fill_y_prob,
            'path_output': mcf_.gen_cfg.outpath,
            }


def predict_main(mcf_: 'ModifiedCausalForest',
                 data_df: pd.DataFrame,
                 exit_after_commonsupport=False,
                 ) -> dict:
    """
    Compute all effects given a causal forest estimated with mcf.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the predictions. It must contain information about features
        (and treatment if effects for treatment specific subpopulations are desired as well).

    exit_after_commonsupport : Boolean, optional
        Programme exits once the common support is determined.
        This is useful to determine common support cut-offs only.

    Returns
    -------
    results : Dictionary.
        Results. This dictionary has the following structure:
        'ate': ATE, 'ate_se': Standard error of ATE.
            None if exit_after_commonsupport is False.
        'ate_effect_list': List of names of estimated effects.
            None if exit_after_commonsupport is False.
        'gate': GATE, 'gate_se': SE of GATE.
            None if exit_after_commonsupport is False.
        'gate_diff': GATE minus ATE.
            None if exit_after_commonsupport is False.
        'gate_diff_se': Standard error of GATE minus ATE.
            None if exit_after_commonsupport is False.
        'cbgate': cbGATE (all covariates balanced).
            None if exit_after_commonsupport is False.
        'cbgate_se': Standard error of CBGATE.
            None if exit_after_commonsupport is False.
        'cbgate_diff': CBGATE minus ATE.
            None if exit_after_commonsupport is False.
        'cbgate_diff_se': Standard error of CBGATE minus ATE.
            None if exit_after_commonsupport is False.
        'bgate': BGATE (only prespecified covariates balanced).
            None if exit_after_commonsupport is False.
        'bgate_se': Standard error of BGATE.
            None if exit_after_commonsupport is False.
        'bgate_diff': BGATE minus ATE.
            None if exit_after_commonsupport is False.
        'bgate_diff_se': Standard errror of BGATE minus ATE.
            None if exit_after_commonsupport is False.
        'gate_names_values': Dictionary: Order of gates parameters, name and values of GATE effects.
            None if exit_after_commonsupport is False.
        'qiate': QIATE, 'qiate_se': Standard error of QIATEs.
            None if exit_after_commonsupport is False.
        'qiate_diff': QIATE minus QIATE at median.
            None if exit_after_commonsupport is False.
        'qiate_diff_se': Standard error of QIATE minus QIATE at median.
            None if exit_after_commonsupport is False.
        'iate_data_df': DataFrame with IATEs.
            None if exit_after_commonsupport is False.
        'iate_names_dic': Dictionary containing names of IATEs.
            None if exit_after_commonsupport is False.
        'bala': Effects of balancing tests.
            None if exit_after_commonsupport is False.
        'bala_se': Standard error of effects of balancing tests.
            None if exit_after_commonsupport is False.
        'bala_effect_list': Names of effects of balancing tests.
            None if exit_after_commonsupport is False.
        'common_support_probabilities: ': pd.DataFrame containing treatment probabilities for all
            treatments, the identifier of the observation, and a dummy variable indicating whether
            the observation is inside or outside the common support.
        'path_output': Pathlib object, location of directory in which output is saved.
        'inputdata_on_support': DataFrame of input data that on the common support.
            None, if exit_after_commonsupport is True.
    """
    time_start = time()
    report = {}

    if not isinstance(exit_after_commonsupport, bool):
        raise TypeError('Keyword "exit_after_commonsupport" must be Boolean.')

    # Reduce sample size to upper limit
    data_df, rnd_reduce, txt_red = check_reduce_dataframe(data_df,
                                                          title='Prediction',
                                                          max_obs=mcf_.int_cfg.max_obs_prediction,
                                                          seed=124535, ignore_index=True,
                                                          )
    if rnd_reduce and mcf_.gen_cfg.with_output:
        mcf_ps.print_mcf(mcf_.gen_cfg, txt_red, summary=True)

    data_df, _ = mcf_data.data_frame_vars_lower(data_df)

    # Initialise again with data information
    data_df = mcf_init_update.p_update_pred(mcf_, data_df)

    # Check treatment data
    if mcf_.p_cfg.d_in_pred:
        data_df = mcf_data.check_recode_treat_variable(mcf_, data_df)
    mcf_init_update.int_update_pred(mcf_, len(data_df))
    mcf_init_update.post_update_pred(mcf_, data_df)
    if mcf_.gen_cfg.with_output:
        mcf_ps.print_dic_values_all(mcf_, summary_top=True, summary_dic=False, train=False)

    # Prepare data: Add and recode variables for GATES (Z)
    #               Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(mcf_, data_df, train=False)
    if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
        mcf_data.print_prime_value_corr(mcf_.data_train_dict, mcf_.gen_cfg, summary=False)

    # Clean data and remove missings and unncessary variables
    if mcf_.dc_cfg.clean_data:
        data_df, report['prediction_obs'] = mcf_data.clean_data(mcf_, data_df, train=False)

    # Descriptives by treatment on common support
    if mcf_.p_cfg.d_in_pred and mcf_.int_cfg.descriptive_stats and mcf_.gen_cfg.with_output:
        mcf_ps.desc_by_treatment(mcf_,
                                 data_df,
                                 summary=False, stage=3, subtreatments=mcf_.gen_tv_cfg.yes
                                 )
    time_1 = time()

    # Common support
    if mcf_.cs_cfg.type_:
        (data_df, _, report['cs_p_share_deleted'], report['cs_p_obs_remain'], _, _, _, cs_pred_prob,
         ) = common_support_p_score(mcf_, data_df, None, train=False, p_score_only=False)
    else:
        cs_pred_prob = None

    if mcf_.gen_tv_cfg.yes:
        # Check if any subtreatment was completely deleted by common support
        mcf_init_update.gen_tv_update_pred(mcf_)

    data_df = data_df.copy().reset_index(drop=True)
    if (mcf_.p_cfg.d_in_pred and mcf_.int_cfg.descriptive_stats
            and mcf_.gen_cfg.with_output):
        mcf_ps.desc_by_treatment(mcf_,
                                 data_df,
                                 summary=True, stage=3, subtreatments=mcf_.gen_tv_cfg.yes
                                 )
    time_2 = time()

    if exit_after_commonsupport:
        return {'ate': None, 'ate_se': None, 'ate_effect_list': None,
                'gate': None, 'gate_se': None, 'gate_diff': None, 'gate_diff_se': None,
                'gate_names_values': None,
                'cbgate': None, 'cbgate_se': None, 'cbgate_diff': None, 'cbgate_diff_se': None,
                'bgate': None, 'bgate_se': None, 'bgate_diff': None, 'bgate_diff_se': None,
                'qiate': None, 'qiate_se': None, 'qiate_mmed': None, 'qiate_mmed_se': None,
                'qiate_mopp': None, 'qiate_mopp_se': None,
                'iate': None, 'iate_se': None, 'iate_eff': None, 'iate_data_df': None,
                'iate_names_dic': None,
                'bala': None, 'bala_se': None, 'bala_effect_list': None,
                'common_support_probabilities': cs_pred_prob,
                'path_output': mcf_.gen_cfg.outpath,
                'inputdata_on_support': data_df,
                }
    # Local centering for IATE (to reverse its effect on potential outcomes)
    if mcf_.lc_cfg.yes and mcf_.lc_cfg.uncenter_po:
        _, _, y_pred_x_df, _ = local_centering(mcf_, data_df, fill_y_df=None, train=False)
    else:
        y_pred_x_df = 0
    time_3 = time()

    # Initialise instances of dataclasses holding results and parameters
    eff_flags_dc = inst_eff_flags(mcf_.gen_cfg)
    # The following two instances will be upated during iterations of folds and rounds
    effect_dc = inst_effect_dicts(eff_flags_dc)
    time_dc = TimeState()

    only_one_fold_one_round = mcf_.cf_cfg.folds == 1 and len(mcf_.cf_cfg.est_rounds) == 1

    # Get effects for single folds and rounds ('efficient' effects)
    for fold in range(mcf_.cf_cfg.folds):
        for round_ in mcf_.cf_cfg.est_rounds:
            idx_round = 0 if round_ == 'regular' else 1
            if only_one_fold_one_round:
                forest_dic = mcf_.forest[fold][0]
            else:
                forest_dic = deepcopy(mcf_.forest[fold][idx_round])

            if mcf_.low_mem_cfg.yes:
                effect_fct = mcf_fsw.effect_with_sequential_weights
                if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
                    print(f'\n\nSequentiell estimation (weight & IATE): {fold+1} / '
                          f'{mcf_.cf_cfg.folds} forests, {round_}'
                          )
            else:
                if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
                    print(f'\n\nWeight maxtrix {fold+1} / {mcf_.cf_cfg.folds} forests, {round_}')
                effect_fct = mcf_fsw.effect_with_full_weights

            effect_fct(mcf_,
                       data_df=data_df,
                       forest_dic=forest_dic,
                       eff_flags=eff_flags_dc,
                       effect=effect_dc,
                       time_=time_dc,
                       round_=round_,
                       idx_round=idx_round,
                       fold=fold,
                       )
        if mcf_.int_cfg.mp_use_old_ray:
            check_ray_shutdown(mcf_.int_cfg.mp_ray_shutdown, mcf_.gen_cfg.mp_parallel)

        if not only_one_fold_one_round and mcf_.int_cfg.del_forest:
            mcf_.forest[fold] = None
            # Without this and the next delete, it becomes impossible to reuse
            # the same forest for several data sets, which is bad.

    if mcf_.int_cfg.del_forest:
        mcf_.forest = None

    # Aggregate dictionaries if needed
    aggregate_effects(effect_dc)

    # ATE
    time_a_start = time()
    ate, ate_se, ate_effect_list = ate_effects_print(mcf_,
                                                     effect_dc.ate_dic, y_pred_x_df,
                                                     balancing_test=False,
                                                     )
    time_dc.ate += time() - time_a_start

    # GATE
    time_g_start = time()
    if mcf_.p_cfg.gate:
        (gate, gate_se, gate_diff, gate_diff_se, report['fig_gate']
         ) = gate_effects_print(mcf_,
                                effect_dic=effect_dc.gate_dic,
                                effect_m_ate_dic=effect_dc.gate_m_ate_dic,
                                gate_est_dic=effect_dc.gate_est_dic, ate=ate, ate_se=ate_se,
                                gate_type='GATE', iv=False,
                                )
    else:
        gate = gate_se = gate_diff = gate_diff_se = effect_dc.gate_est_dic = None

    time_dc.gate += time() - time_g_start

    # BGATE
    time_bgate_start = time()
    if mcf_.p_cfg.bgate:
        (bgate, bgate_se, bgate_diff, bgate_diff_se, report['fig_bgate']
         ) = gate_effects_print(mcf_,
                                effect_dic=effect_dc.bgate_dic,
                                effect_m_ate_dic=effect_dc.bgate_m_ate_dic,
                                gate_est_dic=effect_dc.bgate_est_dic, ate=ate, ate_se=ate_se,
                                gate_type='BGATE', iv=False, special_txt=effect_dc.txt_b,
                                )
    else:
        bgate = bgate_se = bgate_diff = bgate_diff_se = effect_dc.bgate_est_dic = None

    time_dc.bgate += time() - time_bgate_start

    # CBGATE
    time_cbg_start = time()
    if mcf_.p_cfg.cbgate:
        (cbgate, cbgate_se, cbgate_diff, cbgate_diff_se, report['fig_cbgate']
         ) = gate_effects_print(mcf_,
                                effect_dic=effect_dc.cbgate_dic,
                                effect_m_ate_dic=effect_dc.cbgate_m_ate_dic,
                                gate_est_dic=effect_dc.cbgate_est_dic, ate=ate, ate_se=ate_se,
                                gate_type='CBGATE', iv=False, special_txt=effect_dc.txt_am,
                                )
    else:
        cbgate = cbgate_se = cbgate_diff = cbgate_diff_se = effect_dc.cbgate_est_dic = None

    time_dc.cbgate += time() - time_cbg_start

    # Collect some information for results_dic
    if mcf_.p_cfg.gate or mcf_.p_cfg.bgate or mcf_.p_cfg.cbgate:
        gate_names_values = get_names_values(mcf_,
                                             effect_dc.gate_est_dic, effect_dc.bgate_est_dic,
                                             effect_dc.cbgate_est_dic,
                                             )
    else:
        gate_names_values = None

    # QIATE
    time_q_start = time()
    if mcf_.p_cfg.qiate:
        (qiate, qiate_se, qiate_mmed, qiate_mmed_se, qiate_mopp, qiate_mopp_se, report['fig_qiate'],
         ) = qiate_effects_print(mcf_, effect_dc.qiate_dic, effect_dc.qiate_m_med_dic,
                                 effect_dc.qiate_m_opp_dic, effect_dc.qiate_est_dic,
                                 )
    else:
        qiate = qiate_se = effect_dc.qiate_est_dic = None
        qiate_mmed = qiate_mmed_se = qiate_mopp = qiate_mopp_se = None

    time_dc.qiate += time() - time_q_start  # QIATE

    # IATE
    time_i_start = time()
    if mcf_.p_cfg.iate:
        (iate, iate_se, iate_names_dic, iate_df, report['iate_text']    # TODO
         ) = iate_effects_print(mcf_, effect_dc.iate_dic, effect_dc.iate_m_ate_dic, y_pred_x_df)
        data_df.reset_index(drop=True, inplace=True)
        iate_df.reset_index(drop=True, inplace=True)
        iate_pred_df = pd.concat([data_df, iate_df], axis=1)
    else:
        iate = iate_se = iate_df = iate_pred_df = iate_names_dic = None

    time_dc.iate += time() - time_i_start

    # Balancing test
    time_b_start = time()
    if mcf_.p_cfg.bt_yes:
        bala, bala_se, bala_effect_list = ate_effects_print(mcf_,
                                                            effect_dc.bala_dic, None,
                                                            balancing_test=True,
                                                            )
    else:
        bala = bala_se = bala_effect_list = None
    time_dc.bala += time() - time_b_start

    # Collect results
    results = {'ate': ate, 'ate_se': ate_se, 'ate_effect_list': ate_effect_list,
               'gate': gate, 'gate_se': gate_se, 'gate_diff': gate_diff,
               'gate_diff_se': gate_diff_se, 'gate_names_values': gate_names_values,
               'cbgate': cbgate, 'cbgate_se': cbgate_se, 'cbgate_diff': cbgate_diff,
               'cbgate_diff_se': cbgate_diff_se,
               'bgate': bgate, 'bgate_se': bgate_se, 'bgate_diff': bgate_diff,
               'bgate_diff_se': bgate_diff_se,
               'qiate': qiate, 'qiate_se': qiate_se, 'qiate_mmed': qiate_mmed,
               'qiate_mmed_se': qiate_mmed_se, 'qiate_mopp': qiate_mopp,
               'qiate_mopp_se': qiate_mopp_se,
               'iate': iate, 'iate_se': iate_se, 'iate_eff': eff_flags_dc.iate_eff,
               'iate_data_df': iate_pred_df, 'iate_names_dic': iate_names_dic,
               'bala': bala, 'bala_se': bala_se, 'bala_effect_list': bala_effect_list,
               'common_support_probabilities': cs_pred_prob,
               'path_output': mcf_.gen_cfg.outpath,
               'inputdata_on_support': None,
               }
    if mcf_.gen_cfg.with_output:
        report['mcf_pred_results'] = results
    mcf_.report['predict_list'].append(report.copy())
    time_end = time()
    if mcf_.gen_cfg.with_output:
        time_string = ['Data preparation and stats II:                  ',
                       'Common support:                                 ',
                       'Local centering (recoding of Y):                ',
                       'Weights (some double counting with (C)BGATE):   ',
                       'ATEs:                                           ',
                       'GATEs:                                          ',
                       'BGATEs:                                         ',
                       'CBGATEs:                                        ',
                       'QIATEs:                                         ',
                       'IATEs:                                          ',
                       'Balancing test:                                 ',
                       '\nTotal time prediction:                          '
                       ]
        time_difference = [time_1 - time_start, time_2 - time_1, time_3 - time_2,
                           time_dc.weight, time_dc.ate, time_dc.gate, time_dc.bgate, time_dc.cbgate,
                           time_dc.qiate, time_dc.iate, time_dc.bala,
                           time_end - time_start,
                           ]
        mcf_ps.print_mcf(mcf_.gen_cfg, mcf_.time_strings['time_train'])
        time_pred = mcf_ps.print_timing(mcf_.gen_cfg, 'Prediction', time_string, time_difference,
                                        summary=True,
                                        )
        mcf_.time_strings['time_pred'] = time_pred

    return results


def analyse_main(mcf_: 'ModifiedCausalForest', results: dict) -> dict:
    """
    Analyse estimated IATE with various descriptive tools.

    Parameters
    ----------
    results : Dictionary
        Contains estimation results. This dictionary must have the same
        structure as the one returned from the :meth:`~ModifiedCausalForest.predict` method.

    Raises
    ------
    ValueError
        Some of the attribute are not compatible with running this method.

    Returns
    -------
    results_plus_cluster : Dictionary
        Same as the results dictionary, but the DataFrame with estimated
        IATEs contains an additional integer with a group label that comes from k-means clustering.

    """
    report = {}

    # Identify if results come from IV estimation or not.
    iv = mcf_.iv_mcf['firststage'] is not None

    if mcf_.gen_cfg.with_output and mcf_.post_cfg.est_stats and mcf_.gen_cfg.return_iate_sp:
        time_start = time()
        report['fig_iate'] = mcf_post.post_estimation_iate(mcf_, results, iv)
        time_end_corr = time()
        if mcf_.post_cfg.kmeans_yes:
            results_plus_cluster, report['knn_table'] = mcf_post.k_means_of_x_iate(mcf_, results)
        else:
            results_plus_cluster = report['knn_table'] = None
        time_end_km = time()
        if mcf_.post_cfg.random_forest_vi or mcf_.post_cfg.tree:
            mcf_post.random_forest_tree_of_iate(mcf_, results)

        time_string = ['Correlational analysis and plots of IATE:       ',
                       'K-means clustering of IATE:                     ',
                       'Random forest / tree analysis of IATE:          ',
                       '\nTotal time post estimation analysis:            '
                       ]
        time_difference = [time_end_corr - time_start, time_end_km - time_end_corr,
                           time() - time_end_km, time() - time_start
                           ]
        mcf_ps.print_mcf(mcf_.gen_cfg, mcf_.time_strings['time_train'], summary=True)
        mcf_ps.print_mcf(mcf_.gen_cfg, mcf_.time_strings['time_pred'], summary=True)
        mcf_ps.print_timing(mcf_.gen_cfg, 'Analysis of IATE', time_string, time_difference,
                            summary=True,
                            )
        mcf_.report['analyse_list'].append(report.copy())
    else:
        raise ValueError('"Analyse" method produces output only if all of the following parameters '
                         'are True:'
                         f'\nint_with_output: {mcf_.gen_cfg.with_output}'
                         f'\npos_test_stats: {mcf_.post_cfg.est_stats}'
                         f'\nint_return_iate_sp: {mcf_.gen_cfg.return_iate_sp}'
                         )
    return results_plus_cluster
