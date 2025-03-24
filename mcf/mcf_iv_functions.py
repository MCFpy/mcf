"""Created on Thu Oct 31 10:08:06 2024.

Contains IV specific functions.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time

import pandas as pd
from ray import is_initialized, shutdown

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_common_support_functions as mcf_cs
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_feature_selection_functions as mcf_fs
from mcf import mcf_forest_functions as mcf_fo
from mcf.mcf_gate_functions import gate_est, bgate_est
from mcf import mcf_gateout_functions as mcf_gateout
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_init_functions as mcf_init
from mcf import mcf_local_centering_functions as mcf_lc
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_qiate_functions as mcf_qiate
from mcf import mcf_iv_functions_add as mcf_iv_add


def train_iv_main(self, data_df):
    """
    Train the modified causal forest IV on the training data.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the causal forest. It must contain information
        about outcomes, treatment, and features.

    Returns
    -------
    tree_df : DataFrame
        Dataset used to build the forest.

    fill_y_df : DataFrame
        Dataset used to populate the forest with outcomes.

    outpath : pathlib object
        Location of directory in which output is saved.

    """
    time_start = time()
    # Check treatment data
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_data.check_recode_treat_variable(self, data_df)
    # Initialise again with data information. Order of the following
    # init functions important. Change only if you know what you do.

    self.int_dict['iv'] = True
    if not self.var_dict['iv_name']:
        raise ValueError('Instrument must be specified for IV estimation.')

    # These standard updates remain as some are important for the predict method
    mcf_init.var_update_train(self, data_df)
    mcf_init.gen_update_train(self, data_df)
    mcf_init.ct_update_train(self, data_df)
    mcf_init.cs_update_train(self)            # Used updated gen_dict info
    mcf_init.cf_update_train(self, data_df)
    mcf_init.int_update_train(self)
    mcf_init.lc_update_train(self, data_df)
    mcf_init.p_update_train(self)

    if self.int_dict['with_output']:
        mcf_ps.print_dic_values_all(self, summary_top=True, summary_dic=False)

    # Prepare data: Add and recode variables for GATES (Z)
    #             Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(self, data_df, train=True)
    if self.int_dict['with_output'] and self.int_dict['verbose']:
        mcf_data.print_prime_value_corr(self.data_train_dict,
                                        self.gen_dict, summary=False)

    # Clean data and remove missings and unncessary variables
    if self.dc_dict['clean_data']:
        data_df, report = mcf_data.clean_data(
            self, data_df, train=True)
        if self.gen_dict['with_output']:
            self.report['training_obs'] = report
    if self.dc_dict['screen_covariates']:   # Only training
        (self.gen_dict, self.var_dict, self.var_x_type,
         self.var_x_values, report
         ) = mcf_data.screen_adjust_variables(self, data_df)
        if self.gen_dict['with_output']:
            self.report['removed_vars'] = report

    # Ensure that treatment and instrument are binary (and coded as integer)
    data_df = ensure_binary_binary(self, data_df)

    # Descriptives by treatment
    if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
        mcf_ps.desc_by_treatment(self, data_df, summary=False, stage=1)
    time_1 = time()

    # Feature selection on the first stage only
    if self.fs_dict['yes']:
        mcf_inst_fs = deepcopy(self)
        mcf_inst_fs.var_dict['d_name'] = self.var_dict['iv_name']
        mcf_inst_fs.var_dict['y_name'] = self.var_dict['d_name']
        mcf_inst_fs.var_dict['y_tree_name'] = self.var_dict['d_name']
        data_df, report = mcf_fs.feature_selection(mcf_inst_fs, data_df)
        self.var_dict['x_name'] = mcf_inst_fs.var_dict['x_name'].copy()
        self.var_x_type = mcf_inst_fs.var_x_type.copy()
        self.var_x_values = mcf_inst_fs.var_x_values.copy()
        if self.gen_dict['with_output']:
            self.report['fs_vars_deleted'] = report
        del mcf_inst_fs

    # Split sample for tree building and tree-filling-with-y
    tree_df, fill_y_df = mcf_data.split_sample_for_mcf(self, data_df)
    obs = len(data_df)
    del data_df
    time_2 = time()

    # Compute Common support on full sample
    if self.cs_dict['type']:
        mcf_inst_cs = deepcopy(self)
        mcf_inst_cs.var_dict['d_name'] = self.var_dict['iv_name']
        (tree_df, fill_y_df, rep_sh_del, obs_remain, rep_fig
         ) = mcf_cs.common_support(mcf_inst_cs, tree_df, fill_y_df, train=True)
        if self.gen_dict['with_output']:
            self.report['cs_t_share_deleted'] = rep_sh_del
            self.report['cs_t_obs_remain'] = obs_remain
            self.report['cs_t_figs'] = rep_fig
        self.cs_dict = deepcopy(mcf_inst_cs.cs_dict)
        del mcf_inst_cs

    # Descriptives by treatment on common support
    if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
        mcf_ps.desc_by_treatment(self, pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1)
    time_3 = time()

    # Create new instances for 1st stage and reduced form
    mcf_1st, mcf_redf = new_instances_iv_train(self)

    # Local centering
    if self.lc_dict['yes']:
        (tree_df, _, _, report_1st) = mcf_lc.local_centering(
            mcf_1st, tree_df, fill_y_df=None)

        (tree_df, fill_y_df, _, report_redf) = mcf_lc.local_centering(
            mcf_redf, tree_df, fill_y_df=fill_y_df)

        if self.gen_dict['with_output']:
            self.report["lc_r2_1st"] = report_1st
            self.report["lc_r2_redform"] = report_redf
    time_4 = time()

    # Train forests
    # 1st stage: Train forest on tree data only (not honest)
    if self.int_dict['with_output']:
        mcf_ps.variable_features(self, summary=False)
        txt = ('\n' * 2 + '-' * 100 + '\n' + '1st stage forest' + '\n'
               + '-' * 100)
        mcf_ps.print_mcf(self.gen_dict, txt, summary=True)
    (mcf_1st.cf_dict, mcf_1st.forest, time_vi_1st, report_1st
     ) = mcf_fo.train_forest(mcf_1st, tree_df, tree_df)
    time_5 = time()

    # Reduced form (honest)
    if self.int_dict['with_output']:
        txt = ('\n' * 2 + '-' * 100 + '\n' + 'Reduced form forest' + '\n'
               + '-' * 100)
        mcf_ps.print_mcf(self.gen_dict, txt, summary=True)

    (mcf_redf.cf_dict, mcf_redf.forest, time_vi_redf, report_redf
     ) = mcf_fo.train_forest(mcf_redf, tree_df, fill_y_df)

    if self.gen_dict['with_output']:
        self.report['cf_1st'] = report_1st
        self.report['cf_redf'] = report_redf

    self.iv_mcf['firststage'] = mcf_1st
    self.iv_mcf['reducedform'] = mcf_redf

    time_end = time()
    time_string = ['Data preparation and stats I:                   ',
                   'Feature preselection:                           ',
                   'Common support:                                 ',
                   'Local centering (recoding of Y & D):            ',
                   'Training the causal forest (1st stage):         ',
                   '  ... of which is time for variable importance: ',
                   'Training the causal forest (reduced form):      ',
                   '  ... of which is time for variable importance: ',
                   '\nTotal time training:                            ']
    time_difference = [time_1 - time_start, time_2 - time_1,
                       time_3 - time_2, time_4 - time_3,
                       time_5 - time_4, time_vi_1st,
                       time_end - time_5, time_vi_redf,
                       time_end - time_start]
    if self.int_dict['with_output']:
        time_train = mcf_ps.print_timing(
            self.gen_dict, 'Training', time_string, time_difference,
            summary=True)
        self.time_strings['time_train'] = time_train

    if (is_initialized()
        and self.gen_dict['mp_parallel'] > 1
            and obs > self.int_dict['obs_bigdata']):
        shutdown()

    return tree_df, fill_y_df


def predict_iv_main(self, data_df):
    """
    Compute all effects based on causal forests estimated by the train_iv.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the predictions. It must contain information
        about features (and treatment if effects for treatment specific
        subpopulations are desired as well).

    Returns
    -------
    results : Dictionary.
        Results.

    """
    time_start = time()
    report = {}

    # Get instances that contain the results from training
    mcf_1st = self.iv_mcf['firststage']
    mcf_redf = self.iv_mcf['reducedform']
    update_self(self, mcf_1st)

    # Initialise again with data information
    data_df = update_inst_data(self, mcf_1st, mcf_redf, data_df)

    # Print information of dictionaries
    if self.int_dict['with_output']:
        print_dics(self, mcf_1st, mcf_redf)

    # Prepare data and some descriptive stats
    data_df, report = prepare_data_pred(self, data_df, report)
    time_1 = time()

    # Common support
    data_df, report = common_support_pred(self, data_df, report)
    time_2 = time()

    # Local centering for IATE
    if self.lc_dict['yes'] and self.lc_dict['uncenter_po']:
        y_pred_x_df, d_pred_x_df = local_center(mcf_1st, mcf_redf, data_df)
    else:
        y_pred_x_df = d_pred_x_df = 0
    time_3 = time()

    # Initialise various variables
    time_delta_weight = time_delta_ate = time_delta_bala = time_delta_iate = 0
    time_delta_gate = time_delta_cbgate = time_delta_bgate = 0
    time_delta_qiate = 0
    ate_dic = bala_1st_dic = bala_redf_dic = iate_dic = iate_m_ate_dic = None
    iate_eff_dic = ate_1st_dic = iate_1st_dic = ate_redf_dic = None
    iate_redf_dic = iate_m_ate_1st_dic = iate_m_ate_redf_dic = None
    iate_eff_1st_dic = iate_eff_redf_dic = qiate_est_dic = None
    gate_dic = gate_m_ate_dic = cbgate_dic = cbgate_m_ate_dic = None
    bgate_dic = bgate_m_ate_dic = qiate_dic = None
    qiate_m_med_dic = qiate_m_opp_dic = None

    only_one_fold_one_round = (self.cf_dict['folds'] == 1
                               and len(self.cf_dict['est_rounds']) == 1)

    # Start with 1st stage only (as IATEs of 1st stage are needed for scaling)
    iate_1st_dic, iate_eff_1st_dic = mcf_iv_add.get_1st_stage_iate(
        self, mcf_1st, data_df, only_one_fold_one_round)
    time_1stscale = time()
    # All other parameters
    # Estimate effects fold by fold and then average across folds
    for fold in range(self.cf_dict['folds']):
        for round_ in self.cf_dict['est_rounds']:
            time_w_start = time()

            # Get relevant forests
            forest_1st_dic, forest_redf_dic = get_forests(
                self, mcf_1st, mcf_redf, only_one_fold_one_round, fold, round_)

            # Compute weights of reduced form & 1st stage & final estimation
            (weights_1st_dic, weights_redf_dic, weights_dic
             ) = mcf_iv_add.get_weights_late(
                self, mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic,
                iate_1st_dic, iate_eff_1st_dic, data_df, round_)

            time_delta_weight += time() - time_w_start
            time_a_start = time()
            # Compute effects
            weights = (weights_dic, weights_1st_dic, weights_redf_dic,)
            instances = (self, mcf_1st, mcf_redf,)
            # ATE
            if round_ == 'regular':
                (w_ate, ate_dic, w_ate_1st, ate_1st_dic, w_ate_redf,
                 ate_redf_dic) = ate_iv(
                     instances, weights, ate_dic, ate_1st_dic, ate_redf_dic,
                     data_df, fold)
            else:
                w_ate = w_ate_1st = w_ate_redf = None
            time_delta_ate += time() - time_a_start

            # Balancing tests for reduced form and first stage
            time_b_start = time()
            if round_ == 'regular' and self.p_dict['bt_yes']:
                bala_1st_dic, bala_redf_dic = bala_pred(
                    instances, weights, bala_1st_dic, bala_redf_dic,
                    data_df, fold)
            time_delta_bala += time() - time_b_start

            # BGATE & CBGATE
            time_bgate_start = time()
            if round_ == 'regular' and (self.p_dict['bgate']
                                        or self.p_dict['cbgate']):
                late_tuple = (mcf_1st, mcf_redf, forest_1st_dic,
                              forest_redf_dic,)
            else:
                late_tuple = None

            # BGATE
            if round_ == 'regular' and self.p_dict['bgate']:
                bgate_dic, bgate_m_ate_dic, bgate_est_dic, txt_bg = bgate_pred(
                    self, data_df, weights_dic, w_ate, bgate_dic,
                    bgate_m_ate_dic, fold, late_tuple=late_tuple,
                    gate_type='BGATE', title='LBGATE')
            else:
                bgate_est_dic = txt_bg = None
            time_delta_bgate += time() - time_bgate_start

            # CBGATE
            time_cbg_start = time()
            if round_ == 'regular' and self.p_dict['cbgate']:
                (cbgate_dic, cbgate_m_ate_dic, cbgate_est_dic, txt_cbg
                 ) = bgate_pred(self, data_df, weights_dic, w_ate, cbgate_dic,
                                cbgate_m_ate_dic, fold, late_tuple=late_tuple,
                                gate_type='CBGATE', title='LCBGATE')
            else:
                cbgate_est_dic = txt_cbg = None
            time_delta_cbgate += time() - time_cbg_start
            if self.int_dict['del_forest']:
                del forest_1st_dic['forest'], forest_redf_dic['forest']

            # IATE
            time_i_start = time()
            if self.p_dict['iate']:
                y_pot_iate_f = y_pot_iate_1st_f = y_pot_iate_redf_f = None
                # LIATE
                iate_dic, iate_eff_dic, y_pot_iate_f = iate_pred(
                    self, weights_dic, iate_dic, iate_m_ate_dic,
                    iate_eff_dic, w_ate, y_pot_iate_f,
                    round_, fold, late=True, title='LIATE')

                # IATE (1st stage)
                iate_1st_dic, iate_eff_1st_dic, y_pot_iate_1st_f = iate_pred(
                    mcf_1st, weights_1st_dic, iate_1st_dic, iate_m_ate_1st_dic,
                    iate_eff_1st_dic, w_ate_1st, y_pot_iate_1st_f,
                    round_, fold, late=True, title='IATE (1st stage)')

                # IATE (reduced form)
                iate_redf_dic, iate_eff_redf_dic, y_pot_iate_redf_f = iate_pred(
                    mcf_redf, weights_redf_dic, iate_redf_dic,
                    iate_m_ate_redf_dic, iate_eff_redf_dic, w_ate_redf,
                    y_pot_iate_redf_f, round_, fold, late=True,
                    title='IATE (reduced form)')

            time_delta_iate += time() - time_i_start

            # QIATE
            time_q_start = time()
            self.p_dict['qiate'] = False
# =============================================================================
#             Not implemented for QIATE (if implementing, update before)
#             if round_ == 'regular' and self.p_dict['qiate']:
#                 (y_pot_qiate_f, y_pot_var_qiate_f, y_pot_mmed_qiate_f,
#                  y_pot_mmed_var_qiate_f, qiate_est_dic, txt_w_f
#                  ) = mcf_qiate.qiate_est(self, data_df, weights_dic, y_pot_f,
#                                          late=True)
#                 qiate_dic = mcf_est.aggregate_pots(
#                     self, y_pot_qiate_f, y_pot_var_qiate_f, txt_w_f,
#                     qiate_dic, fold, pot_is_list=True, title='QLIATE')
#                 if y_pot_mmed_qiate_f is not None:
#                     qiate_m_med_dic = mcf_est.aggregate_pots(
#                         self, y_pot_mmed_qiate_f, y_pot_mmed_var_qiate_f,
#                         txt_w_f, qiate_m_med_dic, fold, pot_is_list=True,
#                         title='QLIATE minus LIATE(median)')
# =============================================================================
            time_delta_qiate += time() - time_q_start  # QIATE

            # GATE
            time_g_start = time()
            if round_ == 'regular' and self.p_dict['gate']:
                gate_dic, gate_m_ate_dic, gate_est_dic = gate_pred(
                    self, weights_dic, w_ate, gate_dic, gate_m_ate_dic,
                    data_df, fold)
            else:
                gate_est_dic = None
            time_delta_gate += time() - time_g_start

        if not only_one_fold_one_round and self.int_dict['del_forest']:
            self.forest[fold] = mcf_1st.forest[fold] = None
            mcf_redf.forest[fold] = None
            # Without those two deletes, it becomes impossible to reuse
            # the same forest for several data sets, which is bad.

    if self.int_dict['del_forest']:
        self.forest = mcf_1st.forest = mcf_redf.forest = None

    del weights, weights_dic, weights_1st_dic, weights_redf_dic

    # ATE
    time_a_start = time()
    ate, ate_se, ate_effect_list = mcf_ate.ate_effects_print(
        self, ate_dic, y_pred_x_df, balancing_test=False,
        extra_title='(LATE)')

    ate_1st, ate_1st_se, ate_1st_effect_list = mcf_ate.ate_effects_print(
        mcf_1st, ate_1st_dic, d_pred_x_df, balancing_test=False,
        extra_title='(1st stage)')

    ate_redf, ate_redf_se, ate_redf_effect_list = mcf_ate.ate_effects_print(
        mcf_redf, ate_redf_dic, y_pred_x_df, balancing_test=False,
        extra_title='(reduced form)')

    time_delta_ate += time() - time_a_start

    # GATE
    time_g_start = time()
    if self.p_dict['gate']:
        (gate, gate_se, gate_diff, gate_diff_se, report['fig_gate']
         ) = mcf_gateout.gate_effects_print(self, gate_dic, gate_m_ate_dic,
                                            gate_est_dic, ate, ate_se,
                                            gate_type='GATE')
    else:
        gate = gate_se = gate_diff = gate_diff_se = gate_est_dic = None
    time_delta_gate += time() - time_g_start

    # BGATE
    time_bgate_start = time()
    if self.p_dict['bgate']:
        (bgate, bgate_se, bgate_diff, bgate_diff_se, report['fig_bgate']
         ) = mcf_gateout.gate_effects_print(
             self, bgate_dic, bgate_m_ate_dic, bgate_est_dic, ate,
             ate_se, gate_type='BGATE', special_txt=txt_bg)
    else:
        bgate = bgate_se = bgate_diff = bgate_diff_se = None
        bgate_est_dic = None
    time_delta_bgate += time() - time_bgate_start

    # CBGATE
    time_cbg_start = time()
    if self.p_dict['cbgate']:
        (cbgate, cbgate_se, cbgate_diff, cbgate_diff_se,
         report['fig_cbgate']) = mcf_gateout.gate_effects_print(
             self, cbgate_dic, cbgate_m_ate_dic, cbgate_est_dic, ate,
             ate_se, gate_type='CBGATE', special_txt=txt_cbg)
    else:
        cbgate = cbgate_se = cbgate_diff = cbgate_diff_se = None
        cbgate_est_dic = None
    time_delta_cbgate += time() - time_cbg_start
    # Collect some information for results_dic
    if (self.p_dict['gate'] or self.p_dict['bgate']
            or self.p_dict['cbgate']):
        gate_names_values = mcf_gateout.get_names_values(
            self, gate_est_dic, bgate_est_dic, cbgate_est_dic)
    else:
        gate_names_values = None

    # QIATE (not yet(?) available)
    time_q_start = time()
    if self.p_dict['qiate']:
        (qiate, qiate_se, qiate_diff, qiate_diff_se, qiate_mopp, qiate_mopp_se,
         report['fig_qiate']) = mcf_qiate.qiate_effects_print(
             self, qiate_dic, qiate_m_med_dic, qiate_m_opp_dic, qiate_est_dic)

    else:
        qiate = qiate_se = qiate_diff = qiate_diff_se = qiate_est_dic = None
        qiate_mopp = qiate_mopp_se = None
    time_delta_qiate += time() - time_q_start  # QIATE

    # IATE
    time_i_start = time()
    if self.p_dict['iate']:
        (iate, iate_se, iate_eff, iate_names_dic, iate_df,
         report['iate_text']) = mcf_iate.iate_effects_print(
             self, iate_dic, iate_m_ate_dic, iate_eff_dic, y_pred_x_df,
             extra_title='(LIATE)', late=True)
        data_df.reset_index(drop=True, inplace=True)
        iate_df.reset_index(drop=True, inplace=True)
        iate_pred_df = pd.concat([data_df, iate_df], axis=1)
    else:
        iate_eff = iate = iate_se = iate_df = iate_pred_df = None
        iate_names_dic = None

    # IATE (1st stage)
    time_i_start = time()
    if self.p_dict['iate']:
        (iate_1st, iate_1st_se, iate_1st_eff, iate_1st_names_dic, iate_1st_df,
         report['iate_1st_text']) = mcf_iate.iate_effects_print(
             mcf_1st, iate_1st_dic, iate_m_ate_1st_dic, iate_eff_1st_dic,
             d_pred_x_df, extra_title='(1st stage)', late=False)
        data_df.reset_index(drop=True, inplace=True)
        iate_1st_df.reset_index(drop=True, inplace=True)
        iate_1st_pred_df = pd.concat([data_df, iate_1st_df], axis=1)
    else:
        iate_1st_eff = iate_1st = iate_1st_se = iate_1st_df = None
        iate_1st_pred_df = iate_1st_names_dic = None

    # IATE (reduced form)
    time_i_start = time()
    if self.p_dict['iate']:
        (iate_redf, iate_redf_se, iate_redf_eff, iate_redf_names_dic,
         iate_redf_df, report['iate_redf_text']) = mcf_iate.iate_effects_print(
             mcf_redf, iate_redf_dic, iate_m_ate_redf_dic, iate_eff_redf_dic,
             y_pred_x_df, extra_title='(reduced form)', late=False)
        data_df.reset_index(drop=True, inplace=True)
        iate_redf_df.reset_index(drop=True, inplace=True)
        iate_redf_pred_df = pd.concat([data_df, iate_redf_df], axis=1)
    else:
        iate_redf_eff = iate_redf = iate_redf_se = iate_redf_df = None
        iate_redf_pred_df = iate_redf_names_dic = None

    time_delta_iate += time() - time_i_start

    # Balancing test
    time_b_start = time()
    if self.p_dict['bt_yes']:
        (bala_1st, bala_1st_se, bala_1st_effect_list
         ) = mcf_ate.ate_effects_print(self, bala_1st_dic, None,
                                       balancing_test=True)
        (bala_redf, bala_redf_se, bala_redf_effect_list
         ) = mcf_ate.ate_effects_print(self, bala_redf_dic, None,
                                       balancing_test=True)
    else:
        bala_1st = bala_1st_se = bala_1st_effect_list = None
        bala_redf = bala_redf_se = bala_redf_effect_list = None
    time_delta_bala += time() - time_b_start

    # Collect results
    results = {
        'ate': ate, 'ate_se': ate_se, 'ate effect_list': ate_effect_list,
        'ate_1st': ate_1st, 'ate_1st_se': ate_1st_se,
        'ate 1st_effect_list': ate_1st_effect_list,
        'ate_redf': ate_redf, 'ate_redf_se': ate_redf_se,
        'ate redf_effect_list': ate_redf_effect_list,
        'gate': gate, 'gate_se': gate_se,
        'gate_diff': gate_diff, 'gate_diff_se': gate_diff_se,
        'gate_names_values': gate_names_values,
        'cbgate': cbgate, 'cbgate_se': cbgate_se,
        'cbgate_diff': cbgate_diff, 'cbgate_diff_se': cbgate_diff_se,
        'bgate': bgate, 'bgate_se': bgate_se,
        'bgate_diff': bgate_diff, 'bgate_diff_se': bgate_diff_se,
        'qiate': qiate, 'qiate_se': qiate_se,
        'qiate_diff': qiate_diff, 'qiate_diff_se': qiate_diff_se,
        'qiate_mopp': qiate_mopp, 'qiate_mopp_se': qiate_mopp_se,
        'iate': iate, 'iate_se': iate_se, 'iate_eff': iate_eff,
        'iate_1st': iate_1st, 'iate_1st_se': iate_1st_se,
        'iate_redf': iate_redf, 'iate_redf_se': iate_redf_se,
        'iate_1st_eff': iate_1st_eff, 'iate_redf_eff': iate_redf_eff,
        'iate_data_df': iate_pred_df, 'iate_1st_pred_df': iate_1st_pred_df,
        'iate_redf_pred_df': iate_redf_pred_df,
        'iate_names_dic': iate_names_dic,
        'iate_1st_names_dic': iate_1st_names_dic,
        'iate_redf_names_dic': iate_redf_names_dic,
        'bala_1st': bala_1st, 'bala_1st_se': bala_1st_se,
        'bala_1st_effect_list': bala_1st_effect_list,
        'bala_redf': bala_redf, 'bala_redf_se': bala_redf_se,
        'bala_redf_effect_list': bala_redf_effect_list,
               }
    if self.int_dict['with_output']:
        results_dic = results.copy()
        del results_dic['iate_data_df']
        report['mcf_pred_results'] = results
    self.report['predict_list'].append(report.copy())
    time_end = time()
    if self.int_dict['with_output']:
        time_string = [
            'Data preparation and stats II:                  ',
            'Common support:                                 ',
            'Local centering (recoding of Y):                ',
            'First stage scaling :                           ',
            'Weights:                                        ',
            'LATEs:                                           ',
            'LGATEs:                                          ',
            'LBGATEs:                                         ',
            'LCBGATEs:                                        ',
            'LQIATEs:                                         ',
            'LIATEs:                                          ',
            'Balancing test:                                 ',
            '\nTotal time prediction:                          ']
        time_difference = [
            time_1 - time_start, time_2 - time_1, time_3 - time_2,
            time_1stscale - time_3,
            time_delta_weight, time_delta_ate, time_delta_gate,
            time_delta_bgate, time_delta_cbgate, time_delta_qiate,
            time_delta_iate, time_delta_bala, time_end - time_start]
        mcf_ps.print_mcf(self.gen_dict, self.time_strings['time_train'])
        time_pred = mcf_ps.print_timing(
            self.gen_dict, 'Prediction', time_string, time_difference,
            summary=True)
        self.time_strings['time_pred'] = time_pred

    return results


def ensure_binary_binary(mcf_, data_df):
    """Check if treatment and instrument are binary."""
    # Check data is all numeric and convert it integers
    int_series = pd.to_numeric(
        data_df[mcf_.var_dict['iv_name']].squeeze(), errors='raise').astype(int)

    # Check if instrument is binary
    if not set(int_series.unique()).issubset({0, 1}):
        raise ValueError('Instrument must be binary.')
    data_df[mcf_.var_dict['iv_name']] = int_series.to_frame()

    # Check if treatment is binary
    d_series = data_df[mcf_.var_dict['d_name']].squeeze()
    if not set(d_series.unique()).issubset({0, 1}):
        raise ValueError('Treatment must be binary.')

    return data_df


def print_dics(self, mcf_1st, mcf_redf):
    """Print values of dictionaries."""
    mcf_ps.print_dic_values_all(self, summary_top=True, summary_dic=False,
                                train=False)
    mcf_ps.print_dic_values_all(mcf_1st, summary_top=False,
                                summary_dic=False, train=False)
    mcf_ps.print_dic_values_all(mcf_redf, summary_top=True,
                                summary_dic=False, train=False)


def update_self(self, mcf_1st):
    """Update self instance."""
    # Copy some attributes of mcf_1st to self
    self.cf_dict['folds'] = mcf_1st.cf_dict['folds']
    self.cf_dict['est_rounds'] = mcf_1st.cf_dict['est_rounds']
    self.cf_dict['x_name_mcf'] = mcf_1st.cf_dict['x_name_mcf'].copy()


def update_inst_data(self, mcf_1st, mcf_redf, data_df):
    """Update instances and Dataframe."""
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_init.p_update_pred(self, data_df)
    mcf_init.p_update_pred(mcf_1st, data_df)
    mcf_init.p_update_pred(mcf_redf, data_df)

    # Check treatment data
    if self.p_dict['d_in_pred']:
        data_df = mcf_data.check_recode_treat_variable(self, data_df)

    mcf_init.int_update_pred(self, len(data_df))
    mcf_init.post_update_pred(self, data_df)
    mcf_init.int_update_pred(mcf_1st, len(data_df))
    mcf_init.post_update_pred(mcf_1st, data_df)
    mcf_init.int_update_pred(mcf_redf, len(data_df))
    mcf_init.post_update_pred(mcf_redf, data_df)

    return data_df


def local_center(mcf_1st, mcf_redf, data_df):
    """Compute centering variables."""
    (_, _, d_pred_x_df, _) = mcf_lc.local_centering(mcf_1st, data_df, None,
                                                    train=False)
    (_, _, y_pred_x_df, _) = mcf_lc.local_centering(mcf_redf, data_df, None,
                                                    train=False)

    return y_pred_x_df, d_pred_x_df


def get_forests(self, mcf_1st, mcf_redf, only_one_fold_one_round, fold, round_):
    """Extract forsts."""
    if only_one_fold_one_round:
        forest_1st_dic = mcf_1st.forest[fold][0]
        forest_redf_dic = mcf_redf.forest[fold][0]
    else:
        forest_1st_dic = deepcopy(
            mcf_1st.forest[fold][0 if round_ == 'regular' else 1])
        forest_redf_dic = deepcopy(
            mcf_redf.forest[fold][0 if round_ == 'regular' else 1])
    if self.int_dict['with_output'] and self.int_dict['verbose']:
        print(f'\n\nWeight maxtrix (all effects) {fold+1} /',
              f'{self.cf_dict["folds"]} forests, {round_}')

    return forest_1st_dic, forest_redf_dic


def new_instances_iv_train(self):
    """Create instances for 1st stage and reduced form."""
    mcf_1st = deepcopy(self)
    mcf_1st.var_dict['y_name'] = self.var_dict['d_name'].copy()
    mcf_1st.var_dict['y_tree_name'] = mcf_1st.var_dict['y_name']
    mcf_1st.var_dict['d_name'] = self.var_dict['iv_name'].copy()
    mcf_1st.int_dict['verbose'] = False

    mcf_redf = deepcopy(self)
    mcf_redf.var_dict['d_name'] = self.var_dict['iv_name'].copy()
    mcf_redf.int_dict['verbose'] = False

    return mcf_1st, mcf_redf


def prepare_data_pred(self, data_df, report):
    """Prepare data and print some descriptive statistics."""
    # Prepare data: Add and recode variables for GATES (Z)
    #             Recode categorical variables to prime numbers, cont. vars
    data_df = mcf_data.create_xz_variables(self, data_df, train=False)
    if self.int_dict['with_output'] and self.int_dict['verbose']:
        mcf_data.print_prime_value_corr(self.data_train_dict,
                                        self.gen_dict, summary=False)

    # Clean data and remove missings and unncessary variables
    if self.dc_dict['clean_data']:
        data_df, report['prediction_obs'] = mcf_data.clean_data(
            self, data_df, train=False)

    # Descriptives by treatment
    if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
            and self.int_dict['with_output']):
        mcf_ps.desc_by_treatment(self, data_df, summary=False, stage=3)

    return data_df, report


def common_support_pred(self, data_df, report):
    """Adjust common support based on training information."""
    if self.cs_dict['type']:
        (data_df, _, report['cs_p_share_deleted'],
         report['cs_p_obs_remain'], _) = mcf_cs.common_support(
             self, data_df, None, train=False)
    data_df = data_df.copy().reset_index(drop=True)
    if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
            and self.int_dict['with_output']):
        mcf_ps.desc_by_treatment(self, data_df, summary=True, stage=3)

    return data_df, report


def ate_iv(instances, weights, ate_dic, ate_1st_dic, ate_redf_dic, data_df,
           fold):
    """Compute LATE, reduced form and first stage."""
    mcf_, mcf_1st, mcf_redf = instances
    weights_dic, weights_1st_dic, weights_redf_dic = weights

    (w_ate, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_, data_df, weights_dic, late=True)
    # Aggregate ATEs over folds
    ate_dic = mcf_est.aggregate_pots(
        mcf_, y_pot_f, y_pot_var_f, txt_w_f, ate_dic, fold,
        title='LATE')

    # 1st stage
    (w_ate_1st, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_1st, data_df, weights_1st_dic, late=False)
    # Aggregate ATEs over folds
    ate_1st_dic = mcf_est.aggregate_pots(
        mcf_1st, y_pot_f, y_pot_var_f, txt_w_f, ate_1st_dic, fold,
        title='ATE (1st stage)')

    # Reduced form
    (w_ate_redf, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_redf, data_df, weights_redf_dic, late=False)
    # Aggregate ATEs over folds
    ate_redf_dic = mcf_est.aggregate_pots(
        mcf_redf, y_pot_f, y_pot_var_f, txt_w_f, ate_redf_dic, fold,
        title='ATE (reduced form)')

    return w_ate, ate_dic, w_ate_1st, ate_1st_dic, w_ate_redf, ate_redf_dic


def bala_pred(instances, weights, bala_1st_dic, bala_redf_dic, data_df, fold):
    """Perform balancing tests for 1st stage and reduced form."""
    _, mcf_1st, mcf_redf = instances
    _, weights_1st_dic, weights_redf_dic = weights
    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_1st, data_df, weights_1st_dic, balancing_test=True)
    # Aggregate Balancing results over folds
    bala_1st_dic = mcf_est.aggregate_pots(
        mcf_1st, y_pot_f, y_pot_var_f, txt_w_f, bala_1st_dic,
        fold, title='Reduced form balancing check: ')

    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_redf, data_df, weights_redf_dic, balancing_test=True)
    # Aggregate Balancing results over folds
    bala_redf_dic = mcf_est.aggregate_pots(
        mcf_redf, y_pot_f, y_pot_var_f, txt_w_f, bala_redf_dic,
        fold, title='Reduced form balancing check: ')

    return bala_1st_dic, bala_redf_dic


def bgate_pred(self, data_df, weights_dic, w_ate, bgate_dic, bgate_m_ate_dic,
               fold, late_tuple=None, gate_type='BGATE', title='LBGATE'):
    """Compute BGATE."""
    (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f,
     y_pot_mate_var_bgate_f, bgate_est_dic, txt_w_f, txt_b,
     ) = bgate_est(self, data_df, weights_dic, w_ate, None,
                   gate_type=gate_type, late_tuple=late_tuple)
    bgate_dic = mcf_est.aggregate_pots(
        self, y_pot_bgate_f, y_pot_var_bgate_f, txt_w_f,
        bgate_dic, fold, pot_is_list=True, title=title)
    if y_pot_mate_bgate_f is not None:
        bgate_m_ate_dic = mcf_est.aggregate_pots(
            self, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f, txt_w_f,
            bgate_m_ate_dic, fold, pot_is_list=True,
            title=title + ' minus LATE')

    return bgate_dic, bgate_m_ate_dic, bgate_est_dic, txt_b


def iate_pred(self, weights_dic, iate_dic, iate_m_ate_dic, iate_eff_dic,
              w_ate, y_pot_iate_f, round_, fold, late=True, title='LIATE'):
    """Compute IATEs."""
    (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f,
     txt_w_f) = mcf_iate.iate_est_mp(
         self, weights_dic, w_ate, round_ == 'regular', late=True)
    if round_ == 'regular':
        y_pot_iate_f = y_pot_f.copy()
        y_pot_varf = (None if y_pot_var_f is None else y_pot_var_f.copy())
        iate_dic = mcf_est.aggregate_pots(
            self, y_pot_iate_f, y_pot_varf, txt_w_f, iate_dic,
            fold, title=title)
        if y_pot_m_ate_f is not None:
            titel2 = title + ' minus LATE' if late else title + ' minus ATE'
            iate_m_ate_dic = mcf_est.aggregate_pots(
                self, y_pot_m_ate_f, y_pot_m_ate_var_f, txt_w_f, iate_m_ate_dic,
                fold, title=titel2)
    else:
        y_pot_eff = (y_pot_iate_f + y_pot_f) / 2
        iate_eff_dic = mcf_est.aggregate_pots(
            self, y_pot_eff, None, txt_w_f, iate_eff_dic, fold,
            title=title + ' eff')

    return iate_dic, iate_eff_dic, y_pot_iate_f


def gate_pred(self, weights_dic, w_ate, gate_dic, gate_m_ate_dic, data_df,
              fold):
    """Compute GATEs."""
    (y_pot_gate_f, y_pot_var_gate_f, y_pot_mate_gate_f,
     y_pot_mate_var_gate_f, gate_est_dic, txt_w_f) = gate_est(
         self, data_df, weights_dic, w_ate, late=True)
    gate_dic = mcf_est.aggregate_pots(
        self, y_pot_gate_f, y_pot_var_gate_f, txt_w_f,
        gate_dic, fold, pot_is_list=True, title='LGATE')
    if y_pot_mate_gate_f is not None:
        gate_m_ate_dic = mcf_est.aggregate_pots(
            self, y_pot_mate_gate_f, y_pot_mate_var_gate_f,
            txt_w_f, gate_m_ate_dic, fold, pot_is_list=True,
            title='LGATE minus LATE')

    return gate_dic, gate_m_ate_dic, gate_est_dic
