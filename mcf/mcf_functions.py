"""Created on Mon May 8 2023.

Contains the class and the functions needed for running the mcf.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time

import pandas as pd

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_common_support_functions as mcf_cs
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_feature_selection_functions as mcf_fs
from mcf import mcf_forest_functions as mcf_fo
from mcf import mcf_gate_functions as mcf_gate
from mcf import mcf_gateout_functions as mcf_gateout
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_init_functions as mcf_init
from mcf import mcf_local_centering_functions as mcf_lc
from mcf import mcf_post_functions as mcf_post
from mcf import mcf_print_stats_functions as ps
from mcf import mcf_weight_functions as mcf_w


class ModifiedCausalForest:
    """Class used for all estimation steps of the mcf."""

    def __init__(
            self,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05,
            cf_boot=1000, cf_chunks_maxsize=None,
            cf_n_min_grid=None, cf_n_min_max=None,
            cf_n_min_min=None, cf_n_min_treat=None, cf_nn_main_diag_only=False,
            cf_m_grid=None, cf_m_random_poisson=True, cf_m_share_max=None,
            cf_m_share_min=None, cf_match_nn_prog_score=True, cf_mce_vart=1,
            cf_p_diff_penalty=None, cf_subsample_factor_eval=None,
            cf_subsample_factor_forest=None, cf_random_thresholds=None,
            cf_vi_oob_yes=False,
            cs_adjust_limits=None, cs_max_del_train=0.5, cs_min_p=0.1,
            cs_quantil=1, cs_type=1,
            ct_grid_dr=100, ct_grid_nn=10, ct_grid_w=10,
            dc_check_perfectcorr=True, dc_clean_data=True, dc_min_dummy_obs=10,
            dc_screen_covariates=True, fs_rf_threshold=0.0001,
            fs_other_sample=True, fs_other_sample_share=0.25, fs_yes=False,
            gen_d_type=None, gen_iate_eff=True, gen_panel_data=None,
            gen_mp_parallel=None, gen_outfiletext=None, gen_outpath=None,
            gen_output_type=2, gen_panel_in_rf=True, gen_replication=False,
            gen_weighted=False,
            lc_cs_cv=True, lc_cs_cv_k=5, lc_cs_share=0.25,
            lc_uncenter_po=True, lc_yes=True,
            post_bin_corr_threshold=True, post_bin_corr_yes=0.1,
            post_est_stats=True, post_kmeans_no_of_groups=None,
            post_kmeans_max_tries=1000, post_kmeans_replications=10,
            post_kmeans_yes=True, post_random_forest_vi=True,
            post_relative_to_first_group_only=True, post_plots=True,
            p_amgate=False, p_atet=False, p_bgate=False, p_bt_yes=True,
            p_choice_based_sampling=False, p_choice_based_probs=None,
            p_ci_level=0.90, p_cluster_std=False, p_cond_var=True,
            p_gates_minus_previous=False, p_gates_smooth=True,
            p_gates_smooth_bandwidth=1, p_gates_smooth_no_evalu_points=50,
            p_gatet=False, p_gmate_no_evalu_points=50,
            p_gmate_sample_share=None, p_iate=True, p_iate_se=False,
            p_iate_m_ate=False, p_knn=True, p_knn_const=1, p_knn_min_k=10,
            p_nw_bandw=1, p_nw_kern=1, p_max_cats_z_vars=None,
            p_max_weight_share=0.05, p_se_boot_ate=False, p_se_boot_gate=False,
            p_se_boot_iate=False, var_bgate_name=None, var_cluster_name=None,
            var_d_name=None, var_id_name=None, var_w_name=None,
            var_x_balance_name_ord=None, var_x_balance_name_unord=None,
            var_x_name_always_in_ord=None, var_x_name_always_in_unord=None,
            var_x_name_remain_ord=None, var_x_name_remain_unord=None,
            var_x_name_ord=None, var_x_name_unord=None, var_y_name=None,
            var_y_tree_name=None, var_z_name_list=None,
            var_z_name_ord=None, var_z_name_unord=None,
            _int_descriptive_stats=True, _int_dpi=500, _int_fontsize=2,
            _int_no_filled_plot=20, _int_max_cats_cont_vars=None,
            _int_max_save_values=50, _int_mp_ray_del=('refs',),
            _int_mp_ray_objstore_multiplier=1, _int_mp_ray_shutdown=None,
            _int_mp_vim_type=None, _int_mp_weights_tree_batch=None,
            _int_mp_weights_type=1, _int_ray_or_dask='ray',
            _int_red_largest_group_train=False,
            _int_red_largest_group_train_share=0.5,
            _int_red_prediction=False, _int_red_prediction_share=0.5,
            _int_red_split_sample=False, _int_red_split_sample_pred_share=0.5,
            _int_red_training=False, _int_red_training_share=0.5,
            _int_return_iate_sp=False, _int_seed_sample_split=67567885,
            _int_share_forest_sample=0.5, _int_show_plots=True,
            _int_verbose=True, _int_smaller_sample=None,
            _int_weight_as_sparse=True, _int_weight_as_sparse_splits=None,
            _int_with_output=True
            ):
        self.int_dict = mcf_init.int_init(
            descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
            fontsize=_int_fontsize, no_filled_plot=_int_no_filled_plot,
            max_save_values=_int_max_save_values, mp_ray_del=_int_mp_ray_del,
            mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
            mp_weights_tree_batch=_int_mp_weights_tree_batch,
            mp_weights_type=_int_mp_weights_type, ray_or_dask=_int_ray_or_dask,
            return_iate_sp=_int_return_iate_sp,
            seed_sample_split=_int_seed_sample_split,
            share_forest_sample=_int_share_forest_sample,
            show_plots=_int_show_plots, verbose=_int_verbose,
            smaller_sample=_int_smaller_sample,
            weight_as_sparse=_int_weight_as_sparse,
            with_output=_int_with_output, mp_ray_shutdown=_int_mp_ray_shutdown,
            mp_vim_type=_int_mp_vim_type,
            weight_as_sparse_splits=_int_weight_as_sparse_splits,
            max_cats_cont_vars=_int_max_cats_cont_vars)
        gen_dict = mcf_init.gen_init(
            self.int_dict,
            d_type=gen_d_type, iate_eff=gen_iate_eff,
            mp_parallel=gen_mp_parallel, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            replication=gen_replication,  weighted=gen_weighted,
            panel_data=gen_panel_data, panel_in_rf=gen_panel_in_rf)
        self.dc_dict = mcf_init.dc_init(
            check_perfectcorr=dc_check_perfectcorr,
            clean_data=dc_clean_data, min_dummy_obs=dc_min_dummy_obs,
            screen_covariates=dc_screen_covariates)
        self.ct_dict = {'grid_dr': ct_grid_dr, 'grid_nn': ct_grid_nn,
                        'grid_w': ct_grid_w}
        self.fs_dict = mcf_init.fs_init(
            rf_threshold=fs_rf_threshold, other_sample=fs_other_sample,
            other_sample_share=fs_other_sample_share, yes=fs_yes)
        self.cs_dict = mcf_init.cs_init(
            gen_dict,
            max_del_train=cs_max_del_train, min_p=cs_min_p, quantil=cs_quantil,
            type_=cs_type, adjust_limits=cs_adjust_limits)
        self.lc_dict = mcf_init.lc_init(
            cs_cv=lc_cs_cv, cs_cv_k=lc_cs_cv_k, cs_share=lc_cs_share,
            undo_iate=lc_uncenter_po, yes=lc_yes)
        self.cf_dict = mcf_init.cf_init(
            alpha_reg_grid=cf_alpha_reg_grid, alpha_reg_max=cf_alpha_reg_max,
            alpha_reg_min=cf_alpha_reg_min, boot=cf_boot,
            chunks_maxsize=cf_chunks_maxsize,
            nn_main_diag_only=cf_nn_main_diag_only, m_grid=cf_m_grid,
            m_share_max=cf_m_share_max, m_share_min=cf_m_share_min,
            m_random_poisson=cf_m_random_poisson,
            match_nn_prog_score=cf_match_nn_prog_score, mce_vart=cf_mce_vart,
            vi_oob_yes=cf_vi_oob_yes, n_min_grid=cf_n_min_grid,
            n_min_max=cf_n_min_max, n_min_min=cf_n_min_min,
            n_min_treat=cf_n_min_treat,
            p_diff_penalty=cf_p_diff_penalty,
            subsample_factor_eval=cf_subsample_factor_eval,
            subsample_factor_forest=cf_subsample_factor_forest,
            random_thresholds=cf_random_thresholds)
        p_dict = mcf_init.p_init(
            gen_dict,
            amgate=p_amgate, atet=p_atet, bgate=p_bgate, bt_yes=p_bt_yes,
            choice_based_sampling=p_choice_based_sampling,
            choice_based_probs=p_choice_based_probs, ci_level=p_ci_level,
            cluster_std=p_cluster_std, cond_var=p_cond_var,
            gates_minus_previous=p_gates_minus_previous,
            gates_smooth=p_gates_smooth,
            gates_smooth_bandwidth=p_gates_smooth_bandwidth,
            gates_smooth_no_evalu_points=p_gates_smooth_no_evalu_points,
            gatet=p_gatet, gmate_no_evalu_points=p_gmate_no_evalu_points,
            gmate_sample_share=p_gmate_sample_share,
            iate=p_iate, iate_se=p_iate_se, iate_m_ate=p_iate_m_ate, knn=p_knn,
            knn_const=p_knn_const, knn_min_k=p_knn_min_k, nw_bandw=p_nw_bandw,
            nw_kern=p_nw_kern, max_cats_z_vars=p_max_cats_z_vars,
            max_weight_share=p_max_weight_share,
            se_boot_ate=p_se_boot_ate, se_boot_gate=p_se_boot_gate,
            se_boot_iate=p_se_boot_iate)
        self.post_dict = mcf_init.post_init(
            p_dict,
            bin_corr_threshold=post_bin_corr_threshold,
            bin_corr_yes=post_bin_corr_yes, est_stats=post_est_stats,
            kmeans_no_of_groups=post_kmeans_no_of_groups,
            kmeans_max_tries=post_kmeans_max_tries,
            kmeans_replications=post_kmeans_replications,
            kmeans_yes=post_kmeans_yes, random_forest_vi=post_random_forest_vi,
            relative_to_first_group_only=post_relative_to_first_group_only,
            plots=post_plots)
        self.var_dict, self.gen_dict, self.p_dict = mcf_init.var_init(
            gen_dict, self.fs_dict, p_dict,
            bgate_name=var_bgate_name, cluster_name=var_cluster_name,
            d_name=var_d_name, id_name=var_id_name, w_name=var_w_name,
            x_balance_name_ord=var_x_balance_name_ord,
            x_balance_name_unord=var_x_balance_name_unord,
            x_name_always_in_ord=var_x_name_always_in_ord,
            x_name_always_in_unord=var_x_name_always_in_unord,
            x_name_remain_ord=var_x_name_remain_ord,
            x_name_remain_unord=var_x_name_remain_unord,
            x_name_ord=var_x_name_ord, x_name_unord=var_x_name_unord,
            y_name=var_y_name, y_tree_name=var_y_tree_name,
            z_name_list=var_z_name_list, z_name_ord=var_z_name_ord,
            z_name_unord=var_z_name_unord)
        self.data_train_dict = self.var_x_type = self.var_x_values = None
        self.forest, self.time_strings = None, {}

    def train(self, data_df):
        """Training the causal forest."""
        time_start = time()
        # Check treatment data
        data_df, _ = mcf_data.data_frame_vars_upper(data_df)
        data_df = mcf_data.check_recode_treat_variable(self, data_df)
        # Initialise again with data information
        mcf_init.var_update_train(self, data_df)
        mcf_init.p_update_train(self, data_df)
        mcf_init.gen_update_train(self, data_df)
        mcf_init.ct_update_train(self, data_df)
        mcf_init.cs_update_train(self)
        mcf_init.cf_update_train(self, data_df)
        mcf_init.int_update_train(self, len(data_df))

        if self.int_dict['with_output']:
            ps.print_dic_values_all(self, summary_top=True, summary_dic=False)

        # Prepare data: Add and recode variables for GATES (Z)
        #             Recode categorical variables to prime numbers, cont. vars
        data_df = mcf_data.create_xz_variables(self, data_df, train=True)
        if self.int_dict['with_output'] and self.int_dict['verbose']:
            mcf_data.print_prime_value_corr(self.data_train_dict,
                                            self.gen_dict, summary=False)

        # Clean data and remove missings and unncessary variables
        if self.dc_dict['clean_data']:
            data_df = mcf_data.clean_data(self, data_df, train=True)
        if self.dc_dict['screen_covariates']:   # Only training
            (self.gen_dict, self.var_dict, self.var_x_type, self.var_x_values
             ) = mcf_data.screen_adjust_variables(self, data_df)
        time_1 = time()

        # Descriptives by treatment
        if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
            ps.desc_by_treatment(self, data_df, summary=False, stage=1)

        # Feature selection
        if self.fs_dict['yes']:
            data_df = mcf_fs.feature_selection(self, data_df)
        # Split sample for tree building and tree-filling-with-y
        tree_df, fill_y_df = mcf_data.split_sample_for_mcf(self, data_df)
        del data_df
        time_2 = time()

        # Compute Common support
        if self.cs_dict['type']:
            tree_df, fill_y_df = mcf_cs.common_support(self, tree_df,
                                                       fill_y_df, train=True)
        # Descriptives by treatment on common support
        if self.int_dict['descriptive_stats'] and self.int_dict['with_output']:
            ps.desc_by_treatment(self, pd.concat([tree_df, fill_y_df], axis=0),
                                 summary=True, stage=1)
        time_3 = time()

        # Local centering
        if self.lc_dict['yes']:
            (tree_df, fill_y_df, _) = mcf_lc.local_centering(self, tree_df,
                                                             fill_y_df)
        time_4 = time()
        if self.int_dict['with_output']:
            ps.variable_features(self, summary=False)
        self.cf_dict, self.forest, time_vi = mcf_fo.train_forest(
            self, tree_df, fill_y_df)
        time_end = time()
        time_string = ['Data preparation and stats I:                   ',
                       'Feature preselection:                           ',
                       'Common support:                                 ',
                       'Local centering (recoding of Y):                ',
                       'Training the causal forest:                     ',
                       '  ... of which is time for variable importance: ',
                       '\nTotal time training:                            ']
        time_difference = [time_1 - time_start, time_2 - time_1,
                           time_3 - time_2, time_4 - time_3,
                           time_end - time_4, time_vi,
                           time_end - time_start]
        if self.int_dict['with_output']:
            time_train = ps.print_timing(
                self.gen_dict, 'Training', time_string, time_difference,
                summary=True)
            self.time_strings['time_train'] = time_train
        return tree_df, fill_y_df

    def predict(self, data_df):
        """Predictions of parameters."""
        time_start = time()
        data_df, _ = mcf_data.data_frame_vars_upper(data_df)
        # Initialise again with data information
        data_df = mcf_init.p_update_pred(self, data_df)
        # Check treatment data
        if self.p_dict['d_in_pred']:
            data_df = mcf_data.check_recode_treat_variable(self, data_df)
        # self.var_dict = mcf_init.ct_update_pred(self)
        mcf_init.int_update_pred(self, len(data_df))
        mcf_init.post_update_pred(self, data_df)
        if self.int_dict['with_output']:
            ps.print_dic_values_all(self, summary_top=True, summary_dic=False,
                                    train=False)
        # Prepare data: Add and recode variables for GATES (Z)
        #             Recode categorical variables to prime numbers, cont. vars
        data_df = mcf_data.create_xz_variables(self, data_df, train=False)
        if self.int_dict['with_output'] and self.int_dict['verbose']:
            mcf_data.print_prime_value_corr(self.data_train_dict,
                                            self.gen_dict, summary=False)
        # Clean data and remove missings and unncessary variables
        if self.dc_dict['clean_data']:
            data_df = mcf_data.clean_data(self, data_df, train=False)
        # Descriptives by treatment on common support
        if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
                and self.int_dict['with_output']):
            ps.desc_by_treatment(self, data_df, summary=False, stage=3)
        time_1 = time()
        # Check if on common support
        if self.cs_dict['type']:
            data_df, _ = mcf_cs.common_support(self, data_df, None,
                                               train=False)
        data_df = data_df.copy().reset_index()
        if (self.p_dict['d_in_pred'] and self.int_dict['descriptive_stats']
                and self.int_dict['with_output']):
            ps.desc_by_treatment(self, data_df, summary=True, stage=3)
        time_2 = time()
        # Local centering for IATE
        if self.lc_dict['yes'] and self.lc_dict['uncenter_po']:
            (_, _, y_pred_x_df) = mcf_lc.local_centering(self, data_df, None,
                                                         train=False)
        else:
            y_pred_x_df = 0
        time_3 = time()
        time_delta_weight = time_delta_ate = time_delta_bala = 0
        time_delta_iate = time_delta_gate = time_delta_amgate = 0
        time_delta_bgate = 0
        ate_dic = bala_dic = iate_dic = iate_m_ate_dic = iate_eff_dic = None
        gate_dic = gate_m_ate_dic = amgate_dic = amgate_m_ate_dic = None
        bgate_dic = bgate_m_ate_dic = None
        only_one_fold_one_round = (self.cf_dict['folds'] == 1
                                   and len(self.cf_dict['est_rounds']) == 1)
        for fold in range(self.cf_dict['folds']):
            for round_ in self.cf_dict['est_rounds']:
                time_w_start = time()
                if only_one_fold_one_round:
                    forest_dic = self.forest[fold][0]
                else:
                    forest_dic = deepcopy(
                        self.forest[fold][0 if round_ == 'regular' else 1])
                if self.int_dict['with_output'] and self.int_dict['verbose']:
                    print(f'\n\nWeight maxtrix {fold+1} /',
                          f'{self.cf_dict["folds"]} forests, {round_}')
                weights_dic = mcf_w.get_weights_mp(
                    self, data_df, forest_dic, round_ == 'regular')
                time_delta_weight += time() - time_w_start
                time_a_start = time()
                if round_ == 'regular':
                    # Estimate ATE n fold
                    (w_ate, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                        self, data_df, weights_dic)
                    # Aggregate ATEs over folds
                    ate_dic = mcf_est.aggregate_pots(
                        self, y_pot_f, y_pot_var_f, txt_w_f, ate_dic, fold,
                        title='ATE')
                else:
                    w_ate = None
                time_delta_ate += time() - time_a_start
                # Compute balancing tests
                time_b_start = time()
                if self.p_dict['bt_yes']:
                    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                        self, data_df, weights_dic, balancing_test=True)
                    # Aggregate Balancing results over folds
                    bala_dic = mcf_est.aggregate_pots(
                        self, y_pot_f, y_pot_var_f, txt_w_f, bala_dic, fold,
                        title='Balancing check: ')
                time_delta_bala += time() - time_b_start

                # BGATE
                time_bgate_start = time()
                if round_ == 'regular' and self.p_dict['bgate']:
                    (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f,
                     y_pot_mate_var_bgate_f, bgate_est_dic, txt_w_f, txt_b,
                     ) = mcf_gate.bamgate_est(self, data_df, weights_dic,
                                              w_ate, forest_dic,
                                              gate_type='BGATE')
                    bgate_dic = mcf_est.aggregate_pots(
                        self, y_pot_bgate_f, y_pot_var_bgate_f, txt_w_f,
                        bgate_dic, fold, pot_is_list=True, title='BGATE')
                    if y_pot_mate_bgate_f is not None:
                        bgate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
                            txt_w_f, bgate_m_ate_dic, fold, pot_is_list=True,
                            title='BGATE minus ATE')
                time_delta_bgate += time() - time_bgate_start

                # AMGATE
                time_amg_start = time()
                if round_ == 'regular' and self.p_dict['amgate']:
                    (y_pot_amgate_f, y_pot_var_amgate_f, y_pot_mate_amgate_f,
                     y_pot_mate_var_amgate_f, amgate_est_dic, txt_w_f, txt_am,
                     ) = mcf_gate.bamgate_est(self, data_df, weights_dic,
                                              w_ate, forest_dic,
                                              gate_type='AMGATE')
                    amgate_dic = mcf_est.aggregate_pots(
                        self, y_pot_amgate_f, y_pot_var_amgate_f, txt_w_f,
                        amgate_dic, fold, pot_is_list=True, title='AMGATE')
                    if y_pot_mate_amgate_f is not None:
                        amgate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_amgate_f, y_pot_mate_var_amgate_f,
                            txt_w_f, amgate_m_ate_dic, fold, pot_is_list=True,
                            title='AMGATE minus ATE')
                time_delta_amgate += time() - time_amg_start
                del forest_dic['forest']

                # IATE
                time_i_start = time()
                if self.p_dict['iate']:
                    y_pot_eff = None
                    (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f,
                     txt_w_f) = mcf_iate.iate_est_mp(
                         self, weights_dic, w_ate, round_ == 'regular')
                    if round_ == 'regular':
                        y_pot_iate_f = y_pot_f.copy()
                        y_pot_varf = (None if y_pot_var_f is None
                                      else y_pot_var_f.copy())
                        iate_dic = mcf_est.aggregate_pots(
                            self, y_pot_iate_f, y_pot_varf, txt_w_f, iate_dic,
                            fold, title='IATE')
                        if y_pot_m_ate_f is not None:
                            iate_m_ate_dic = mcf_est.aggregate_pots(
                                self, y_pot_m_ate_f, y_pot_m_ate_var_f,
                                txt_w_f, iate_m_ate_dic, fold,
                                title='IATE minus ATE')
                    else:
                        y_pot_eff = (y_pot_iate_f + y_pot_f) / 2
                        iate_eff_dic = mcf_est.aggregate_pots(
                            self, y_pot_eff, None, txt_w_f, iate_eff_dic, fold,
                            title='IATE eff')
                time_delta_iate += time() - time_i_start

                # GATE
                time_g_start = time()
                if round_ == 'regular' and self.p_dict['gate']:
                    (y_pot_gate_f, y_pot_var_gate_f, y_pot_mate_gate_f,
                     y_pot_mate_var_gate_f, gate_est_dic, txt_w_f
                     ) = mcf_gate.gate_est(self, data_df, weights_dic, w_ate)
                    gate_dic = mcf_est.aggregate_pots(
                        self, y_pot_gate_f, y_pot_var_gate_f, txt_w_f,
                        gate_dic, fold, pot_is_list=True, title='GATE')
                    if y_pot_mate_gate_f is not None:
                        gate_m_ate_dic = mcf_est.aggregate_pots(
                            self, y_pot_mate_gate_f, y_pot_mate_var_gate_f,
                            txt_w_f, gate_m_ate_dic, fold, pot_is_list=True,
                            title='GATE minus ATE')
                time_delta_gate += time() - time_g_start
            if not only_one_fold_one_round:
                self.forest[fold] = None
        self.forest = None
        del weights_dic

        # ATE
        time_a_start = time()
        ate, ate_se, ate_effect_list = mcf_ate.ate_effects_print(
            self, ate_dic, y_pred_x_df, balancing_test=False)
        time_delta_ate += time() - time_a_start

        # GATE
        time_g_start = time()
        if self.p_dict['gate']:
            (gate, gate_se, gate_diff, gate_diff_se
             ) = mcf_gateout.gate_effects_print(self, gate_dic, gate_m_ate_dic,
                                                gate_est_dic, ate, ate_se,
                                                gate_type='GATE')
        else:
            gate = gate_se = gate_diff = gate_diff_se = None
        time_delta_gate += time() - time_g_start

        # BGATE
        time_bgate_start = time()
        if self.p_dict['bgate']:
            (bgate, bgate_se, bgate_diff, bgate_diff_se
             ) = mcf_gateout.gate_effects_print(
                 self, bgate_dic, bgate_m_ate_dic, bgate_est_dic, ate,
                 ate_se, gate_type='BGATE', special_txt=txt_b)
        else:
            bgate = bgate_se = bgate_diff = bgate_diff_se = None
        time_delta_bgate += time() - time_bgate_start

        # AMGATE
        time_amg_start = time()
        if self.p_dict['amgate']:
            (amgate, amgate_se, amgate_diff, amgate_diff_se
             ) = mcf_gateout.gate_effects_print(
                 self, amgate_dic, amgate_m_ate_dic, amgate_est_dic, ate,
                 ate_se, gate_type='AMGATE', special_txt=txt_am)
        else:
            amgate = amgate_se = amgate_diff = amgate_diff_se = None
        time_delta_amgate += time() - time_amg_start

        # IATE
        time_i_start = time()
        if self.p_dict['iate']:
            (iate, iate_se, iate_eff, iate_names_dic, iate_df
             ) = mcf_iate.iate_effects_print(self, iate_dic, iate_m_ate_dic,
                                             iate_eff_dic, y_pred_x_df)
            data_df.reset_index(drop=True, inplace=True)
            iate_df.reset_index(drop=True, inplace=True)
            iate_pred_df = pd.concat([data_df, iate_df], axis=1)
        else:
            iate_eff = iate = iate_se = iate_df = iate_pred_df = None
            iate_names_dic = None
        time_delta_iate += time() - time_i_start

        # Balancing test
        time_b_start = time()
        if self.p_dict['bt_yes']:
            bala, bala_se, bala_effect_list = mcf_ate.ate_effects_print(
                self, bala_dic, None, balancing_test=True)
        else:
            bala = bala_se = bala_effect_list = None
        time_delta_bala += time() - time_b_start
        # Collect results
        results = {
            'ate': ate, 'ate_se': ate_se, 'ate effect_list': ate_effect_list,
            'gate': gate, 'gate_se': gate_se,
            'gate_diff': gate_diff, 'gate_diff_se': gate_diff_se,
            'amgate': amgate, 'amgate_se': amgate_se,
            'amgate_diff': amgate_diff, 'amgate_diff_se': amgate_diff_se,
            'bgate': bgate, 'bgate_se': bgate_se,
            'bgate_diff': bgate_diff, 'bgate_diff_se': bgate_diff_se,
            'iate': iate, 'iate_se': iate_se, 'iate_eff': iate_eff,
            'iate_data_df': iate_pred_df, 'iate_names_dic': iate_names_dic,
            'bala': bala, 'bala_se': bala_se, 'bala_effect_list':
                bala_effect_list
                   }
        time_end = time()
        if self.int_dict['with_output']:
            time_string = [
                'Data preparation and stats II:                  ',
                'Common support:                                 ',
                'Local centering (recoding of Y):                ',
                'Weights:                                        ',
                'ATEs:                                           ',
                'GATEs:                                          ',
                'BGATEs:                                         ',
                'AMGATEs:                                        ',
                'IATEs:                                          ',
                'Balancing test:                                 ',
                '\nTotal time prediction:                          ']
            time_difference = [
                time_1 - time_start, time_2 - time_1, time_3 - time_2,
                time_delta_weight, time_delta_ate, time_delta_gate,
                time_delta_bgate, time_delta_amgate, time_delta_iate,
                time_delta_bala, time_end - time_start]
            ps.print_mcf(self.gen_dict, self.time_strings['time_train'])
            time_pred = ps.print_timing(
                self.gen_dict, 'Prediction', time_string, time_difference,
                summary=True)
            self.time_strings['time_pred'] = time_pred
        return results

    def analyse(self, results):
        """Print and analyse the results."""
        if (self.int_dict['with_output'] and self.post_dict['est_stats'] and
                self.int_dict['return_iate_sp']):
            time_start = time()
            mcf_post.post_estimation_iate(self, results)
            time_end_corr = time()
            if self.post_dict['kmeans_yes']:
                results_plus_cluster = mcf_post.k_means_of_x_iate(self,
                                                                  results)
            time_end_km = time()
            if self.post_dict['random_forest_vi']:
                mcf_post.random_forest_of_iate(self, results)
            time_string = [
                'Correlational analysis and plots of IATE:       ',
                'K-means clustering of IATE:                     ',
                'Random forest analysis of IATE:                 ',
                '\nTotal time post estimation analysis:            ']
            time_difference = [
                time_end_corr - time_start, time_end_km - time_end_corr,
                time() - time_end_km, time() - time_start]
            ps.print_mcf(self.gen_dict, self.time_strings['time_train'],
                         summary=True)
            ps.print_mcf(self.gen_dict, self.time_strings['time_pred'],
                         summary=True)
            ps.print_timing(self.gen_dict, 'Analysis of IATE', time_string,
                            time_difference, summary=True)
        else:
            raise ValueError(
                '"Analyse" method produces output only if all of the following'
                ' parameters are True:'
                f'\nint_with_output: {self.int_dict["with_output"]}'
                f'\npos_test_stats: {self.post_dict["est_stats"]}'
                f'\nint_return_iate_sp: {self.int_dict["return_iate_sp"]}')
        return results_plus_cluster
