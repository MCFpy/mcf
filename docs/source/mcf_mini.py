import numpy as np
import pandas as pd
import io


class ModifiedCausalForest:
    """
    The :class:`ModifiedCausalForest` contains all methods necessary for a complete mcf estimation.

    Parameters
    ----------
    cf_alpha_reg_grid : Integer (or None), optional
        Minimum remaining share when splitting leaf: Number of grid values.
        If grid is used, optimal value is determined by out-of-bag
        estimation of objective function. 
        Default (or ``None``) is ``1``.
        
        .. versionadded:: 0.4.4
        

    cf_alpha_reg_max : Float (or None), optional
        Minimum remaining share when splitting leaf: Largest value of
        grid (keep it below ``0.2``).
        Example of inline mathematics :math:`a^2 + b^2 = c^2`. 
        
        .. math:: 
            
            \\text{cf_chunks_maxsize} = \\text{round}(60000 + \\sqrt{\\text{number of observations} - 60000})
            
        Default (or ``None``) is ``0.15``.

    cf_alpha_reg_min : Integer (or None), optional
        Minimum leaf size: Largest minimum leaf size.
        If None: 
            
        .. math::
                        
            \\text{A} = \\frac{\\sqrt{\\text{{number of observations in the smallest treatment group}}^{0.5}}}{10},
            
            \\text{cf_n_min_max} = \\text{round}(A * \\text{number of treatments})  
                                         
        Default is ``None``.

    """

    def __init__(
            self,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05
            ):
        """
        Define Constructor for ModifiedCausalForest class.

        """
        self.int_dict = mcf_init.int_init(
            del_forest=_int_del_forest,
            descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
            fontsize=_int_fontsize, no_filled_plot=_int_no_filled_plot,
            max_save_values=_int_max_save_values, mp_ray_del=_int_mp_ray_del,
            mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
            mp_weights_tree_batch=_int_mp_weights_tree_batch,
            mp_weights_type=_int_mp_weights_type,
            return_iate_sp=_int_return_iate_sp,
            seed_sample_split=_int_seed_sample_split,
            share_forest_sample=_int_share_forest_sample,
            show_plots=_int_show_plots, verbose=_int_verbose,
            weight_as_sparse=_int_weight_as_sparse, keep_w0=_int_keep_w0,
            with_output=_int_with_output, mp_ray_shutdown=_int_mp_ray_shutdown,
            mp_vim_type=_int_mp_vim_type,
            output_no_new_dir=_int_output_no_new_dir,
            weight_as_sparse_splits=_int_weight_as_sparse_splits,
            max_cats_cont_vars=_int_max_cats_cont_vars,
            p_ate_no_se_only=p_ate_no_se_only)
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
            ate_no_se_only=p_ate_no_se_only, amgate=p_amgate, atet=p_atet,
            bgate=p_bgate, bt_yes=p_bt_yes,
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
        self.blind_dict = self.sens_dict = None
        self.data_train_dict = self.var_x_type = self.var_x_values = None
        self.forest, self.time_strings = None, {}
        
        
    @property
    def blind_dict(self):
        """
        ``Dictionary``, parameters to compute (partially) blinded IATEs. 
        """
        return self._blind_dict    
    
    @property
    def cf_dict(self):
        """
        ``Dictionary``, parameters used in training the forest (directly).
        """
        return self._cf_dict    
    
    @property
    def cs_dict(self):
        """
        ``Dictionary``, parameters used in common support adjustments
        """
        return self._cs_dict 


    def train(self, data_df):
        """
        Build the modified causal forest on the training data.

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

        """
        time_start = time()
        # Check treatment data
        data_df, _ = mcf_data.data_frame_vars_upper(data_df)
        data_df = mcf_data.check_recode_treat_variable(self, data_df)
        # Initialise again with data information. Order of the following
        # init functions important. Change only if you know what you do.
        mcf_init.var_update_train(self, data_df)
        mcf_init.gen_update_train(self, data_df)
        mcf_init.ct_update_train(self, data_df)
        mcf_init.cs_update_train(self)            # Used updated gen_dict info
        mcf_init.cf_update_train(self, data_df)
        mcf_init.int_update_train(self)
        mcf_init.p_update_train(self)

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

        # Train forest
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
        """
        Compute all effects given a causal forest estimated with train method.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
                            subpopulations are desired as well).

        Returns
        -------
        results : Dictionary.
            Results. This dictionary has the following structure:
            'ate': ATE, 'ate_se': Standard error of ATE,
            'ate effect_list': List of names of estimated effects,
            'gate': GATE, 'gate_se': SE of GATE,
            'gate_diff': GATE minus ATE,
            'gate_diff_se': Standard error of GATE minus ATE,
            'amgate': AMGATE (all covariates balanced),
            'amgate_se': Standard error of AMGATE,
            'amgate_diff': AMGATE minus ATE,
            'amgate_diff_se': Standard error of AMGATE minus ATE,
            'bgate': BGATE (only prespecified covariates balanced),
            'bgate_se': Standard error of BGATE,
            'bgate_diff': BGATE minus ATE,
            'bgate_diff_se': Standard errror of BGATE minus ATE,
            'gate_names_values': Dictionary: Order of gates parameters and
                        name and values of GATE effects.
            'iate': IATE, 'iate_se': Standard error of IATE,
            'iate_eff': (More) Efficient IATE (IATE estimated twice and
                        averaged where role of tree_building and tree_filling
                        sample is exchanged),
            'iate_data_df': DataFrame with IATEs,
            'iate_names_dic': Dictionary containing names of IATEs,
            'bala': Effects of balancing tests,
            'bala_se': Standard error of effects of balancing tests,
            'bala_effect_list': Names of effects of balancing tests
        """
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

        # Common support
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
                if round_ == 'regular':
                    if self.p_dict['bt_yes']:
                        (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
                            self, data_df, weights_dic, balancing_test=True)
                        # Aggregate Balancing results over folds
                        bala_dic = mcf_est.aggregate_pots(
                            self, y_pot_f, y_pot_var_f, txt_w_f, bala_dic,
                            fold, title='Balancing check: ')
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
                if self.int_dict['del_forest']:
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
            if not only_one_fold_one_round and self.int_dict['del_forest']:
                self.forest[fold] = None
                # Without those two deletes, it becomes impossible to reuse
                # the same forest for several data sets, which is bad.
        if self.int_dict['del_forest']:
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
            gate = gate_se = gate_diff = gate_diff_se = gate_est_dic = None
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
            bgate_est_dic = None
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
            amgate_est_dic = None
        time_delta_amgate += time() - time_amg_start
        # Collect some information for results_dic
        if (self.p_dict['gate'] or self.p_dict['bgate']
                or self.p_dict['amgate']):
            gate_names_values = mcf_gateout.get_names_values(
                self, gate_est_dic, bgate_est_dic, amgate_est_dic)
        else:
            gate_names_values = None
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
            'gate_names_values': gate_names_values,
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
        """
        Analyse estimated IATE with various descriptive tools.

        Parameters
        ----------
        results : Dictionary
            Contains estimation results. This dictionary must have the same
            structure as the one returned from the predict method.

        Raises
        ------
        ValueError
            Some of the attribute are not compatible with running this method.

        Returns
        -------
        results_plus_cluster : Dictionary
            Same as the results dictionary, but the DataFrame with estimated
            IATEs contains an additional integer with a group label that comes
            from k-means clustering.

        """
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

    def blinder_iates(
        self, data_df, blind_var_x_protected_name=None,
        blind_var_x_policy_name=None, blind_var_x_unrestricted_name=None,
        blind_weights_of_blind=None, blind_obs_ref_data=50,
            blind_seed=123456):
        """
        Compute IATEs that causally depend less on protected variables.

        Parameters
        ----------
        data_df : DataFrame.
            Contains data needed to predict the various adjusted IATES.
        blind_var_x_protected_name : List of strings (or None), optional
            Names of protected variables. Names that are
            explicitly denoted as blind_var_x_unrestricted_name or as
            blind_var_x_policy_name and used to compute IATEs will be
            automatically added to this list. Default is None.
        blind_var_x_policy_name : List of strings (or None), optional
            Names of decision variables. Default is None.
        blind_var_x_unrestricted_name : List of strings (or None), optional
            Names of unrestricted variables. Default is None.
        blind_weights_of_blind : Tuple of float (or None), optional
            Weights to compute weighted means of blinded and unblinded IATEs.
            Between 0 and 1. 1 implies all weight goes to fully blinded IATEs.
        blind_obs_ref_data : Integer (or None), optional
            Number of observations to be used for blinding. Runtime of
            programme is almost linear in this parameter. Default is 50.
        blind_seed : Integer, optional.
            Seed for the random selection of the reference data.
            Default is 123456.

        Returns
        -------
        blinded_dic : Dictionary.
            Contains potential outcomes that do not fully
            depend on some protected attributes (and non-modified IATE).
        data_on_support_df : DataFrame.
            Features that are on the common support.
        var_x_policy_ord_name : List of strings, optional
            Ordered variables to be used to build the decision rules.
        var_x_policy_unord_name : List of strings.
            Unordered variables to be used to build the decision rules.
        var_x_blind_ord_name : List of strings, optional
            Ordered variables to be used to blind potential outcomes.
        var_x_blind_unord_name : List of strings.
            Unordered variables to be used to blind potential outcomes.
        """
        self.blind_dict = mcf_init.blind_init(
            var_x_protected_name=blind_var_x_protected_name,
            var_x_policy_name=blind_var_x_policy_name,
            var_x_unrestricted_name=blind_var_x_unrestricted_name,
            weights_of_blind=blind_weights_of_blind,
            obs_ref_data=blind_obs_ref_data,
            seed=blind_seed)

        if self.int_dict['with_output']:
            time_start = time()
        with_output = self.int_dict['with_output']

        (blinded_dic, data_on_support_df, var_x_policy_ord_name,
         var_x_policy_unord_name, var_x_blind_ord_name, var_x_blind_unord_name
         ) = mcf_fair.make_fair_iates(self, data_df, with_output=with_output)

        self.int_dict['with_output'] = with_output
        if self.int_dict['with_output']:
            time_difference = [time() - time_start]
            time_string = ['Total time for blinding IATEs:                  ']
            ps.print_timing(self.gen_dict, 'Blinding IATEs', time_string,
                            time_difference, summary=True)
        return (blinded_dic, data_on_support_df, var_x_policy_ord_name,
                var_x_policy_unord_name, var_x_blind_ord_name,
                var_x_blind_unord_name)

    def sensitivity(self, train_df, predict_df=None, sens_amgate=None,
                    sens_bgate=None, sens_gate=None, sens_iate=None,
                    sens_iate_se=None, sens_scenarios=None, sens_cv_k=None,
                    sens_replications=2, sens_reference_population=None):
        """
        Compute simulation based sensitivity indicators.

        Parameters
        ----------
        train_df : DataFrame.
            Data with real outcomes, treatments, and covariates. Data will be
            transformed to compute sensitivity indicators.
        predict_df : DataFrame (or None), optinal.
            Prediction data to compute all effects for. This data will not be
            changed in the computation process. Only covariate information is
            used from this dataset. If predict_df is not a DataFrame,
            train_df will be used instead.
        sens_amgate : Boolean (or None), optional
            Compute AMGATEs for sensitivity analysis. Default is False.
        sens_bgate : Boolean (or None), optional
            Compute BGATEs for sensitivity analysis. Default is False.
        sens_gate : Boolean (or None), optional
            Compute GATEs for sensitivity analysis. Default is False.
        sens_iate : Boolean (or None), optional
            Compute IATEs for sensitivity analysis. Default is False.
        sens_iate_se : Boolean (or None), optional
            Compute Standard errors of IATEs for sensitivity analysis. Default
            is False.
        sens_scenarios : List or tuple of strings, optional.
            Different scenarios considered. Default is ('basic',).
            'basic' : Use estimated treatment probabilities for simulations.
                      No confounding.
        sens_cv_k : Integer (or None), optional
            Data to be used for any cross-validation: Number of folds in
            cross-validation. Default (or None) is 5.
        sens_replications : Integer (or None), optional.
            Number of replications for simulating placebo treatments. Default
            is 2.
        sens_reference_population: integer or float (or None)
            Defines the treatment status of the reference population used by
            the sensitivity analysis. Default is to use the treatment with most
            observed observations.

        Returns
        -------
        results_avg : Dictionary
            Same content as for the predict method but (if applicable) averaged
            over replications.

        """
        if not isinstance(predict_df, pd.DataFrame):
            predict_df = train_df.copy()
        self.sens_dict = mcf_init.sens_init(
            self.p_dict, amgate=sens_amgate, bgate=sens_bgate, gate=sens_gate,
            iate=sens_iate, iate_se=sens_iate_se, scenarios=sens_scenarios,
            cv_k=sens_cv_k, replications=sens_replications,
            reference_population=sens_reference_population)
        if self.int_dict['with_output']:
            time_start = time()
        results_avg = mcf_sens.sensitivity_analysis(
            self, train_df, predict_df, self.int_dict['with_output'],
            seed=9345467)
        if self.int_dict['with_output']:
            time_difference = [time() - time_start]
            time_string = ['Total time for sensitivity analysis:            ']
            ps.print_timing(self.gen_dict, 'Sensitiviy analysis', time_string,
                            time_difference, summary=True)
        return results_avg