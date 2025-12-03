from ray import is_initialized, shutdown
from pandas import DataFrame

from mcf.mcf_general import check_reduce_dataframe
from mcf.mcf_iv_functions import train_iv_main, predict_iv_main
from mcf import mcf_init_functions as mcf_init

from mcf.mcf_inf_for_alloc_functions import predict_different_allocations_main
from mcf.mcf_print_stats_functions import print_mcf
from mcf.mcf_sensitivity_functions import sensitivity_main
from mcf.mcf_unconfound_functions import train_main, predict_main, analyse_main
from mcf.mcf_unconfound_functions import blinder_iates_main

class ModifiedCausalForest:
    """
    Optimal policy learning

    Parameters
    ----------
    dc_screen_covariates : Boolean (or None), optional
        Check features.
        Default (or None) is True.

    Attributes
    ----------

    <NOT-ON-API>
    
    version : String
        Version of mcf module used to create the instance.

    blind_dict : Dictionary
        Parameters to compute (partially) blinded IATEs.

    cf_dict : Dictionary
        Parameters used in training the forest (directly).

    cs_dict : Dictionary
        Parameters used in common support adjustments.

    ct_dict : Dictionary
        Parameters used in dealing with continuous treatments.

    data_train_dict : Dictionary

    dc_dict : Dictionary
        Parameters used in data cleaning.

    fs_dict : Dictionary
        Parameters used in feature selection.

    forest : List
        List of lists containing the estimated causal forest.

    gen_dict : Dictionary
        General parameters used in various parts of the programme.

    int_dict : Dictionary
        Internal parameters used in various parts of the class.

    iv_mcf : Dictionary
        Internal instances of instrumental mcf containing for first stage
        and reduced form.

    lc_dict : Dictionary
        Parameters used in local centering.

    p_dict : Dictionary
        Parameters used in prediction method.

    post_dict : Dictionary
        Parameters used in analyse method.

    report :
        Provides information for McfOptPolReports to construct reports.

    sens_dict : Dictionary
        Parameters used in sensitivity method.

    time_strings : String
        Detailed information on how long the different methods needed.

    var_dict : Dictionary
        Variable names.

    var_x_type : Dictionary
        Types of covariates (internal).

    var_x_values : Dictionary
        Values of covariates (internal).

    </NOT-ON-API>

    """

    def __init__(
            self,
            var_d_name=None, var_id_name=None, var_iv_name=None,
            var_w_name=None,
            var_x_name_always_in_ord=None, var_x_name_always_in_unord=None,
            var_x_name_balance_test_ord=None,
            var_x_name_balance_test_unord=None, var_x_name_remain_ord=None,
            var_x_name_remain_unord=None, var_x_name_ord=None,
            var_x_name_unord=None, var_y_name=None, var_y_tree_name=None,
            var_z_name_cont=None, var_z_name_ord=None, var_z_name_unord=None,
            cf_alpha_reg_grid=1, cf_alpha_reg_max=0.15, cf_alpha_reg_min=0.05,
            cf_boot=1000, cf_chunks_maxsize=None, cf_compare_only_to_zero=False,
            cf_n_min_grid=1, cf_n_min_max=None, cf_n_min_min=None,
            cf_n_min_treat=None, cf_nn_main_diag_only=False, cf_m_grid=1,
            cf_m_random_poisson=True, cf_m_share_max=0.6, cf_m_share_min=0.1,
            cf_match_nn_prog_score=True, cf_mce_vart=1,
            cf_random_thresholds=None, cf_p_diff_penalty=None,
            cf_penalty_type='mse_d',
            cf_subsample_factor_eval=None, cf_subsample_factor_forest=1,
            cf_tune_all=False, cf_vi_oob_yes=False,
            cs_adjust_limits=None, cs_detect_const_vars_stop=True,
            cs_max_del_train=0.5, cs_min_p=0.01, cs_quantil=1, cs_type=1,
            ct_grid_dr=100, ct_grid_nn=10, ct_grid_w=10,
            dc_check_perfectcorr=True, dc_clean_data=True, dc_min_dummy_obs=10,
            dc_screen_covariates=True, fs_rf_threshold=1, fs_other_sample=True,
            fs_other_sample_share=0.33, fs_yes=False,
            gen_d_type='discrete', gen_iate_eff=False, gen_panel_data=False,
            gen_mp_parallel=None, gen_outfiletext=None, gen_outpath=None,
            gen_output_type=2, gen_panel_in_rf=True, gen_weighted=False,
            lc_cs_cv=True, lc_cs_cv_k=None, lc_cs_share=0.25,
            lc_estimator='RandomForest', lc_yes=True, lc_uncenter_po=True,
            p_ate_no_se_only=False, p_atet=False, p_bgate=False,
            p_bgate_sample_share=None, p_bt_yes=True, p_cbgate=False,
            p_choice_based_sampling=False, p_choice_based_probs=None,
            p_ci_level=0.95, p_cluster_std=False, p_cond_var=True,
            p_gates_minus_previous=False, p_gates_smooth=True,
            p_gates_smooth_bandwidth=1, p_gates_smooth_no_evalu_points=50,
            p_gates_no_evalu_points=50, p_gatet=False,
            p_iate=True, p_iate_se=False, p_iate_m_ate=False,
            p_iv_aggregation_method=('local', 'global',),
            p_knn=True, p_knn_const=1, p_knn_min_k=10,
            p_nw_bandw=1, p_nw_kern=1,
            p_max_cats_z_vars=None, p_max_weight_share=0.05,
            p_qiate=False, p_qiate_se=False, p_qiate_m_mqiate=False,
            p_qiate_m_opp=False,
            p_qiate_no_of_quantiles=99, p_qiate_smooth=True,
            p_qiate_smooth_bandwidth=1,
            p_qiate_bias_adjust=True,
            p_se_boot_ate=None, p_se_boot_gate=None,
            p_se_boot_iate=None, p_se_boot_qiate=None,
            var_x_name_balance_bgate=None, var_cluster_name=None,
            post_bin_corr_threshold=0.1, post_bin_corr_yes=True,
            post_est_stats=True, post_kmeans_no_of_groups=None,
            post_kmeans_max_tries=1000, post_kmeans_min_size_share=None,
            post_kmeans_replications=10, post_kmeans_single=False,
            post_kmeans_yes=True, post_random_forest_vi=True,
            post_relative_to_first_group_only=True, post_plots=True,
            post_tree=True,
            _int_cuda=False, _int_del_forest=False,
            _int_descriptive_stats=True, _int_dpi=500, _int_fontsize=2,
            _int_iate_chunk_size=None,
            _int_keep_w0=False, _int_no_filled_plot=20,
            _int_max_cats_cont_vars=None, _int_max_save_values=50,
            _int_max_obs_training=float('inf'), _int_max_obs_prediction=250000,
            _int_max_obs_kmeans=200000, _int_max_obs_post_rel_graphs=50000,
            _int_mp_ray_del=('refs',), _int_mp_ray_objstore_multiplier=1,
            _int_mp_ray_shutdown=None, _int_mp_vim_type=None,
            _int_mp_weights_tree_batch=None, _int_mp_weights_type=1,
            _int_obs_bigdata=1000000,
            _int_output_no_new_dir=False, _int_red_largest_group_train=False,
            _int_replication=False, _int_report=True, _int_return_iate_sp=False,
            _int_seed_sample_split=67567885, _int_share_forest_sample=0.5,
            _int_show_plots=True, _int_verbose=True,
            _int_weight_as_sparse=True, _int_weight_as_sparse_splits=None,
            _int_with_output=True
            ):

        self.__version__ = '0.8.0'

        self.int_dict = mcf_init.int_init(
            cuda=_int_cuda, cython=False,  # Cython turned off for now
            del_forest=_int_del_forest,
            descriptive_stats=_int_descriptive_stats, dpi=_int_dpi,
            fontsize=_int_fontsize,
            max_save_values=_int_max_save_values, mp_ray_del=_int_mp_ray_del,
            mp_ray_objstore_multiplier=_int_mp_ray_objstore_multiplier,
            mp_weights_tree_batch=_int_mp_weights_tree_batch,
            mp_weights_type=_int_mp_weights_type,
            no_filled_plot=_int_no_filled_plot,
            return_iate_sp=_int_return_iate_sp, replication=_int_replication,
            seed_sample_split=_int_seed_sample_split,
            share_forest_sample=_int_share_forest_sample,
            show_plots=_int_show_plots, verbose=_int_verbose,
            weight_as_sparse=_int_weight_as_sparse, keep_w0=_int_keep_w0,
            with_output=_int_with_output, mp_ray_shutdown=_int_mp_ray_shutdown,
            mp_vim_type=_int_mp_vim_type,
            obs_bigdata=_int_obs_bigdata,
            output_no_new_dir=_int_output_no_new_dir, report=_int_report,
            weight_as_sparse_splits=_int_weight_as_sparse_splits,
            max_cats_cont_vars=_int_max_cats_cont_vars,
            iate_chunk_size=_int_iate_chunk_size,
            max_obs_training=_int_max_obs_training,
            max_obs_prediction=_int_max_obs_prediction,
            max_obs_kmeans=_int_max_obs_kmeans,
            max_obs_post_rel_graphs=_int_max_obs_post_rel_graphs,
            p_ate_no_se_only=p_ate_no_se_only
            )
        gen_dict = mcf_init.gen_init(
            self.int_dict,
            d_type=gen_d_type, iate_eff=gen_iate_eff,
            mp_parallel=gen_mp_parallel, outfiletext=gen_outfiletext,
            outpath=gen_outpath, output_type=gen_output_type,
            weighted=gen_weighted, panel_data=gen_panel_data,
            panel_in_rf=gen_panel_in_rf)
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
            type_=cs_type, adjust_limits=cs_adjust_limits,
            detect_const_vars_stop=cs_detect_const_vars_stop)
        self.lc_dict = mcf_init.lc_init(
            cs_cv=lc_cs_cv, cs_cv_k=lc_cs_cv_k, cs_share=lc_cs_share,
            undo_iate=lc_uncenter_po, yes=lc_yes, estimator=lc_estimator)
        self.cf_dict = mcf_init.cf_init(
            alpha_reg_grid=cf_alpha_reg_grid, alpha_reg_max=cf_alpha_reg_max,
            alpha_reg_min=cf_alpha_reg_min, boot=cf_boot,
            chunks_maxsize=cf_chunks_maxsize,
            compare_only_to_zero=cf_compare_only_to_zero,
            nn_main_diag_only=cf_nn_main_diag_only, m_grid=cf_m_grid,
            m_share_max=cf_m_share_max, m_share_min=cf_m_share_min,
            m_random_poisson=cf_m_random_poisson,
            match_nn_prog_score=cf_match_nn_prog_score, mce_vart=cf_mce_vart,
            vi_oob_yes=cf_vi_oob_yes, n_min_grid=cf_n_min_grid,
            n_min_max=cf_n_min_max, n_min_min=cf_n_min_min,
            n_min_treat=cf_n_min_treat,
            p_diff_penalty=cf_p_diff_penalty, penalty_type=cf_penalty_type,
            subsample_factor_eval=cf_subsample_factor_eval,
            subsample_factor_forest=cf_subsample_factor_forest,
            tune_all=cf_tune_all,
            random_thresholds=cf_random_thresholds)
        p_dict = mcf_init.p_init(
            gen_dict,
            ate_no_se_only=p_ate_no_se_only, cbgate=p_cbgate, atet=p_atet,
            bgate=p_bgate, bt_yes=p_bt_yes,
            choice_based_sampling=p_choice_based_sampling,
            choice_based_probs=p_choice_based_probs, ci_level=p_ci_level,
            cluster_std=p_cluster_std, cond_var=p_cond_var,
            gates_minus_previous=p_gates_minus_previous,
            gates_smooth=p_gates_smooth,
            gates_smooth_bandwidth=p_gates_smooth_bandwidth,
            gates_smooth_no_evalu_points=p_gates_smooth_no_evalu_points,
            gatet=p_gatet, gate_no_evalu_points=p_gates_no_evalu_points,
            bgate_sample_share=p_bgate_sample_share,
            iate=p_iate, iate_se=p_iate_se, iate_m_ate=p_iate_m_ate, knn=p_knn,
            knn_const=p_knn_const, knn_min_k=p_knn_min_k, nw_bandw=p_nw_bandw,
            nw_kern=p_nw_kern, max_cats_z_vars=p_max_cats_z_vars,
            max_weight_share=p_max_weight_share,
            se_boot_ate=p_se_boot_ate, se_boot_gate=p_se_boot_gate,
            se_boot_iate=p_se_boot_iate,
            qiate=p_qiate, qiate_se=p_qiate_se,
            qiate_m_mqiate=p_qiate_m_mqiate, qiate_m_opp=p_qiate_m_opp,
            qiate_no_of_quantiles=p_qiate_no_of_quantiles,
            se_boot_qiate=p_se_boot_qiate, qiate_smooth=p_qiate_smooth,
            qiate_smooth_bandwidth=p_qiate_smooth_bandwidth,
            qiate_bias_adjust=p_qiate_bias_adjust,
            iv_aggregation_method=p_iv_aggregation_method)
        self.post_dict = mcf_init.post_init(
            p_dict,
            bin_corr_threshold=post_bin_corr_threshold,
            bin_corr_yes=post_bin_corr_yes, est_stats=post_est_stats,
            kmeans_no_of_groups=post_kmeans_no_of_groups,
            kmeans_max_tries=post_kmeans_max_tries,
            kmeans_replications=post_kmeans_replications,
            kmeans_yes=post_kmeans_yes, kmeans_single=post_kmeans_single,
            kmeans_min_size_share=post_kmeans_min_size_share,
            random_forest_vi=post_random_forest_vi,
            relative_to_first_group_only=post_relative_to_first_group_only,
            plots=post_plots, tree=post_tree)
        self.var_dict, self.gen_dict, self.p_dict = mcf_init.var_init(
            gen_dict, self.fs_dict, p_dict,
            x_name_balance_bgate=var_x_name_balance_bgate,
            cluster_name=var_cluster_name,
            d_name=var_d_name, id_name=var_id_name, w_name=var_w_name,
            iv_name=var_iv_name,
            x_name_balance_test_ord=var_x_name_balance_test_ord,
            x_name_balance_test_unord=var_x_name_balance_test_unord,
            x_name_always_in_ord=var_x_name_always_in_ord,
            x_name_always_in_unord=var_x_name_always_in_unord,
            x_name_remain_ord=var_x_name_remain_ord,
            x_name_remain_unord=var_x_name_remain_unord,
            x_name_ord=var_x_name_ord, x_name_unord=var_x_name_unord,
            y_name=var_y_name, y_tree_name=var_y_tree_name,
            z_name_cont=var_z_name_cont, z_name_ord=var_z_name_ord,
            z_name_unord=var_z_name_unord)
        self.blind_dict = self.sens_dict = None
        self.data_train_dict = self.var_x_type = self.var_x_values = None
        self.forest, self.time_strings = None, {}
        self.report = {'predict_list': [],   # Needed for multiple predicts
                       'analyse_list': []
                       }
        self.iv_mcf = {'firststage': None, 'reducedform': None}
        self.predict_done = False
        self.predict_iv_done = False
        self.predict_different_allocations_done = False

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
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'tree_df' : DataFrame
                Dataset used to build the forest.
            'fill_y_df' : DataFrame
                Dataset used to populate the forest with outcomes.
            'common_support_probabilities_tree': pd.DataFrame containing
                treatment probabilities for all treatments,
                the identifier of the observation, and a dummy variable
                indicating whether the observation is inside or outside the
                common support. This is for the data used to build the trees.
                None if _int_with_output is False.
            'common_support_probabilities_fill_y': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to fill the trees with outcome values.
                None if _int_with_output is False.
            'path_output' : Pathlib object
                Location of directory in which output is saved.

        """
        results = train_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def train_iv(self, data_df):
        """
        Train the IV modified causal forest on the training data.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the causal forest. It must contain information
            about outcomes, treatment, and features.

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'tree_df' : DataFrame
                Dataset used to build the forest.
            'fill_y_df' : DataFrame
                Dataset used to populate the forest with outcomes.
            'common_support_probabilities_tree': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to build the trees.
                None if _int_with_output is False.
            'common_support_probabilities_fill_y': pd.DataFrame containing
                treatment probabilities for all treatments, the identifier of
                the observation, and a dummy variable indicating
                whether the observation is inside or outside the common support.
                This is for the data used to fill the trees with outcome values.
                None if _int_with_output is False.
            'path_output' : Pathlib object
                Location of directory in which output is saved.

        """
        results = train_iv_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict(self, data_df):
        """
        Compute all effects.

        meth:`~ModifiedCausalForest.train` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        Returns
        -------
        results : Dictionary.
            Contains the results. This dictionary has the following structure:
            'ate': ATE, 'ate_se': Standard error of ATE,
            'ate_effect_list': List of names of estimated effects,
            'gate': GATE, 'gate_se': SE of GATE,
            'gate_diff': GATE minus ATE,
            'gate_diff_se': Standard error of GATE minus ATE,
            'cbgate': cbGATE (all covariates balanced),
            'cbgate_se': Standard error of CBGATE,
            'cbgate_diff': CBGATE minus ATE,
            'cbgate_diff_se': Standard error of CBGATE minus ATE,
            'bgate': BGATE (only prespecified covariates balanced),
            'bgate_se': Standard error of BGATE,
            'bgate_diff': BGATE minus ATE,
            'bgate_diff_se': Standard errror of BGATE minus ATE,
            'gate_names_values': Dictionary: Order of gates parameters
            and name and values of GATE effects.
            'qiate': QIATE, 'qiate_se': Standard error of QIATEs,
            'qiate_diff': QIATE minus QIATE at median,
            'qiate_diff_se': Standard error of QIATE minus QIATE at median,
            'iate_eff': (More) Efficient IATE (IATE estimated twice and
            averaged where role of tree_building and tree_filling
            sample is exchanged),
            'iate_data_df': DataFrame with IATEs,
            'iate_names_dic': Dictionary containing names of IATEs,
            'bala': Effects of balancing tests,
            'bala_se': Standard error of effects of balancing tests,
            'bala_effect_list': Names of effects of balancing tests.
            'common_support_probabilities' : pd.DataFrame containing treatment
            probabilities for all treatments, the identifier of the observation,
            and a dummy variable indicating whether the observation is inside or
            outside the common support. None if _int_with_output is False.
            'path_output': Pathlib object, location of directory in which output
            is saved.

        """
        self.predict_done = True
        results = predict_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict_different_allocations(self,
                                      data_df,
                                      allocations_df: bool = None
                                      ):
        """
        Predict average potential outcomes for different allocations.

        meth:`~ModifiedCausalForest.train` method must be run beforehand. The
        details of this methods are described in the working paper by
        Busshoff and Lechner (2025).

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        allocations_df : Dataframe or None, optional
            Different allocations which are to be evaluated. The length of this
            dataframe must be the same as the length of data_df.
            Default is None.

        Returns
        -------
        results : Dictionary
            Results. This dictionary has the following structure:
            'ate': Average treatment effects
            'ate_se': Standard error of average treatment effects
            'ate_effect_list': List with name with estiamted effects
            'alloc_df': Dataframe with value and variance of value for all
                        allocations investigated.
            'outpath' : Pathlib object. Location of directory in which output
                        is saved.

        """
        self.predict_different_allocations_done = True
        results, self.gen_dict['outpath'] = predict_different_allocations_main(
            self, data_df, allocations_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results

    def predict_iv(self, data_df):
        """
        Compute all effects for instrument mcf (possibly in 2 differnt ways).

        :meth:`~ModifiedCausalForest.train_iv` method must be run beforehand.

        Parameters
        ----------
        data_df : DataFrame
            Data used to compute the predictions. It must contain information
            about features (and treatment if effects for treatment specific
            subpopulations are desired as well).

        Returns
        -------
        results_global : Dictionary.
            Contains the results. This dictionary has the following structure:
            'ate': LATE, 'ate_se': Standard error of LATE,
            'ate_effect_list': List of names of estimated effects,
            'ate_1st': ATE 1st stage, 'ate_1st_se': Standard error of ATE (1st)
            'ate 1st_effect_list': List of names of estimated effects (1st),
            'ate_redf': ATE reduced form, 'ate_redf_se': Standard error of ATE
            of reduced form,
            'ate redf_effect_list': List of names of estimated effects (red.f.),
            'gate': LGATE, 'gate_se': SE of LGATE,
            'gate_diff': LGATE minus LATE,
            'gate_diff_se': Standard error of LGATE minus LATE,
            'cbgate': LCBGATE (all covariates balanced),
            'cbgate_se': Standard error of LCBGATE,
            'cbgate_diff': LCBGATE minus LATE,
            'cbgate_diff_se': Standard error of LCBGATE minus LATE,
            'bgate': LBGATE (only prespecified covariates balanced),
            'bgate_se': Standard error of LBGATE,
            'bgate_diff': LBGATE minus LATE,
            'bgate_diff_se': Standard errror of LBGATE minus LATE,
            'gate_names_values': Dictionary: Order of gates parameters
            and name and values of LGATE effects.
            'iate': LIATE, 'iate_se': Standard error of LIATE,
            'iate_1st': IATE (1st stage), 'iate_1st_se': Standard error of
            IATE (1st stage),
            'iate_redf': IATE (reduced form), 'iate_redf_se': Standard error of
            IATE (reduced form),
            'iate_eff': (More) Efficient LIATE (LIATE estimated twice and
            averaged where role of tree_building and tree_filling
            sample is exchanged),
            iate_1st_eff': (More) Efficient IATE (1st stage),
            iate_redf_eff': (More) Efficient IATE (reduced form),
            'iate_data_df': DataFrame with LIATEs,
            'iate_1st_data_df': DataFrame with IATEs (1st stage),
            'iate_redf_data_df': DataFrame with IATEs (reduced form),
            'iate_names_dic': Dictionary containing names of LIATEs,
            'iate_1st_names_dic': Dictionary containing names of IATEs (1st),
            'iate_redf_dic': Dictionary containing names of LIATEs (red.f.),
            'qiate': QLIATE, 'qiate_se': Standard error of QLIATE,
            'bala_1st': Effects of balancing tests (1st stage),
            'bala_1st_se': Standard error of effects of balancing tests (1st),
            'bala_1st_effect_list': Names of effects of balancing tests (1st),
            'bala_redf': Effects of balancing tests (reduced form),
            'bala_redf_se': Standard error of effects of balancing tests (red.),
            'bala_redf_effect_list': Names of effects of balancing tests (red.).
            'common_support_probabilities': pd.DataFrame containing treatment
            probabilities for all treatments, the identifier of the observation,
            and a dummy variable indicating whether the observation is inside or
            outside the common support. None if _int_with_output is False.
            'path_output': Pathlib object, location of directory in which output
            is saved.

            It is empty if the IV estimation method 'global' has not been
            used.

        results_local : Dictionary.
            Same content as results_wald.
            It is empty if the IV estimation method 'local' has not been
            used.

        """
        self.predict_iv_done = True
        # Reduce sample size to upper limit
        data_df, rnd_reduce, txt_red = check_reduce_dataframe(
            data_df, title='Prediction',
            max_obs=self.int_dict['max_obs_prediction'],
            seed=124535, ignore_index=True)
        if rnd_reduce and self.int_dict['with_output']:
            print_mcf(self.gen_dict, txt_red, summary=True)

        results_global, results_local = predict_iv_main(self, data_df)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_global, results_local

    def analyse(self, results):
        """
        Analyse estimated IATEs with various descriptive tools.

        Parameters
        ----------
        results : Dictionary
            Contains estimation results. This dictionary must have the same
            structure as the one returned from the
            :meth:`~ModifiedCausalForest.predict` method.

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
        results_plus_cluster = analyse_main(self, results)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_plus_cluster

    def blinder_iates(
        self, data_df, blind_var_x_protected_name=None,
        blind_var_x_policy_name=None, blind_var_x_unrestricted_name=None,
        blind_weights_of_blind=None, blind_obs_ref_data=50,
            blind_seed=123456):
        """
        Compute IATEs that causally depend less on protected variables.

        WARNING
        This method is deprecated and will be removed in future
        versions. Use the method fairscores of the OptimalPolicy class instead.

        Parameters
        ----------
        data_df : DataFrame.
            Contains data needed to
            :meth:`~ModifiedCausalForest.predict` the various adjusted IATES.

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

        var_x_policy_ord_name : List of strings
            Ordered variables to be used to build the decision rules.

        var_x_policy_unord_name : List of strings.
            Unordered variables to be used to build the decision rules.

        var_x_blind_ord_name : List of strings
            Ordered variables to be used to blind potential outcomes.

        var_x_blind_unord_name : List of strings.
            Unordered variables to be used to blind potential outcomes.

        outpath : Pathlib object
            Location of directory in which output is saved.

        """
        print('This method of reducing dependence on protected '
              'variables is deprecated. Use the method '
              'fairscores of the OptimalPolicy class instead.')

        (blinded_dic, data_on_support_df, var_x_policy_ord_name,
         var_x_policy_unord_name, var_x_blind_ord_name,
         var_x_blind_unord_name, self.gen_dict['outpath']
         ) = blinder_iates_main(
             self,
             data_df, blind_var_x_protected_name=blind_var_x_protected_name,
             blind_var_x_policy_name=blind_var_x_policy_name,
             blind_var_x_unrestricted_name=blind_var_x_unrestricted_name,
             blind_weights_of_blind=blind_weights_of_blind,
             blind_obs_ref_data=blind_obs_ref_data, blind_seed=blind_seed)

        return (blinded_dic, data_on_support_df, var_x_policy_ord_name,
                var_x_policy_unord_name, var_x_blind_ord_name,
                var_x_blind_unord_name, self.gen_dict['outpath'])

    def sensitivity(
                    self,
                    train_df,
                    predict_df: None = None,
                    results: None = None,
                    sens_cbgate: None = None,
                    sens_bgate: None = None,
                    sens_gate: None = None,
                    sens_iate: None = None,
                    sens_iate_se: None = None,
                    sens_scenarios: None = None,
                    sens_cv_k: None = None,
                    sens_replications: int = 2,
                    sens_reference_population: None = None,
                    ):
        """
        Compute simulation-based sensitivity indicators.
    
        Parameters
        ----------
        train_df : DataFrame
            Data with real outcomes, treatments, and covariates. Data will be
            transformed to compute sensitivity indicators.
    
        predict_df : DataFrame, optional
            Prediction data to compute all effects for. This data will not be
            changed in the computation process. Only covariate information is
            used from this dataset. If None, ``train_df`` will be used.
    
        results : dict, optional
            Output dictionary from :meth:`~ModifiedCausalForest.predict`.
            If it contains estimated IATEs, they are used for the no-effect
            (basic) scenario and compared to those in the dictionary.
            Otherwise, passing it has no effect.
    
        sens_cbgate : bool, optional
            If True, compute CBGATEs for sensitivity analysis. Default is False.
    
        sens_bgate : bool, optional
            If True, compute BGATEs for sensitivity analysis. Default is False.
    
        sens_gate : bool, optional
            If True, compute GATEs for sensitivity analysis. Default is False.
    
        sens_iate : bool, optional
            If True, compute IATEs for sensitivity analysis.
            If ``results`` contains IATEs, default is True, otherwise False.
    
        sens_iate_se : bool, optional
            If True, compute standard errors of IATEs for sensitivity analysis.
            Default is False.
    
        sens_scenarios : list or tuple of str, optional
            Different scenarios considered. Default is ('basic',).
            - 'basic': Use estimated treatment probabilities for simulations
              (no confounding).
    
        sens_cv_k : int, optional
            Number of folds in cross-validation. Default is 5.
    
        sens_replications : int, optional
            Number of replications for simulating placebo treatments.
            Default is 2.
    
        sens_reference_population : int or float, optional
            Treatment status of the reference population used by the
            sensitivity analysis. Default is the treatment with the most
            observed observations.
    
        Returns
        -------
        results_avg : dict
            Same structure as :meth:`~ModifiedCausalForest.predict`, but
            averaged over replications (if applicable).
        """
        results_avg = sensitivity_main(
            self, train_df, predict_df=predict_df, results=results,
            sens_cbgate=sens_cbgate, sens_bgate=sens_bgate, sens_gate=sens_gate,
            sens_iate=sens_iate, sens_iate_se=sens_iate_se,
            sens_scenarios=sens_scenarios, sens_cv_k=sens_cv_k,
            sens_replications=sens_replications,
            sens_reference_population=sens_reference_population)

        if (self.int_dict['mp_ray_shutdown']
            and self.gen_dict['mp_parallel'] > 1
                and is_initialized()):
            shutdown()

        return results_avg
