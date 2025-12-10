"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from mcf.mcf_init_update_helper_functions import isinstance_scalar, bootstrap
from mcf import mcf_init_values_cfg_functions as mcf_initvals

type NumberLike = int | float | None
type IntLike = int | None
type BoolLike = bool | None
type StrLike = str | None
type GridLike = list[float | int] | tuple[float | int, ...] | NDArray | None
type ListLike = list[Any] | None
type ListofStrLike = list[str] | None
type NameLike = str | ListofStrLike | tuple[str, ...]


@dataclass(slots=True, kw_only=True)
class PCfg:
    """Effect-prediction configuration."""

    # canonical / toggles
    ate_no_se_only: bool = False
    atet: bool = False
    gatet: bool = False
    bgate: bool = False
    bt_yes: bool = False
    cbgate: bool = False

    # sampling
    choice_based_sampling: bool = False
    choice_based_probs: float | list[float] | tuple[float, ...] = 1

    # inference
    ci_level: float = 0.95
    cluster_std: bool = False
    cond_var: bool = True
    se_boot_ate: bool | NumberLike = False
    se_boot_gate: bool | NumberLike = False
    se_boot_iate: bool | NumberLike = False
    se_boot_qiate: bool | NumberLike = False

    # gates
    gates_minus_previous: bool = False
    gates_smooth: bool = True
    gates_smooth_bandwidth: NumberLike = 1
    gates_smooth_no_evalu_points: int = 50
    gate_no_evalu_points: int = 50

    # iate / qiate
    iate: bool = True
    iate_se: bool = False
    iate_m_ate: bool = False

    qiate: bool = False
    qiate_se: bool = False
    qiate_m_mqiate: bool = False
    qiate_m_opp: bool = False
    qiate_no_of_quantiles: int = 99
    qiate_quantiles: NDArray = field(
        default_factory=lambda: np.arange(99) / 99 + 0.5 / 99
        )
    qiate_smooth: bool = True
    qiate_smooth_bandwidth: NumberLike = 1
    qiate_bias_adjust: bool = False
    qiate_bias_adjust_draws: int = 1  # current logic uses 1

    # kernels / neighbors
    knn: bool = True
    knn_min_k: NumberLike = 10
    knn_const: NumberLike = 1
    nw_bandw: NumberLike = 1
    nw_kern: NumberLike = 1  # 1: Epanechnikov, 2: Normal

    # misc numeric limits
    max_cats_z_vars: NumberLike = None
    max_weight_share: NumberLike = 0.05
    bgate_sample_share: NumberLike = None

    # IV aggregation
    iv_aggregation_method: tuple[str, ...] = ('local', 'global')

    # quantile weights
    q_w: list[float] = field(
        default_factory=lambda: [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
        )
    # output paths injected by mcf_sys.get_fig_path
    paths: dict[str, Any] = field(default_factory=dict)

    # to be initialised later
    combi: tuple[Any, ...] | ListLike = field(default=None, init=False)
    directory_effect: Path | str | None = field(default=None, init=False)
    number_of_decimal_places: IntLike = field(default=None, init=False)
    number_of_stats: IntLike = field(default=None, init=False)
    number_of_treatments: IntLike = field(default=None, init=False)
    treatment_names: ListofStrLike = field(default=None, init=False)
    multiplier_rows: IntLike = field(default=None, init=False)
    gate: BoolLike = field(default=None, init=False)
    d_in_pred: BoolLike = field(default=None, init=False)

    @classmethod
    def from_args(
        cls,
        gen_cfg: Any,
        *,
        ate_no_se_only: BoolLike = None,
        atet: BoolLike = None,
        bgate: BoolLike = None,
        bgate_sample_share: NumberLike = None,
        bt_yes: BoolLike = None,
        cbgate: BoolLike = None,
        choice_based_sampling: BoolLike = None,
        choice_based_probs: list[float] | tuple[float, ...] | None = None,
        ci_level: NumberLike = None,
        cluster_std: BoolLike = None,
        cond_var: BoolLike = None,
        gates_minus_previous: BoolLike = None,
        gates_smooth: BoolLike = None,
        gates_smooth_bandwidth: NumberLike = None,
        gates_smooth_no_evalu_points: NumberLike = None,
        gatet: BoolLike = None,
        gate_no_evalu_points: NumberLike = None,
        iate: BoolLike = None,
        iate_se: BoolLike = None,
        iate_m_ate: BoolLike = None,
        iv_aggregation_method: Any = None,
        knn: BoolLike = None,
        knn_const: NumberLike = None,
        knn_min_k: NumberLike = None,
        nw_bandw: NumberLike = None,
        nw_kern: NumberLike = None,
        max_cats_z_vars: NumberLike = None,
        max_weight_share: NumberLike = None,
        qiate: BoolLike = None,
        qiate_se: BoolLike = None,
        qiate_m_mqiate: BoolLike = None,
        qiate_m_opp: BoolLike = None,
        qiate_no_of_quantiles: IntLike = None,
        qiate_smooth: BoolLike = None,
        qiate_smooth_bandwidth: NumberLike = None,
        qiate_bias_adjust: BoolLike = None,
        se_boot_ate: bool | NumberLike = None,
        se_boot_gate: bool | NumberLike = None,
        se_boot_iate: bool | NumberLike = None,
        se_boot_qiate: bool | NumberLike = None,
            ) -> 'PCfg':
        """Get input and normalize parameters."""
        # toggles with dependencies
        atet_b = atet is True
        gatet_b = gatet is True
        ate_no_se_only_b = ate_no_se_only is True
        if ate_no_se_only_b:
            atet_b = gatet_b = cbgate_b = bgate_b = bt_yes_b = False
            cluster_std_b = False
            gates_smooth_b = False
            iate_b = iate_se_b = iate_m_ate_b = False
            qiate_b = qiate_se_b = qiate_m_mqiate_b = qiate_m_opp_b = False
            se_boot_ate_v = se_boot_gate_v = False
            se_boot_iate_v = se_boot_qiate_v = False
        else:
            cbgate_b = cbgate is True
            bgate_b = bgate is True
            bt_yes_b = bt_yes is True
            iate_b = iate is not False
            iate_se_b = iate_se is True
            iate_m_ate_b = iate_m_ate is True
            qiate_b = qiate is True
            qiate_se_b = qiate_se is True
            qiate_m_mqiate_b = qiate_m_mqiate is True
            qiate_m_opp_b = qiate_m_opp is True
            gates_smooth_b = gates_smooth is not False
            se_boot_ate_v = se_boot_ate
            se_boot_gate_v = se_boot_gate
            se_boot_iate_v = se_boot_iate
            se_boot_qiate_v = se_boot_qiate

        if gatet_b:
            atet_b = True
        if qiate_b:
            iate_b = True
        if not iate_b:
            iate_se_b = iate_m_ate_b = False

        # sampling
        if choice_based_sampling is True:
            if gen_cfg.d_type != 'discrete':
                raise NotImplementedError(
                    'No choice based sample with continuous treatments.'
                    )
            cbs_b = True
            cbp_v = choice_based_probs if choice_based_probs is not None else 1
        else:
            cbs_b = False
            cbp_v = 1

        # ci level
        ci_v = (0.95 if ci_level is None or not (0.5 < ci_level < 0.99999999)
                else float(ci_level)
                )
        # cluster std and cond var
        cluster_std_b = cluster_std is True or gen_cfg.panel_data
        cond_var_b = cond_var is not False

        # gates smooth params
        gsbw_v = (1 if gates_smooth_bandwidth is None
                  or gates_smooth_bandwidth <= 0
                  else gates_smooth_bandwidth
                  )
        gsnev_v = (50 if gates_smooth_no_evalu_points is None
                   or gates_smooth_no_evalu_points < 2
                   else round(gates_smooth_no_evalu_points)
                   )
        gnev_v = (50 if gate_no_evalu_points is None or gate_no_evalu_points < 2
                  else round(gate_no_evalu_points)
                  )
        # qiate params
        qn_v = (99 if (qiate_no_of_quantiles is None
                       or not isinstance(qiate_no_of_quantiles, int)
                       or qiate_no_of_quantiles < 10)
                else qiate_no_of_quantiles
                )
        qq_v = np.arange(qn_v) / qn_v + 0.5 / qn_v
        q_smooth_b = qiate_smooth is not False
        qsbw_v = (1 if qiate_smooth_bandwidth is None
                  or qiate_smooth_bandwidth <= 0
                  else qiate_smooth_bandwidth
                  )
        q_bias_b = qiate_bias_adjust is True
        if q_bias_b or qiate_se_b:
            iate_se_b = True
        q_bad_v = 1  # current bias adj uses 1 draw

        # kNN / NW
        knn_b = knn is not False
        knn_min_v = 10 if knn_min_k is None or knn_min_k < 0 else knn_min_k
        knn_const_v = 1 if knn_const is None or knn_const < 0 else knn_const
        nw_bandw_v = 1 if nw_bandw is None or nw_bandw < 0 else nw_bandw
        nw_kern_v = 1 if nw_kern is None or nw_kern != 2 else 2

        # bootstrap sizes (or False)
        se_boot_ate_v = bootstrap(se_boot_ate_v, 49, 199, cluster_std_b)
        se_boot_gate_v = bootstrap(se_boot_gate_v, 49, 199, cluster_std_b)
        se_boot_iate_v = bootstrap(se_boot_iate_v, 49, 199, cluster_std_b)
        se_boot_qiate_v = bootstrap(se_boot_qiate_v, 49, 199, cluster_std_b)

        # weights / limits
        mws_v = (0.05 if max_weight_share is None or max_weight_share <= 0
                 else max_weight_share
                 )
        q_w_v = [0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]

        # IV aggregation method

        ivm_v = mcf_initvals.iv_agg_method_fct(iv_aggregation_method)
        # default_iv = ('local', 'global')

        # valid_iv = ('local', 'global')
        # if isinstance(iv_aggregation_method, (str, int, float,)):
        #     if iv_aggregation_method in valid_iv:
        #         ivm_v = (iv_aggregation_method,)
        #     else:
        #         raise ValueError(
        #             f'{iv_aggregation_method} is not a valid method for IV '
        #             f'estimation. Valid methods are {" ".join(valid_iv)}.'
        #             )
        # elif isinstance(iv_aggregation_method, (list, tuple)):
        #     if not set(iv_aggregation_method) == set(valid_iv):
        #         raise ValueError(
        #             f'{iv_aggregation_method} is not a valid method for IV '
        #             f'estimation. Valid methods are {" ".join(valid_iv)}.'
        #             )
        #     ivm_v = tuple(iv_aggregation_method)
        # else:
        #     ivm_v = default_iv

        # figure paths
        paths_v: dict[str, Any] = {}
        if gen_cfg.with_output:
            # paths_v = get_fig_path({}, gen_cfg.outpath,
            #                        'ate_iate', gen_cfg.with_output
            #                        )
            # paths_v = get_fig_path(paths_v, gen_cfg.outpath,
            #                        'gate', gen_cfg.with_output
            #                        )
            # if qiate_b:
            #     paths_v = get_fig_path(paths_v, gen_cfg.outpath,
            #                            'qiate', gen_cfg.with_output
            #                            )
            # if cbgate_b:
            #     paths_v = get_fig_path(paths_v, gen_cfg.outpath,
            #                            'cbgate', gen_cfg.with_output)
            # if bgate_b:
            #     paths_v = get_fig_path(paths_v, gen_cfg.outpath, 'bgate',
            #                            gen_cfg.with_output
            #                            )
            paths_v = mcf_initvals.get_directories_for_output(gen_cfg.outpath,
                                                              True,
                                                              qiate=qiate_b,
                                                              cbgate=cbgate_b,
                                                              bgate=bgate_b,
                                                              )
        return cls(
            ate_no_se_only=ate_no_se_only_b,
            cbgate=cbgate_b,
            atet=atet_b,
            bgate=bgate_b,
            bt_yes=bt_yes_b,
            ci_level=ci_v,
            choice_based_sampling=cbs_b,
            choice_based_probs=cbp_v,
            cluster_std=cluster_std_b,
            cond_var=cond_var_b,
            bgate_sample_share=bgate_sample_share,
            gates_minus_previous=gates_minus_previous is True,
            gates_smooth=gates_smooth_b,
            gates_smooth_bandwidth=gsbw_v,
            gates_smooth_no_evalu_points=gsnev_v,
            gatet=gatet_b,
            gate_no_evalu_points=gnev_v,
            iate=iate_b,
            iate_se=iate_se_b,
            iate_m_ate=iate_m_ate_b,
            iv_aggregation_method=ivm_v,
            knn=knn_b,
            knn_const=knn_const_v,
            knn_min_k=knn_min_v,
            nw_bandw=nw_bandw_v,
            nw_kern=nw_kern_v,
            max_cats_z_vars=max_cats_z_vars,
            max_weight_share=mws_v,
            q_w=q_w_v,
            qiate=qiate_b,
            qiate_se=qiate_se_b,
            qiate_m_mqiate=qiate_m_mqiate_b,
            qiate_m_opp=qiate_m_opp_b,
            qiate_no_of_quantiles=qn_v,
            qiate_quantiles=qq_v,
            qiate_smooth=q_smooth_b,
            qiate_smooth_bandwidth=qsbw_v,
            qiate_bias_adjust=q_bias_b,
            qiate_bias_adjust_draws=q_bad_v,
            se_boot_ate=se_boot_ate_v,
            se_boot_gate=se_boot_gate_v,
            se_boot_iate=se_boot_iate_v,
            se_boot_qiate=se_boot_qiate_v,
            paths=paths_v,
            )


@dataclass(slots=True, kw_only=True)
class PBiasAdjustmentCfg:
    """Configuration for bias adjustments."""
    yes: bool
    use_prop_score: bool
    use_prog_score: bool
    estimator: str
    cv_k: int
    use_x: bool
    adj_method: StrLike
    pos_weights_only: bool

    # to be initialised later (data needed to perform bias adjustments)
    x_ba_eval: NDArray | None = field(default=None, init=False)
    prog_score_eval: NDArray | None = field(default=None, init=False)
    prop_score_eval: NDArray | None = field(default=None, init=False)

    # allowed estimators (case-insensitive match)
    OK_ESTIMATORS: ClassVar[tuple[str, ...]] = (
        'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5',
        'SupportVectorMachine', 'SupportVectorMachineC2',
        'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200',
        'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12', 'LASSO',
        'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
        'Mean', 'automatic'
        )

    @classmethod
    def from_args(
        cls,
        *,
        yes: BoolLike=False,
        pos_weights_only: BoolLike=False,
        use_prop_score: BoolLike=True,
        use_prog_score: BoolLike=True,
        use_x: BoolLike=False,
        estimator: StrLike='RandomForest',
        cv_k: NumberLike=None,
        adj_method: StrLike='w_obs',
        clustering: bool = False,
        weighted: bool = False,
        continuous: bool = False,
        qiate: bool = False,
        ) -> 'PBiasAdjustmentCfg':
        """Get input and normalize parameters."""
        adj_method_n = mcf_initvals.p_ba_adjust_method(adj_method)

        yes_n = yes is True
        use_prop_score_n = use_prop_score is not False
        use_prog_score_n = use_prog_score is not False

        if estimator is None or not use_prog_score_n:
            estimator_n: StrLike = 'RandomForest'
        else:
            names = cls.OK_ESTIMATORS
            names_cf = tuple(n.casefold() for n in names)
            try:
                pos = names_cf.index(estimator.casefold())
                estimator_n = names[pos]
            except ValueError:
                print(
                    'Estimator specified for bias adjustment is not valid.\n'
                    f'Specified estimator {estimator}.\n'
                    f'Allowed estimators: {" ".join(names)}'
                    )
                estimator_n = None  # mirrors: no key set in the original dict

        if cv_k is not None and not isinstance_scalar(cv_k):
            raise TypeError('Number of folds for cross-validation must be '
                            'integer, float or None'
                            )
        if not isinstance(cv_k, (int, float)) or cv_k < 1:
            cv_k_n: IntLike = None
        else:
            cv_k_n = round(float(cv_k))

        use_x_n = use_x is True
        if yes_n and not (use_prop_score_n or use_prog_score_n or use_x_n):
            raise ValueError('Specify which variables or scores to use for '
                             'bias adjustments.'
                             )
        if yes_n and (clustering or weighted or continuous or qiate):
            raise ValueError('Clustering, weighting, QIATE estimation, and '
                             'continuous treatments are not allowed together '
                             'with bias adjustment.'
                             )
        return cls(
            yes=yes_n,
            pos_weights_only=pos_weights_only is True,
            use_prop_score=use_prop_score_n,
            use_prog_score=use_prog_score_n,
            use_x=use_x_n,
            adj_method=adj_method_n,
            estimator=estimator_n,
            cv_k=cv_k_n,
            )


@dataclass(slots=True, kw_only=True)
class PostCfg:
    """Parameters of post estimation analysis."""

    # normalized values (set by from_args)
    add_pred_to_data_file: bool = False
    bin_corr_threshold: NumberLike = 0.1
    bin_corr_yes: bool = True
    est_stats: bool = False
    kmeans_no_of_groups: NumberLike = None
    kmeans_max_tries: int = 1000
    kmeans_single: bool = False
    kmeans_replications: int = 10
    kmeans_yes: bool = True
    kmeans_min_size_share: NumberLike = 1
    random_forest_vi: bool = True
    plots: bool = True
    relative_to_first_group_only: bool = True
    tree: bool = True
    tree_depths: tuple[int, ...] = (2, 3, 4, 5)

    @classmethod
    def from_args(
        cls,
        p_cfg,  # transient; not stored
        *,
        bin_corr_threshold: NumberLike = None,
        bin_corr_yes: BoolLike = None,
        est_stats: BoolLike = None,
        kmeans_no_of_groups: NumberLike = None,
        kmeans_max_tries: NumberLike = None,
        kmeans_replications: NumberLike = None,
        kmeans_yes: BoolLike = None,
        kmeans_single: BoolLike = None,
        kmeans_min_size_share: NumberLike = None,
        random_forest_vi: BoolLike = None,
        relative_to_first_group_only: BoolLike = None,
        plots: BoolLike = None,
        tree: BoolLike = None,
            ) -> 'PostCfg':
        """Get input and normalize parameters."""
        est_stats_b = est_stats is not False
        if not p_cfg.iate:
            est_stats_b = False

        bin_corr_yes_b = bin_corr_yes is not False
        if bin_corr_threshold is None or not 0 <= bin_corr_threshold <= 1:
            bin_corr_threshold_v = 0.1
        else:
            bin_corr_threshold_v = bin_corr_threshold

        plots_b = plots is not False
        rel_first_b = relative_to_first_group_only is not False
        kmeans_yes_b = kmeans_yes is not False
        kmeans_single_b = kmeans_single is True

        if kmeans_replications is None or kmeans_replications < 0:
            kmeans_replications_v = 10
        else:
            kmeans_replications_v = int(round(kmeans_replications))

        if kmeans_max_tries is None:
            kmeans_max_tries_v = 1000
        else:
            kmeans_max_tries_v = max(int(kmeans_max_tries), 10)

        if (kmeans_min_size_share is None
            or not isinstance_scalar(kmeans_min_size_share)
                or not 0 < kmeans_min_size_share < 33):
            kmeans_min_size_share_v = 1
        else:
            kmeans_min_size_share_v = kmeans_min_size_share

        add_pred_b = est_stats_b
        rf_vi_b = random_forest_vi is not False
        tree_b = tree is not False

        return cls(
            add_pred_to_data_file=add_pred_b,
            bin_corr_threshold=bin_corr_threshold_v,
            bin_corr_yes=bin_corr_yes_b,
            est_stats=est_stats_b,
            kmeans_no_of_groups=kmeans_no_of_groups,
            kmeans_max_tries=kmeans_max_tries_v,
            kmeans_single=kmeans_single_b,
            kmeans_replications=kmeans_replications_v,
            kmeans_yes=kmeans_yes_b,
            kmeans_min_size_share=kmeans_min_size_share_v,
            random_forest_vi=rf_vi_b,
            plots=plots_b,
            relative_to_first_group_only=rel_first_b,
            tree=tree_b,
        )


@dataclass(slots=True, kw_only=True)
class SensCfg:
    """Parameters of sensitivity (post-estimation) analysis."""

    cbgate: bool = False
    bgate: bool = False
    gate: bool = False
    iate: bool = False
    iate_se: bool = False
    reference_population: NumberLike = None
    cv_k: int = 5
    replications: int = 2
    scenarios: tuple[str, ...] = ('basic',)

    @classmethod
    def from_args(
        cls,
        p_cfg,  # transient; not stored
        *,
        bgate: BoolLike = None,
        cbgate: BoolLike = None,
        cv_k: NumberLike = None,
        gate: BoolLike = None,
        iate: BoolLike = None,
        iate_se: BoolLike = None,
        replications: NumberLike = 2,
        scenarios: NameLike = None,
        reference_population: NumberLike = None,
        iate_df: DataFrame | None = None,
            ) -> 'SensCfg':
        """Get input and normalize parameters."""
        # type checks (preserve original semantics)
        if cv_k is not None and not isinstance_scalar(cv_k):
            raise TypeError('Number of folds for cross-validation must be '
                            'integer, float or None'
                            )
        if replications is not None and not isinstance_scalar(replications):
            raise TypeError('Number of replication must be integer, float or '
                            'None'
                            )
        if scenarios is not None and not isinstance(scenarios,
                                                    (list, tuple, str)):
            raise TypeError('Names of scenarios must be string or None')
        if cbgate is not None and not isinstance(cbgate, bool):
            raise TypeError('cbgate must be boolean or None')
        if bgate is not None and not isinstance(bgate, bool):
            raise TypeError('bgate must be boolean or None')
        if gate is not None and not isinstance(gate, bool):
            raise TypeError('gate must be boolean or None')
        if iate is not None and not isinstance(iate, bool):
            raise TypeError('iate must be boolean or None')
        if iate_se is not None and not isinstance(iate_se, bool):
            raise TypeError('iate_se must be boolean or None')
        if (reference_population is not None
                and not isinstance_scalar(reference_population)):
            raise TypeError('reference_population must be a Number or None')

        cbgate_b = cbgate is True
        bgate_b = bgate is True
        gate_b = (gate is True) or cbgate_b or bgate_b
        iate_b = True if iate_df is not None else (iate is True)
        iate_se_b = iate_se is True

        if cbgate_b and not getattr(p_cfg, 'cbgate', False):
            raise ValueError(
                'p_cbgate must be set to True if sens_cbgate is True'
                )
        if gate_b and not getattr(p_cfg, 'bgate', False):
            raise ValueError(
                'p_bgate must be set to True if sens_bgate is True'
                )

        refpop_v = (None if reference_population is None
                    else reference_population
                    )
        cv_k_v = 5 if cv_k is None or cv_k < 0.5 else round(cv_k)
        replications_v = (2 if replications is None or replications < 0.5
                          else round(replications)
                          )
        match scenarios:
            case None:
                scens_v = ('basic',)
            case str() as s:
                scens_v = (s,)
            case _:
                raise NotImplementedError(f'Sensitivity scenario {scenarios!r} '
                                          'not implemented'
                                          )
        eligible_scenarios = ('basic',)
        wrong_scenarios = [scen for scen in scens_v
                           if scen not in eligible_scenarios
                           ]
        if wrong_scenarios:
            raise ValueError(f'{wrong_scenarios} '
                             f'{"are" if len(wrong_scenarios) > 1 else "is"} '
                             'ineligable for sensitivity analysis'
                             )
        return cls(
            cbgate=cbgate_b,
            bgate=bgate_b,
            gate=gate_b,
            iate=iate_b,
            iate_se=iate_se_b,
            reference_population=refpop_v,
            cv_k=cv_k_v,
            replications=replications_v,
            scenarios=scens_v,
            )
