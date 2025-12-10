"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from dataclasses import dataclass, field
from typing import Any, ClassVar

from numpy.typing import NDArray

from mcf.mcf_init_update_helper_functions import get_alpha
from mcf import mcf_general_sys as mcf_sys

type NumberLike = int | float | None
type IntLike = int | None
type FloatLike = float | None
type BoolLike = bool | None
type StrLike = str | None
type GridLike = list[float | int] | tuple[float | int, ...] | NDArray | None
type ListLike = list[Any] | None
type ListofStrLike = list[str] | None


@dataclass(slots=True, kw_only=True)
class FsCfg:
    """Feature-selection configuration (canonical form).

    Notes
    -----
    - rf_threshold is stored as a fraction in [0, 1] (percent / 100).
    - other_sample_share is in [0, 0.5], but forced to 0 if yes=False or
      other_sample=False.
    """

    yes: bool = False
    rf_threshold: float = 0.01          # corresponds to 1%
    other_sample: bool = True
    other_sample_share: float = 0.33

    def __post_init__(self) -> None:
        """Keep values valid even if fields are modified after init."""
        if not 0.0 <= self.rf_threshold <= 1.0:
            self.rf_threshold = 0.01

        if not 0.0 <= self.other_sample_share <= 0.5:
            self.other_sample_share = 0.33

        if (not self.other_sample) or (not self.yes):
            self.other_sample_share = 0.0

    @classmethod
    def from_args(
        cls,
        rf_threshold: NumberLike = None,
        other_sample: BoolLike = None,
        other_sample_share: FloatLike = None,
        yes: BoolLike = None,
        ) -> 'FsCfg':
        """Get input and normalize parameters."""
        # yes only if explicitly True
        yes_b = yes is True

        # rf_threshold: percent -> fraction; fallback 1% if invalid
        if rf_threshold is None or rf_threshold <= 0 or rf_threshold > 100:
            rf_frac = 0.01
        else:
            rf_frac = float(rf_threshold) / 100.0

        # other_sample defaults to True unless explicitly False
        other_b = other_sample is not False

        # share in [0, 0.5], else default 0.33
        if other_sample_share is None or not 0.0 <= other_sample_share <= 0.5:
            share = 0.33
        else:
            share = float(other_sample_share)

        # disable share if not using other sample or FS not active
        if (not other_b) or (not yes_b):
            share = 0.0

        return cls(
            yes=yes_b,
            rf_threshold=rf_frac,
            other_sample=other_b,
            other_sample_share=share,
            )


@dataclass(slots=True, kw_only=True)
class CsCfg:
    """Common-support configuration."""
    # canonical values
    type_: int = 1
    quantil: float = 1.0
    min_p: float = 0.01
    max_del_train: float = 0.5
    # data dependent / to be adjusted later
    adjust_limits: NumberLike = None
    cut_offs: dict | None = None
    forests: ListLike = None
    # misc
    detect_const_vars_stop: bool = True
    yes: bool = True
    # whatever get_fig_path injects (e.g., figure/csv paths)
    paths: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(
        cls,
        gen_cfg: Any,
        *,
        adjust_limits: NumberLike = None,
        detect_const_vars_stop: BoolLike = None,
        max_del_train: FloatLike = None,
        min_p: FloatLike = None,
        quantil: FloatLike = None,
        type_: IntLike = None,
            ) -> 'CsCfg':
        """Get input and normalize parameters."""
        # type_: 0/1/2; 'continuous' -> 0 (no CS check)
        t = (0 if gen_cfg.d_type == 'continuous'
             else (type_ if type_ in (0, 1, 2) else 1)
             )
        q = (1.0 if (quantil is None or not 0.0 <= quantil <= 1.0)
             else float(quantil)
             )
        mp = (0.01 if (min_p is None or not 0.0 <= min_p <= 0.5)
              else float(min_p)
              )
        mdt = (0.5 if max_del_train is None or not 0.0 < max_del_train <= 1.0
               else float(max_del_train)
               )
        dcs = (detect_const_vars_stop if isinstance(detect_const_vars_stop,
                                                    bool)
               else True
               )
        paths: dict[str, Any] = {}
        if gen_cfg.outpath is not None:
            paths = mcf_sys.get_fig_path(
                {}, gen_cfg.outpath, 'common_support',
                gen_cfg.with_output, no_csv=False
                )
        return cls(
            type_=t,
            quantil=q,
            min_p=mp,
            max_del_train=mdt,
            adjust_limits=adjust_limits,
            detect_const_vars_stop=dcs,
            cut_offs=None,
            forests=None,
            paths=paths,
            )


@dataclass(slots=True, kw_only=True)
class LcCfg:
    """Local centering / common support / cross-validation config.

    Semantics preserved from the original lc_init:
    - cs_cv, yes, uncenter_po use '... is not False' logic (default True).
    - uncenter_po is forced to False when yes is False.
    - estimator stays None if an invalid name is provided.
    """

    # canonical values
    cs_cv: bool = True
    cs_share: float = 0.25
    cs_cv_k: IntLike = None

    # local centering
    yes: bool = True
    estimator: StrLike = 'RandomForest'
    uncenter_po: bool = True

    # To be filled later
    forests: ListLike = field(default=None, init=False)

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
        cs_cv: BoolLike = None,
        cs_cv_k: NumberLike = None,
        cs_share: NumberLike = None,
        estimator: StrLike = None,
        undo_iate: BoolLike = None,
        yes: BoolLike = None,
            ) -> 'LcCfg':
        """Get input and normalize parameters."""
        cscv = cs_cv is not False

        share = (
            0.25 if (cs_share is None or not 0.0999 < float(cs_share) < 0.9001)
            else float(cs_share)
            )
        if not isinstance(cs_cv_k, (int, float)) or cs_cv_k < 1:
            k_val: IntLike = None
        else:
            k_val = round(float(cs_cv_k))

        yes_b = yes is not False

        if estimator is None:
            est: StrLike = 'RandomForest'
        else:
            names = cls.OK_ESTIMATORS
            names_cf = tuple(n.casefold() for n in names)
            try:
                pos = names_cf.index(estimator.casefold())
                est = names[pos]
            except ValueError:
                print(
                    'Estimator specified for local centering is not valid.\n'
                    f'Specified estimator {estimator}.\n'
                    f'Allowed estimators: {" ".join(names)}'
                    )
                est = None  # mirrors: no key set in the original dict

        uncenter = undo_iate is not False
        if not yes_b:
            uncenter = False

        return cls(
            cs_cv=cscv,
            cs_share=share,
            cs_cv_k=k_val,
            yes=yes_b,
            estimator=est,
            uncenter_po=uncenter,
            )


@dataclass(slots=True, kw_only=True)
class CfCfg:
    """Parameters for causal-forest building."""

    # canonical values (filled by from_args)
    boot: int = 1000
    match_nn_prog_score: bool = True
    nn_main_diag_only: bool = False
    compare_only_to_zero: bool = False

    m_grid: int = 1
    n_min_grid: int = 1

    alpha_reg_grid: int = 1
    alpha_reg_max: NumberLike = None
    alpha_reg_min: NumberLike = None
    alpha_reg_values: Any = field(default_factory=list)

    tune_all: bool = False

    m_share_min: float = 0.1
    m_share_max: float = 0.6
    m_random_poisson: bool = True
    m_random_poisson_min: int = 10

    mce_vart: IntLike = None
    penalty_type: str = 'mse_d'

    # derived / descriptive
    mtot: int = 1
    mtot_no_mce: int = 0
    estimator_str: str = 'MSE & MCE'

    # pass-through / optional extras
    chunks_maxsize: NumberLike = None
    vi_oob_yes: bool = False
    n_min_max: NumberLike = None
    n_min_min: NumberLike = None
    n_min_treat: NumberLike = None
    p_diff_penalty: NumberLike = None
    subsample_factor_eval: NumberLike | bool = None
    subsample_factor_forest: NumberLike | bool = None
    random_thresholds: NumberLike = None

    # To be set later during training and prediction
    baseline: IntLike = field(default=None, init=False)
    n_train: IntLike = field(default=None, init=False)
    n_train_eff: IntLike = field(default=None, init=False)
    subsample_share_forest: NumberLike = field(default=None, init=False)
    subsample_share_eval: NumberLike = field(default=None, init=False)
    n_min_values: NumberLike = field(default=None, init=False)
    forests: list | tuple | None = field(default=None, init=False)
    est_rounds: ListofStrLike | tuple[str, ...] = field(default=None, init=False
                                                        )
    folds: NumberLike = field(default=None, init=False)
    m_values: list[int] | None = field(default=None, init=False)
    x_name_mcf: ListofStrLike = field(default=None, init=False)

    @classmethod
    def from_args(
        cls,
        *,
        alpha_reg_grid: NumberLike = None,
        alpha_reg_max: NumberLike = None,
        alpha_reg_min: NumberLike = None,
        boot: NumberLike = None,
        chunks_maxsize: NumberLike = None,
        compare_only_to_zero: BoolLike = None,
        nn_main_diag_only: BoolLike = None,
        n_min_grid: NumberLike = None,
        n_min_max: NumberLike = None,
        n_min_min: NumberLike = None,
        n_min_treat: NumberLike = None,
        m_grid: NumberLike = None,
        m_share_max: NumberLike = None,
        m_share_min: NumberLike = None,
        m_random_poisson: BoolLike = None,
        match_nn_prog_score: BoolLike = None,
        mce_vart: IntLike = None,
        p_diff_penalty: NumberLike = None,
        penalty_type: StrLike = None,
        subsample_factor_eval: NumberLike | bool = None,
        subsample_factor_forest: NumberLike = None,
        tune_all: BoolLike = None,
        random_thresholds: NumberLike = None,
        vi_oob_yes: BoolLike = None,
        zero_tol: float = 1e-15,
            ) -> 'CfCfg':
        """Get input and normalize parameters."""
        # integers / grids
        boot_i = 1000 if boot is None or boot < 1 else round(boot)
        m_grid_i = 1 if m_grid is None or m_grid < 1 else round(m_grid)
        n_min_grid_i = (1 if n_min_grid is None or n_min_grid < 1
                        else round(n_min_grid)
                        )
        arg_grid_chk = (1 if alpha_reg_grid is None or alpha_reg_grid < 1
                        else round(alpha_reg_grid)
                        )
        tune_all_b = tune_all is True
        if tune_all_b:
            no_vals = 3
            m_grid_i = max(m_grid_i, no_vals)
            n_min_grid_i = max(n_min_grid_i, no_vals)
            arg_grid_chk = max(arg_grid_chk, no_vals)

        # get alpha grid/max/min/values from helper
        (alpha_reg_grid_o, alpha_reg_max_o, alpha_reg_min_o, alpha_reg_values_o,
         ) = get_alpha(arg_grid_chk, alpha_reg_max, alpha_reg_min,
                       )
        # shares
        m_share_min_f = (
            0.1 if m_share_min is None or not (0 < m_share_min <= 1)
            else float(m_share_min)
            )
        m_share_max_f = (
            0.6 if m_share_max is None or not (0 < m_share_max <= 1)
            else float(m_share_max)
            )
        # poisson toggle + min
        if m_random_poisson is False:
            m_random_poisson_b = False
            m_random_poisson_min_i = 1_000_000
        else:
            m_random_poisson_b = True
            m_random_poisson_min_i = 10

        # objective function choice via mce_vart
        match mce_vart:
            case None | 1:
                mtot_i, mtot_no_mce_i = 1, 0
                estimator_str_s = 'MSE & MCE'
            case 2:
                mtot_i, mtot_no_mce_i = 2, 1
                estimator_str_s = '-Var(effect)'
            case 0:
                mtot_i, mtot_no_mce_i = 3, 1
                estimator_str_s = 'MSE'
            case 3:
                mtot_i, mtot_no_mce_i = 4, 0
                estimator_str_s = 'MSE, MCE or penalty (random)'
            case _:
                raise ValueError('Inconsistent MTOT definition of MCE_VarT.')

        penalty_type_s = (
            'mse_d' if penalty_type is None or penalty_type != 'diff_d'
            else penalty_type
            )
        return cls(
            boot=boot_i,
            match_nn_prog_score=(match_nn_prog_score is not False),
            nn_main_diag_only=(nn_main_diag_only is True),
            compare_only_to_zero=(compare_only_to_zero is True),
            m_grid=m_grid_i,
            n_min_grid=n_min_grid_i,
            alpha_reg_grid=alpha_reg_grid_o,
            alpha_reg_max=alpha_reg_max_o,
            alpha_reg_min=alpha_reg_min_o,
            alpha_reg_values=alpha_reg_values_o,
            tune_all=tune_all_b,
            m_share_min=m_share_min_f,
            m_share_max=m_share_max_f,
            m_random_poisson=m_random_poisson_b,
            m_random_poisson_min=m_random_poisson_min_i,
            mce_vart=mce_vart,
            penalty_type=penalty_type_s,
            mtot=mtot_i,
            mtot_no_mce=mtot_no_mce_i,
            estimator_str=estimator_str_s,
            chunks_maxsize=chunks_maxsize,
            vi_oob_yes=(vi_oob_yes is True),
            n_min_max=n_min_max,
            n_min_min=n_min_min,
            n_min_treat=n_min_treat,
            p_diff_penalty=p_diff_penalty,
            subsample_factor_eval=subsample_factor_eval,
            subsample_factor_forest=subsample_factor_forest,
            random_thresholds=random_thresholds,
            )
