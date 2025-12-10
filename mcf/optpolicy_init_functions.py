"""Created on Sat July 15 10:03:15 2023.

Contains the functions needed for initialising the parameters.
@author: MLechner
-*- coding: utf-8 -*-
"""
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from psutil import cpu_count

from mcf import mcf_general as gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_general_sys as mcf_sys

type NameLike = str | list[str] | tuple[str, ...] | None
type NumberLike = int | float | None
type BoolLike = bool | None
type StrLike = str | None

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


# Useful enums
class FontLabel(StrEnum):
    """Define labels for plots."""

    XX_SMALL = 'xx-small'
    X_SMALL = 'x-small'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    X_LARGE = 'x-large'
    XX_LARGE = 'xx-large'


class LegendLoc(StrEnum):
    """Define location for legend in plots."""

    BEST = 'best'
    UPPER_RIGHT = 'upper right'
    UPPER_LEFT = 'upper left'
    LOWER_LEFT = 'lower left'
    LOWER_RIGHT = 'lower right'
    RIGHT = 'right'
    CENTER_LEFT = 'center left'
    CENTER_RIGHT = 'center right'
    LOWER_CENTER = 'lower center'
    UPPER_CENTER = 'upper center'
    CENTER = 'center'


@dataclass(slots=True, kw_only=True)
class DataCleanCfg:
    """Parameters for data cleaning (normalized)."""

    check_perfectcorr: bool
    clean_data: bool
    screen_covariates: bool
    min_dummy_obs: int

    @classmethod
    def from_args(
        cls,
        *,
        check_perfectcorr: BoolLike = None,
        clean_data: BoolLike = None,
        min_dummy_obs: NumberLike = None,
        screen_covariates: BoolLike = None,
            ) -> 'DataCleanCfg':
        """Define parameters from input."""
        return cls(
            check_perfectcorr=(check_perfectcorr is not False),
            clean_data=(clean_data is not False),
            screen_covariates=(screen_covariates is not False),
            min_dummy_obs=(
                10
                if (min_dummy_obs is None or float(min_dummy_obs) < 1)
                else int(round(float(min_dummy_obs)))
            ),
        )


@dataclass(slots=True, kw_only=True)
class EstRiskCfg:
    """Initialise parameters for estimation risk adjustments."""

    # always starts False
    estrisk_used: bool = field(default=False, init=False)

    # numeric value, defaults to 1 if input isn't numeric
    value: int | float = 1

    @classmethod
    def from_args(cls, *, value: object | None = None) -> 'EstRiskCfg':
        """Modify input."""
        val = value if isinstance(value, (int, float)) else 1
        return cls(value=val)


@dataclass(slots=True, kw_only=True)
class FairCfg:
    """Parameters for fair allocations with protected variables."""

    # canonical (normalized) fields
    cont_min_values: int = 20
    adjust_target: str = 'xvariables'
    regression_method: str = 'RandomForest'
    adj_type: str = 'Quantiled'

    # discretization
    discretization_methods: tuple[str, ...] = ('EqualCell', 'Kmeans',
                                               'NoDiscretization',
                                               )
    default_disc_method: str = 'Kmeans'
    protected_disc_method: str = 'Kmeans'
    material_disc_method: str = 'Kmeans'

    # other controls
    consistency_test: bool = False
    protected_max_groups: int = 5
    material_max_groups: int = 5

    # flags populated elsewhere
    fairscores_used: bool = False
    solvefair_used: bool = False

    # to be populated later (if at all)
    decision_vars_fair_org_name: list[str] | tuple[str] | None = (
        field(default=None, init=False)
        )
    x_ord_org_name: list[str] | tuple[str] | None = field(default=None,
                                                          init=False)
    fair_strata: DataFrame | None = field(default=None, init=False)
    decision_vars_org_df: DataFrame | None = field(default=None, init=False)
    data_train_for_pred_df: DataFrame | None = field(default=None, init=False)
    protected_matrel: DataFrame | None = field(default=None, init=False)
    org_name_dict: dict | None = field(default=None, init=False)

    @classmethod
    def from_args(
        cls,
        gen_cfg: Any,
        *,
        adjust_target: StrLike = None,
        consistency_test: bool | None = None,
        cont_min_values: NumberLike = None,
        material_disc_method: StrLike = None,
        material_max_groups: NumberLike = None,
        protected_disc_method: StrLike = None,
        protected_max_groups: NumberLike = None,
        regression_method: StrLike = None,
        adj_type: StrLike = None,
            ) -> 'FairCfg':
        """Normalize and modify input."""
        ok_methods = (
            'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5',
            'SupportVectorMachine', 'SupportVectorMachineC2',
            'SupportVectorMachineC4',
            'AdaBoost', 'AdaBoost100', 'AdaBoost200',
            'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12',
            'LASSO',
            'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
            'Mean',
            'automatic',
            )
        ok_adjustments = ('scores', 'xvariables', 'scores_xvariables',)
        ok_types = ('Mean', 'MeanVar', 'Quantiled',)
        ok_disc_methods = ('NoDiscretization', 'EqualCell', 'Kmeans',)

        # cont_min_values
        if (cont_min_values is None
            or not isinstance(cont_min_values, (int, float))
                or cont_min_values < 1):
            cmv = 20
        else:
            cmv = int(round(cont_min_values))

        # validated strings
        adj_tgt = check_valid_user_str(
            gen_cfg,
            title='fair_adjust_target',
            user_input=adjust_target,
            default='xvariables',
            valid_strings=ok_adjustments,
            )
        regr = check_valid_user_str(
            gen_cfg,
            title='fair_regression_method',
            user_input=regression_method,
            default='RandomForest',
            valid_strings=ok_methods,
            )
        atype = check_valid_user_str(
            gen_cfg,
            title='fair_adj_type',
            user_input=adj_type,
            default='Quantiled',
            valid_strings=ok_types,
            )
        default_disc = 'Kmeans'
        prot_disc = check_valid_user_str(
            gen_cfg,
            title='fair_protected_disc_method',
            user_input=protected_disc_method,
            default=default_disc,
            valid_strings=ok_disc_methods,
            )
        mat_disc = check_valid_user_str(
            gen_cfg,
            title='fair_material_disc_method',
            user_input=material_disc_method,
            default=default_disc,
            valid_strings=ok_disc_methods,
            )
        # booleans and group sizes (no brackets around one-liners)
        cons_test = consistency_test is True

        if isinstance(protected_max_groups, (int, float)):
            prot_max = int(round(protected_max_groups))
        else:
            prot_max = 5

        if isinstance(material_max_groups, (int, float)):
            mat_max = round(material_max_groups)
        else:
            mat_max = 5

        return cls(
            cont_min_values=cmv,
            adjust_target=adj_tgt,
            regression_method=regr,
            adj_type=atype,
            protected_disc_method=prot_disc,
            material_disc_method=mat_disc,
            consistency_test=cons_test,
            protected_max_groups=prot_max,
            material_max_groups=mat_max,
            discretization_methods=('EqualCell', 'Kmeans', 'NoDiscretization',),
            default_disc_method=default_disc,
            fairscores_used=False,
            solvefair_used=False,
            )

# def init_fair(gen_cfg: Any,
#               adjust_target: StrLike = None,
#               consistency_test: BoolLike = None,
#               cont_min_values: NumberLike = None,
#               material_disc_method: StrLike = None,
#               material_max_groups: NumberLike = None,
#               protected_disc_method: StrLike = None,
#               protected_max_groups: NumberLike = None,
#               regression_method: StrLike = None,
#               adj_type: StrLike = None
#               ) -> dict:
#     """Initialise parameters for fair allocations with protected variables."""
#     ok_methods = ('RandomForest', 'RandomForestNminl5',
#                   'RandomForestNminls5',
#                   'SupportVectorMachine', 'SupportVectorMachineC2',
#                   'SupportVectorMachineC4',
#                   'AdaBoost', 'AdaBoost100', 'AdaBoost200',
#                   'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',
#                   'LASSO',
#                   'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger',
#                   'Mean', 'automatic')
#     ok_adjustments = ('scores', 'xvariables', 'scores_xvariables',)
#     ok_types = ('Mean', 'MeanVar', 'Quantiled',)
#     ok_disc_methods = ('NoDiscretization', 'EqualCell', 'Kmeans',)
#     dic = {}

#     if (cont_min_values is None
#         or not isinstance(cont_min_values, (int, float))
#             or cont_min_values < 1):

#         dic['cont_min_values'] = 20
#     else:
#         dic['cont_min_values'] = int(round(cont_min_values))

#     dic['adjust_target'] = check_valid_user_str(
#         gen_cfg,
#         title='fair_adjust_target',
#         user_input=adjust_target,
#         default='xvariables',
#         valid_strings=ok_adjustments
#         )
#     dic['regression_method'] = check_valid_user_str(
#         gen_cfg,
#         title='fair_regression_method',
#         user_input=regression_method,
#         default='RandomForest',
#         valid_strings=ok_methods
#         )
#     dic['adj_type'] = check_valid_user_str(
#         gen_cfg,
#         title='fair_adj_type',
#         user_input=adj_type,
#         default='Quantiled',
#         valid_strings=ok_types
#         )
#     # Discretization methods
#    dic['discretization_methods'] = ('EqualCell', 'Kmeans','NoDiscretization',)
#     dic['default_disc_method'] = 'Kmeans'
#     dic['protected_disc_method'] = check_valid_user_str(
#         gen_cfg,
#         title='fair_protected_disc_method',
#         user_input=protected_disc_method,
#         default=dic['default_disc_method'],
#         valid_strings=ok_disc_methods
#         )
#     dic['material_disc_method'] = check_valid_user_str(
#         gen_cfg,
#         title='fair_material_disc_method',
#         user_input=material_disc_method,
#         default=dic['default_disc_method'],
#         valid_strings=ok_disc_methods
#         )

#     dic['consistency_test'] = consistency_test is True

#     if isinstance(protected_max_groups, (int, float)):
#         dic['protected_max_groups'] = int(round(protected_max_groups))
#     else:
#         dic['protected_max_groups'] = 5

#     if isinstance(material_max_groups, (int, float)):
#         dic['material_max_groups'] = round(material_max_groups)
#     else:
#         dic['material_max_groups'] = 5

#     dic['fairscores_used'] = False   # Will be overwritten if fairscores
#     #                                  method is used
#     dic['solvefair_used'] = False    # Will be overwritten if solvefair
#     #                                  method is used

#     return dic


def check_valid_user_str(gen_cfg: Any,
                         title: str = '',
                         user_input: StrLike = None,
                         default: str = '',
                         valid_strings: tuple[str, ...] = ('',),
                         ) -> str:
    """Check if user specified value is valid or whether to use default."""
    # No user input -> default
    if user_input is None or user_input == '':
        return default

    # Correct user input
    if user_input in valid_strings:
        return user_input

    # incorrect string as user input, check if useful
    if isinstance(user_input, str):
        valid_strings_cf = [name.casefold() for name in valid_strings]
        user_input_cf = user_input.casefold()
        if user_input_cf in valid_strings_cf:  # return corrected user input
            return valid_strings[valid_strings_cf.index(user_input_cf)]
        # print warning to file and return default
        mcf_ps.print_mcf(gen_cfg,
                         f'\n{title}: WARNING '
                         f'\n{user_input} is not among the valid methods: '
                         f'{" ".join(valid_strings)}',
                         summary=True
                         )
        return default

    # In any other case return default
    return default


@dataclass(slots=True, kw_only=True)
class GenCfg:
    """Define configuration parameters."""

    method: Literal['best_policy_score', 'policy_tree',
                    'policy tree old', 'bps_classifier']
    mp_parallel: int
    variable_importance: bool
    output_type: int
    print_to_file: bool
    print_to_terminal: bool
    with_output: BoolLike
    dir_nam: str
    outpath: Path | None
    outfiletext: Path | None
    outfilesummary: Path | None
    d_values: tuple[np.int32, ...] | None = None
    no_of_treat: int = 2
    x_cont_flag: bool = False
    x_continuous: bool = False
    x_ord_flag: bool = False
    x_unord_flag: bool = False

    @classmethod
    def from_args(
        cls,
        method: StrLike = None,
        mp_parallel: int | float | None = None,
        outfiletext: StrLike = None,
        outpath: Path | StrLike = None,
        output_type: int | float | None = None,
        variable_importance: BoolLike = None,
        with_output: BoolLike = None,
        new_outpath: Path | StrLike = None,
    ) -> 'GenCfg':
        """Get the input parameters and transform them if needed."""
        # mp_parallel
        if mp_parallel is None or not isinstance(mp_parallel, (int, float)):
            mp_parallel_n = round(cpu_count(logical=True) * 0.8)
        elif mp_parallel <= 1.5:
            mp_parallel_n = 1
        else:
            mp_parallel_n = round(float(mp_parallel))

        # method
        m = 'best_policy_score' if method is None else method
        valid = ('best_policy_score', 'policy_tree',
                 'policy tree old', 'bps_classifier')
        if m not in valid:
            raise ValueError(f'{m} is not a valid method.')

        match m:
            case 'best_policy_score': dir_nam = 'BPS'
            case 'policy_tree':       dir_nam = 'PT'
            case 'policy tree old':   dir_nam = 'PT_OLD'
            case 'bps_classifier':    dir_nam = 'BPS_CLASSIF'
            case _:                   dir_nam = ''

        # variable importance: True unless explicitly False
        vi = variable_importance is not False

        # output_type -> print switches
        out_t = 2 if output_type is None else int(output_type)
        match out_t:
            case 0: ptf, ptt = False, True
            case 1: ptf, ptt = True, False
            case _: ptf, ptt = True, True

        if not with_output:
            ptf = ptt = False

        # outpath + files
        if with_output:
            if outpath is None:
                outpath_p = mcf_sys.define_outpath(None, new_outpath)
            else:
                outpath_p = Path(outpath)
                outpath_p = mcf_sys.define_outpath(outpath_p / dir_nam,
                                                   new_outpath)
        else:
            outpath_p = None

        name = 'txtFileWithOutput' if outfiletext is None else outfiletext
        of_txt = outpath_p / f'{name}.txt' if outpath_p else None
        of_sum = outpath_p / f'{name}_Summary.txt' if outpath_p else None

        if with_output and outpath_p:
            mcf_sys.delete_file_if_exists(of_txt)
            mcf_sys.delete_file_if_exists(of_sum)

        return cls(
            method=m,
            mp_parallel=mp_parallel_n,
            variable_importance=vi,
            output_type=out_t,
            print_to_file=ptf,
            print_to_terminal=ptt,
            with_output=with_output,
            dir_nam=dir_nam,
            outpath=outpath_p,
            outfiletext=of_txt,
            outfilesummary=of_sum,
        )


def init_gen_solve(optp_: 'OptimalPolicy', data_df: DataFrame) -> None:
    """Add and update some dictionary entry based on data."""
    var_cfg, gen_cfg = optp_.var_cfg, optp_.gen_cfg
    if var_cfg.d_name and var_cfg.d_name[0] in data_df.columns:
        d_dat = data_df[var_cfg.d_name].to_numpy()
        gen_cfg.d_values = tuple(np.unique(np.round(d_dat).astype(np.int32)))

    else:
        gen_cfg.d_values = tuple(np.int32(range(len(var_cfg.polscore_name))))
    gen_cfg.no_of_treat = len(gen_cfg.d_values)

    optp_.gen_cfg = gen_cfg


@dataclass(frozen=True, slots=True, kw_only=True)
class IntCfg:
    """Define basic configuration variables for optimal policy class."""

    cuda: bool = False
    output_no_new_dir: bool = False
    report: bool = True
    with_numba: bool = True
    with_output: bool = True
    xtr_parallel: bool = False
    dpi: int = 500
    fontsize: FontLabel = FontLabel.X_SMALL
    legend_loc: LegendLoc = LegendLoc.BEST
    zero_tol: float = 1e-15
    sum_tol: float = 1e-12     # for sum about 1/0 checks

    # definition order of the enum gives us the mapping 1..7
    FONT_ORDER: ClassVar[tuple[FontLabel, ...]] = tuple(FontLabel)

    @staticmethod
    def _bool_or_default(value: BoolLike, default: bool) -> bool:
        return default if value is None else bool(value)

    @staticmethod
    def _coerce_dpi(dpi: NumberLike) -> int:
        return 500 if (dpi is None or dpi < 10) else int(round(dpi))

    @staticmethod
    def _coerce_fontsize(fs: NumberLike | FontLabel | str) -> FontLabel:
        if isinstance(fs, FontLabel):
            return fs
        if isinstance(fs, str):
            try:
                return FontLabel(fs)
            except ValueError as e:
                raise ValueError(f'unknown fontsize label: {fs!r}') from e
        # numeric path: 0.5 < fs < 7.5, round to 1..7; default 2 -> 'x-small'
        if fs is not None and 0.5 < fs < 7.5:
            idx = int(round(fs)) - 1
        else:
            idx = 1
        return IntCfg.FONT_ORDER[idx]

    @staticmethod
    def _coerce_legend_loc(loc: LegendLoc | StrLike) -> LegendLoc:
        if loc is None:
            return LegendLoc.BEST
        if isinstance(loc, LegendLoc):
            return loc
        try:
            return LegendLoc(loc)
        except ValueError as e:
            raise ValueError(f'unknown legend location: {loc!r}') from e

    @classmethod
    def from_args(cls,
                  *,
                  cuda: BoolLike = None,
                  output_no_new_dir: BoolLike = None,
                  report: BoolLike = None,
                  with_numba: BoolLike = None,
                  with_output: BoolLike = None,
                  xtr_parallel: BoolLike = True,
                  dpi: NumberLike = 500,
                  fontsize: NumberLike | FontLabel | str = 2,
                  legend_loc: LegendLoc | StrLike = None) -> 'IntCfg':
        """Allow flexible types in inputs."""
        if cuda:
            raise NotImplementedError('GPU is not used for Optimal Policy')
        return cls(
            cuda=False,
            output_no_new_dir=cls._bool_or_default(output_no_new_dir, False),
            report=cls._bool_or_default(report, True),
            with_numba=cls._bool_or_default(with_numba, True),
            with_output=cls._bool_or_default(with_output, True),
            xtr_parallel=cls._bool_or_default(xtr_parallel, True),
            dpi=cls._coerce_dpi(dpi),
            fontsize=cls._coerce_fontsize(fontsize),
            legend_loc=cls._coerce_legend_loc(legend_loc),
            )

    # convenience for matplotlib interop
    @property
    def mpl_fontsize(self) -> str:
        """Define property."""
        return self.fontsize.value

    @property
    def mpl_legend_loc(self) -> str:
        """Define property."""
        return self.legend_loc.value


def init_rnd_shares(optp_:
                    'OptimalPolicy', data_df: DataFrame,
                    d_in_data: bool,
                    ) -> None:
    """Reinitialise the shares if they are not consistent with use."""
    no_of_treat = optp_.gen_cfg.no_of_treat
    rnd_cfg, var_cfg = optp_.rnd_cfg, optp_.var_cfg
    if rnd_cfg.shares is None or len(rnd_cfg.shares) < no_of_treat:
        if d_in_data:
            obs_shares = data_df[var_cfg.d_name].value_counts(normalize=True
                                                              ).sort_index()
            rnd_cfg.shares = obs_shares.tolist()
        else:
            rnd_cfg.shares = [1/no_of_treat] * no_of_treat
    if sum(rnd_cfg.shares) < 0.999999 or sum(rnd_cfg.shares) > 1.0000001:
        raise ValueError('"random shares" do not add to 1.')
    optp_.rnd_cfg = rnd_cfg


@dataclass(slots=True, kw_only=True)
class PtCfg:
    """Parameters related to policy trees (translated from init_pt)."""

    # canonical / stored values
    no_of_evalupoints: int = 100
    depth_tree_1: int = 4
    depth_tree_2: int = 2
    min_leaf_size: NumberLike = None  # to be set later elsewhere
    select_values_cat: bool = False
    enforce_restriction: bool = False
    eva_cat_mult: float = 2.0

    # derived fields
    depth: int = field(default=0, init=False)
    depth_tree_1_adj: int = field(default=0, init=False)
    depth_tree_2_adj: int = field(default=0, init=False)
    total_depth_adj: int = field(default=0, init=False)

    # Initialised later
    cost_of_treat_restrict: NDArray | None = field(default=None, init=False)
    policy_tree: list[Any] | None = field(default=None, init=False)

    @classmethod
    def from_args(
        cls,
        *,
        depth_tree_1: NumberLike = None,
        depth_tree_2: NumberLike = None,
        enforce_restriction: BoolLike = None,
        eva_cat_mult: NumberLike = None,
        no_of_evalupoints: NumberLike = None,
        min_leaf_size: NumberLike = None,
        select_values_cat: BoolLike = None,
            ) -> 'PtCfg':
        """Get and transform input."""
        # no_of_evalupoints
        if no_of_evalupoints is None or no_of_evalupoints < 5:
            noe = 100
        else:
            noe = int(round(no_of_evalupoints))

        # depths (note the +1 then round, matching original)
        if depth_tree_1 is None or depth_tree_1 < 1:
            d1 = 4
        else:
            d1 = int(round(depth_tree_1 + 1))

        if depth_tree_2 is None or depth_tree_2 < 0:
            d2 = 2
        else:
            d2 = int(round(depth_tree_2 + 1))

        # booleans with "is True" semantics
        svc = select_values_cat is True
        enf = enforce_restriction is True
        if enf and d2 > 1:
            enf = False

        # eva_cat_mult
        if (eva_cat_mult is None
            or not isinstance(eva_cat_mult, (int, float))
                or eva_cat_mult < 0.1):
            eva = 2.0
        else:
            eva = float(eva_cat_mult)

        cfg = cls(
            no_of_evalupoints=noe,
            depth_tree_1=d1,
            depth_tree_2=d2,
            min_leaf_size=min_leaf_size,
            select_values_cat=svc,
            enforce_restriction=enf,
            eva_cat_mult=eva,
            )
        # derived fields
        cfg.depth = cfg.depth_tree_1 + cfg.depth_tree_2 - 1
        cfg.depth_tree_1_adj = cfg.depth_tree_1 - 1
        cfg.depth_tree_2_adj = cfg.depth_tree_2 - 1
        cfg.total_depth_adj = cfg.depth_tree_1_adj + cfg.depth_tree_2_adj

        return cfg


@dataclass(slots=True, kw_only=True)
class RndCfg:
    """Contains shares for random allocation."""

    shares: list[NumberLike] | tuple[NumberLike] | None = None

    @classmethod
    def from_args(cls, *,
                  rnd_shares: list[NumberLike] | tuple[NumberLike] | None = None
                  ) -> 'RndCfg':
        """Get input."""
        return cls(shares=rnd_shares)


def init_pt_solve(optp_: 'OptimalPolicy', no_of_obs: int | float) -> None:
    """Initialise parameters related to policy tree."""
    if (optp_.pt_cfg.min_leaf_size is None
            or optp_.pt_cfg.min_leaf_size < 0):
        optp_.pt_cfg.min_leaf_size = 0.1 * no_of_obs / (
            (optp_.pt_cfg.depth - 1) * 2)
        if optp_.other_cfg.restricted:
            min_share = np.min(optp_.other_cfg.max_shares)
            optp_.pt_cfg.min_leaf_size = min(round(
                optp_.pt_cfg.min_leaf_size * min_share), 100
                )
    else:
        optp_.pt_cfg.min_leaf_size = round(optp_.pt_cfg.min_leaf_size)


@dataclass(slots=True, kw_only=True)
class OtherCfg:
    """Cost data."""

    costs_of_treat: list[NumberLike] | tuple[NumberLike] | None = None
    costs_of_treat_mult: list[NumberLike] | tuple[NumberLike] | None = None
    max_shares: list[NumberLike] | tuple[NumberLike] | None = None

    # To be initialized later
    restricted: BoolLike = field(default=None, init=False)
    max_by_treat: NDArray | None = field(default=None, init=False)

    @classmethod
    def from_args(
        cls,
        *,
        other_costs_of_treat=None,
        other_costs_of_treat_mult=None,
        other_max_shares=None,
            ) -> 'OtherCfg':
        """Get inputs."""
        return cls(
            costs_of_treat=other_costs_of_treat,
            costs_of_treat_mult=other_costs_of_treat_mult,
            max_shares=other_max_shares,
            )


def init_other_solve(optp_: 'OptimalPolicy') -> None:
    """Initialise treatment costs (needs info on number of treatments."""
    no_of_treat = optp_.gen_cfg.no_of_treat
    other_cfg = optp_.other_cfg
    if other_cfg.max_shares is None or len(other_cfg.max_shares) < no_of_treat:
        other_cfg.max_shares = [1] * no_of_treat
    no_zeros = sum(1 for share in other_cfg.max_shares if share == 0)
    if no_zeros == len(other_cfg.max_shares):
        raise ValueError('All restrictions are zero. No allocation possible.')
    if sum(other_cfg.max_shares) < 1:
        raise ValueError('Sum of restrictions < 1. No allocation possible.')
    other_cfg.restricted = any(share < 1 for share in other_cfg.max_shares)
    if (other_cfg.costs_of_treat is None
            or len(other_cfg.costs_of_treat) < no_of_treat):
        other_cfg.costs_of_treat = [0] * no_of_treat

    if (other_cfg.costs_of_treat_mult is None
            or len(other_cfg.costs_of_treat_mult) < no_of_treat):
        mult = 1
        other_cfg.costs_of_treat_mult = [mult] * no_of_treat
    if any(cost <= 0 for cost in other_cfg.costs_of_treat_mult):
        raise ValueError('Cost multiplier must be positive.')
    optp_.other_cfg = other_cfg


@dataclass(slots=True, kw_only=True)
class VarCfg:
    """Configure the variable names."""

    bb_restrict_name: list[str]
    d_name: list[str]
    effect_vs_0: list[str]
    effect_vs_0_se: list[str]
    id_name: list[str]
    material_ord_name: list[str]
    material_unord_name: list[str]
    polscore_name: list[str]
    polscore_desc_name: list[str]
    polscore_se_name: list[str]
    protected_ord_name: list[str]
    protected_unord_name: list[str]
    vi_x_name: list[str]
    vi_to_dummy_name: list[str]
    x_ord_name: list[str]
    x_unord_name: list[str]

    # Initialised by from_args()
    name_ordered: list[str] = field(default_factory=list)
    name_unordered: list[str] = field(default_factory=list)
    x_name_balance_test: list[str] = field(default_factory=list)
    x_name_always_in: list[str] = field(default_factory=list)
    z_name: list[str] = field(default_factory=list)
    x_name_remain: list[str] = field(default_factory=list)

    # May be initialized later by the programme
    x_name: list[str] = field(default_factory=list, init=False)
    prot_mat_no_dummy_name: list[str] = field(default_factory=list, init=False)
    protected_name: list[str] = field(default_factory=list, init=False)
    material_name: list[str] = field(default_factory=list, init=False)

    @classmethod
    def from_args(
        cls,
        *,
        bb_restrict_name: NameLike = None,
        d_name: NameLike = None,
        effect_vs_0: NameLike = None,
        effect_vs_0_se: NameLike = None,
        id_name: NameLike = None,
        material_ord_name: NameLike = None,
        material_unord_name: NameLike = None,
        polscore_name: NameLike = None,
        polscore_desc_name: NameLike = None,
        polscore_se_name: NameLike = None,
        protected_ord_name: NameLike = None,
        protected_unord_name: NameLike = None,
        vi_x_name: NameLike = None,
        vi_to_dummy_name: NameLike = None,
        x_ord_name: NameLike = None,
        x_unord_name: NameLike = None,
    ) -> 'VarCfg':
        """Read and transform input."""
        return cls(
            bb_restrict_name=check_var(bb_restrict_name),
            d_name=check_var(d_name),
            effect_vs_0=check_var(effect_vs_0),
            effect_vs_0_se=check_var(effect_vs_0_se),
            id_name=check_var(id_name),
            polscore_desc_name=check_var(polscore_desc_name),
            polscore_name=check_var(polscore_name),
            polscore_se_name=check_var(polscore_se_name),
            x_ord_name=check_var_no_none(x_ord_name),
            x_unord_name=check_var_no_none(x_unord_name),
            vi_x_name=check_var(vi_x_name),
            vi_to_dummy_name=check_var(vi_to_dummy_name),
            protected_ord_name=check_var_no_none(protected_ord_name),
            protected_unord_name=check_var_no_none(protected_unord_name),
            material_ord_name=check_var_no_none(material_ord_name),
            material_unord_name=check_var_no_none(material_unord_name),
        )


def check_var_no_none(var: NameLike) -> NameLike:
    """Capitalise and clean variable names and remove None's."""
    return [] if var is None else check_var(var)


def check_var(variable: NameLike) -> NameLike:
    """Capitalise and clean variable names."""
    if variable is None or variable == []:
        return variable
    variable = gp.to_list_if_needed(variable)
    variable = gp.cleaned_var_names(variable)

    return variable
