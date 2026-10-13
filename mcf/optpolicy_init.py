"""Created on Sat July 15 10:03:15 2023.

Contains the functions needed for initialising the parameters.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from pathlib import Path
import sys
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
try:
    import ray  # type: ignore[import]
except (ImportError, OSError):
    ray = None  # type: ignore[assignment]
from pandas import DataFrame
from psutil import cpu_count

from mcf import mcf_general as gp
from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_general_sys as mcf_sys

type NameLike = str | list[str] | tuple[str, ...] | None
type NumberLike = int | float | None
type BoolLike = bool | None
type StrLike = str | None
type IntLike = int | None

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
    """Parameters for covariate screening (normalized)."""

    check_perfectcorr: bool
    min_dummy_obs: int

    @classmethod
    def from_args(cls: type['DataCleanCfg'], *,
                  check_perfectcorr: BoolLike = None,
                  min_dummy_obs: NumberLike = None,
                  ) -> 'DataCleanCfg':
        """Define parameters from input."""
        return cls(check_perfectcorr=check_perfectcorr is not False,
                   min_dummy_obs=(10 if (min_dummy_obs is None or float(min_dummy_obs) < 1)
                                  else int(round(float(min_dummy_obs))))
                   )


@dataclass(slots=True, kw_only=True)
class EstRiskCfg:
    """Initialise parameters for estimation risk adjustments."""

    # always starts False
    estrisk_used: bool = field(default=False, init=False)
    polscore_name_org: list[str] | None = field(default=None, init=False)
    var_cfg_original: 'VarCfg | None' = field(default=None, init=False, repr=False)
    # numeric value, defaults to 1 if input isn't numeric
    value: int | float = 1

    @classmethod
    def from_args(cls, *, value: object | None = None) -> 'EstRiskCfg':
        """Modify input."""
        val = value if isinstance(value, (int, float)) else 1
        if isinstance(val, float) and not np.isfinite(val):
            raise ValueError('estrisk_value must be finite.')

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
    discretization_methods: tuple[str, ...] = ('EqualCell', 'Kmeans', 'ExactCell',)
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
    decision_vars_fair_org_name: list[str] | tuple[str] | None = field(default=None, init=False)
    x_ord_org_name: list[str] | tuple[str] | None = field(default=None, init=False)
    fair_strata: DataFrame | None = field(default=None, init=False)
    decision_vars_org_df: DataFrame | None = field(default=None, init=False)
    data_train_for_pred_df: DataFrame | None = field(default=None, init=False)
    protected_matrel: DataFrame | None = field(default=None, init=False)
    org_name_dict: dict | None = field(default=None, init=False)
    var_cfg_original: 'VarCfg | None' = field(default=None, init=False, repr=False)
    settings_original: dict[str, str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_args(cls,
                  gen_cfg: 'GenCfg', *,
                  adjust_target: StrLike = None,
                  consistency_test: BoolLike = None,
                  cont_min_values: NumberLike = None,
                  material_disc_method: StrLike = None,
                  material_max_groups: NumberLike = None,
                  protected_disc_method: StrLike = None,
                  protected_max_groups: NumberLike = None,
                  regression_method: StrLike = None,
                  adj_type: StrLike = None,
                  ) -> 'FairCfg':
        """Normalize and modify input."""
        ok_methods = ('RandomForest', 'RandomForestNminl5', 'RandomForestNminls5',
                      'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4',
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
        default_adjust_target = ('scores' if gen_cfg.method == 'best_policy_score' else 'xvariables'
                                 )
        adj_tgt = check_valid_user_str(gen_cfg,
                                       title='fair_adjust_target',
                                       user_input=adjust_target,
                                       default=default_adjust_target,
                                       valid_strings=ok_adjustments,
                                       )
        if gen_cfg.method == 'best_policy_score' and adj_tgt == 'xvariables':
            raise ValueError("fair_adjust_target='xvariables' has no effect with "
                             "gen_method='best_policy_score'. Use 'scores' or "
                             "'scores_xvariables'."
                             )
        regr = check_valid_user_str(gen_cfg,
                                    title='fair_regression_method',
                                    user_input=regression_method,
                                    default='RandomForest',
                                    valid_strings=ok_methods,
                                    )
        atype = check_valid_user_str(gen_cfg,
                                     title='fair_adj_type',
                                     user_input=adj_type,
                                     default='Quantiled',
                                     valid_strings=ok_types,
                                     )
        default_disc = 'Kmeans'
        prot_disc = check_valid_user_str(gen_cfg,
                                         title='fair_protected_disc_method',
                                         user_input=protected_disc_method,
                                         default=default_disc,
                                         valid_strings=ok_disc_methods,
                                         )
        mat_disc = check_valid_user_str(gen_cfg,
                                        title='fair_material_disc_method',
                                        user_input=material_disc_method,
                                        default=default_disc,
                                        valid_strings=ok_disc_methods,
                                        )
        # booleans and group sizes (no brackets around one-liners)
        cons_test = consistency_test is True

        if isinstance(protected_max_groups, (int, float)):
            if isinstance(protected_max_groups, float) and not np.isfinite(protected_max_groups):
                raise ValueError('fair_protected_max_groups must be finite.')
            prot_max = max(1, int(round(protected_max_groups)))
        else:
            prot_max = 5

        if isinstance(material_max_groups, (int, float)):
            if isinstance(material_max_groups, float) and not np.isfinite(material_max_groups):
                raise ValueError('fair_material_max_groups must be finite.')
            mat_max = max(1, int(round(material_max_groups)))
        else:
            mat_max = 5

        return cls(cont_min_values=cmv,
                   adjust_target=adj_tgt,
                   regression_method=regr,
                   adj_type=atype,
                   protected_disc_method=prot_disc,
                   material_disc_method=mat_disc,
                   consistency_test=cons_test,
                   protected_max_groups=prot_max,
                   material_max_groups=mat_max,
                   discretization_methods=('EqualCell', 'Kmeans', 'ExactCell',),
                   default_disc_method=default_disc,
                   fairscores_used=False,
                   solvefair_used=False,
                   )


@dataclass(slots=True, kw_only=True)
class GenCfg:
    """Define configuration parameters."""

    method: Literal['best_policy_score', 'policy_tree', 'policy tree old', 'bps_classifier']
    mp_parallel: int
    variable_importance: bool
    output_type: int
    print_to_file: bool
    print_to_terminal: bool
    with_output: BoolLike
    verbose: BoolLike
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
    def from_args(cls, *,
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
        valid = ('best_policy_score', 'policy_tree', 'policy tree old', 'bps_classifier')
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
        if output_type is not None and not isinstance(output_type, Real):
            raise TypeError('gen_output_type must be specified as a number.')
        out_t = 2 if output_type is None else int(round(output_type))
        if out_t not in (0, 1, 2):
            raise TypeError('gen_output_type must round to one of: 0, 1, 2.')
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
                outpath_p = mcf_sys.define_outpath(outpath_p / dir_nam, new_outpath)
        else:
            outpath_p = None

        name = 'txtFileWithOutput' if outfiletext is None else outfiletext
        of_txt = outpath_p / f'{name}.txt' if outpath_p else None
        of_sum = outpath_p / f'{name}_Summary.txt' if outpath_p else None

        if with_output and outpath_p:
            mcf_sys.delete_file_if_exists(of_txt)
            mcf_sys.delete_file_if_exists(of_sum)

        return cls(method=m,
                   mp_parallel=mp_parallel_n,
                   variable_importance=vi,
                   output_type=out_t,
                   print_to_file=ptf,
                   print_to_terminal=ptt,
                   with_output=with_output,
                   verbose=with_output,           # for compatibility reasons
                   dir_nam=dir_nam,
                   outpath=outpath_p,
                   outfiletext=of_txt,
                   outfilesummary=of_sum,
                   )


@dataclass(frozen=True, slots=True, kw_only=True)
class IntCfg:
    """Define basic configuration variables for optimal policy class."""

    cuda: bool = False
    output_no_new_dir: bool = False
    show_plots: bool = True
    with_numba: bool = True
    with_output: bool = True
    xtr_parallel: bool = False
    mp_backend: Literal['ray', 'joblib', 'sequential'] = 'joblib'
    mp_batches: IntLike | str = 'automatic'
    mp_memmap_min_bytes: int = 4 * 1024 * 1024
    mp_memmap_dir: Path | None = Path.cwd() / 'joblibtemp'
    mp_use_old_ray: bool = False
    dpi: int = 500
    fontsize: FontLabel = FontLabel.X_SMALL
    legend_loc: LegendLoc = LegendLoc.BEST
    zero_tol: float = 1e-10
    sum_tol: float = 1e-8     # for sum about 1/0 checks

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
                  show_plots: BoolLike = None,
                  with_numba: BoolLike = None,
                  with_output: BoolLike = None,
                  xtr_parallel: BoolLike = True,
                  mp_use_old_ray: BoolLike = False,
                  mp_backend: str | None = 'joblib',
                  mp_batches: IntLike | str = 'automatic',
                  mp_memmap_min_bytes: NumberLike = None,
                  mp_memmap_dir: Path | None = None,
                  dpi: NumberLike = 500,
                  fontsize: NumberLike | FontLabel | str = 2,
                  legend_loc: LegendLoc | StrLike = None) -> 'IntCfg':
        """Allow flexible types in inputs."""
        if cuda:
            raise NotImplementedError('GPU is not used for Optimal Policy')
        # old ray switch
        mp_use_old_ray_b = mp_use_old_ray is True

        # backend
        if mp_backend is None:
            mp_backend_s = 'joblib' if sys.platform.startswith('win') or ray is None else 'ray'
        else:
            if not isinstance(mp_backend, str):
                raise ValueError('Backend must be specified as string.')
            valid_backends = ('ray', 'joblib', 'sequential')
            if mp_backend not in valid_backends:
                raise ValueError(f'{mp_backend} is invalid value. '
                                 f'Valid values are {", ".join(valid_backends)}'
                                 )
            mp_backend_s = mp_backend

        if mp_batches is None:
            mp_batches_s = -1
        else:
            if not isinstance(mp_batches, (int, float, str)):
                raise ValueError('mp_batches must be specified as integer or string.')
            if isinstance(mp_batches, (int, float)):
                if mp_batches < 1:
                    raise ValueError(f'mp_batches must be positive. Current value is {mp_batches}')
                mp_batches_s = int(mp_batches)
            else:
                if mp_batches == 'automatic':
                    mp_batches_s = -1
                else:
                    raise ValueError('If mp_batches are strings, the only value allowed is '
                                     f'"automatic". Current value is {mp_batches}.'
                                     )
        # Joblib: When to switch to memory maps
        if not isinstance(mp_memmap_min_bytes, (int, float)) or mp_memmap_min_bytes < 0:
            mp_memmap_min_bytes_b = 4 * 1024 * 1024
        else:
            mp_memmap_min_bytes_b = int(mp_memmap_min_bytes)

        if isinstance(mp_memmap_dir, Path):
            mp_memmap_dir_b = mp_memmap_dir
        elif isinstance(mp_memmap_dir, str):
            mp_memmap_dir_b = Path(mp_memmap_dir)
        else:
            mp_memmap_dir_b = Path.cwd() / 'joblibtemp'

        return cls(cuda=False,
                   output_no_new_dir=cls._bool_or_default(output_no_new_dir, False),
                   show_plots=cls._bool_or_default(show_plots, True),
                   with_numba=cls._bool_or_default(with_numba, True),
                   with_output=cls._bool_or_default(with_output, True),
                   xtr_parallel=cls._bool_or_default(xtr_parallel, True),
                   mp_backend=mp_backend_s,
                   mp_batches=mp_batches_s,
                   mp_memmap_min_bytes=mp_memmap_min_bytes_b,
                   mp_memmap_dir=mp_memmap_dir_b,
                   mp_use_old_ray=mp_use_old_ray_b,
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


def init_rnd_shares(optp_: 'OptimalPolicy', data_df: DataFrame, d_in_data: bool,) -> None:
    """Reinitialise the shares if they are not specified."""
    no_of_treat = optp_.gen_cfg.no_of_treat
    d_values = optp_.gen_cfg.d_values
    rnd_cfg, var_cfg = optp_.rnd_cfg, optp_.var_cfg

    if not rnd_cfg.shares_user_specified:
        if d_in_data:
            observed_treat = data_df[var_cfg.d_name[0]]
            invalid_treat = np.setdiff1d(observed_treat.unique(), d_values)
            if invalid_treat.size:
                raise ValueError(f'Unknown observed treatment values: {invalid_treat.tolist()}')

            obs_shares = observed_treat.value_counts(normalize=True)
            obs_shares = obs_shares.reindex(d_values, fill_value=0)
            rnd_cfg.shares = obs_shares.tolist()
        else:
            rnd_cfg.shares = [1/no_of_treat] * no_of_treat
    elif len(rnd_cfg.shares) != no_of_treat:
        raise ValueError(f'rnd_shares must contain exactly {no_of_treat} values.')

    shares_np = np.asarray(rnd_cfg.shares, dtype=float)
    if shares_np.shape != (no_of_treat,):
        raise ValueError(f'rnd_shares must contain exactly {no_of_treat} scalar values.')
    if not np.all(np.isfinite(shares_np)) or np.any(shares_np < 0):
        raise ValueError('Random shares must be finite and nonnegative.')

    shares_sum = shares_np.sum()
    if not np.isclose(shares_sum, 1, rtol=0, atol=1e-6):
        raise ValueError('Random shares do not add to 1.')

    shares_np /= shares_sum
    rnd_cfg.shares = shares_np.tolist()
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
    def from_args(cls, *,
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

        cfg = cls(no_of_evalupoints=noe,
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
    shares_user_specified: bool = field(default=False, init=False)

    @classmethod
    def from_args(cls, *,
                  rnd_shares: list[NumberLike] | tuple[NumberLike] | None = None,
                  ) -> 'RndCfg':
        """Get input."""
        cfg = cls(shares=rnd_shares)
        cfg.shares_user_specified = rnd_shares is not None

        return cfg


def init_pt_solve(optp_: 'OptimalPolicy', no_of_obs: int | float) -> None:
    """Initialise parameters related to policy tree."""
    if optp_.pt_cfg.min_leaf_size is None or optp_.pt_cfg.min_leaf_size < 0:
        no_of_leaves = 2 ** optp_.pt_cfg.total_depth_adj
        min_leaf_size = 0.1 * no_of_obs / no_of_leaves

        if optp_.other_cfg.restricted:
            positive_shares = [share for share in optp_.other_cfg.max_shares if share > 0]
            min_leaf_size *= min(positive_shares)

        optp_.pt_cfg.min_leaf_size = max(round(min_leaf_size), 1)
    else:
        optp_.pt_cfg.min_leaf_size = max(round(optp_.pt_cfg.min_leaf_size), 1)


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
    def from_args(cls, *,
                  other_costs_of_treat=None,
                  other_costs_of_treat_mult=None,
                  other_max_shares=None,
                  ) -> 'OtherCfg':
        """Get inputs."""

        return cls(costs_of_treat=other_costs_of_treat,
                   costs_of_treat_mult=other_costs_of_treat_mult,
                   max_shares=other_max_shares,
                   )


def init_other_solve(optp_: 'OptimalPolicy', no_obs: int) -> None:
    """Initialise treatment costs (needs info on number of treatments."""
    no_of_treat = optp_.gen_cfg.no_of_treat
    other_cfg = optp_.other_cfg

    if other_cfg.max_shares is None:
        other_cfg.max_shares = [1] * no_of_treat
    elif len(other_cfg.max_shares) != no_of_treat:
        raise ValueError(f'other_max_shares must contain exactly {no_of_treat} values.')

    max_shares_np = np.asarray(other_cfg.max_shares, dtype=float)
    if (not np.all(np.isfinite(max_shares_np))
            or np.any(max_shares_np < 0)
            or np.any(max_shares_np > 1)
            ):
        raise ValueError('Maximum treatment shares must be finite and between 0 and 1.')
    if np.all(max_shares_np == 0):
        raise ValueError('All restrictions are zero. No allocation possible.')
    max_shares_sum = max_shares_np.sum()
    if (max_shares_sum < 1 and not np.isclose(max_shares_sum, 1, rtol=0, atol=1e-12)):
        raise ValueError('Sum of restrictions < 1. No allocation possible.')

    max_by_cat = np.floor(no_obs * max_shares_np).astype(np.int64)

    capacity_shortfall = no_obs - max_by_cat.sum()

    if capacity_shortfall == 1:
        next_feasible_shares = (max_by_cat + 1) / no_obs
        required_increase = next_feasible_shares - max_shares_np

        adjustable = np.flatnonzero(max_shares_np > 0)
        adjustment_idx = adjustable[np.argmin(required_increase[adjustable])]
        max_by_cat[adjustment_idx] += 1

    elif capacity_shortfall > 1:
        raise ValueError('Restrictions are too tight. They do not allow the allocation of all '
                         'observations. Relax them somewhat.')

    other_cfg.max_shares = max_shares_np.tolist()

    other_cfg.restricted = any(share < 1 for share in other_cfg.max_shares)

    if other_cfg.costs_of_treat is None:
        other_cfg.costs_of_treat = [0] * no_of_treat
    elif len(other_cfg.costs_of_treat) != no_of_treat:
        raise ValueError(f'other_costs_of_treat must contain exactly {no_of_treat} values.')
    other_cfg.costs_of_treat = [float(cost) for cost in other_cfg.costs_of_treat]
    if not np.all(np.isfinite(other_cfg.costs_of_treat)):
        raise ValueError('Treatment costs must be finite.')

    if other_cfg.costs_of_treat_mult is None:
        other_cfg.costs_of_treat_mult = [1] * no_of_treat
    elif len(other_cfg.costs_of_treat_mult) != no_of_treat:
        raise ValueError(f'other_costs_of_treat_mult must contain exactly {no_of_treat} values.')
    other_cfg.costs_of_treat_mult = [float(mult) for mult in other_cfg.costs_of_treat_mult]
    if not np.all(np.isfinite(other_cfg.costs_of_treat_mult)):
        raise ValueError('Treatment cost multipliers must be finite.')
    if any(mult <= 0 for mult in other_cfg.costs_of_treat_mult):
        raise ValueError('Cost multiplier must be positive.')

    other_cfg.max_by_treat = max_by_cat
    optp_.other_cfg = other_cfg


@dataclass(slots=True, kw_only=True)
class VarCfg:
    """Configure the variable names."""

    bb_restrict_name: list[str]
    d_name: list[str]
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
    def from_args(cls: type['VarCfg'], *,
                  bb_restrict_name: NameLike = None,
                  d_name: NameLike = None,
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
        return cls(bb_restrict_name=check_var(bb_restrict_name),
                   d_name=check_var(d_name),
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


@dataclass(slots=True, kw_only=True)
class VersionCfg:
    """Configure treatment-version inputs and parameters."""

    policyscores_dict: dict
    depth_version_tree: list[int]
    params_optpol: dict
    d_name: list[str] | None = None

    @classmethod
    def from_args(cls, *,
                  policyscores_dict: dict | None,
                  depth_version_tree: list[NumberLike] | tuple[NumberLike, ...] | NumberLike = None,
                  params_optpol: dict | None = None,
                  d_name: list[str] | tuple[str, ...] | None = None,
                  ) -> 'VersionCfg':
        """Read and transform inputs."""
        if not isinstance(policyscores_dict, dict):
            raise TypeError('policyscores_dict must be specified as a '
                            'dictionary with main treatments as keys and '
                            'corresponding treatment versions as values.'
                            )
        if not policyscores_dict:
            raise ValueError('policyscores_dict must not be empty.')

        for key, values in policyscores_dict.items():
            if not isinstance(key, str):
                raise TypeError('The keys of policyscores_dict must be strings.')

            if values is None or isinstance(values, str):
                continue

            if not isinstance(values, (list, tuple)):
                raise TypeError('The values of policyscores_dict must be None, '
                                'a string, or a tuple/list of strings.'
                                )
            if not values:
                raise ValueError('Treatment-version lists must not be empty.')

            if not all(isinstance(value, str) for value in values):
                raise TypeError('All elements of treatment-version lists must be strings.')

        no_main_treatments = len(policyscores_dict)

        if depth_version_tree is None:
            depth_version_tree_n = [2] * no_main_treatments

        elif isinstance(depth_version_tree, (int, float)):
            if not 2 <= depth_version_tree <= 10:
                raise ValueError('Tree depth must be between 2 and 10.')
            depth_version_tree_n = [round(depth_version_tree)] * no_main_treatments

        elif isinstance(depth_version_tree, (list, tuple)):
            if len(depth_version_tree) != no_main_treatments:
                raise ValueError('When depth_version_tree is a list or tuple, '
                                 'it must have the same length as the number '
                                 'of keys in policyscores_dict.'
                                 )
            if not all(isinstance(depth, (int, float)) for depth in depth_version_tree):
                raise TypeError('All elements of depth_version_tree must be numeric.')
            if any(not 2 <= depth <= 10 for depth in depth_version_tree):
                raise ValueError('All elements of depth_version_tree must be between 2 and 10.')
            depth_version_tree_n = [round(depth) for depth in depth_version_tree]

        else:
            raise TypeError('depth_version_tree must be numeric, a list or tuple of numbers, '
                            'or None.'
                            )
        if params_optpol is None:
            params_optpol_n = {'gen_method': 'best_policy_score',}
        elif not isinstance(params_optpol, dict):
            raise TypeError('params_optpol must be a dictionary or None.')
        else:
            params_optpol_n = deepcopy(params_optpol)
            params_optpol_n.setdefault('gen_method', 'best_policy_score')

        if d_name is None:
            d_name_n = None
        else:
            if (not isinstance(d_name, (list, tuple)) or len(d_name) != 2):
                raise ValueError('d_name must contain exactly two elements.')
            if not all(isinstance(name, str) for name in d_name):
                raise TypeError('Both elements of d_name must be strings.')
            d_name_n = [name.casefold() for name in d_name]

        return cls(policyscores_dict=policyscores_dict,
                   depth_version_tree=depth_version_tree_n,
                   params_optpol=params_optpol_n,
                   d_name=d_name_n,
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


def check_valid_user_str(gen_cfg: 'GenCfg',
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


def init_gen_solve(optp_: 'OptimalPolicy', data_df: DataFrame) -> None:
    """Add and update some dictionary entry based on data."""
    var_cfg, gen_cfg = optp_.var_cfg, optp_.gen_cfg
    no_of_treat = len(var_cfg.polscore_name)
    gen_cfg.d_values = tuple(np.int32(range(no_of_treat)))
    gen_cfg.no_of_treat = no_of_treat

    if var_cfg.d_name and var_cfg.d_name[0] in data_df.columns:
        d_dat = data_df[var_cfg.d_name[0]].to_numpy(copy=False)
        try:
            d_dat_float = np.asarray(d_dat, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Observed treatment values must be numeric.') from exc

        d_dat_rounded = np.rint(d_dat_float)
        if (not np.all(np.isfinite(d_dat_float))
                or not np.allclose(d_dat_float, d_dat_rounded, rtol=0, atol=1e-8)
                ):
            raise ValueError('Observed treatment values must be finite integers.')

        observed_values = np.unique(d_dat_rounded.astype(np.int64))
        invalid_values = np.setdiff1d(observed_values, gen_cfg.d_values)
        if invalid_values.size:
            raise ValueError('Observed treatment values are inconsistent with the policy scores. '
                             f'Unknown values: {invalid_values.tolist()}'
                             )

    optp_.gen_cfg = gen_cfg
