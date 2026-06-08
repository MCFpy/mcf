"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from typing import Any, Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from psutil import virtual_memory, cpu_count

try:
    import torch      # type: ignore[import]
except (ImportError, OSError):
    torch = None      # type: ignore[assignment]

from mcf import mcf_general as mcf_gp
from mcf.mcf_init_update_helper import isinstance_scalar, get_ray_del_defaults, name_unique
from mcf import mcf_init_values_cfg as mcf_initvals
from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_general_sys as mcf_sys

if TYPE_CHECKING:
    from mcf.mcf_init_predict import PCfg

type NumberLike = int | float | None
type IntLike = int | None
type FloatLike = float | None
type BoolLike = bool | None
type StrLike = str | None
type Scalar = float | int
type GridLike = list[Scalar] | tuple[Scalar, ...] | NDArray | None
type ArrayLike = NDArray[Any] | None
type ListofStrLike = list[str] | None
type NameLike = str | ListofStrLike | tuple[str, ...]


@dataclass(slots=True, kw_only=True)
class IntCfg:
    """Normalized internal parameters. Initial setup."""

    # output / plots
    descriptive_stats: bool = True
    dpi: int = 500
    fontsize: str = 'x-small'
    all_fonts: tuple[str, ...] = field(default=('xx-small', 'x-small', 'small', 'medium',
                                                'large', 'x-large', 'xx-large',
                                                ),
                                       init=False,
                                       )
    legend_loc: str = field(default='best', init=False)
    no_filled_plot: int = 20
    show_plots: bool = True
    with_output: bool = True
    verbose: bool = False
    memory_print: bool = False
    output_no_new_dir: bool = False
    report: bool = True

    # implementation / backend
    cuda: bool = False
    cython: bool = False
    del_forest: bool = False
    mp_ray_del: Any = None
    mp_ray_shutdown: BoolLike = None
    mp_ray_objstore_multiplier: NumberLike = 1
    no_ray_in_forest_building: bool = False
    mp_backend: Literal['ray', 'joblib', 'sequential'] | None = None
    mp_batches: IntLike | str = 'automatic'
    mp_memmap_min_bytes: int = 4 * 1024 * 1024
    mp_memmap_dir: Path | None = Path.cwd() / 'joblibtemp'
    mp_use_old_ray: bool = False

    low_memory_predict: BoolLike = True

    # memory / internal store, filled later
    mem_object_store_1: NumberLike = field(default=None, init=False)
    mem_object_store_2: NumberLike = field(default=None, init=False)
    mem_object_store_3: NumberLike = field(default=None, init=False)

    # weights / multiprocessing
    max_save_values: NumberLike = 50
    keep_w0: bool = False
    mp_weights_tree_batch: int = 0
    mp_weights_type: int = 1
    weight_as_sparse: bool = True
    weight_as_sparse_splits: IntLike = None
    mp_vim_type: IntLike = None

    # training / sampling
    iate_chunk_size: NumberLike = None
    share_forest_sample: float = 0.5
    seed_sample_split: int = 67567885
    max_cats_cont_vars: IntLike = None
    replication: bool = False

    # big data handling
    smaller_sample: None = None
    obs_bigdata: NumberLike = 1_000_000
    bigdata_train: bool = False
    max_obs_training: NumberLike = float('inf')
    max_obs_prediction: NumberLike = None
    max_obs_post_kmeans: NumberLike = 100_000
    max_obs_post_rel_graphs: NumberLike = 50_000

    # tolerances
    zero_tol: float = 1e-10
    sum_tol: float = 1e-8

    @classmethod
    def from_args(cls, *,
                  cuda: BoolLike = None,
                  cython: BoolLike = None,
                  del_forest: BoolLike = None,
                  descriptive_stats: BoolLike = None,
                  dpi: IntLike = None,
                  fontsize: NumberLike = None,
                  iate_chunk_size: NumberLike = None,
                  keep_w0: BoolLike = None,
                  low_memory_predict: BoolLike = True,
                  no_filled_plot: NumberLike = None,
                  max_cats_cont_vars: IntLike = None,
                  max_obs_kmeans: NumberLike = None,
                  max_obs_prediction: NumberLike = None,
                  max_obs_post_rel_graphs: NumberLike = None,
                  max_obs_training: NumberLike = None,
                  max_save_values: NumberLike = None,
                  memory_print: bool = False,
                  mp_use_old_ray: BoolLike = False,
                  mp_backend: str | None = None,
                  mp_batches: IntLike | str = 'automatic',
                  mp_memmap_min_bytes: NumberLike = None,
                  mp_memmap_dir: Path | None = None,
                  mp_ray_del: IntLike | tuple[str, ...] = None,
                  mp_ray_objstore_multiplier: NumberLike = None,
                  mp_ray_shutdown: BoolLike = None,
                  mp_vim_type: IntLike = None,
                  mp_weights_tree_batch: bool | NumberLike = None,
                  mp_weights_type: NumberLike = None,
                  obs_bigdata: NumberLike = None,
                  output_no_new_dir: BoolLike = None,
                  replication: BoolLike = None,
                  report: BoolLike = None,
                  seed_sample_split: NumberLike = None,
                  share_forest_sample: FloatLike = None,
                  show_plots: BoolLike = None,
                  weight_as_sparse: BoolLike = None,
                  weight_as_sparse_splits: NumberLike = None,
                  with_output: BoolLike = None,
                  verbose: BoolLike = None,
                  ) -> 'IntCfg':
        """Read, check, and normalize inputs."""
        # cuda
        cuda_b = False

        if cuda is True:
            if torch is None:
                warnings.warn('cuda=True was requested, but PyTorch could not be imported. '
                              'CUDA is disabled; continuing with int_cfg.cuda=False.',
                              UserWarning,
                              stacklevel=2,
                              )
            else:
                torch_cuda = getattr(torch, 'cuda', None)
                cuda_available = bool(torch_cuda and torch_cuda.is_available())

                if not cuda_available:
                    warnings.warn('cuda=True was requested, but torch.cuda.is_available() is '
                                  'False. CUDA is disabled; continuing with int_cfg.cuda=False.',
                                  UserWarning,
                                  stacklevel=2,
                                  )
                cuda_b = cuda_available

        # simple booleans
        cython_b = cython is not False
        if cython_b:
            raise ValueError('Cython currently not used.')
        del_forest_b = del_forest is True
        keep_w0_b = keep_w0 is True
        descriptive_stats_b = descriptive_stats is not False

        # output toggles
        with_output_b = with_output is not False
        verbose_b = verbose is True
        if not with_output_b:
            verbose_b = False

        show_plots_b = show_plots is not False
        report_b = report is not False
        output_no_new_dir_b = output_no_new_dir is True
        replication_b = replication is True

        # dpi
        dpi_i = 500 if (dpi is None or dpi < 10) else int(round(dpi))

        # fontsize
        if isinstance_scalar(fontsize) and 0.5 < fontsize < 7.5:
            idx = int(round(fontsize))
        else:
            idx = 2
        all_fonts = ('xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large',)
        fontsize_s = all_fonts[idx - 1] if 1 <= idx <= len(all_fonts) else 'x-small'

        # no filled plot
        no_filled_plot_i = (20 if (no_filled_plot is None or no_filled_plot < 5)
                            else int(round(no_filled_plot))
                            )
        # ray deletion defaults
        mp_ray_del_out = get_ray_del_defaults(mp_ray_del)

        # object store multiplier
        mp_ray_objstore_multiplier_f = (1 if (mp_ray_objstore_multiplier is None
                                              or mp_ray_objstore_multiplier < 0)
                                        else mp_ray_objstore_multiplier
                                        )
        # weights tree batch
        mtb = mp_weights_tree_batch
        if mtb is False:
            mtb = 1
        if isinstance_scalar(mtb) and mtb > 0.5:
            mp_weights_tree_batch_i = int(round(mtb))
        else:
            mp_weights_tree_batch_i = 0

        # share forest sample
        share_forest_sample_f = (0.5 if (share_forest_sample is None
                                         or not 0.01 < share_forest_sample < 0.99)
                                 else float(share_forest_sample)
                                 )
        # Avoid storing weight matrix - some parameters or methods may not be with this method
        low_memory_predict_b = low_memory_predict is not False

        # sparse weights
        weight_as_sparse_b = weight_as_sparse is not False

        weight_as_sparse_splits_i = (int(weight_as_sparse_splits)
                                     if isinstance(weight_as_sparse_splits, int)
                                     else None
                                     )
        if weight_as_sparse_splits_i is None and low_memory_predict_b:
            weight_as_sparse_splits_i = 1

        # max save values
        max_save_values_out =  50 if max_save_values is None else max_save_values

        # weights type
        mp_weights_type_i = 2 if mp_weights_type == 2 else 1

        # seed
        seed_sample_split_i = (67567885 if seed_sample_split is None
                               else int(round(seed_sample_split))
                               )
        # iate chunk size
        iate_chunk_size_out = (iate_chunk_size if (isinstance_scalar(iate_chunk_size)
                                                   and iate_chunk_size > 0
                                                   )
                               else None
                               )
        # sizes / caps
        obs_bigdata_out = (obs_bigdata if isinstance_scalar(obs_bigdata) and obs_bigdata > 10
                           else 1_000_000
                           )
        max_obs_training_out = (max_obs_training if (isinstance_scalar(max_obs_training)
                                                     and max_obs_training > 100
                                                     )
                                else float('inf')
                                )
        max_obs_prediction_out = (max_obs_prediction if (isinstance_scalar(max_obs_prediction)
                                                         and max_obs_prediction > 100
                                                         )
                                  else (1_000_000 if low_memory_predict_b else 250_000)
                                  )
        max_obs_post_kmeans_out = (max_obs_kmeans if (isinstance_scalar(max_obs_kmeans)
                                                      and max_obs_kmeans > 100
                                                      )
                                   else 200_000
                                   )
        max_obs_post_rel_graphs_out = (
            max_obs_post_rel_graphs if (isinstance_scalar(max_obs_post_rel_graphs)
                                        and max_obs_post_rel_graphs > 100
                                        )
            else 50_000
            )
        # old ray switch
        mp_use_old_ray_b = mp_use_old_ray is True

        # backend
        if mp_backend is None:
            mp_backend_s = None  # Default value depends on number of training observations
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

        return cls(cuda=cuda_b,
                   cython=cython_b,
                   descriptive_stats=descriptive_stats_b,
                   del_forest=del_forest_b,
                   dpi=dpi_i,
                   fontsize=fontsize_s,
                   iate_chunk_size=iate_chunk_size_out,
                   keep_w0=keep_w0_b,
                   low_memory_predict = low_memory_predict_b,
                   memory_print=memory_print is True,
                   max_cats_cont_vars=max_cats_cont_vars,
                   max_obs_prediction=max_obs_prediction_out,
                   max_obs_post_kmeans=max_obs_post_kmeans_out,
                   max_obs_post_rel_graphs=max_obs_post_rel_graphs_out,
                   max_obs_training=max_obs_training_out,
                   mp_backend=mp_backend_s,
                   mp_batches=mp_batches_s,
                   mp_memmap_min_bytes=mp_memmap_min_bytes_b,
                   mp_memmap_dir=mp_memmap_dir_b,
                   mp_ray_del=mp_ray_del_out,
                   mp_ray_objstore_multiplier=mp_ray_objstore_multiplier_f,
                   mp_ray_shutdown=mp_ray_shutdown,
                   max_save_values=max_save_values_out,
                   mp_use_old_ray=mp_use_old_ray_b,
                   mp_vim_type=mp_vim_type,
                   mp_weights_tree_batch=mp_weights_tree_batch_i,
                   mp_weights_type=mp_weights_type_i,
                   no_filled_plot=no_filled_plot_i,
                   obs_bigdata=obs_bigdata_out,
                   output_no_new_dir=output_no_new_dir_b,
                   replication=replication_b,
                   report=report_b,
                   show_plots=show_plots_b,
                   seed_sample_split=seed_sample_split_i,
                   share_forest_sample=share_forest_sample_f,
                   verbose=verbose_b,
                   weight_as_sparse=weight_as_sparse_b,
                   weight_as_sparse_splits=weight_as_sparse_splits_i,
                   with_output=with_output_b,
                   )


@dataclass(slots=True, kw_only=True)
class GenCfg:
    """Define the general parameters."""

    # core
    d_type: Literal['discrete', 'continuous'] = 'discrete'
    iv: bool = False  # set later by train_iv

    # effects
    ate_eff: bool = False
    gate_eff: bool = False
    iate_eff: bool = False
    qiate_eff: bool = False
    any_eff: bool = False

    # parallelism
    mp_parallel: int = 1
    mp_automatic: bool = True
    sys_share: float = 0.0

    # output & verbosity
    with_output: bool = True
    verbose: bool = True
    return_iate_sp: bool = False
    outpath: Path | None = None
    outfiletext: Path | None = None
    outfilesummary: Path | None = None

    # printing
    output_type: int = 2  # 0: terminal, 1: file, else both
    print_to_file: bool = True
    print_to_terminal: bool = True

    # weighting / panel
    weighted: bool = False
    panel_data: bool = False
    panel_in_rf: bool = False

    # to be initialized at the beginning of training
    agg_yes: bool = field(default=False, init=False)
    d_values: tuple[np.integer, ...] | None = field(default=None, init=False)
    no_of_treat: int = field(default=2, init=False)
    x_type_0: bool = field(default=False, init=False)
    x_type_1: bool = field(default=False, init=False)
    x_type_2: bool = field(default=False, init=False)

    @classmethod
    def from_args(cls,
                  int_cfg: IntCfg, *,
                  ate_eff: BoolLike = None,
                  d_type: StrLike = None,
                  gate_eff: BoolLike = None,
                  iate_eff: BoolLike = None,
                  mp_parallel: NumberLike = None,
                  outfiletext: StrLike = None,
                  outpath: Path | None = None,
                  output_type: IntLike = None,
                  p_ate_no_se_only: BoolLike = None,
                  return_iate_sp: BoolLike = None,
                  panel_data: BoolLike = None,
                  panel_in_rf: BoolLike = None,
                  qiate_eff: BoolLike = None,
                  verbose: BoolLike = None,
                  weighted: BoolLike = None,
                  with_output: BoolLike = None,
                  ) -> 'GenCfg':
        """Read in a flexible way."""
        # d_type
        match d_type:
            case None | 'discrete':
                d_type_final: Literal['discrete', 'continuous'] = 'discrete'
            case 'continuous':
                d_type_final = 'continuous'
            case _:
                raise ValueError(f'{d_type} is wrong treatment type.')

        # effects (explicit True only)
        ate_eff_b = ate_eff is True
        gate_eff_b = gate_eff is True
        iate_eff_b = iate_eff is True
        qiate_eff_b = qiate_eff is True
        any_eff_b = ate_eff_b or gate_eff_b or iate_eff_b or qiate_eff_b
        if any_eff_b:
            ate_eff_b = True  # ensure ATE weights available

        # parallelism
        match mp_parallel:
            case x if x is None or not isinstance_scalar(x):
                mp_parallel_i = round(cpu_count(logical=True) * 0.8)
                mp_auto = True
            case (int() | float()) as x if x <= 1.5:
                mp_parallel_i, mp_auto = 1, False
            case (int() | float()) as x:
                mp_parallel_i, mp_auto = int(round(x)), False

        # system share
        sys_share_f = 0.7 * virtual_memory().percent / 100.0

        # output & verbosity
        with_output_b = with_output is not False
        verbose_b = verbose is not False
        return_iate_sp_b = return_iate_sp is True
        if p_ate_no_se_only:
            return_iate_sp_b = False
        if not with_output_b:
            verbose_b = False
        if with_output_b:
            return_iate_sp_b = True

        # paths & files
        if with_output_b:
            outpath_final = mcf_sys.define_outpath(outpath, not int_cfg.output_no_new_dir)
            base = 'txtFileWithOutput' if outfiletext is None else outfiletext
            outfiletext_path = outpath_final / f'{base}.txt'
            outfilesummary_path = outfiletext_path.with_name(
                f'{outfiletext_path.stem}_Summary.txt'
                )
            mcf_sys.delete_file_if_exists(outfiletext_path)
            mcf_sys.delete_file_if_exists(outfilesummary_path)
        else:
            outpath_final = outfiletext_path = outfilesummary_path = None

        # printing
        output_type_i = 2 if output_type is None else int(output_type)
        match output_type_i:
            case 0:
                print_to_file, print_to_terminal = False, True
            case 1:
                print_to_file, print_to_terminal = True, False
            case _:
                print_to_file = print_to_terminal = True
        if not with_output_b:
            print_to_file = print_to_terminal = False

        # weighting / panel
        weighted_b = weighted is True
        if panel_data is True:
            panel_data_b = True
            panel_in_rf_b = panel_in_rf is not False
        else:
            panel_data_b = panel_in_rf_b = False

        return cls(d_type=d_type_final,
                   iv=False,
                   ate_eff=ate_eff_b,
                   gate_eff=gate_eff_b,
                   iate_eff=iate_eff_b,
                   qiate_eff=qiate_eff_b,
                   any_eff=any_eff_b,
                   mp_parallel=mp_parallel_i,
                   mp_automatic=mp_auto,
                   sys_share=sys_share_f,
                   with_output=with_output_b,
                   verbose=verbose_b,
                   return_iate_sp=return_iate_sp_b,
                   outpath=outpath_final,
                   outfiletext=outfiletext_path,
                   outfilesummary=outfilesummary_path,
                   output_type=output_type_i,
                   print_to_file=print_to_file,
                   print_to_terminal=print_to_terminal,
                   weighted=weighted_b,
                   panel_data=panel_data_b,
                   panel_in_rf=panel_in_rf_b,
                   )


@dataclass(slots=True, kw_only=True)
class VarCfg:
    """Get the variable names and put them into lists."""

    # external state
    gen_cfg: GenCfg
    p: Any
    fs_yes: bool
    p_ba_yes: bool
    p_ba_use_x: bool

    # canonical inputs (lists; normalized by from_arg)
    cluster_name: list[str] = field(default_factory=list)
    d_name: list[str] = field(default_factory=list)
    id_name: list[str] = field(default_factory=list)
    iv_name: list[str] = field(default_factory=list)
    w_name: list[str] = field(default_factory=list)

    x_name_balance_test_ord: list[str] = field(default_factory=list)
    x_name_balance_test_unord: list[str] = field(default_factory=list)
    x_name_always_in_ord: list[str] = field(default_factory=list)
    x_name_always_in_unord: list[str] = field(default_factory=list)
    x_name_balance_bgate: list[str] = field(default_factory=list)
    x_name_ba: list[str] = field(default_factory=list)
    x_name_tv: list[str] = field(default_factory=list)
    x_name_remain_ord: list[str] = field(default_factory=list)
    x_name_remain_unord: list[str] = field(default_factory=list)
    x_name_ord: list[str] = field(default_factory=list)
    x_name_unord: list[str] = field(default_factory=list)

    y_name: list[str] = field(default_factory=list)
    y_tree_name: list[str] = field(default_factory=list)

    z_name_cont: list[str] = field(default_factory=list)
    z_name_ord: list[str] = field(default_factory=list)
    z_name_unord: list[str] = field(default_factory=list)

    # derived inputs
    name_unordered: list[str] = field(init=False)
    name_ordered: list[str] = field(init=False)
    names_to_check_train: list[str] = field(init=False)
    names_to_check_pred: list[str] = field(init=False)
    x_name_balance_test: list[str] = field(init=False)
    x_name_always_in: list[str] = field(init=False)
    x_name_remain: list[str] = field(init=False)
    x_name: list[str] = field(init=False)
    x_name_in_tree: list[str] = field(init=False)
    z_name: list[str] = field(init=False)

    # May be initialized later by the programme
    y_name_lc: list[str] = field(default_factory=list, init=False)
    y_name_ey_x: list[str] = field(default_factory=list, init=False)
    y_tree_name_unc: list[str] = field(default_factory=list, init=False)
    y_match_name: list[str] = field(default_factory=list, init=False)

    grid_nn_name: list[str] = field(default_factory=list, init=False)
    grid_w_name: list[str] = field(default_factory=list, init=False)
    grid_dr_name: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Perform some simple transformation with already cleaned variables."""
        # required fields
        if not self.y_name:
            raise ValueError('y_name must be specified.')
        if not self.d_name:
            raise ValueError('d_name must be specified.')

        # at least one x source
        if not (self.x_name_ord or self.x_name_unord):
            raise ValueError('x_name_ord or x_name_unord must be specified.')

        # cluster names required if cluster std or panel data
        if self.p.cluster_std or self.gen_cfg.panel_data:
            if not self.cluster_name:
                raise ValueError('cluster_name must be specified.')

        # weights
        if self.gen_cfg.weighted:
            if not self.w_name:
                raise ValueError('No name for sample weights specified.')
        else:
            self.w_name = []

        # compose x-sets
        self.x_name = name_unique(mcf_gp.cleaned_var_names(
                deepcopy(self.x_name_ord + self.x_name_unord))
            )
        self.x_name_in_tree = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_always_in_ord + self.x_name_always_in_unord)
            )
        self.x_name_balance_test = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_balance_test_ord + self.x_name_balance_test_unord)
            )
        if not self.x_name_balance_test:
            self.p.bt_yes = False

        self.x_name_remain = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_remain_ord + self.x_name_remain_unord +
                     self.x_name_in_tree + self.x_name_balance_test)
            )
        self.x_name_always_in = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_always_in_ord + self.x_name_always_in_unord)
            )
        self.name_ordered = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_ord + self.x_name_always_in_ord + self.x_name_remain_ord)
            )
        self.name_unordered = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_unord + self.x_name_always_in_unord + self.x_name_remain_unord)
            )
        # feature selection interaction
        if self.fs_yes and self.p.bt_yes:
            self.x_name_remain = mcf_gp.cleaned_var_names(
                deepcopy(self.x_name_balance_test + self.x_name_remain)
                )
        # ensure in-tree vars are included
        if self.x_name_in_tree:
            self.x_name_remain = mcf_gp.cleaned_var_names(
                deepcopy(self.x_name_in_tree + self.x_name_remain)
                )
            self.x_name = mcf_gp.cleaned_var_names(deepcopy(self.x_name_in_tree + self.x_name))
        # ordered vs unordered overlap
        if self.name_ordered and self.name_unordered:
            if any(v for v in self.name_ordered if v in self.name_unordered):
                raise ValueError('Remove overlap in ordered + unordered variables')
        # names to check
        self.names_to_check_train = self.d_name + self.y_name + self.x_name
        self.names_to_check_pred = self.x_name[:]

        # z and gate flags
        if (not self.z_name_cont) and (not self.z_name_ord) and (not self.z_name_unord):
            self.gen_cfg.agg_yes = self.p.gate = False
            self.z_name = []
        else:
            self.gen_cfg.agg_yes = self.p.gate = True
            if self.z_name_cont:
                self.names_to_check_train.extend(self.z_name_cont)
                self.names_to_check_pred.extend(self.z_name_cont)
            if self.z_name_ord:
                self.names_to_check_train.extend(self.z_name_ord)
                self.names_to_check_pred.extend(self.z_name_ord)
            if self.z_name_unord:
                self.names_to_check_train.extend(self.z_name_unord)
                self.names_to_check_pred.extend(self.z_name_unord)
            self.z_name = self.z_name_cont + self.z_name_ord + self.z_name_unord
        # gate-family toggles and message
        txt = ''
        if self.p.bgate and not self.p.gate:
            txt += '\nBGATEs can only be computed if GATEs are computed.'
            self.p.bgate = False
        if self.p.cbgate and not self.p.gate:
            txt += '\nCBGATEs can only be computed if GATEs are computed.'
            self.p.cbgate = False
        if self.p.gatet and not self.p.gate:
            txt += '\nGATETs can only be computed if GATEs are computed.'
            self.p.gatet = False
        if txt:
            mcf_ps.print_mcf(self.gen_cfg, txt, summary=True)

        if self.p.bgate and self.x_name_balance_bgate == self.z_name and len(self.z_name) == 1:
            self.p.bgate = False

        if self.p.bgate:
            if not self.x_name_balance_bgate:
                if len(self.z_name) > 1:
                    self.x_name_balance_bgate = self.z_name[:]
                else:
                    self.p.bgate = False
                    self.x_name_balance_bgate = []
            else:
                self.names_to_check_train.extend(self.x_name_balance_bgate)
                self.names_to_check_pred.extend(self.x_name_balance_bgate)
        else:
            self.x_name_balance_bgate = []

        if not self.x_name_balance_bgate and len(self.z_name) == 1:
            self.p.bgate = False

        if self.p_ba_yes and self.p_ba_use_x:
            if self.x_name_ba:
                self.x_name_ba = mcf_gp.cleaned_var_names(
                    deepcopy(self.x_name_ba))
                if not set(self.x_name_ba) <= set(self.x_name):
                    missing_vars = list(set(self.x_name_ba) - set(self.x_name))
                    raise ValueError('Variables specified for bias adjustment '
                                     'must be contained in variables used '
                                     'to build causal forest. Problematic '
                                     f'variable(s): {" ".join(missing_vars)}'
                                     )
            else:
                raise ValueError('No individual feature (X) specificed for '
                                 'bias correction although keyword p_ba_x '
                                 'indicates that such variables should be '
                                 'included.'
                                 )
        if len(self.d_name) == 2 and self.x_name_tv:  # Treatment versions & X
            self.x_name_tv = mcf_gp.cleaned_var_names(deepcopy(self.x_name_tv))
            if not set(self.x_name_tv) <= set(self.x_name):
                missing_vars = list(set(self.x_name_tv) - set(self.x_name))
                raise ValueError('Variables specified for regression step of treatment '
                                 'versions adjustment must be contained in variables used '
                                 'to build causal forest. Problematic '
                                 f'variable(s): {" ".join(missing_vars)}'
                                 )

        # final uniqueness for checks
        self.names_to_check_train = name_unique(self.names_to_check_train[:])
        self.names_to_check_pred = name_unique(self.names_to_check_pred[:])
    # ---------- keyword-only factory ----------

    @classmethod
    def from_args(cls, *,
                  gen_cfg: 'GenCfg',
                  p_cfg: 'PCfg',
                  fs_yes: bool,
                  p_ba_yes: bool,
                  p_ba_use_x: bool,
                  cluster_name: NameLike = None,
                  d_name: NameLike = None,
                  id_name: NameLike = None,
                  iv_name: NameLike = None,
                  w_name: NameLike = None,
                  x_name_balance_test_ord: NameLike = None,
                  x_name_balance_test_unord: NameLike = None,
                  x_name_always_in_ord: NameLike = None,
                  x_name_always_in_unord: NameLike = None,
                  x_name_balance_bgate: NameLike = None,
                  x_name_ba: NameLike = None,
                  x_name_remain_ord: NameLike = None,
                  x_name_remain_unord: NameLike = None,
                  x_name_ord: NameLike = None,
                  x_name_tv: NameLike = None,
                  x_name_unord: NameLike = None,
                  y_name: NameLike = None,
                  y_tree_name: NameLike = None,
                  z_name_cont: NameLike = None,
                  z_name_ord: NameLike = None,
                  z_name_unord: NameLike = None,
                  ) -> 'VarCfg':
        """Get input from keywords and transform them."""
        # short-cut names for functions
        norm = mcf_gp.cleaned_var_names
        to_list = mcf_gp.to_list_if_needed

        # y_tree defaulting based on y_name (mirrors original)
        yt_raw = y_tree_name
        if not yt_raw or yt_raw == []:
            yt_raw = [y_name[0]] if isinstance(y_name, (list, tuple)) else [y_name]
        y_tree = norm(to_list(yt_raw))

        y_all = to_list(y_name)
        y_all.extend(y_tree)
        y_name_n = norm(y_all)

        # cluster names only if needed
        cluster_name_n = norm(cluster_name) if p_cfg.cluster_std or gen_cfg.panel_data else None

        # ate_no_se_only zeroes certain inputs
        if p_cfg.ate_no_se_only is True:
            x_name_balance_test_ord = None
            x_name_balance_test_unord = None
            z_name_cont = z_name_ord = z_name_unord = None
            x_name_balance_bgate = None

        # core normals
        d_name_n = norm(to_list(d_name))

        # id_name shaping (ensure first entry is str)
        id_raw = [id_name] if not isinstance(id_name, (list, tuple)) else list(id_name)
        if not id_raw or not isinstance(id_raw[0], str):
            id_raw = ['ID'] if not id_raw else 'ID'
        id_name_n = norm(id_raw)

        # iv list shaping
        iv_raw = [iv_name] if not isinstance(iv_name, (list, tuple)) else list(iv_name)
        iv_name_n = norm(iv_raw)

        # z parts
        z_name_cont_n = norm(z_name_cont)
        z_name_ord_n = norm(z_name_ord)
        z_name_unord_n = norm(z_name_unord)

        # x parts (+ augment with z to mirror original behavior)
        x_name_ord_n = norm(x_name_ord)
        x_name_unord_n = norm(x_name_unord)
        if z_name_cont_n or z_name_ord_n:
            x_name_ord_n = x_name_ord_n + z_name_cont_n + z_name_ord_n
            x_name_unord_n = x_name_unord_n + z_name_unord_n

        # always/remain/balance sets
        x_ai_ord_n = norm(x_name_always_in_ord)
        x_ai_unord_n = norm(x_name_always_in_unord)
        x_rem_ord_n = norm(x_name_remain_ord)
        x_rem_unord_n = norm(x_name_remain_unord)
        x_bt_ord_n = norm(x_name_balance_test_ord)
        x_bt_unord_n = norm(x_name_balance_test_unord)

        # bgate balance set only if bgate on
        x_bgate_n = norm(x_name_balance_bgate) if p_cfg.bgate else []

        x_name_ba_n = norm(to_list(x_name_ba)) if p_ba_yes and p_ba_use_x else []

        x_name_tv_n = norm(to_list(x_name_tv)) if x_name_tv else []

        # weights (clean; presence validated in __post_init__)
        w_name_n = norm(w_name)

        return cls(gen_cfg=gen_cfg,
                   p=p_cfg,
                   fs_yes=fs_yes,
                   p_ba_yes=p_ba_yes,
                   p_ba_use_x=p_ba_use_x,
                   cluster_name=cluster_name_n,
                   d_name=d_name_n,
                   id_name=id_name_n,
                   iv_name=iv_name_n,
                   w_name=w_name_n,
                   x_name_balance_test_ord=x_bt_ord_n,
                   x_name_balance_test_unord=x_bt_unord_n,
                   x_name_always_in_ord=x_ai_ord_n,
                   x_name_always_in_unord=x_ai_unord_n,
                   x_name_balance_bgate=x_bgate_n,
                   x_name_ba=x_name_ba_n,
                   x_name_remain_ord=x_rem_ord_n,
                   x_name_remain_unord=x_rem_unord_n,
                   x_name_ord=x_name_ord_n,
                   x_name_tv=x_name_tv_n,
                   x_name_unord=x_name_unord_n,
                   y_name=y_name_n,
                   y_tree_name=y_tree,
                   z_name_cont=z_name_cont_n,
                   z_name_ord=z_name_ord_n,
                   z_name_unord=z_name_unord_n,
                   )


@dataclass(slots=True, kw_only=True)
class DCCfg:
    """Parameters for data cleaning (normalized)."""

    screen_covariates: bool
    check_perfectcorr: bool
    clean_data: bool
    min_dummy_obs: int

    @classmethod
    def from_args(cls, *,
                  screen_covariates: BoolLike = None,
                  check_perfectcorr: BoolLike = None,
                  clean_data: BoolLike = None,
                  min_dummy_obs: NumberLike = None,
                  ) -> 'DCCfg':
        """Get input and normalize parameters."""
        return cls(screen_covariates=screen_covariates is not False,
                   check_perfectcorr=check_perfectcorr is not False,
                   clean_data=clean_data is not False,
                   min_dummy_obs=(10 if (min_dummy_obs is None or float(min_dummy_obs) < 1)
                                  else int(round(float(min_dummy_obs)))
                                  ),
                   )


@dataclass(slots=True, kw_only=True)
class CtGrid:
    """Just some values for continous treatment."""

    grid_dr: GridLike | int = None
    grid_nn: GridLike | int = None
    grid_w:  GridLike | int = None

    # To be filled later
    grid_nn_val: GridLike = field(default=None, init=False)
    grid_w_val: GridLike = field(default=None, init=False)
    grid_dr_val: GridLike = field(default=None, init=False)
    no_of_treat: IntLike = field(default=None, init=False)
    d_values: GridLike = field(default=None, init=False)
    w_to_dr_int_w01: ArrayLike = field(default=None, init=False)
    w_to_dr_int_w10: ArrayLike = field(default=None, init=False)
    w_to_dr_index_full: ArrayLike = field(default=None, init=False)
    d_values_dr_list: list[Scalar] | None = field(default=None, init=False)
    d_values_dr_np: ArrayLike = field(default=None, init=False)
    grid_nn_name: ListofStrLike = field(default=None, init=False)
    grid_w_name: ListofStrLike = field(default=None, init=False)
    grid_dr_name: ListofStrLike = field(default=None, init=False)

    @classmethod
    def from_args(cls, *,
                  grid_dr: GridLike = None,
                  grid_nn: GridLike = None,
                  grid_w:  GridLike = None,
                  ) -> 'CtGrid':
        """Just save values."""
        return cls(grid_dr=grid_dr, grid_nn=grid_nn, grid_w=grid_w,)


@dataclass(slots=True, kw_only=True)
class GenTvCfg:
    """Parameters for dealing with versions of treatment."""

    yes: bool = False
    estimator: StrLike = 'ridge'
    cv_k: IntLike = None
    tv_min_subtreat: IntLike = 10
    specification: str = 'interacted'
    penalize_version: bool = False

    # To be filled later
    d_main_values: GridLike = field(default=None, init=False)
    d_sub_values: GridLike = field(default=None, init=False)
    d_dict: dict | None = field(default=None, init=False)
    no_of_treat_all: IntLike = field(default=None, init=False)
    d_all_values: ArrayLike = field(default=None, init=False)
    d_all_values_one_label: list[int] | None = field(default=None, init=False)
    no_of_treat_per_main: list[int] | None = field(default=None, init=False)

    @classmethod
    def from_args(cls, *,
                  d_name: GridLike = None,
                  y_name: GridLike = None,
                  estimator: StrLike = 'ridge',
                  penalize_version: bool | list[bool] | tuple[bool, ...] = False,
                  cv_k: IntLike = None,
                  tv_min_subtreat: IntLike = 10,
                  specification: str = 'interacted',
                  clustering: bool = False,
                  weighted: bool = False,
                  continuous: bool = False,
                  qiate: bool = False,
                  p_ba: bool = False,
                  ) -> 'GenTvCfg':
        """Check, modify and save input values."""
        yes_n = len(d_name) > 1     # Second treatment variable required

        if yes_n:
            if clustering or weighted or continuous or qiate or p_ba:
                raise ValueError('Clustering, weighting, QIATE estimation, bias adjustment and '
                                 'continuous treatments are not allowed together with treatment '
                                 'versions.'
                                 )
            if isinstance(y_name, (list, tuple)) and len(y_name) > 1:
                raise NotImplementedError('Version estimation is currently implemented for '
                                          'single outcomes only.'
                                          )
            estimator_n = mcf_initvals.gen_tv_valid_estimators(estimator)
            specification_n = mcf_initvals.gen_tv_valid_specification(specification)
            cv_k_n = mcf_initvals.gen_tv_valid_cv_k(cv_k)
            match tv_min_subtreat:
                case None:
                    tv_min_subtreat_n = 10
                case k if isinstance(k, int) and k > 0:
                    tv_min_subtreat_n = k
                case _:
                    raise ValueError('tv_min_subtreat must be a positive Integer or None.')

            if penalize_version is not None:
                if not isinstance(penalize_version, (bool, list, tuple)):
                    raise ValueError('penalize_version must be Boolean, list, or tuple (or None).')
                if isinstance(penalize_version, (list, tuple)):
                    bool_ok = all(isinstance(pen, bool) for pen in penalize_version)
                    if not bool_ok:
                        raise ValueError('Elements of penalize_version must be Boolean' )
            penalize_version_n = False if penalize_version is None else penalize_version
            if estimator_n == 'ols':
                penalize_version_n = False
            else:
                if isinstance(penalize_version_n, bool):
                    penalize_version_n = [penalize_version_n]
                if isinstance(penalize_version_n, tuple):
                    penalize_version_n = list(penalize_version_n)
        else:
            tv_min_subtreat_n = penalize_version_n = estimator_n = specification_n = cv_k_n = None

        return cls(yes=yes_n,
                   estimator=estimator_n,
                   cv_k=cv_k_n,
                   penalize_version=penalize_version_n,
                   tv_min_subtreat=tv_min_subtreat_n,
                   specification=specification_n,
                   )
