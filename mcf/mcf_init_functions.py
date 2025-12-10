"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from psutil import virtual_memory, cpu_count

from mcf import mcf_general as mcf_gp
from mcf.mcf_init_update_helper_functions import (isinstance_scalar,
                                                  name_unique,
                                                  get_ray_del_defaults
                                                  )
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_general_sys as mcf_sys

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
    """Normalized internal, immutable parameters. Initial setup.

    It is currently written such as to work with frozen dataclass. Not needed.

    All inputs are InitVars (not stored).
    The public attributes are the normalized values used by the rest of the
    code.
    """

    # ---- public, normalized fields (what you used to return in dic) ----
    descriptive_stats: bool = True

    dpi: int = 500
    fontsize: str = 'x-small'
    all_fonts: tuple[str, ...] = (
        'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        )
    legend_loc: str = 'best'
    no_filled_plot: int = 20
    show_plots: bool = True
    with_output: bool = True
    verbose: bool = False  # will be forced False if with_output is False
    output_no_new_dir: bool = False
    report: bool = True

    cuda: bool = False
    cython: bool = True
    del_forest: bool = False
    mp_ray_del: Any = None
    mp_ray_shutdown: BoolLike = None
    mp_ray_objstore_multiplier: NumberLike = 1
    no_ray_in_forest_building: bool = False
    mem_object_store_1: NumberLike = None  # To be filled later
    mem_object_store_2: NumberLike = None  # To be filled later
    mem_object_store_3: NumberLike = None  # To be filled later

    max_save_values: NumberLike = 50
    keep_w0: bool = False
    mp_weights_tree_batch: int = 0  # 0 means 'off' per original logic
    mp_weights_type: int = 1
    weight_as_sparse: bool = True
    weight_as_sparse_splits: IntLike = None
    mp_vim_type: IntLike = None

    iate_chunk_size: NumberLike = None
    share_forest_sample: float = 0.5
    seed_sample_split: int = 67567885
    max_cats_cont_vars: IntLike = None
    replication: bool = False

    smaller_sample: None = None  # placeholder kept for compatibility
    obs_bigdata: NumberLike = 1_000_000
    bigdata_train: bool = False
    max_obs_training: NumberLike = float('inf')
    max_obs_prediction: NumberLike = 250_000
    max_obs_post_kmeans: NumberLike = 200_000
    max_obs_post_rel_graphs: NumberLike = 50_000

    zero_tol: float = 1e-15    # for checking if single variables are about 0
    sum_tol: float = 1e-12     # for sum about 1/0 checks

    # ---- InitVars: mirror the original function signature (inputs only) ----
    cuda_in: InitVar[BoolLike] = None
    cython_in: InitVar[BoolLike] = None
    del_forest_in: InitVar[BoolLike] = None
    descriptive_stats_in: InitVar[BoolLike] = None
    dpi_in: InitVar[IntLike] = None
    fontsize_in: InitVar[NumberLike] = None
    iate_chunk_size_in: InitVar[NumberLike] = None
    keep_w0_in: InitVar[BoolLike] = None
    no_filled_plot_in: InitVar[NumberLike] = None
    max_cats_cont_vars_in: InitVar[IntLike] = None
    max_obs_kmeans_in: InitVar[NumberLike] = None
    max_obs_prediction_in: InitVar[NumberLike] = None
    max_obs_post_rel_graphs_in: InitVar[NumberLike] = None
    max_obs_training_in: InitVar[NumberLike] = None
    max_save_values_in: InitVar[NumberLike] = None
    mp_ray_del_in: InitVar[IntLike | tuple[str, ...]] = None
    mp_ray_objstore_multiplier_in: InitVar[NumberLike] = None
    mp_ray_shutdown_in: InitVar[BoolLike] = None
    mp_vim_type_in: InitVar[IntLike] = None
    mp_weights_tree_batch_in: InitVar[bool | NumberLike] = None
    mp_weights_type_in: InitVar[NumberLike] = None
    obs_bigdata_in: InitVar[NumberLike] = None
    output_no_new_dir_in: InitVar[BoolLike] = None
    replication_in: InitVar[BoolLike] = None
    report_in: InitVar[BoolLike] = None
    seed_sample_split_in: InitVar[NumberLike] = None
    share_forest_sample_in: InitVar[FloatLike] = None
    show_plots_in: InitVar[BoolLike] = None
    weight_as_sparse_in: InitVar[BoolLike] = None
    weight_as_sparse_splits_in: InitVar[NumberLike] = None

    def __post_init__(
        self,
        cuda_in, cython_in, del_forest_in, descriptive_stats_in, dpi_in,
        fontsize_in, iate_chunk_size_in, keep_w0_in, no_filled_plot_in,
        max_cats_cont_vars_in, max_obs_kmeans_in, max_obs_prediction_in,
        max_obs_post_rel_graphs_in, max_obs_training_in, max_save_values_in,
        mp_ray_del_in, mp_ray_objstore_multiplier_in, mp_ray_shutdown_in,
        mp_vim_type_in, mp_weights_tree_batch_in, mp_weights_type_in,
        obs_bigdata_in, output_no_new_dir_in,
        replication_in, report_in, seed_sample_split_in,
        share_forest_sample_in, show_plots_in,
        weight_as_sparse_in, weight_as_sparse_splits_in,
            ):
        """Check and transform inputs."""
        set_ = object.__setattr__

        # cuda
        if cuda_in is True:
            try:
                import torch  # type: ignore[import]
            except (ImportError, OSError):
                ok = False
            else:
                ok = bool(getattr(torch, 'cuda', None)
                          and torch.cuda.is_available())
                set_(self, 'cuda', ok)
        else:
            set_(self, 'cuda', False)

        # cython flag
        set_(self, 'cython', cython_in is not False)

        # simple booleans
        set_(self, 'del_forest', del_forest_in is True)
        set_(self, 'keep_w0', keep_w0_in is True)
        set_(self, 'descriptive_stats', descriptive_stats_in is not False)

        # dpi
        dpi_val = 500 if (dpi_in is None or dpi_in < 10) else round(dpi_in)
        set_(self, 'dpi', int(dpi_val))

        # fontsize → label
        if isinstance_scalar(fontsize_in) and 0.5 < fontsize_in < 7.5:
            idx = int(round(fontsize_in))
        else:
            idx = 2
        fonts = self.all_fonts
        label = fonts[idx - 1] if 1 <= idx <= len(fonts) else 'x-small'
        set_(self, 'fontsize', label)
        set_(self, 'legend_loc', 'best')

        # no_filled_plot
        nfp = (20 if (no_filled_plot_in is None or no_filled_plot_in < 5)
               else int(round(no_filled_plot_in))
               )
        set_(self, 'no_filled_plot', nfp)

        mp_ray_del_out = get_ray_del_defaults(mp_ray_del_in)
        set_(self, 'mp_ray_del', mp_ray_del_out)

        # object store multiplier
        mul = (1 if (mp_ray_objstore_multiplier_in is None
                     or mp_ray_objstore_multiplier_in < 0)
               else mp_ray_objstore_multiplier_in)
        set_(self, 'mp_ray_objstore_multiplier', mul)

        # weights tree batch
        mtb = mp_weights_tree_batch_in
        if mtb is False:
            mtb = 1
        if isinstance_scalar(mtb) and mtb > 0.5:
            set_(self, 'mp_weights_tree_batch', int(round(mtb)))
        else:
            set_(self, 'mp_weights_tree_batch', 0)

        # share_forest_sample
        sfs = (0.5 if (share_forest_sample_in is None
                       or not 0.01 < share_forest_sample_in < 0.99)
               else float(share_forest_sample_in)
               )
        set_(self, 'share_forest_sample', sfs)

        # weight_as_sparse
        set_(self, 'weight_as_sparse', weight_as_sparse_in is not False)

        # output toggles
        set_(self, 'show_plots', show_plots_in is not False)
        set_(self, 'report', report_in is not False)

        # max_save_values (kept as NumberLike for parity)
        msv = 50 if max_save_values_in is None else max_save_values_in
        set_(self, 'max_save_values', msv)

        # mp_weights_type
        set_(self, 'mp_weights_type', 2 if (mp_weights_type_in == 2) else 1)

        # seed
        seed = (67567885 if seed_sample_split_in is None
                else int(round(seed_sample_split_in))
                )
        set_(self, 'seed_sample_split', seed)

        # weight_as_sparse_splits
        was = (int(weight_as_sparse_splits_in)
               if isinstance(weight_as_sparse_splits_in, int) else None
               )
        set_(self, 'weight_as_sparse_splits', was)

        # passthroughs
        set_(self, 'mp_ray_shutdown', mp_ray_shutdown_in)
        set_(self, 'mp_vim_type', mp_vim_type_in)
        set_(self, 'max_cats_cont_vars', max_cats_cont_vars_in)

        set_(self, 'output_no_new_dir', output_no_new_dir_in is True)
        set_(self, 'replication', replication_in is True)

        # iate_chunk_size
        ics = (iate_chunk_size_in if (isinstance_scalar(iate_chunk_size_in)
                                      and iate_chunk_size_in > 0)
               else None
               )
        set_(self, 'iate_chunk_size', ics)

        # sizes / caps
        obd = (obs_bigdata_in if (isinstance_scalar(obs_bigdata_in)
                                  and obs_bigdata_in > 10)
               else 1_000_000
               )
        set_(self, 'obs_bigdata', obd)

        mot = (max_obs_training_in if (isinstance_scalar(max_obs_training_in)
                                       and max_obs_training_in > 100)
               else float('inf')
               )
        set_(self, 'max_obs_training', mot)

        mop = (max_obs_prediction_in if (isinstance_scalar(max_obs_prediction_in
                                                           )
                                         and max_obs_prediction_in > 100)
               else 250_000
               )
        set_(self, 'max_obs_prediction', mop)

        mok = (max_obs_kmeans_in if (isinstance_scalar(max_obs_kmeans_in)
                                     and max_obs_kmeans_in > 100)
               else 200_000
               )
        set_(self, 'max_obs_post_kmeans', mok)

        mor = (max_obs_post_rel_graphs_in
               if isinstance_scalar(max_obs_post_rel_graphs_in)
               and max_obs_post_rel_graphs_in > 100
               else 50_000
               )
        set_(self, 'max_obs_post_rel_graphs', mor)


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
    def from_args(
        cls,
        int_cfg: Any,
        *,
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
            outpath_final = mcf_sys.define_outpath(
                outpath, not int_cfg.output_no_new_dir
                )
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

        return cls(
            d_type=d_type_final,
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
    gen_cfg: Any
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
            deepcopy(self.x_name_balance_test_ord +
                     self.x_name_balance_test_unord)
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
            deepcopy(self.x_name_ord + self.x_name_always_in_ord +
                     self.x_name_remain_ord)
            )
        self.name_unordered = mcf_gp.cleaned_var_names(
            deepcopy(self.x_name_unord + self.x_name_always_in_unord +
                     self.x_name_remain_unord)
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
            self.x_name = mcf_gp.cleaned_var_names(
                deepcopy(self.x_name_in_tree + self.x_name)
                )
        # ordered vs unordered overlap
        if self.name_ordered and self.name_unordered:
            if any(v for v in self.name_ordered if v in self.name_unordered):
                raise ValueError(
                    'Remove overlap in ordered + unordered variables'
                    )
        # names to check
        self.names_to_check_train = self.d_name + self.y_name + self.x_name
        self.names_to_check_pred = self.x_name[:]

        # z and gate flags
        if ((not self.z_name_cont) and (not self.z_name_ord)
                and (not self.z_name_unord)):
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
            self.z_name = (self.z_name_cont + self.z_name_ord
                           + self.z_name_unord
                           )
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

        if (self.p.bgate and self.x_name_balance_bgate == self.z_name
                and len(self.z_name) == 1):
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
        # final uniqueness for checks
        self.names_to_check_train = name_unique(self.names_to_check_train[:])
        self.names_to_check_pred = name_unique(self.names_to_check_pred[:])
    # ---------- keyword-only factory ----------

    @classmethod
    def from_args(
        cls, *,
        gen_cfg: Any,
        p_cfg: Any,
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
            if isinstance(y_name, (list, tuple)):
                yt_raw = [y_name[0]]
            else:
                yt_raw = [y_name]
        y_tree = norm(to_list(yt_raw))

        y_all = to_list(y_name)
        y_all.extend(y_tree)
        y_name_n = norm(y_all)

        # cluster names only if needed
        if p_cfg.cluster_std or gen_cfg.panel_data:
            cluster_name_n = norm(cluster_name)
        else:
            cluster_name_n = []

        # ate_no_se_only zeroes certain inputs
        if p_cfg.ate_no_se_only is True:
            x_name_balance_test_ord = None
            x_name_balance_test_unord = None
            z_name_cont = z_name_ord = z_name_unord = None
            x_name_balance_bgate = None

        # core normals
        d_name_n = norm(to_list(d_name))

        # id_name shaping (ensure first entry is str)
        if not isinstance(id_name, (list, tuple)):
            id_raw = [id_name]
        else:
            id_raw = list(id_name)
        if not id_raw or not isinstance(id_raw[0], str):
            id_raw = ['ID'] if not id_raw else 'ID'
        id_name_n = norm(id_raw)

        # iv list shaping
        if not isinstance(iv_name, (list, tuple)):
            iv_raw = [iv_name]
        else:
            iv_raw = list(iv_name)
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
        if p_cfg.bgate:
            x_bgate_n = norm(x_name_balance_bgate)
        else:
            x_bgate_n = []

        if p_ba_yes and p_ba_use_x:
            x_name_ba_n = norm(to_list(x_name_ba))
        else:
            x_name_ba_n = []

        # weights (clean; presence validated in __post_init__)
        w_name_n = norm(w_name)

        return cls(
            gen_cfg=gen_cfg,
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
    def from_args(
        cls,
        *,
        screen_covariates: BoolLike = None,
        check_perfectcorr: BoolLike = None,
        clean_data: BoolLike = None,
        min_dummy_obs: NumberLike = None,
            ) -> 'DCCfg':
        """Get input and normalize parameters."""
        return cls(
            screen_covariates=screen_covariates is not False,
            check_perfectcorr=check_perfectcorr is not False,
            clean_data=clean_data is not False,
            min_dummy_obs=(
                10 if (min_dummy_obs is None or float(min_dummy_obs) < 1)
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
    def from_args(
        cls,
        grid_dr: GridLike | None = None,
        grid_nn: GridLike | None = None,
        grid_w:  GridLike | None = None,
             ) -> 'CtGrid':
        """Just save values."""
        return cls(
            grid_dr=grid_dr,
            grid_nn=grid_nn,
            grid_w=grid_w,
            )
