"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from numbers import Real
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from psutil import virtual_memory

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_init_values_cfg_functions as mcf_initvals

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def int_update_train(mcf_: 'ModifiedCausalForest') -> None:
    """Update internal parameters before training."""
    int_cfg, cf_cfg = mcf_.int_cfg, mcf_.cf_cfg

    int_cfg.mp_ray_shutdown = ray_shut_down(int_cfg.mp_ray_shutdown,
                                            cf_cfg.n_train_eff
                                            )
    if int_cfg.max_cats_cont_vars is None or int_cfg.max_cats_cont_vars < 1:
        int_cfg.max_cats_cont_vars = cf_cfg.n_train_eff + 100_000_000
    else:
        int_cfg.max_cats_cont_vars = round(int_cfg.max_cats_cont_vars)
    if int_cfg.mp_vim_type not in (1, 2):
        int_cfg.mp_vim_type = 1 if cf_cfg.n_train_eff < 20_000 else 2
    # 1: MP over var's, fast, lots RAM; 2: MP over bootstraps.
    if cf_cfg.n_train_eff < 20_000:
        int_cfg.mem_object_store_1 = int_cfg.mem_object_store_2 = None
    else:
        memory = virtual_memory()
        # Obs. & number of trees as determinants of obj.store when forest build
        min_obj_str_n_1 = (
            (cf_cfg.n_train_eff / 60_000) * (cf_cfg.boot / 1_000)
            * (cf_cfg.m_grid * cf_cfg.n_min_grid
               * cf_cfg.alpha_reg_grid / 12)
            * (120 * 1024 * 1024 * 1024) * 5
            )
        min_obj_str_n_2 = ((cf_cfg.n_train_eff / 60_000)
                           * (cf_cfg.boot / 1_000)
                           * (120 * 1024 * 1024 * 1024)
                           )
        int_cfg.mem_object_store_1 = min(memory.available*0.5, min_obj_str_n_1)
        int_cfg.mem_object_store_2 = min(memory.available*0.5, min_obj_str_n_2)
        if int_cfg.mp_ray_objstore_multiplier > 0:
            int_cfg.mem_object_store_1 *= int_cfg.mp_ray_objstore_multiplier
            int_cfg.mem_object_store_2 *= int_cfg.mp_ray_objstore_multiplier
        int_cfg.mem_object_store_1 = min(0.7 * memory.available,
                                         int_cfg.mem_object_store_1
                                         )
        int_cfg.mem_object_store_2 = min(0.5 * memory.available,
                                         int_cfg.mem_object_store_2
                                         )
        int_cfg.mem_object_store_1 = int(int_cfg.mem_object_store_1)
        int_cfg.mem_object_store_2 = int(int_cfg.mem_object_store_2)

    int_cfg.bigdata_train = cf_cfg.n_train > int_cfg.obs_bigdata

    mcf_.int_cfg = int_cfg  # probably redundant


def int_update_pred(mcf_: 'ModifiedCausalForest', n_pred: int) -> None:
    """Update internal parameters before prediction."""
    int_cfg = mcf_.int_cfg
    # Adjusted for effective training sample size
    n_pred_adj = n_pred * (mcf_.cf_cfg.n_train_eff / n_pred)**0.5
    int_cfg.mp_ray_shutdown = ray_shut_down(int_cfg.mp_ray_shutdown, n_pred_adj)
    memory = virtual_memory()
    if n_pred_adj < 20000:
        int_cfg.mem_object_store_3 = None
    else:
        min_obj_str_n_3 = (n_pred_adj / 60000) * (120 * 1024 * 1024 * 1024)
        int_cfg.mem_object_store_3 = min(memory.available * 0.5, min_obj_str_n_3
                                         )
        if int_cfg.mp_ray_objstore_multiplier > 0:
            int_cfg.mem_object_store_3 *= int_cfg.mp_ray_objstore_multiplier
        if int_cfg.mem_object_store_3 > 0.5 * memory.available:
            int_cfg.mem_object_store_3 = int(0.5 * memory.available)
        int_cfg.mem_object_store_3 = int(int_cfg.mem_object_store_3)

    mcf_.int_cfg = int_cfg  # probably redunant


def gen_update_train(mcf_: 'ModifiedCausalForest',
                     data_df: DataFrame,
                     ) -> None:
    """Add and update some dictionary entry based on data."""
    gen_cfg, var_cfg = mcf_.gen_cfg, mcf_.var_cfg
    d_dat = data_df[var_cfg.d_name].to_numpy()
    gen_cfg.d_values = np.unique(np.int32(np.round(d_dat))).tolist()
    gen_cfg.no_of_treat = len(gen_cfg.d_values)
    if (len(data_df) > mcf_.int_cfg.obs_bigdata
            and gen_cfg.mp_parallel > 1):
        gen_cfg.mp_parallel = int(gen_cfg.p_parallel * 0.75)

    mcf_.gen_cfg = gen_cfg


def var_update_train(mcf_: 'ModifiedCausalForest', data_df: DataFrame) -> None:
    """Update var_cfg with training data if needed."""
    var_cfg = mcf_.var_cfg
    var_cfg.d_name[0] = mcf_gp.adjust_var_name(var_cfg.d_name[0],
                                               data_df.columns.tolist()
                                               )
    mcf_.var_cfg = var_cfg


def ct_update_train(mcf_: 'ModifiedCausalForest', data_df: DataFrame) -> None:
    """Initialise dictionary with parameters of continuous treatment."""
    gen_cfg, ct_cfg, var_cfg = mcf_.gen_cfg, mcf_.ct_cfg, mcf_.var_cfg
    zero_tol = mcf_.int_cfg.zero_tol
    d_dat = data_df[var_cfg.d_name].to_numpy()
    if gen_cfg.d_type == 'continuous':
        grid_nn, grid_w, grid_dr = 10, 10, 100
        ct_cfg.grid_nn = ct_grid(grid_nn, grid_nn)
        ct_cfg.grid_w = ct_grid(grid_w, grid_w)
        ct_cfg.grid_dr = ct_grid(grid_dr, grid_dr)
        ct_cfg.grid_dr = max(ct_cfg.grid_dr, ct_cfg.grid_w)
        ct_cfg.grid_nn_val = grid_val_fct(ct_cfg.grid_nn, d_dat, zero_tol)
        ct_cfg.grid_w_val = grid_val_fct(ct_cfg.grid_w, d_dat, zero_tol)
        ct_cfg.grid_dr_val = grid_val_fct(ct_cfg.grid_dr, d_dat, zero_tol)
        ct_cfg.grid_nn = len(ct_cfg.grid_nn_val)
        ct_cfg.grid_w = len(ct_cfg.grid_w_val)
        ct_cfg.grid_dr = len(ct_cfg.grid_dr_val)
        gen_cfg.no_of_treat = len(ct_cfg.grid_nn_val)
        gen_cfg.d_values = None
        precision_of_cont_treat = 4
        (ct_cfg.w_to_dr_int_w01, ct_cfg.w_to_dr_int_w10,
         ct_cfg.w_to_dr_index_full, ct_cfg.d_values_dr_list,
         ct_cfg.d_values_dr_np) = interpol_weights(
            ct_cfg.grid_dr, ct_cfg.grid_w, ct_cfg.grid_w_val,
            precision_of_cont_treat
            )
        var_cfg.grid_nn_name = grid_name(var_cfg.d_name, ct_cfg.grid_nn)
        var_cfg.grid_w_name = grid_name(var_cfg.d_name, ct_cfg.grid_w)
        var_cfg.grid_dr_name = grid_name(var_cfg.d_name, ct_cfg.grid_dr)
    else:
        ct_cfg.grid_nn = ct_cfg.grid_nn_val = ct_cfg.grid_w = None
        ct_cfg.grid_dr = ct_cfg.grid_dr_val = ct_cfg.grid_w_val = None
    mcf_.gen_cfg = gen_cfg
    mcf_.ct_cfg = ct_cfg


def lc_update_train(mcf_: 'ModifiedCausalForest', data_df):
    """Adjust lc for number of training observations."""
    obs = len(data_df)
    if mcf_.lc_cfg.cs_cv_k is None:
        match obs:
            case _ if obs < 100_000: mcf_.lc_cfg.cs_cv_k = 5
            case _ if obs < 250_000: mcf_.lc_cfg.cs_cv_k = 4
            case _ if obs < 500_000: mcf_.lc_cfg.cs_cv_k = 3
            case _:                  mcf_.lc_cfg.cs_cv_k = 2

def pa_ba_update_train(cv_k, obs):
    """Adjust folds for CV for number of training observations."""
    if cv_k is None:
        match obs:
            case _ if obs < 100_000: return 5
            case _ if obs < 250_000: return 4
            case _ if obs < 500_000: return 3
            case _:                  return 2


def cs_update_train(mcf_: 'ModifiedCausalForest') -> None:
    """Adjust cs for number of treatments."""
    cs_cfg = mcf_.cs_cfg
    if cs_cfg.adjust_limits is None:
        cs_cfg.adjust_limits = (mcf_.gen_cfg.no_of_treat - 2) * 0.05
    if cs_cfg.adjust_limits < 0:
        raise ValueError('Negative common support adjustment factor is not'
                         ' possible.'
                         )
    mcf_.cs_cfg = cs_cfg


def cf_update_train(mcf_: 'ModifiedCausalForest', data_df: DataFrame) -> None:
    """Update cf parameters that need information from training data."""
    cf_cfg, gen_cfg, lc_cfg = mcf_.cf_cfg, mcf_.gen_cfg, mcf_.lc_cfg
    fs_cfg, var_cfg, int_cfg = mcf_.fs_cfg, mcf_.var_cfg, mcf_.int_cfg
    n_train = len(data_df)
    # Number of obs in treatments before any selection
    vcount = data_df.groupby(var_cfg.d_name).size()  # pylint:disable=E1101
    obs_by_treat = vcount.to_numpy()
    if abs(n_train - obs_by_treat.sum()) > len(obs_by_treat):
        raise RuntimeError(
            f'Counting treatments does not work. n_d_sum:'
            f' {obs_by_treat.sum()}, n_train: {n_train}. Difference'
            ' could be due to missing values in treatment.')

    # Adjust for smaller effective training samples due to feature selection
    # and possibly local centering and common support
    cf_cfg.n_train = reduce_effective_n_train(mcf_, n_train)
    if fs_cfg.yes and fs_cfg.other_sample:
        obs_by_treat = obs_by_treat * (1 - fs_cfg.other_sample_share)
    if (not isinstance(cf_cfg.chunks_maxsize, (int, float))
            or cf_cfg.chunks_maxsize < 100):
        cf_cfg.baseline = 100_000
        cf_cfg.chunks_maxsize = get_chunks_maxsize_forest(
            cf_cfg.baseline, cf_cfg.n_train,
            mcf_.gen_cfg.no_of_treat)
    else:
        cf_cfg.chunks_maxsize = round(cf_cfg.chunks_maxsize)

    # Effective sample sizes per chuck
    no_of_chuncks = int(np.ceil(cf_cfg.n_train / cf_cfg.chunks_maxsize))
    # Actual number of chuncks could be smaller if lot's of data is deleted in
    # common support adjustment
    # This will be updated in the train method, adjusting for common support
    cf_cfg.n_train_eff = np.int32(cf_cfg.n_train / no_of_chuncks)
    obs_by_treat_eff = np.int32(obs_by_treat / no_of_chuncks)

    # size of subsampling samples         n/2: size of forest sample
    cf_cfg.subsample_share_forest = sub_size(
        cf_cfg.n_train_eff, cf_cfg.subsample_factor_forest, 0.67)

    match cf_cfg.subsample_factor_eval:
        case None | True:
            cf_cfg.subsample_factor_eval = 2
        case False:
            cf_cfg.subsample_factor_eval = 1_000_000_000
        case x if isinstance(x, Real) and x < 0.01:
            cf_cfg.subsample_factor_eval = 1_000_000_000
        case _:
            pass  # keep provided numeric factor

    cf_cfg.subsample_share_eval = min(
        cf_cfg.subsample_share_forest * cf_cfg.subsample_factor_eval, 1
        )
    n_d_subsam = (obs_by_treat_eff.min()
                  * int_cfg.share_forest_sample
                  * cf_cfg.subsample_share_forest)

    # Further adjustments when data is used for other purposes
    if mcf_.cs_cfg.type_ > 0 and not lc_cfg.cs_cv:
        n_d_subsam *= (1 - lc_cfg.cs_share)
    if lc_cfg.yes and not lc_cfg.cs_cv:
        n_d_subsam *= (1 - lc_cfg.cs_share)

    # Check only random thresholds to save computation time when building CF
    if cf_cfg.random_thresholds is None or cf_cfg.random_thresholds < 0:
        cf_cfg.random_thresholds = round(4 + cf_cfg.n_train_eff**0.2)

    # Penalty multiplier in CF building
    if (cf_cfg.p_diff_penalty is None or cf_cfg.p_diff_penalty < 0):
        if cf_cfg.mtot == 4:
            cf_cfg.p_diff_penalty = 0.5
        else:                                   # Approx 1 for N = 1000
            cf_cfg.p_diff_penalty = (
                2 * ((cf_cfg.n_train_eff * cf_cfg.subsample_share_forest)**0.9)
                / (cf_cfg.n_train_eff * cf_cfg.subsample_share_forest))
            if cf_cfg.mtot == 2:
                cf_cfg.p_diff_penalty = 100 * cf_cfg.p_diff_penalty
            if gen_cfg.d_type == 'discrete':
                cf_cfg.p_diff_penalty *= np.sqrt(
                    gen_cfg.no_of_treat * (gen_cfg.no_of_treat - 1) / 2)
    elif cf_cfg.p_diff_penalty == 0:
        if cf_cfg.mtot == 4:
            cf_cfg.mtot = 1  # No random mixing  prob of MSE+MCE rule== 1
    else:
        if cf_cfg.mtot == 4:
            if cf_cfg.p_diff_penalty > 1:  # if accidently scaled %
                cf_cfg.p_diff_penalty = cf_cfg.p_diff_penalty / 100
            if not 0 <= cf_cfg.p_diff_penalty <= 1:
                raise ValueError('Probability of using p-score > 1. Programm'
                                 ' stopped.')
    if cf_cfg.p_diff_penalty:
        if cf_cfg.penalty_type == 'mse_d':
            cf_cfg.estimator_str += ' Penalty "MSE of treatment variable"'
        else:
            cf_cfg.estimator_str += f' Penalty {cf_cfg.penalty_type}'

    # Minimum leaf size
    if cf_cfg.n_min_min is None or cf_cfg.n_min_min < 1:
        cf_cfg.n_min_min = round(max((n_d_subsam**0.4) / 10, 1.5)
                                 * gen_cfg.no_of_treat)
    else:
        cf_cfg.n_min_min = round(cf_cfg.n_min_min)
    if cf_cfg.n_min_max is None or cf_cfg.n_min_max < 1:
        cf_cfg.n_min_max = round(max(n_d_subsam**0.5 / 10, 2)
                                 * gen_cfg.no_of_treat)
    else:
        cf_cfg.n_min_max = round(cf_cfg.n_min_max)
    cf_cfg.n_min_max = max(cf_cfg.n_min_min, cf_cfg.n_min_max)
    if gen_cfg.d_type == 'discrete':
        if cf_cfg.n_min_treat is None or cf_cfg.n_min_treat < 1:
            cf_cfg.n_min_treat = round(
                max((cf_cfg.n_min_min + cf_cfg.n_min_max) / 2
                    / gen_cfg.no_of_treat / 10, 1))
        else:
            cf_cfg.n_min_treat = round(cf_cfg.n_min_treat)
        min_leaf_size = cf_cfg.n_min_treat * gen_cfg.no_of_treat
        if cf_cfg.n_min_min < min_leaf_size:
            cf_cfg.n_min_min = min_leaf_size
            mcf_ps.print_mcf(gen_cfg, 'Minimum leaf size adjusted. Smallest ',
                             f' leafsize set to: {cf_cfg.n_min_min}',
                             summary=True)
    else:
        cf_cfg.n_min_treat = 0
    if cf_cfg.n_min_grid is None or cf_cfg.n_min_grid < 1:
        cf_cfg.n_min_grid = 1
    else:
        cf_cfg.n_min_grid = round(cf_cfg.n_min_grid)

    if cf_cfg.n_min_min == cf_cfg.n_min_max:
        cf_cfg.n_min_grid = 1
    if cf_cfg.n_min_grid == 1:
        cf_cfg.n_min_min = cf_cfg.n_min_max = round(
            (cf_cfg.n_min_min + cf_cfg.n_min_max) / 2)
        cf_cfg.n_min_values = cf_cfg.n_min_min
    else:
        if cf_cfg.n_min_grid == 2:
            n_min = np.hstack((cf_cfg.n_min_min, cf_cfg.n_min_max))
        else:
            n_min = np.linspace(cf_cfg.n_min_min,
                                cf_cfg.n_min_max,
                                cf_cfg.n_min_grid
                                )
        n_min = list(np.unique(np.round(n_min)))
        cf_cfg.n_min_min, cf_cfg.n_min_max = n_min[0], n_min[-1]
        cf_cfg.n_min_grid = len(n_min)
        cf_cfg.n_min_values = n_min
    cf_cfg.forests = None              # To be filled at the end of training
    mcf_.cf_cfg = cf_cfg


def p_update_train(mcf_: 'ModifiedCausalForest') -> None:
    """Update parameters with sample size information."""
    p_cfg = mcf_.p_cfg
    # Categorise continuous gate variables
    if p_cfg.max_cats_z_vars is None or p_cfg.max_cats_z_vars < 1:
        p_cfg.max_cats_z_vars = round(mcf_.cf_cfg.n_train_eff ** 0.3)
    else:
        p_cfg.max_cats_z_vars = round(p_cfg.max_cats_z_vars)
    mcf_.p_cfg = p_cfg


def p_update_pred(mcf_: 'ModifiedCausalForest',
                  data_df: DataFrame
                  ) -> DataFrame:
    """Update parameters of p_cfg with data_df related information."""
    gen_cfg, p_cfg, var_cfg = mcf_.gen_cfg, mcf_.p_cfg, mcf_.var_cfg
    n_pred = len(data_df)
    if p_cfg.bgate_sample_share is None or p_cfg.bgate_sample_share <= 0:
        p_cfg.bgate_sample_share = (
            1 if n_pred < 1000 else (1000 + ((n_pred-1000) ** 0.75)) / n_pred)
    d_name = (var_cfg.d_name[0] if isinstance(var_cfg.d_name, (list, tuple))
              else var_cfg.d_name
              )
    # Capitalise all variable names
    data_df = data_df.rename(columns=lambda x: x.casefold())
    # Check if treatment is included
    if gen_cfg.d_type == 'continuous':
        p_cfg.d_in_pred = False
    else:
        p_cfg.d_in_pred = d_name in data_df.columns
    if not p_cfg.d_in_pred:
        if p_cfg.atet or p_cfg.gatet:
            mcf_ps.print_mcf(gen_cfg,
                             'Treatment variable not in prediction data. ATET '
                             'and GATET cannot be computed.',
                             summary=True
                             )
        p_cfg.atet = p_cfg.gatet = False
        if p_cfg.choice_based_sampling:
            raise ValueError('Choice based sampling relates only to prediction '
                             'data. It requires treatment information in '
                             'prediction data, WHICH IS MISSING!'
                             )
    if p_cfg.choice_based_sampling is True:
        if len(p_cfg.choice_based_probs) != gen_cfg.no_of_treat:
            raise ValueError('Choice based sampling. Rows in choice '
                             'probabilities do not correspond to number of '
                             'treatments.'
                             )
        if any(v <= 0 for v in p_cfg.choice_based_probs):
            raise ValueError('Choice based sampling active. Not possible to '
                             'have zero or negative choice probability. '
                             )
        # Normalize
        p_cfg.choice_based_probs = mcf_initvals.p_cb_normalize(
            p_cfg.choice_based_probs, gen_cfg.no_of_treat
            )

    else:
        p_cfg.choice_based_sampling, p_cfg.choice_based_probs = False, 1
    mcf_.p_cfg = p_cfg

    return data_df


def post_update_pred(mcf_: 'ModifiedCausalForest', data_df: DataFrame) -> None:
    """Update entries in post_cfg that need info from prediction data."""
    n_pred = len(data_df)
    post_cfg = mcf_.post_cfg
    if isinstance(post_cfg.kmeans_no_of_groups, (int, float)):
        post_cfg.kmeans_no_of_groups = [(post_cfg.kmeans_no_of_groups)]
    if (post_cfg.kmeans_no_of_groups is None
        or len(post_cfg.kmeans_no_of_groups) == 1
            or post_cfg.kmeans_no_of_groups[0] < 2):
        if n_pred < 10_000:
            middle = 5
        elif n_pred > 100_000:
            middle = 10
        else:
            middle = 5 + round(n_pred/20_000)
        if middle < 7:
            post_cfg.kmeans_no_of_groups = [middle-2, middle-1,
                                            middle,
                                            middle+1, middle+2
                                            ]
        else:
            post_cfg.kmeans_no_of_groups = [middle-4, middle-2,
                                            middle,
                                            middle+2, middle+4
                                            ]
    else:
        if not isinstance(post_cfg.kmeans_no_of_groups, list):
            post_cfg.kmeans_no_of_groups = list(post_cfg.kmeans_no_of_groups)
            post_cfg.kmeans_no_of_groups = [
                round(a) for a in post_cfg.kmeans_no_of_groups
                ]
    mcf_.post_cfg = post_cfg


def name_unique(all_names: list[str]) -> list[str]:
    """Remove any duplicates."""
    seen = set()
    name_unique_ = [
        item for item in all_names if item not in seen and not seen.add(item)]
    return name_unique_


def get_ray_del_defaults(_mp_ray_del_user: Any) -> Any:
    """Get values for :mp_ray_del."""
    if _mp_ray_del_user is None:
        _mp_ray_del = ('refs',)
    else:
        possible_vals = ('refs', 'rest', 'none')

        match _mp_ray_del_user:
            case str() as s:    _mp_ray_del = (s,)
            case list() as lst: _mp_ray_del = tuple(lst)
            case _:             _mp_ray_del = _mp_ray_del_user

        if len(_mp_ray_del) > 2:
            raise ValueError(
                f'Too many parameters for _mp_ray_del{_mp_ray_del}')
        if not isinstance(_mp_ray_del, tuple):
            raise ValueError(f'mp_ray_del is no Tuple {_mp_ray_del}')
        if not all(i in possible_vals for i in _mp_ray_del):
            raise ValueError(f'Wrong values for _mp_ray_del {_mp_ray_del}')

    return _mp_ray_del


def ray_shut_down(ray_shutdown: bool | None, len_data: int) -> bool:
    """Define mimimum sample size for ray_shut_down."""
    if ray_shutdown is None:
        ray_shutdown = not len_data < 150000
    if not isinstance(ray_shutdown, bool):
        raise ValueError('mp_ray_shutdown must be either None or Boolean')

    return ray_shutdown


def get_alpha(alpha_reg_grid: int | float | None,
              alpha_reg_max: float | None,
              alpha_reg_min: float | None,
              ) -> tuple[int | float,
                         float, float,
                         float | NDArray[float] | list[float]
                         ]:
    """Get the alphas for the CF."""
    if alpha_reg_min is None or not 0 <= alpha_reg_min < 0.4:
        alpha_reg_min = 0.05
    if alpha_reg_max is None or not 0 <= alpha_reg_max < 0.4:
        alpha_reg_max = 0.15
    alpha_reg_grid = (1 if alpha_reg_grid is None or alpha_reg_grid < 1
                      else round(alpha_reg_grid))
    if alpha_reg_min >= alpha_reg_max:
        alpha_reg_grid, alpha_reg_max = 1, alpha_reg_min

    match alpha_reg_grid:
        case 1: alpha_reg = (alpha_reg_max + alpha_reg_min) / 2
        case 2: alpha_reg = np.array([alpha_reg_min, alpha_reg_max])
        case _:
            alpha_reg = np.linspace(alpha_reg_min, alpha_reg_max, alpha_reg_grid
                                    )
            alpha_reg = list(np.unique(alpha_reg))
            alpha_reg_grid = len(alpha_reg)

    return alpha_reg_grid, alpha_reg_max, alpha_reg_min, alpha_reg


def grid_val_fct(grid: float | int,
                 d_dat: NDArray[Any],
                 zero_tol: float = 1e-15
             ) -> NDArray[Any]:
    """Help for initialisation."""
    quantile = np.linspace(1/(grid)/2, 1-1/grid/2, num=grid)
    d_dat_min = d_dat.min()
    d_dat_r = d_dat - d_dat_min if d_dat_min != 0 else d_dat
    gridvalues = np.around(
        np.quantile(d_dat_r[d_dat_r > zero_tol], quantile), decimals=6)
    gridvalues = np.insert(gridvalues, 0, 0)

    return gridvalues


def ct_grid(user_grid: list[int | float],
            defaultgrid: list[int | float]
            ) -> list[int | float]:
    """Help for initialisation."""
    if isinstance(user_grid, int):
        grid = defaultgrid if user_grid < 1 else user_grid
    else:
        grid = defaultgrid

    return grid


def interpol_weights(ct_grid_dr: float | int,
                     ct_grid_w: int,
                     ct_grid_w_val: list[float | int] | NDArray[Any],
                     precision_of_treat: int
                     ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any],
                                list[int | float],
                                NDArray[Any],
                                ]:
    """Generate interpolation measures for continuous treatments."""
    interpol_points = round(ct_grid_dr / ct_grid_w) + 1
    int_w01 = np.linspace(0, 1, interpol_points, endpoint=False)
    int_w10 = 1 - int_w01
    treat_val_list, j_all = [], 0
    index_full = np.zeros((ct_grid_w, len(int_w01)))
    for i, (val, val1) in enumerate(zip(ct_grid_w_val[:-1],
                                        ct_grid_w_val[1:])):
        for j in range(interpol_points):
            value = int_w10[j] * val + int_w01[j] * val1
            treat_val_list.append(round(value, precision_of_treat))
            index_full[i, j] = j_all
            j_all = j_all + 1    # do not use +=
    treat_val_list.append(ct_grid_w_val[-1])
    treat_val_np = np.around(np.array(treat_val_list), precision_of_treat)
    if len(treat_val_np) != len(np.unique(treat_val_np)):
        raise ValueError('Continuous treatment needs higher precision')
    index_full[ct_grid_w-1, 0] = j_all
    index_full = np.int32(index_full)

    return int_w01, int_w10, index_full, treat_val_list, treat_val_np


def grid_name(d_name: list[str] | tuple[str],
              add_name: str | int | float
              ) -> list[str]:
    """Help for initialisation."""
    grid_name_tmp = d_name[0] + str(add_name)
    grid_name_l = [grid_name_tmp.casefold()]

    return grid_name_l


def sub_size(n_train: int, share_mult: float, max_share: float | int) -> float:
    """Help for initialisation."""
    if share_mult is None or share_mult <= 0:
        share_mult = 1
    subsam_share = min(4 * ((n_train / 2)**0.85) / n_train, 0.67) * share_mult
    subsam_share = max(min(subsam_share, max_share),
                       (2 * (n_train / 2)**0.5) / n_train
                       )
    return subsam_share


def bootstrap(se_boot: int | None,
              cut_off: int,
              bnr: int,
              cluster_std: bool,
              ) -> int | bool:
    """Check and correct bootstrap level."""
    if se_boot is None:
        se_boot = bnr if cluster_std else False
    if 0 < se_boot < cut_off:
        return bnr
    if se_boot >= cut_off:
        return round(se_boot)

    return False


def reduce_effective_n_train(mcf_: 'ModifiedCausalForest', n_train: int) -> int:
    """Compute effective training sample size."""
    if mcf_.fs_cfg.yes and mcf_.fs_cfg.other_sample:
        n_train *= 1 - mcf_.fs_cfg.other_sample_share
    if mcf_.cs_cfg.type_ > 0 and not mcf_.lc_cfg.cs_cv:
        n_train *= (1 - mcf_.lc_cfg.cs_share)
    if mcf_.lc_cfg.yes and not mcf_.lc_cfg.cs_cv:
        n_train *= (1 - mcf_.lc_cfg.cs_share)

    return int(n_train)


def get_chunks_maxsize_forest(base_level: int,
                              obs: int,
                              no_of_treat: int
                              ) -> int:
    """Compute optimal chunksize for forest splitting."""
    return round(base_level
                 + (max(obs - base_level, 0) ** 0.8) / (no_of_treat - 1))


def var_helper(var_cfg: Any) -> tuple[Any, Any, Any]:
    """Remove unncessary elements from var_cfg."""
    gen_cfg = deepcopy(var_cfg.gen_cfg)
    p_cfg = deepcopy(var_cfg.p)
    var_cfg.gen_cfg = var_cfg.p = None  # Not needed as part of var_cfg

    return gen_cfg, p_cfg, var_cfg


def isinstance_scalar(data: Any) -> bool:
    """Check if the input data is a scalar."""
    SCALAR_TYPES = (float, int, np.floating, np.integer)

    return isinstance(data, SCALAR_TYPES)
