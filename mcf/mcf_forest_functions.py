"""
Contains functions for building the forest.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import ray

from mcf.mcf_bias_adjustment_functions import (
    compute_scores_x, get_ba_data_train, demean_col
    )
from mcf import mcf_forest_data_functions as mcf_fo_data
from mcf import mcf_forest_add_functions as mcf_fo_add
from mcf import mcf_forest_objfct_functions as mcf_fo_obj
from mcf import mcf_forest_asdict_functions as mcf_fo_asdict
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_variable_importance_functions as mcf_vi

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def train_forest(mcf_: 'ModifiedCausalForest',
                 tree_df: DataFrame,
                 fill_y_df: DataFrame,
                 title: str = ''
                 ) -> tuple[dict, list, float, dict]:
    """Train the forest and do variable importance measures."""
    gen_cfg, cf_cfg, forest_list = mcf_.gen_cfg, mcf_.cf_cfg, mcf_.forest
    p_ba_cfg = mcf_.p_ba_cfg
    seed, time_vi = 9324561, 0
    if gen_cfg.with_output:
        mcf_ps.print_mcf(gen_cfg,
                         '=' * 100
                         + f'\nTraining of Modified Causal Forest {title}'
                         )
    if gen_cfg.any_eff:
        cf_cfg.est_rounds = ('regular', 'additional')
    else:
        cf_cfg.est_rounds = ('regular', )
    obs = len(tree_df) + len(fill_y_df)

    if p_ba_cfg.yes:
        # Note: For large data, this is done before forests are computed in
        #       folds. Therefore, we need to save the index to be able to
        #       relate the data to the correct folds in the prediction method
        #       (if needed)
        # These values will be stored in the respective forest_dic defined below
        prog_r, prop_r, x_r = compute_scores_x(mcf_,
                                               train_df=tree_df,
                                               predict_df=fill_y_df,
                                               )
        prog_r, prop_r = demean_col(prog_r), demean_col(prop_r)
        x_r = demean_col(x_r)
        # Define the evaluation points
        match p_ba_cfg.adj_method:
            case 'zeros':
                p_ba_cfg.x_ba_eval = None
                p_ba_cfg.prog_score_eval = None
                p_ba_cfg.prop_score_eval = None
            case 'obs' | 'w_obs':
                p_ba_cfg.x_ba_eval = demean_col(x_r)
                p_ba_cfg.prog_score_eval = demean_col(prog_r)
                p_ba_cfg.prop_score_eval = demean_col(prop_r)
            case _:
                raise ValueError('Invalid Bias Adjustment Method')

        if gen_cfg.any_eff:
            # Change role of samples
            prog_a, prop_a, x_a = compute_scores_x(mcf_,
                                                   train_df=fill_y_df,
                                                   predict_df=tree_df,
                                                   )
            prog_a, prop_a = demean_col(prog_a), demean_col(prop_a)
            x_a = demean_col(x_a)
        else:
            prog_a = prop_a = x_a = None
    else:
        prog_r = prop_r = x_r = prog_a = prop_a = x_a = None

    if (folds := int(np.ceil(obs / cf_cfg.chunks_maxsize))) > 1:
        index_tr, index_y = tree_df.index, fill_y_df.index
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(index_tr.to_numpy())
        rng.shuffle(index_y.to_numpy())
        index_folds_tr = np.array_split(index_tr, folds)
        index_folds_y = np.array_split(index_y, folds)
    else:
        index_folds_tr = index_folds_y = None
    # Similar quantity is defined in mcf_init_functions, but without accounting
    # for common support.
    cf_cfg.folds = folds
    for splits in range(folds):
        if folds > 1:
            tree_fold_df = tree_df.loc[index_folds_tr[splits]]
            fill_y_fold_df = fill_y_df.loc[index_folds_y[splits]]
        else:
            tree_fold_df, fill_y_fold_df = tree_df, fill_y_df

        for round_ in cf_cfg.est_rounds:
            reg_round = round_ == 'regular'
            if p_ba_cfg.yes:  # Bias adjustment
                if reg_round:
                    if folds > 1:
                        prog_dat, prop_dat, x_ba_dat = get_ba_data_train(
                            idx_sub=index_folds_y[splits],
                            idx=fill_y_df.index,
                            prog=prog_r, prop=prop_r, x_ba=x_r,
                            )
                    else:
                        prog_dat, prop_dat, x_ba_dat = prog_r, prop_r, x_r
                else:
                    if folds > 1:
                        prog_dat, prop_dat, x_ba_dat = get_ba_data_train(
                            idx_sub=index_folds_tr[splits],
                            idx=tree_df.index,
                            prog=prog_a, prop=prop_a, x_ba=x_a,
                            )
                    else:
                        prog_dat, prop_dat, x_ba_dat = prog_a, prop_a, x_a
            else:
                prog_dat = prop_dat = x_ba_dat = None

            if not reg_round:
                # Reverse training and fill_with_y_file
                tree_fold_df, fill_y_fold_df = efficient_iate(
                    mcf_, fill_y_fold_df, tree_fold_df, summary=False)

            # Data preparation and stats II (regular, efficient IATE)

            tree_fold_df, mcf_.var_cfg = mcf_fo_data.nn_matched_outcomes(
               mcf_, tree_fold_df, print_out=reg_round and splits == 0)

            # Estimate forest structure (regular, efficient IATE)
            if gen_cfg.with_output:
                print(f'\nBuilding {splits+1} / {folds} forests, {round_}')
            forest, x_name_mcf, report_ = build_forest(mcf_, tree_fold_df)
            if reg_round and splits == 0:
                cf_cfg.x_name_mcf = x_name_mcf
            # Variable importance  ONLY REGULAR
            if all((cf_cfg.vi_oob_yes, gen_cfg.with_output, reg_round,
                    splits == 0)):
                time_start = time()
                mcf_vi.variable_importance(mcf_, tree_fold_df, forest,
                                           x_name_mcf
                                           )
                time_vi = time() - time_start
            else:
                time_vi = 0
            forest = mcf_fo_asdict.delete_data_from_forest(forest)
            # Fill tree with outcomes(regular, , efficient IATE)
            if gen_cfg.with_output:
                print(f'Filling {splits+1} / {folds} forests, {round_}')
            forest, _, share_merged = mcf_fo_add.fill_trees_with_y_indices_mp(
                mcf_, fill_y_fold_df, forest)    # Fill
            forest_dic = mcf_fo_add.train_save_data(mcf_, fill_y_fold_df,
                                                    forest,
                                                    prog_dat, prop_dat, x_ba_dat
                                                    )
            forest_list = mcf_fo_add.save_forests_in_cf_cfg(
                forest_dic, forest_list, splits, folds, reg_round,
                gen_cfg.any_eff,
                )
            if gen_cfg.with_output:
                if reg_round and splits == 0:
                    report = report_
                    report["share_leaf_merged"] = share_merged
            else:
                report = None
            if gen_cfg.with_output and gen_cfg.verbose:
                mcf_sys.print_mememory_statistics(
                    gen_cfg, 'Forest Building: End of forests loop.'
                    )
    return cf_cfg, p_ba_cfg, forest_list, time_vi, report


def build_forest(mcf_: 'ModifiedCausalForest',
                 tree_df: DataFrame,
                 ) -> tuple[list, list, dict]:
    """Build MCF (not yet populated by w and outcomes)."""
    int_cfg, gen_cfg, cf_cfg = mcf_.int_cfg, mcf_.gen_cfg, mcf_.cf_cfg
    cuda, cython = int_cfg.cuda, int_cfg.cython
    with_ray = not int_cfg.no_ray_in_forest_building
    bigdata_train = int_cfg.bigdata_train
    if not with_ray:
        if gen_cfg.with_output and gen_cfg.verbose:
            mcf_ps.print_mcf(gen_cfg, '\nNo use of ray in forest building.',
                             summary=False)
    (x_name, x_type, x_values, cf_cfg, pen_mult, data_np, y_i, y_nn_i, x_i,
     x_ind, x_ai_ind, d_i, w_i, cl_i, d_grid_i
     ) = mcf_fo_data.prepare_data_for_forest(mcf_, tree_df)
    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_sys.find_no_of_workers(gen_cfg.mp_parallel,
                                                 gen_cfg.sys_share,
                                                 zero_tol=int_cfg.zero_tol,
                                                 )
                      if gen_cfg.mp_automatic else gen_cfg.mp_parallel)
    if gen_cfg.with_output and gen_cfg.verbose:
        mcf_ps.print_mcf(gen_cfg,
                         '\nNumber of parallel processes (forest): '
                         f'{maxworkers}',
                         summary=False)
    forest = [None for _ in range(cf_cfg.boot)]

    # Initialise save seeds for forest computation
    child_seeds = np.random.SeedSequence(1233456).spawn(cf_cfg.boot)

    if maxworkers == 1:
        for idx in range(cf_cfg.boot):
            forest[idx] = build_tree_mcf(
                data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type,
                x_values, x_ind, x_ai_ind, gen_cfg, cf_cfg, mcf_.ct_cfg, idx,
                pen_mult, cuda, cython, bigdata_train, seedseq=child_seeds[idx],
                zero_tol=int_cfg.zero_tol,
                )
            if gen_cfg.with_output and gen_cfg.verbose:
                mcf_gp.share_completed(idx+1, cf_cfg.boot)
    else:
        if with_ray:
            if int_cfg.mem_object_store_1 is None:
                if not ray.is_initialized():
                    mcf_sys.init_ray_with_fallback(
                        maxworkers, int_cfg, gen_cfg,
                        ray_err_txt='Ray did not start in forest building.'
                        )
            else:
                if not ray.is_initialized():
                    mcf_sys.init_ray_with_fallback(
                        maxworkers, int_cfg, gen_cfg,
                        mem_object_store=int_cfg.mem_object_store_1,
                        ray_err_txt='Ray did not start in forest building.'
                        )
                if gen_cfg.with_output and gen_cfg.verbose:
                    print("Size of Ray Object Store: ", round(
                        int_cfg.mem_object_store_1/(1024*1024)), " MB"
                        )
            data_np_ref = ray.put(data_np)
            still_running = [
                ray_build_tree_mcf.remote(
                    data_np_ref, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, gen_cfg, cf_cfg,
                    mcf_.ct_cfg, boot, pen_mult, cuda, cython,
                    bigdata_train, seedseq=child_seeds[boot],
                    zero_tol=int_cfg.zero_tol,
                    )
                for boot in range(cf_cfg.boot)
                ]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for ret_all_i in finished_res:
                    forest[jdx] = ret_all_i
                    if gen_cfg.with_output and gen_cfg.verbose:
                        mcf_gp.share_completed(jdx+1, cf_cfg.boot)
                    jdx += 1
                if jdx % 50 == 0:   # every 50'th tree
                    mcf_sys.auto_garbage_collect(50)  # do if half mem full
            if 'refs' in int_cfg.mp_ray_del:
                del data_np_ref
            if 'rest' in int_cfg.mp_ray_del:
                del finished_res, finished
        else:
            raise RuntimeError('USE RAY')
        if len(forest) != cf_cfg.boot:
            raise RuntimeError(f'Forest has wrong size: {len(forest)}'
                               'Bug in Multiprocessing.')
    # find best forest given the saved oob values
    forest_final, m_n_final = best_m_n_min_alpha_reg(forest, gen_cfg, cf_cfg,
                                                     int_cfg.cython)
    del forest    # Free memory
    # Describe final forest
    if gen_cfg.with_output:
        report = mcf_fo_add.describe_forest(
            forest_final, m_n_final, mcf_.var_cfg, cf_cfg, gen_cfg, pen_mult)
    else:
        report = None

    # x_name: List. Order of x_name as used by tree building
    return forest_final, x_name, report


@ray.remote
def ray_build_tree_mcf(data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                       x_type, x_values, x_ind, x_ai_ind, gen_cfg, cf_cfg,
                       ct_cfg, boot, pen_mult, cuda, cython, bigdata_train,
                       seedseq=None, zero_tol=1e-15,
                       ):
    """Prepare function for Ray."""
    return build_tree_mcf(data_np, y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i,
                          x_type, x_values, x_ind, x_ai_ind, gen_cfg, cf_cfg,
                          ct_cfg, boot, pen_mult, cuda, cython, bigdata_train,
                          seedseq=seedseq, zero_tol=zero_tol,
                          )


def build_tree_mcf(data_np: NDArray[np.floating],
                   y_i: int, y_nn_i: NDArray[np.intp],
                   x_i: NDArray[np.intp],
                   d_i: int, d_grid_i: int | None,
                   cl_i: int | None, w_i: int | None,
                   x_type: NDArray[np.int8],
                   x_values: list[list],
                   x_ind: NDArray[np.intp],
                   x_ai_ind: NDArray[np.intp] | list[int],
                   gen_cfg: Any, cf_cfg: Any, ct_cfg: Any,
                   boot: int, pen_mult: float | np.floating,
                   cuda: bool, cython: bool, bigdata_train: bool = False,
                   seedseq: np.random.SeedSequence | int | None = None,
                   zero_tol: float = 1e-15,
                   ) -> list[dict]:
    """Build single trees for all values of tuning parameters.

    Parameters
    ----------
    data_np : Numpy 2D array. Data.
    y_i : Position of Outcome in DATA.
    y_nn_i: Position of Matched outcomes.
    x_i : Position of Covariates.
    d_i : Position of Treatment.
    d_grid_i: Position of discretized treatments (continuous case)
    cl_i : Position of Cluster variable.
    x_type : List of INT. Type of variable: 0,1,2
    x_values: List of lists. Values of variable (if few or categorical)
    x_ind : List of INT. Identifier of variables
    x_ai_ind : List of INT. 1 if variable is included in every split
    ...dict : Dict. Control parameters
    boot : INT. Counter for bootstrap replication (currently not used)
    ....
    cuda : Boolean. If True, use CUDA for MSE calculations (if available).
    cython : Boolean. Use Cython for speed gains.

    Returns
    -------
    tree_all : Dict (m_grid x N_min_grid x alpha_grid) with trees for all
               values of tuning parameters
    """
    # split data into OOB and tree data
    n_obs = data_np.shape[0]
    # Random number initialisation. This seeds rnd generator within process.
    if seedseq is None:
        rng = np.random.default_rng(seed=(10+boot)**2+121)
    else:
        rng = np.random.default_rng(seed=seedseq)
    if gen_cfg.panel_in_rf:
        cl_unique = np.unique(data_np[:, cl_i])
        n_cl = cl_unique.shape[0]
        n_train = round(n_cl * cf_cfg.subsample_share_forest)
        indices_cl = list(rng.choice(n_cl, size=n_train, replace=False))
        indices = [i for i in range(n_obs) if data_np[i, cl_i] in indices_cl]
    else:
        n_train = round(n_obs * cf_cfg.subsample_share_forest)
        indices = list(rng.choice(n_obs, size=n_train, replace=False))
    indices_oob = np.delete(np.arange(n_obs), indices, axis=0)
    tree_empty = mcf_fo_asdict.make_default_tree_dict(
        np.min(cf_cfg.n_min_values), len(x_i), indices, indices_oob,
        bigdata_train=bigdata_train)
    # build trees for all m,n combinations
    grid_for_m = mcf_gp.check_if_iterable(cf_cfg.m_values)
    grid_for_n_min = mcf_gp.check_if_iterable(cf_cfg.n_min_values)
    grid_for_alpha_reg = mcf_gp.check_if_iterable(cf_cfg.alpha_reg_values)
    tree_all = [None for _ in range(len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg))]
    j = 0
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                tree_all[j] = build_single_tree(
                    data_np, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, cf_cfg, gen_cfg,
                    ct_cfg.grid_nn_val, m_idx, n_min, alpha_reg,
                    deepcopy(tree_empty), pen_mult, rng, cuda, cython,
                    zero_tol=zero_tol,
                    )
                j += 1

    return deepcopy(tree_all)


def build_single_tree(data: NDArray[np.float64],
                      y_i: int,
                      y_nn_i: NDArray[np.intp],
                      d_i: int,
                      d_grid_i: int | None,
                      x_i: NDArray[np.intp],
                      w_i: int | None,
                      x_type: NDArray[np.int8],
                      x_values: list[list[float | int |
                                          np.floating | np.integer]],
                      x_ind: NDArray[np.intp],
                      x_ai_ind: NDArray[np.intp] | list[int],
                      cf_cfg: Any,
                      gen_cfg: Any,
                      ct_grid_nn_val: list,
                      mmm: int,
                      n_min: int,
                      alpha_reg: float,
                      tree_dict_global: dict,
                      pen_mult: float,
                      rng: np.random.Generator,
                      cuda: bool,
                      cython: bool,
                      zero_tol: float = 1e-15,
                      ) -> dict:
    """Build single tree given random sample split.

    Parameters
    ----------
    data : Nympy array. Training data
    data_oob : Numpy array. OOB data
    y_i : Int. Position of y in numpy array.
    y_nn_i : List. Position of y_nn in numpy array.
    d_i : Int. Position of d in numpy array.
    d_i_grid: Int. Position of d_grid in numpy array.
    x_i : List. Position of x in numpy array.
    x_type : NDArray[np.int8]. Type of covariate (0,1,2).
    x_values: List of lists. Values of covariate (if not too many)
    x_ind : List. Postion of covariate in x for easy reference.
    x_ai_ind : List. Postion of covariate always-in in x for easy reference.
    cf_cfg : CfCfg dataclass. Parameters.
    gen_cfg : GenCfg dataclass. Parameters.
    m : INT. Number of covariates to be included.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. alpha regularity.
    tree_dict : Dict. Initial tree.
    pen_mult : Float. Multiplier of penalty.
    rng : Default random number generator object.
    cuda : Boolean. If True, use CUDA for MSE calculations (if available).
    cython : Boolean. Use Cython for speed gains.

    Returns
    -------
    tree_dict : Dictionary. Final tree.
    """
    # leaf_info_int[:, 0] = leaf_ids
    # leaf_info_int 0: ID of leaf
    #               1: ID of parent (or -1 for root)
    #               2: ID of left daughter (values <=, cats of values
    #                                                  included in Prime)
    #               3: ID of right daughter (values >, cats of values
    #                                                  not included in Prime)
    #               4: Index of splitting variable to go to daughter
    #               5: Type of splitting variable to go to daughter
    #                   (Ordered or categorical)
    #               6: 1: Terminal leaf. 0: To be split again.
    #               7: 2: Active leaf (to be split again); 0: Already split
    #                  1: Terminal leaf.
    #               8: Leaf size of training data in leaf
    #               9: Leaf size of OOB data in leaf
    # leaf_info_float = -np.ones((leaves_max, max_cols_float),dtype=np.float64)
    # leaf_info_float[:, 0] = leaf_ids
    # leaf_info_float 0: ID of leaf
    #                 1: Cut-of value of ordered variables
    #                 2: OOB value of leaf
    # causal_tree_empty_dic = {
    # 'leaf_info_int': leaf_info_int, 2dnarray with leaf info, integers
    # 'leaf_info_float': leaf_info_float, 2dnarray, with leaf info for floats
    # 'cats_prime': cats_prime, list, prime value for categorical features
    # 'oob_indices': indices_oob, 1D ndarray, indices of tree-specific OOB data
    # 'train_data_list': indices_train,
    #          Indices of data needed during tree building, will be removed use
    # 'oob_data_list': indices_oob,
    # Indices of oob data needed during tree building, will be removed afteruse
    # 'fill_y_indices_list': None,
    #      list (dim: # of leaves) of leaf-specific indices (to be filled after
    #      forest building) - for terminal leaves only

    tree_dict = deepcopy(tree_dict_global)
    leaf_ids = tree_dict['leaf_info_int'][:, 0]
    availabe_leaf_id_daughters = tree_dict['leaf_info_int'][1:, 0]
    for leaf_idx, idx in enumerate(leaf_ids):
        if leaf_idx != idx:
            raise ValueError('ID in Tree differes from position of leaf in '
                             'tree. Something went totally wrong.')
        if tree_dict['leaf_info_int'][leaf_idx, 7] != 2:  # Leaf not to split
            continue

        data_leaf = data[tree_dict['train_data_list'][leaf_idx], :]
        data_oob_leaf = data[tree_dict['oob_data_list'][leaf_idx], :]

        (terminal, split_var_i, split_type, split_n_l, split_n_r, split_leaf_l,
         split_leaf_r, split_leaf_oob_l, split_leaf_oob_r, split_n_oob_l,
         split_n_oob_r, split_value) = next_split(
             data_leaf, data_oob_leaf, y_i, y_nn_i, d_i, d_grid_i, x_i, w_i,
             x_type, x_values, x_ind, x_ai_ind, cf_cfg, gen_cfg,
             ct_grid_nn_val, mmm, n_min, alpha_reg, pen_mult, rng,
             cuda, cython, zero_tol=zero_tol,
             )
        if terminal:
            leaf_id_daughters = None
        else:
            leaf_id_daughters = availabe_leaf_id_daughters[0:2]
            availabe_leaf_id_daughters = availabe_leaf_id_daughters[2:].copy()
            if leaf_id_daughters is None or len(leaf_id_daughters) < 2:
                raise RuntimeError('Not enough daughter leaves available for '
                                   'further splitting: '
                                   f'\n{leaf_id_daughters=:}'
                                   f'\n{availabe_leaf_id_daughters=:}'
                                   f'\n{leaf_ids=:}'
                                   f'\n{leaf_idx=:}'
                                   )
        tree_dict = update_tree(
             data_oob_leaf, tree_dict, leaf_idx,
             split_var_i, split_type, split_value,
             split_n_l, split_n_r, split_leaf_l, split_leaf_r,
             split_leaf_oob_l, split_leaf_oob_r, split_n_oob_l, split_n_oob_r,
             terminal, leaf_id_daughters, d_i, w_i, d_grid_i, y_nn_i, y_i,
             ct_grid_nn_val, gen_cfg, cf_cfg, rng, cuda, cython,
             zero_tol=zero_tol,
             )
    tree_dict = mcf_fo_asdict.cut_back_empty_cells_tree(tree_dict)

    return tree_dict


def best_m_n_min_alpha_reg(forest: list[dict],
                           gen_cfg: Any,
                           cf_cfg: Any,
                           cython: bool
                           ) -> tuple[list, list]:
    """Get best forest for the tuning parameters m_try, n_min, alpha_reg.

    Parameters
    ----------
    forest : List of dictionaries. Estimated forests.
    gen_cfg : GenCfg dataclass. Parameters.
    c : Dict. Parameters.

    Returns
    -------
    forest_final : List of dictionaries. OOB-optimal forest.
    m_n_final : List. Optimal values of m and n_min.

    """
    grid_for_m = mcf_gp.check_if_iterable(cf_cfg.m_values)
    grid_for_n_min = mcf_gp.check_if_iterable(cf_cfg.n_min_values)
    grid_for_alpha_reg = mcf_gp.check_if_iterable(cf_cfg.alpha_reg_values)
    m_n_min_ar_combi = []
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                m_n_min_ar_combi.append([m_idx, n_min, alpha_reg])
    dim_m_n_min_ar = len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg)
    if (dim_m_n_min_ar) > 1:       # Find best of trees
        mse_oob = np.zeros(dim_m_n_min_ar)
        trees_without_oob = np.zeros(dim_m_n_min_ar)
        if gen_cfg.d_type == 'continuous':
            no_of_treat = 2
        else:
            no_of_treat = gen_cfg.no_of_treat
        for trees_m_n_min_ar in forest:                  # different forests
            for j, tree in enumerate(trees_m_n_min_ar):  # trees within forest
                n_lost = n_total = 0
                if no_of_treat is not None:
                    mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
                    obs_t_tree = np.zeros(no_of_treat)
                tree_mse = 0
                for leaf_no, leaf_int in enumerate(tree['leaf_info_int']):
                    leaf_float_2 = tree['leaf_info_float'][leaf_no, 2]
                    leaf_float_2_dim1 = isinstance(
                        leaf_float_2, (float, np.float32, np.float64))
                    if leaf_int[7] == 1:  # Terminal leafs
                        n_total += np.sum(leaf_int[9])
                        if leaf_float_2 is None or leaf_float_2 <= 0:
                            if no_of_treat is None or leaf_float_2_dim1:
                                n_lost += leaf_int[9]
                            else:
                                n_lost += np.sum(leaf_int[9])
                        else:
                            if no_of_treat is None or leaf_float_2_dim1:
                                tree_mse += (leaf_int[9] * leaf_float_2)
                            else:
                                mse_mce_tree, obs_t_tree = (
                                    mcf_fo_obj.add_rescale_mse_mce(
                                        leaf_float_2, leaf_int[9],
                                        cf_cfg.mtot, no_of_treat,
                                        mse_mce_tree, obs_t_tree,
                                        cf_cfg.compare_only_to_zero)
                                    )
                if n_lost > 0:
                    if no_of_treat is None or leaf_float_2_dim1:
                        tree_mse = tree_mse * n_total / (n_total - n_lost)
                    else:
                        if (n_total - n_lost) < 1:
                            trees_without_oob[j] += 1
                if no_of_treat is not None and not leaf_float_2_dim1:
                    mse_mce_tree = mcf_fo_obj.get_avg_mse_mce(
                        mse_mce_tree, obs_t_tree, cf_cfg.mtot, no_of_treat,
                        cf_cfg.compare_only_to_zero
                        )
                    if cython:
                        raise ValueError('Cython currently not used.')
                        # tree_mse = mcf_cy.compute_mse_mce_cy(
                        #     mse_mce_tree, cf_cfg.mtot, no_of_treat,
                        #     cf_cfg.compare_only_to_zero)

                    tree_mse = mcf_fo_obj.compute_mse_mce(
                        mse_mce_tree, cf_cfg.mtot, no_of_treat,
                        cf_cfg.compare_only_to_zero)
                mse_oob[j] += tree_mse     # Add MSE to MSE of forest j
        if np.any(trees_without_oob) > 0:
            for j, trees_without_oob_j in enumerate(trees_without_oob):
                if trees_without_oob_j > 0:
                    mse_oob[j] = mse_oob[j] * (
                        cf_cfg.boot / (cf_cfg.boot - trees_without_oob_j)
                        )
        # Change value of mse to infinity if OOB-based mse could not be computed
        mse_oob = np.nan_to_num(mse_oob, np.inf)
        mse_oob[mse_oob < 0.01 * np.mean(mse_oob)] = np.inf
        min_i = np.argmin(mse_oob)
        mse_oob = mse_oob / cf_cfg.boot
        cf_cfg.n_min_values = mcf_gp.check_if_iterable(cf_cfg.n_min_values)
        cf_cfg.m_values = mcf_gp.check_if_iterable(cf_cfg.m_values)
        cf_cfg.alpha_reg_values = mcf_gp.check_if_iterable(
            cf_cfg.alpha_reg_values)
        if gen_cfg.with_output:
            txt = '\n' * 2 + '-' * 100
            txt += ('\nOOB MSE (without penalty) for M_try, minimum leafsize'
                    ' and alpha_reg combinations'
                    '\nNumber of vars / min. leaf size / alpha reg. / '
                    'OOB value. Trees without OOB\n')
            j = 0
            for m_idx in cf_cfg.m_values:
                for n_min in cf_cfg.n_min_values:
                    for alpha_reg in cf_cfg.alpha_reg_values:
                        txt += (f'\n{m_idx:>12} {n_min:>12}'
                                f' {alpha_reg:15.3f}'
                                f' {mse_oob[j]:8.3f}'
                                f' {trees_without_oob[j]:4.0f}')
                        j += 1
            txt += (f'\nMinimum OOB MSE:     {mse_oob[min_i]:7.3f}'
                    f'\nNumber of variables: {m_n_min_ar_combi[min_i][0]}'
                    f'\nMinimum leafsize:    {m_n_min_ar_combi[min_i][1]}'
                    f'\nAlpha regularity:    {m_n_min_ar_combi[min_i][2]}')
            txt += '\n' + '-' * 100
            mcf_ps.print_mcf(gen_cfg, txt, summary=True)
        forest_final = [trees_m_n_min[min_i] for trees_m_n_min in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[min_i]
    else:       # Find best of trees
        forest_final = [trees_m_n_min_ar[0] for trees_m_n_min_ar in forest]
        m_n_min_ar_opt = m_n_min_ar_combi[0]

    return forest_final, m_n_min_ar_opt


def efficient_iate(mcf_: 'ModifiedCausalForest',
                   fill_y_df: DataFrame,
                   tree_df: DataFrame,
                   summary: bool = False
                   ) -> tuple[DataFrame, DataFrame]:
    """Get more efficient iates (switch role of samples)."""
    if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
        mcf_ps.print_mcf(mcf_.gen_cfg,
                         '\nSecond round of estimation to get more efficient '
                         'effects',
                         summary=summary)
    # Switch the role of the subsamples
    fill_y_df, tree_df = tree_df, fill_y_df

    return tree_df, fill_y_df


def next_split(data_train: NDArray[np.floating],
               data_oob: NDArray[np.floating],
               y_i: int,
               y_nn_i: NDArray[np.intp],
               d_i: int,
               d_grid_i: int | None,
               x_i: NDArray[np.intp],
               w_i: int | None,
               x_type: NDArray[np.intp],
               x_values: list[list[float]],
               x_ind: NDArray[np.intp],
               x_ai_ind: NDArray[np.intp] | list[int],
               cf_cfg: Any,
               gen_cfg: Any,
               ct_grid_nn_val: dict | None,
               mmm: int,
               n_min: int,
               alpha_reg: float,
               pen_mult: np.floating,
               rng: np.random.Generator,
               cuda: bool,
               cython: bool,
               zero_tol: float = 1e-15,
               ) -> tuple[bool,
                          int | None,
                          int | None,
                          int | None,
                          int | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          int | None,
                          int | None,
                          float | list[int] | None,
                          ]:
    """Find best next split of leaf (or terminate splitting for this leaf).

    Parameters
    ----------
    data_train : Numpy array. Training data of leaf.
    data_oob : Numpy array: OOB data of leaf.
    y_i : int. Location of Y in data matrix.
    y_nn_i :  Numpy array. Location of Y_NN in data matrix.
    d_i : INT. Location of D in data matrix.
    d_grid_i : INT. Location of D_grid in data matrix.
    x_i : Numpy array. Location of X in data matrix.
    x_type : Numpy array.(0,1,2). Type of X.
    x_ind : Numpy array. Location of X in X matrix.
    x_ai_ind : Numpy array. Location of X_always in X matrix.
    ... : DICT. Parameters.
    mmm : INT. Number of X-variables to choose for splitting.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. Alpha regularity.
    pen_mult : Float. Penalty multiplier.
    rng : Numpy default random number generator object.
    cuda : Boolean. Use cuda if True.
    cython : Boolean. Use cython for faster usage.

    Returns
    -------
    left : List of lists. Information about left leaf.
    right : List of lists. Information about right leaf.
    current : List of lists. Updated information about this leaf.
    terminal : INT. 1: No splits for this leaf. 0: Leaf splitted
    ...
    """
    # cache config & funcs once
    compare_only_to_zero = cf_cfg.compare_only_to_zero
    n_min_treat = cf_cfg.n_min_treat
    weighted = gen_cfg.weighted
    penalty_type = cf_cfg.penalty_type
    mtot = cf_cfg.mtot
    random_thresholds = cf_cfg.random_thresholds

    count_nonzero = np.count_nonzero
    mcf_mse_fn = mcf_fo_obj.mcf_mse
    add_mse_mce_split_fn = mcf_fo_obj.add_mse_mce_split
    compute_mse_mce_fn = mcf_fo_obj.compute_mse_mce
    mcf_penalty_fn = mcf_fo_obj.mcf_penalty
    not_enough_treated_fn = mcf_fo_add.not_enough_treated
    match_cont_fn = mcf_fo_add.match_cont

    terminal = split_done = False
    leaf_size_train = data_train.shape[0]
    leaf_size_oob = data_oob.shape[0]

    pen_mult_d = pen_mult if (penalty_type == 'mse_d' and pen_mult > 0) else 0
    if leaf_size_train < (2 * n_min):
        terminal = True
    elif np.all(data_train[:, d_i] == data_train[0, d_i]):
        terminal = True
    else:
        if leaf_size_train < 200:  # Otherwise, too slow:
            if gen_cfg.d_type == 'continuous':
                terminal = not (2 <= np.sum(data_train[:, d_i] == 0)
                                <= leaf_size_train - 2
                                )
            else:
                ret = np.unique(data_train[:, d_i], return_counts=True)
                terminal = (len(ret[0]) < gen_cfg.no_of_treat
                            or np.any(ret[1] < 2 * n_min_treat)
                            )
    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
        d_split_in_x_ind = np.max(x_ind) + 1
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        d_bin_dat, continuous, d_split_in_x_ind = None, False, None
    if not terminal:
        obs_min = max(round(leaf_size_train * alpha_reg), n_min)
        best_mse = np.inf
        total_attempts = 2   # New in version 0.9.0; it was 5 before
        all_vars = False
        x_already_used: set[int] = set()   # For faster look-up below
        # Find valid splitting values, try again if failed
        for idx in range(total_attempts):
            if idx == 0:
                x_ind_idx = x_ind.copy()
            if idx == total_attempts - 1:  # last iteration
                all_vars = True
            x_ind_split = mcf_fo_add.rnd_variable_for_split(
                x_ind_idx, x_ai_ind, cf_cfg, mmm, rng, all_vars=all_vars,
                first_attempt=idx == 0
                )
            x_type_split = x_type[x_ind_split].copy()
            x_values_split = [x_values[v_idx].copy() for v_idx in x_ind_split]
            # Check if split is possible ... sequential order to minimize costs
            # Check if enough variation in the data to do splitting (costly)
            with_d_oob = continuous
            (y_dat, _, d_dat, d_oob, d_grid_dat, _, x_dat, x_oob,
             terminal, terminal_x, x_no_varia) = term_or_data(
                 data_train, data_oob, y_i, d_i, d_grid_i,
                 x_i[x_ind_split], no_of_treat, with_d_oob=with_d_oob,
                 cont_treat=continuous, zero_tol=zero_tol,
                 )
            if terminal:  # No variation in d or/and y, game over.
                break
            if not terminal_x:  # Variation in x, find splits
                break

            x_already_used.update(x_ind_split)

            if len(x_already_used) >= len(x_ind):
                break

            x_ind_idx = [ind for ind in x_ind_idx if ind not in x_already_used]

        if terminal_x:
            terminal = True  # No variation in X. Splitting stops.

    best_var_i = best_type = best_n_l = best_n_r = best_leaf_l = None
    best_leaf_r = best_leaf_oob_l = best_leaf_oob_r = best_n_oob_l = None
    best_n_oob_r = best_value = None
    if not terminal:
        # Added Oct., 22, 2025
        best = {'j': None, 'type': None, 'value': None, 'mse': np.inf,
                'd_cont_split': False,
                'n_l': 0, 'n_r': 0, 'n_oob_l': 0, 'n_oob_r': 0,
                # for continuous-D only:
                'zeros_l': None, 'zeros_l_oob': None
                }
        if mtot in (1, 4):
            y_nn = data_train[:, y_nn_i]
        else:
            y_nn = y_nn_l = y_nn_r = 0

        if weighted:
            if w_i is None:
                raise ValueError('weighted=True but w_i is None')
            w_dat = data_train[:, [w_i]]
        else:
            w_dat = [1]

        if continuous:
            d_bin_dat = d_dat > zero_tol   # Binary treatment indicator
            x_no_varia.append(np.all(d_bin_dat == d_bin_dat[0]))
            x_ind_split.append(d_split_in_x_ind)
            x_type_split = np.append(x_type_split, np.int8(0))
        p_x = len(x_ind_split)  # indices refer to order of x in data_*
        d_cont_split = False
        # x_dat_copy, x_oob_copy = np.copy(x_dat), np.copy(x_oob) NOT NEEDED
        for j in range(p_x):  # Loops over the variables
            if not x_no_varia[j]:  # No variation of this x -> no split
                d_cont_split = continuous and (j == p_x - 1)
                if d_cont_split:
                    x_j, x_oob_j = d_dat, d_oob
                    x_j_pos = x_j[x_j > zero_tol]  # Positive treatment values
                    nr_pos, nr_all = x_j_pos.size,  x_j.size
                    nr_0 = nr_all - nr_pos
                    nr_all_oob = x_oob_j.shape[0]
                    if nr_0 < 2 or nr_pos < 2:  # Too few controls
                        continue
                    split_values = np.unique(x_j_pos).tolist()
                    if len(split_values) > 1:
                        split_values = split_values[:-1]  # max not included

                    # Randomly allocate half the controls to left leaf
                    # rnd_in = rng.choice([True, False], size=(nr_all, 1))
                    # # Somewhat inefficient as it is also applied to treated
                    # treat_0 = (x_j - zero_tol) <= 0
                    # zeros_l = treat_0 & rnd_in
                    # rnd_in_oob = rng.choice([True, False], size=(nr_all_oob,
                    #                                              1))
                    # # Somewhat inefficient as it is also applied to treated
                    # treat_0_oob = (x_oob_j - zero_tol) <= 0
                    # zeros_l_oob = treat_0_oob & rnd_in_oob
                    # 1-D masks; no .flatten() needed later
                    treat_0 = x_j <= zero_tol
                    rnd_in = rng.random(nr_all) < 0.5
                    zeros_l = treat_0 & rnd_in

                    treat_0_oob = x_oob_j <= zero_tol
                    rnd_in_oob = rng.random(nr_all_oob) < 0.5
                    zeros_l_oob = treat_0_oob & rnd_in_oob

                else:
                    x_j, x_oob_j = x_dat[:, j], x_oob[:, j]
                    if x_type_split[j] > 0:
                        x_j = x_j.astype(np.int32, copy=False)
                        x_oob_j = x_oob_j.astype(np.int32, copy=False)
                    split_values = get_split_values(
                        y_dat, w_dat,
                        x_j, x_type_split[j], x_values_split[j],
                        leaf_size_train,
                        random_thresholds, weighted,
                        rng=rng
                        )
                split_values_unord_j = []
                cat_mask, cat_mask_oob = None, None
                # Loops over values of variables
                for val in split_values:
                    if x_type_split[j] == 0:
                        val_eps = np.nextafter(val, np.inf)
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l = ~treat_0 & (x_j <= val_eps)
                            # leaf_l = (treated_l | zeros_l).flatten()
                            leaf_l = treated_l | zeros_l
                        else:
                            leaf_l = x_j <= val_eps
                            # because of float
                    else:                      # ordered with few vals.
                        # Categorial variable: Either in group or not
                        split_values_unord_j.append(val)
                        # leaf_l = np.isin(x_j, split_values_unord_j)
                        if cat_mask is None:
                            cat_mask = x_j == val
                        else:
                            cat_mask |= (x_j == val)

                        leaf_l = cat_mask

                    n_l = count_nonzero(leaf_l)
                    n_r = leaf_size_train - n_l
                    # Check if enough observations available
                    if (n_l < obs_min) or (n_r < obs_min):
                        continue
                    # Next we check if any obs in each treatment
                    d_dat_l = (d_bin_dat[leaf_l]
                               if continuous else d_dat[leaf_l])
                    if not_enough_treated_fn(continuous, n_min_treat, d_dat_l,
                                             no_of_treat
                                             ):
                        continue
                    leaf_r = ~leaf_l  # Reverses True to False
                    d_dat_r = (d_bin_dat[leaf_r]
                               if continuous else d_dat[leaf_r])
                    if not_enough_treated_fn(continuous, n_min_treat, d_dat_r,
                                             no_of_treat
                                             ):
                        continue

                    if x_type_split[j] == 0:
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l_oob = ~treat_0_oob & (x_oob_j <= val_eps)
                            # leaf_oob_l = (treated_l_oob | zeros_l_oob
                            #               ).flatten()
                            leaf_oob_l = treated_l_oob | zeros_l_oob
                        else:
                            leaf_oob_l = x_oob_j <= val_eps
                    else:
                        # leaf_oob_l = np.isin(x_oob_j, split_values_unord_j)
                        if cat_mask_oob is None:
                            cat_mask_oob = x_oob_j == val
                        else:
                            cat_mask_oob |= (x_oob_j == val)

                        leaf_oob_l = cat_mask_oob

                    n_oob_l = count_nonzero(leaf_oob_l)
                    n_oob_r = leaf_size_oob - n_oob_l

                    # leaf_oob_r = ~leaf_oob_l
                    if mtot in (1, 4):
                        if continuous:
                            y_nn_l = match_cont_fn(d_grid_dat[leaf_l],
                                                   y_nn[leaf_l, :],
                                                   ct_grid_nn_val, rng
                                                   )
                            y_nn_r = match_cont_fn(d_grid_dat[leaf_r],
                                                   y_nn[leaf_r, :],
                                                   ct_grid_nn_val, rng
                                                   )
                        else:
                            y_nn_l, y_nn_r = y_nn[leaf_l, :], y_nn[leaf_r, :]
                    else:
                        y_nn_l = y_nn_r = 0
                    if weighted:
                        w_l, w_r = w_dat[leaf_l], w_dat[leaf_r]
                    else:
                        w_l = w_r = 0
                    # compute objective functions given particular method
                    mse_mce_l, shares_l, obs_by_treat_l = mcf_mse_fn(
                        y_dat[leaf_l], y_nn_l, d_dat_l, w_l, n_l, mtot,
                        no_of_treat, d_values, weighted, False, cuda, cython,
                        compare_only_to_zero, pen_mult_d
                        )
                    mse_mce_r, shares_r, obs_by_treat_r = mcf_mse_fn(
                        y_dat[leaf_r], y_nn_r, d_dat_r, w_r, n_r, mtot,
                        no_of_treat, d_values, weighted, False, cuda, cython,
                        compare_only_to_zero, pen_mult_d
                        )
                    mse_mce = add_mse_mce_split_fn(
                        mse_mce_l, mse_mce_r, obs_by_treat_l,
                        obs_by_treat_r, mtot, no_of_treat,
                        compare_only_to_zero
                        )
                    if cython:
                        raise ValueError('Cython currently not used.')
                        # mse_split = mcf_cy.compute_mse_mce_cy(
                        #     mse_mce, mtot, no_of_treat,
                        #     compare_only_to_zero)

                    mse_split = compute_mse_mce_fn(mse_mce, mtot, no_of_treat,
                                                   compare_only_to_zero
                                                   )
                    # add penalty for this split if 'diff_d'
                    if (penalty_type == 'diff_d'
                        and ((mtot == 1)
                             or ((mtot == 4) and (rng.random() > 0.5))
                             )):
                        penalty = mcf_penalty_fn(shares_l, shares_r)
                        mse_split = mse_split + pen_mult * penalty

                    if mse_split < best_mse:
                        split_done = True
                        best_mse = mse_split
                        # changed Oct., 22, 2025 to reduce copying
                        best['j'] = int(j)
                        best['type'] = int(x_type_split[j])
                        best['d_cont_split'] = bool(d_cont_split)
                        best['n_l'], best['n_r'] = n_l, n_r
                        best['n_oob_l'], best['n_oob_r'] = n_oob_l, n_oob_r
                        if x_type_split[j] == 0:
                            best['value'] = float(val)
                            if d_cont_split:
                                # persist control assignments;
                                # they are part of the split definition
                                best['zeros_l'] = zeros_l.copy()
                                best['zeros_l_oob'] = zeros_l_oob.copy()
                        else:
                            # keep the current category set without
                            # recomputing from scratch
                            best['value'] = split_values_unord_j[:]
                            # list of included categories

        if not split_done:
            return (True, None, None, None, None, None, None, None, None, None,
                    None, None
                    )
        j = best['j']
        if best['type'] == 0:
            val_eps = np.nextafter(best['value'], np.inf)
            if best['d_cont_split']:
                x_j = d_dat
                x_oob_j = d_oob
                treated_l = (~(x_j <= zero_tol)) & (x_j <= val_eps)
                leaf_l = treated_l | best['zeros_l']
                treated_l_oob = (~(x_oob_j <= zero_tol)) & (x_oob_j <= val_eps)
                leaf_oob_l = treated_l_oob | best['zeros_l_oob']
            else:
                x_j = x_dat[:, j]
                x_oob_j = x_oob[:, j]
                leaf_l = x_j <= val_eps
                leaf_oob_l = x_oob_j <= val_eps
        else:
            cats = best['value']
            x_j = x_dat[:, j]
            x_oob_j = x_oob[:, j]
            leaf_l = np.isin(x_j, cats)
            leaf_oob_l = np.isin(x_oob_j, cats)

        leaf_r = ~leaf_l
        leaf_oob_r = ~leaf_oob_l

        # assign to variables to be returned
        best_var_i = x_ind_split[j]
        best_type = best['type']
        best_n_l = best['n_l']
        best_n_r = best['n_r']
        best_leaf_l = leaf_l
        best_leaf_r = leaf_r
        best_leaf_oob_l = leaf_oob_l
        best_leaf_oob_r = leaf_oob_r
        best_n_oob_l = best['n_oob_l']
        best_n_oob_r = best['n_oob_r']
        best_value = best['value']

    return (terminal, best_var_i, best_type, best_n_l, best_n_r,
            best_leaf_l, best_leaf_r, best_leaf_oob_l, best_leaf_oob_r,
            best_n_oob_l, best_n_oob_r, best_value
            )


def update_tree(
        data_oob_parent: NDArray[np.float64],
        tree_dict_global: dict,
        parent_idx: int,
        split_var_i: int | None,
        split_type: int | None,
        split_value: int | float | np.floating | np.integer | None,
        split_n_l: int | None,
        split_n_r: int | None,
        split_leaf_l: NDArray[bool] | None,
        split_leaf_r: NDArray[bool] | None,
        split_leaf_oob_l: NDArray[bool] | None,
        split_leaf_oob_r: NDArray[bool] | None,
        split_n_oob_l: int | None,
        split_n_oob_r: int | None,
        terminal: bool,
        leaf_id_daughters: NDArray[np.integer] | None,
        d_i: int,
        w_i: int | None,
        d_grid_i: int | None,
        y_nn_i: NDArray[np.intp],
        y_i: int,
        ct_grid_nn_val: list[Any] | None,
        gen_cfg: Any,
        cf_cfg: Any,
        rng: np.random.Generator,
        cuda: bool,
        cython: bool,
        zero_tol: float = 1e-15,
        ) -> dict:
    """Assign values obtained from splitting to parent & daughter leaves."""
    tree = deepcopy(tree_dict_global)
    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        continuous = False
    if terminal:
        tree['leaf_info_int'][parent_idx, 7] = 1  # Terminal
        tree['leaf_info_int'][parent_idx, 6] = 1  # Terminal
        w_oob = data_oob_parent[:, [w_i]] if gen_cfg.weighted else 0
        n_oob = data_oob_parent.shape[0]
        if continuous:
            d_oob = data_oob_parent[:, d_i] > zero_tol
        else:
            d_oob = data_oob_parent[:, d_i]
        if len(np.unique(d_oob)) < no_of_treat:
            obj_fct_oob, obs_oob = 0, n_oob      # MSE cannot be computed
        else:
            if continuous:
                y_nn = mcf_fo_add.match_cont(data_oob_parent[:, d_grid_i],
                                             data_oob_parent[:, y_nn_i],
                                             ct_grid_nn_val, rng)
            else:
                y_nn = data_oob_parent[:, y_nn_i]
            mse_mce, _, obs_oob_list = mcf_fo_obj.mcf_mse(
                data_oob_parent[:, y_i], y_nn, d_oob, w_oob, n_oob,
                cf_cfg.mtot, no_of_treat, d_values, gen_cfg.weighted,
                False, cuda, cython, cf_cfg.compare_only_to_zero)
            if cython:
                raise ValueError('Cython currently not used.')
                # obj_fct_oob = mcf_cy.compute_mse_mce_cy(
                #     mse_mce, cf_cfg.mtot, no_of_treat,
                #     cf_cfg.compare_only_to_zero)

            obj_fct_oob = mcf_fo_obj.compute_mse_mce(
                mse_mce, cf_cfg.mtot, no_of_treat,
                cf_cfg.compare_only_to_zero)
            obs_oob = np.sum(obs_oob_list)
        tree['leaf_info_float'][parent_idx, 2] = obj_fct_oob
        tree['leaf_info_int'][parent_idx, 9] = obs_oob
        tree['train_data_list'][parent_idx] = None
        if not cf_cfg.vi_oob_yes:
            tree['train_data_list'][parent_idx] = None
    else:
        train_list = np.array(tree['train_data_list'][parent_idx], copy=True)
        oob_list = np.array(tree['oob_data_list'][parent_idx], copy=True)
        # Change information in parent leave
        #    Assign IDs
        tree['leaf_info_int'][parent_idx, 2:4] = leaf_id_daughters
        #    Assign status and splitting information
        tree['leaf_info_int'][parent_idx, 7] = 0
        #                             not active, not terminal - intermediate
        tree['leaf_info_int'][parent_idx, 4] = split_var_i
        if split_type > 0:  # Save as product of primes
            tree['cats_prime'][parent_idx] = mcf_gp.list_product(split_value)
        else:
            tree['leaf_info_float'][parent_idx, 1] = split_value
        tree['leaf_info_int'][parent_idx, 5] = split_type

        # Initialise daughter leaves
        id_l, id_r = leaf_id_daughters
        # Assign IDs
        tree['leaf_info_int'][id_l, 1] = tree['leaf_info_int'][parent_idx, 0]
        tree['leaf_info_int'][id_r, 1] = tree['leaf_info_int'][parent_idx, 0]
        tree['leaf_info_int'][id_l, 6] = tree['leaf_info_int'][id_r, 6] = 0
        tree['leaf_info_int'][id_l, 7] = tree['leaf_info_int'][id_r, 7] = 2
        tree['leaf_info_int'][id_l, 8] = split_n_l
        tree['leaf_info_int'][id_r, 8] = split_n_r
        tree['leaf_info_int'][id_l, 9] = split_n_oob_l
        tree['leaf_info_int'][id_r, 9] = split_n_oob_r

        tree['train_data_list'][id_l] = train_list[split_leaf_l]
        tree['train_data_list'][id_r] = train_list[split_leaf_r]
        tree['oob_data_list'][id_l] = oob_list[split_leaf_oob_l]
        tree['oob_data_list'][id_r] = oob_list[split_leaf_oob_r]

        tree['train_data_list'][parent_idx] = None
        tree['oob_data_list'][parent_idx] = None

    return tree


def get_split_values(y_dat: NDArray[np.float64 | np.intp],
                     w_dat: NDArray[np.float64 | np.intp] | None,
                     x_dat: NDArray[np.float64 | np.intp],
                     x_type: np.int8 | int,
                     x_values: list[float | int | np.floating | np.integer],
                     leaf_size: int,
                     random_thresholds: int,
                     w_yes: bool,
                     rng: np.random.Generator | None = None,
                     ) -> list:
    """Determine the values used for splitting.

    Parameters
    ----------
    y_dat : Numpy array. Outcome.
    x_dat : 1-d Numpy array. Splitting variable.
    w_dat : 1-d Numpy array. Weights.
    x_type : Int. Type of variables used for splitting.
    x_values: List.
    leaf_size. INT. Size of leaf.
    c_dict: Dict. Parameters
    rng : Random Number Generator object

    Returns
    -------
    splits : List. Splitting values to use.
    """
    if rng is None:
        rng = np.random.default_rng()
        print("Warning: unseeded random numbers used - impossible to replicate")

    if x_type == 0:   # Continuous variable
        if bool(x_values):  # Continuous var with limited number of values
            min_x, max_x = np.amin(x_dat), np.amax(x_dat)
            xv = np.asarray(x_values)
            mask = (xv >= min_x) & (xv <= max_x)
            splits_x = xv[mask].tolist()

            if len(splits_x) == 1:
                return splits_x

            if len(splits_x) == 0:  # Super unlikely to happen
                return x_values[:]

            # More than 1 splitting value
            splits_x = splits_x[:-1]
            if 0 < random_thresholds < len(splits_x):
                splits = np.unique(
                    rng.choice(splits_x, size=random_thresholds,
                               replace=False, shuffle=False))
            else:
                splits = splits_x

            return splits

        # Continoues variable with very many values; x_values empty
        if 0 < random_thresholds < (leaf_size - 1):
            x_vals_np = rng.choice(
                x_dat, size=random_thresholds, replace=False, shuffle=False
                )
            splits = np.unique(x_vals_np).tolist()
        else:
            splits = np.unique(x_dat).tolist()
            if len(splits) > 1:
                splits = splits[:-1]

        return splits

    # Discrete variable
    y_mean_by_cat = np.empty(len(x_values))  # x_vals comes as list
    x_vals_np = np.array(x_values, dtype=np.int32, copy=True)
    used_values = []
    for v_idx, val in enumerate(x_vals_np):
        value_equal = np.isclose(x_dat, val)
        if np.any(value_equal):  # Position of empty cells do not matter
            if w_yes:
                y_mean_by_cat[v_idx] = np.average(
                    y_dat[value_equal], weights=w_dat[value_equal])
            else:
                y_mean_by_cat[v_idx] = np.average(y_dat[value_equal])
            used_values.append(v_idx)
    x_vals_np = x_vals_np[used_values]
    sort_ind = np.argsort(y_mean_by_cat[used_values])
    x_vals_np = x_vals_np[sort_ind]
    splits = x_vals_np.tolist()
    splits = splits[:-1]  # Last category not needed

    return splits


def term_or_data(data_tr_ns: NDArray[np.float64],
                 data_oob_ns: NDArray[np.float64],
                 y_i: int,
                 d_i: int,
                 d_grid_i: int | list[int] | None,
                 x_i_ind_split: NDArray[np.intp],
                 no_of_treat: int,
                 with_d_oob: bool = True,
                 cont_treat: bool = False,
                 zero_tol: float = 1e-15,
                 ) -> tuple[NDArray[np.float64],         # y_dat
                            NDArray[np.float64] | None,  # y_oob
                            NDArray[np.float64] | None,  # d_dat
                            NDArray[np.float64] | None,  # d_oob
                            NDArray[Any] | None,         # d_grid_dat
                            NDArray[Any] | None,         # d_grid_oob
                            NDArray[np.float64] | None,  # x_dat
                            NDArray[np.float64] | None,  # x_oob
                            bool,                        # terminal
                            bool,                        # terminal_x
                            list[bool]                   # x_no_variation
                            ]:
    """Check if terminal leaf. If not, provide data.

    Parameters
    ----------
    data_tr_ns : Numpy array. Data used for splitting.
    data_oob_ns : Numpy array. OOB Data.
    y_i : INT. Index of y in data.
    d_i : INT. Index of d in data.
    d_grid_i : List of INT. Indices of d_grid in data.
    x_i_ind_split : List of INT. Ind. of x used for splitting. Pos. in data.
    no_of_treat: INT.

    Returns
    -------
    y_dat : Numpy array. Data.
    y_oob : Numpy array. OOB Data.
    d_dat : Numpy array. Data.
    d_oob : Numpy array. OOB Data.
    x_dat : Numpy array. Data.
    x_oob : Numpy array. OOB Data.
    terminal : Boolean. True if no further split possible. End splitting.
    terminal_x : Boolean. No variation in X. Try new variables.
    ...
    """
    y_dat = data_tr_ns[:, y_i]
    if np.allclose(y_dat, y_dat[0]):    # all elements are equal
        return (y_dat, None, None, None, None, None, None, None, True, False, []
                )
    d_grid_dat = None
    y_oob = data_oob_ns[:, y_i]
    d_dat = data_tr_ns[:, d_i]
    if d_grid_i is not None:
        d_grid_dat = data_tr_ns[:, d_grid_i]

    # terminal = len(np.unique(d_dat)) < no_of_treat   ... slow
    # counts = np.bincount(d_dat.ravel().astype(np.int64, copy=False),
    #                      minlength=no_of_treat
    #                      )
    # terminal = np.any(counts == 0)

    if cont_treat:
        # treat as binary indicator > 0 with a tiny tolerance
        d_bin = d_dat > zero_tol
        terminal = (np.all(d_bin) or np.all(~d_bin))
    else:
        # discrete labels 0..no_of_treat-1
        d_int = d_dat.astype(np.int64, copy=False)
        counts = np.bincount(d_int.ravel(), minlength=no_of_treat)
        terminal = np.any(counts[:no_of_treat] == 0)

    if terminal:
        return (y_dat, y_oob, d_dat, None, d_grid_dat, None, None, None,
                True, False, []
                )
    d_oob = d_grid_oob = x_oob = None
    x_no_variation = []
    if with_d_oob:
        d_oob = data_oob_ns[:, d_i]
        if d_grid_i is not None:
            d_grid_oob = data_oob_ns[:, d_grid_i]

    x_dat = data_tr_ns[:, x_i_ind_split]
    # x_no_variation = [np.allclose(x_dat[:, cols], x_dat[0, cols])
    #                   for cols, _ in enumerate(x_i_ind_split)
    #                   ]
    # if np.all(x_no_variation):
    #     terminal_x = True
    # else:
    #     x_oob = data_oob_ns[:, x_i_ind_split]
    # return (y_dat, y_oob, d_dat, d_oob, d_grid_dat, d_grid_oob, x_dat, x_oob,
    #         terminal, terminal_x, x_no_variation
    #         )

    # vectorized "all close to first row" per column
    # (works for float and int; defaults match np.allclose)
    x_first = x_dat[0, :]
    if np.issubdtype(x_dat.dtype, np.integer):
        x_no_variation_arr = np.all(x_dat == x_first, axis=0)
    else:
        x_no_variation_arr = np.all(np.isclose(x_dat, x_first,
                                               rtol=1e-05, atol=1e-08),
                                    axis=0
                                    )
    x_no_variation = x_no_variation_arr.tolist()

    terminal_x = bool(np.all(x_no_variation_arr))
    if not terminal_x:
        x_oob = data_oob_ns[:, x_i_ind_split]

    return (y_dat, y_oob, d_dat, d_oob, d_grid_dat, d_grid_oob, x_dat, x_oob,
            False, terminal_x, x_no_variation)


def oob_in_tree(obs_in_leaf,
                y_dat: NDArray[np.floating | np.integer],
                y_nn: NDArray[np.floating | np.integer],
                d_dat: NDArray[np.floating | np.integer],
                w_dat: NDArray[np.floating | np.integer] | None,
                mtot: int,
                no_of_treat: int,
                treat_values: list[int] | NDArray[np.floating | np.integer],
                w_yes: bool,
                cont: bool = False,
                cuda: bool = False,
                cython: bool = True,
                compare_only_to_zero: bool = False
                ) -> np.floating:
    """Compute OOB values for a tree.

    Parameters
    ----------
    obs_in_leaf : List of int. Terminal leaf no of observation
    y : Numpy array.
    y_nn : Numpy array.
    d : Numpy array.
    w : Numpy array.
    mtot : Integer. Method used.
    no_of_treat : Integer.
    treat_values : List of integer.
    w_yes : Boolean.
    cont : Boolean. Default is False.
    cuda : Boolean. MSE computation with Cuda if True. Default is False.
    # cython : Boolean. MSE computation with Cython if True. Default is True.
    compare_only_to_zero : Boolean. Reduced covariance matrix.

    Returns
    -------
    oob_tree : INT. OOB value of the MSE of the tree

    """
    leaf_no = np.unique(obs_in_leaf[:, 1])
    oob_tree = n_lost = n_total = 0
    mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
    obs_t_tree = np.zeros(no_of_treat)
    for leaf in leaf_no:
        in_leaf = obs_in_leaf[:, 1] == leaf
        w_l = w_dat[in_leaf] if w_yes else 0
        n_l = np.count_nonzero(in_leaf)
        d_dat_in_leaf = d_dat[in_leaf]  # makes a copy
        if n_l < no_of_treat:
            enough_data_in_leaf = False
        else:
            enough_data_in_leaf = True
            if n_l < 40:          # this is done for efficiency reasons
                if set(d_dat_in_leaf.reshape(-1)) < set(treat_values):
                    enough_data_in_leaf = False
            else:
                if len(np.unique(d_dat_in_leaf)) < no_of_treat:  # No MSE
                    enough_data_in_leaf = False
        if enough_data_in_leaf:
            mse_mce_leaf, _, obs_by_treat_leaf = mcf_fo_obj.mcf_mse(
                y_dat[in_leaf], y_nn[in_leaf], d_dat_in_leaf, w_l, n_l,
                mtot, no_of_treat, treat_values, w_yes, cont, cuda, cython,
                compare_only_to_zero)

            mse_mce_tree, obs_t_tree = mcf_fo_obj.add_rescale_mse_mce(
                mse_mce_leaf, obs_by_treat_leaf, mtot, no_of_treat,
                mse_mce_tree, obs_t_tree, compare_only_to_zero)
        else:
            n_lost += n_l
        n_total += n_l
    mse_mce_tree = mcf_fo_obj.get_avg_mse_mce(mse_mce_tree, obs_t_tree, mtot,
                                              no_of_treat, compare_only_to_zero)
    if cython:
        raise ValueError('Cython currently not used.')
        # oob_tree = mcf_cy.compute_mse_mce_cy(mse_mce_tree, mtot, no_of_treat,
        #                                      compare_only_to_zero)

    oob_tree = mcf_fo_obj.compute_mse_mce(mse_mce_tree, mtot, no_of_treat,
                                          compare_only_to_zero)
    return oob_tree
