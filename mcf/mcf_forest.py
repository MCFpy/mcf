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
try:
    import ray
except ImportError:
    ray = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from mcf import mcf_forest_data as mcf_fo_data
from mcf import mcf_forest_add as mcf_fo_add
from mcf import mcf_forest_objfct as mcf_fo_obj
from mcf import mcf_forest_asdict as mcf_fo_asdict
from mcf import mcf_forest_splitting as mcf_fo_split
from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_variable_importance as mcf_vi
from mcf import mcfoptp_parallel_backend_ray_classical as mcf_ray
from mcf.mcf_bias_adjustment import compute_ba_arrays, get_ba_data_train
from mcf.mcf_general_sys import print_mememory_statistics, find_no_of_workers
from mcf.mcfoptp_parallel_backend_forest_executor import make_forest_executor
from mcf.mcfoptp_parallel_backends_base import (batched_tasks, batch_size_fct, TaskSpec,
                                                ParallelExecutor, print_runtime_info,
                                                )
if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg, CtGrid
    from mcf.mcf_init_train import CfCfg


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
        mcf_ps.print_mcf(gen_cfg, '=' * 100 + f'\nTraining of Modified Causal Forest {title}')
    if gen_cfg.any_eff:
        cf_cfg.est_rounds = ('regular', 'additional',)
    else:
        cf_cfg.est_rounds = ('regular', )

    obs = len(tree_df) + len(fill_y_df)

    if p_ba_cfg.yes:
        prog_r, prop_r, x_r, prog_a, prop_a, x_a = compute_ba_arrays(mcf_,
                                                                     train_df=tree_df,
                                                                     predict_df=fill_y_df,
                                                                     )
    else:
        prog_r = prop_r = x_r = prog_a = prop_a = x_a = None

    if (folds := int(np.ceil(obs / cf_cfg.chunks_maxsize))) > 1:
        index_tr, index_y = tree_df.index.to_numpy(copy=True), fill_y_df.index.to_numpy(copy=True)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(index_tr)
        rng.shuffle(index_y)
        index_folds_tr = np.array_split(index_tr, folds)
        index_folds_y = np.array_split(index_y, folds)
    else:
        index_folds_tr = index_folds_y = None
    # Similar quantity is defined in mcf_init, but without accounting
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
                            idx=tree_df.index, prog=prog_a, prop=prop_a, x_ba=x_a,
                            )
                    else:
                        prog_dat, prop_dat, x_ba_dat = prog_a, prop_a, x_a
            else:
                prog_dat = prop_dat = x_ba_dat = None

            if not reg_round:
                # Reverse training and fill_with_y_file
                tree_fold_df, fill_y_fold_df = efficient_iate(mcf_, fill_y_fold_df, tree_fold_df,
                                                              summary=False,
                                                              )
            # Data preparation and stats II (regular, efficient IATE)
            # if gen_cfg.d_type != 'continuous':
            tree_fold_df, mcf_.var_cfg = mcf_fo_data.nn_matched_outcomes(
               mcf_, tree_fold_df, print_out=reg_round and splits == 0,
               )
            # Estimate forest structure (regular, efficient IATE)
            if gen_cfg.with_output:
                print(f'\nBuilding {splits+1} / {folds} forests, {round_}')
            forest, x_name_mcf, report_ = build_forest(mcf_, tree_fold_df)

            if reg_round and splits == 0:
                cf_cfg.x_name_mcf = x_name_mcf
            # Variable importance  ONLY REGULAR
            if cf_cfg.vi_oob_yes and gen_cfg.with_output and reg_round and splits == 0:
                time_start = time()
                mcf_vi.variable_importance(mcf_, tree_fold_df, forest, x_name_mcf)
                time_vi = time() - time_start
            else:
                time_vi = 0
            forest = mcf_fo_asdict.delete_data_from_forest(forest)
            # Fill tree with outcomes(regular, , efficient IATE)
            if gen_cfg.with_output:
                print(f'Filling {splits+1} / {folds} forests, {round_}')

            forest, _, share_merged = mcf_fo_add.fill_trees_with_y_indices_mp(mcf_, fill_y_fold_df,
                                                                              forest,
                                                                              )    # Fill
            forest_dic = mcf_fo_add.train_save_data(mcf_, fill_y_fold_df, forest,
                                                    prog_dat=prog_dat, prop_dat=prop_dat,
                                                    x_ba_dat=x_ba_dat,
                                                    )
            forest_list = mcf_fo_add.save_forests_in_cf_cfg(forest_dic, forest_list, splits, folds,
                                                            reg_round,
                                                            eff_iate=gen_cfg.any_eff,
                                                            )
            if gen_cfg.with_output:
                if reg_round and splits == 0:
                    report = report_
                    report["share_leaf_merged"] = share_merged
            else:
                report = None
            if gen_cfg.with_output and gen_cfg.verbose and mcf_.int_cfg.memory_print:
                print_mememory_statistics(gen_cfg, 'Forest Building: End of forests loop.')

    return cf_cfg, p_ba_cfg, forest_list, time_vi, report


def build_forest(mcf_: 'ModifiedCausalForest',
                 tree_df: DataFrame,
                 ) -> tuple[list, list, dict]:
    """Build MCF (not yet populated by w and outcomes)."""
    int_cfg, gen_cfg, cf_cfg = mcf_.int_cfg, mcf_.gen_cfg, mcf_.cf_cfg
    cuda, cython = int_cfg.cuda, int_cfg.cython
    bigdata_train = int_cfg.bigdata_train

    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, gen_cfg.with_output and gen_cfg.verbose)

    if int_cfg.no_ray_in_forest_building and int_cfg.mp_use_old_ray:
        raise RuntimeError('no_ray_in_forest_building=True conflicts with mp_use_old_ray=True.')

    (x_name, x_type, x_values, cf_cfg, pen_mult, data_np, y_i, y_nn_i, x_i, x_ind, x_ai_ind, d_i,
     w_i, cl_i, d_grid_i,
     ) = mcf_fo_data.prepare_data_for_forest(mcf_, tree_df)

    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        if gen_cfg.mp_automatic:
            maxworkers = find_no_of_workers(gen_cfg.mp_parallel, gen_cfg.sys_share,
                                            zero_tol=int_cfg.zero_tol,
                                            )
        else:
            maxworkers = gen_cfg.mp_parallel

    print_runtime_info(gen_cfg, int_cfg, maxworkers, txt_method='build forest')

    forest = [None for _ in range(cf_cfg.boot)]

    # Initialise save seeds for forest computation
    child_seeds = np.random.SeedSequence(1233456).spawn(cf_cfg.boot)

    kw_tree = {'y_i': y_i, 'y_nn_i': y_nn_i, 'x_i': x_i, 'd_i': d_i, 'd_grid_i': d_grid_i,
               'cl_i': cl_i, 'w_i': w_i, 'x_type': x_type, 'x_values': x_values, 'x_ind': x_ind,
               'x_ai_ind': x_ai_ind, 'gen_cfg': gen_cfg, 'cf_cfg': cf_cfg, 'ct_cfg': mcf_.ct_cfg,
               'pen_mult': pen_mult, 'cuda': cuda, 'cython': cython, 'bigdata_train': bigdata_train,
               }
    if maxworkers == 1:
        iterator = range(cf_cfg.boot)
        if use_tqdm:
            iterator = tqdm(iterator, total=cf_cfg.boot, desc='Building trees', leave=False,
                            dynamic_ncols=True,
                            )
        for idx in iterator:
            forest[idx] = build_tree_mcf(data_np,
                                         **kw_tree,
                                         boot=idx, seedseq=child_seeds[idx],
                                         zero_tol=int_cfg.zero_tol,
                                         )
            mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx+1, total=cf_cfg.boot)
    else:
        if int_cfg.mp_use_old_ray:
            if ray is None or ray_build_tree_mcf is None:
                raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                  'Install ray or use the new backend with mp_use_old_ray=False.'
                                  )
            ray_err_txt='Ray did not start in forest building.'
            if not ray.is_initialized():
                mcf_ray.init_ray_with_fallback(maxworkers, gen_cfg,
                                               mem_object_store=int_cfg.mem_object_store_1,
                                               mem_object_store_2=int_cfg.mem_object_store_2,
                                               ray_err_txt=ray_err_txt,
                                               )
                if int_cfg.mem_object_store_1 is not None:
                    mcf_ray.print_object_store(gen_cfg, int_cfg.mem_object_store_1)

            data_np_ref = mcf_ray.ray_put_all(data_np)

            still_running = [ray_build_tree_mcf.remote(data_np_ref,
                                                       **kw_tree,
                                                       boot=boot, seedseq=child_seeds[boot],
                                                       zero_tol=int_cfg.zero_tol,
                                                       )
                             for boot in range(cf_cfg.boot)
                             ]
            jdx = 0
            pbar = (tqdm(total=cf_cfg.boot, desc='Building trees', leave=False, dynamic_ncols=True)
                    if use_tqdm else None
                    )
            try:
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running, num_returns=1)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        forest[jdx] = ret_all_i
                        jdx += 1
                        if pbar is not None:
                            pbar.update(1)
                        mcf_gp.progress_clean_memory(output=output_clean, current_idx=jdx,
                                                     total=cf_cfg.boot, clean_mem=True,
                                                     )
            finally:
                if pbar is not None:
                    pbar.close()
            (data_np_ref, finished_res, finished        # pylint: disable=unbalanced-tuple-unpacking
             ) = mcf_ray.ray_del_refs(data_np_ref,
                                      f1=finished_res, f2=finished, mp_ray_del=int_cfg.mp_ray_del,
                                      )
        else:
            ray_remote_kwargs = {'num_cpus': 1, 'max_retries': 2,}  # Config for single ray worker
            executor = make_forest_executor(int_cfg=int_cfg,
                                            maxworkers=maxworkers,
                                            memmap_min_bytes=int_cfg.mp_memmap_min_bytes,
                                            joblib_backend_large='loky',
                                            temp_dir=int_cfg.mp_memmap_dir,
                                            cleanup_on_shutdown=True,  # for joblib
                                            ray_err_txt='Ray did not start in forest building',
                                            ray_init_kwargs=None,
                                            ray_remote_kwargs=ray_remote_kwargs,
                                            )
            try:
                forest = build_forest_parallel(executor=executor,
                                               data_np=data_np,
                                               kw_tree=kw_tree,
                                               child_seeds=child_seeds,
                                               cf_cfg=cf_cfg,
                                               int_cfg=int_cfg,
                                               gen_cfg=gen_cfg,
                                               forest=forest,
                                               maxworkers=maxworkers,
                                               )
            finally:
                executor.shutdown()

    # find best forest given the saved oob values
    forest_final, m_n_final = best_m_n_min_alpha_reg(forest, gen_cfg, cf_cfg, int_cfg.cython)
    del forest    # Free memory
    # Describe final forest
    if gen_cfg.with_output:
        report = mcf_fo_add.describe_forest(forest_final, m_n_final,
                                            mcf_.var_cfg, cf_cfg, gen_cfg, pen_mult=pen_mult,
                                            summary=True,
                                            )
    else:
        report = None

    # x_name: List. Order of x_name as used by tree building
    return forest_final, x_name, report


def build_forest_parallel(*,
                          executor: ParallelExecutor,
                          data_np: Any,
                          kw_tree: dict[str, Any],
                          child_seeds: list[Any],
                          cf_cfg: Any,
                          int_cfg: Any,
                          gen_cfg: Any,
                          forest: list[Any],
                          maxworkers: int=1,
                          ) -> list[Any]:
    """Build forest with backend-agnostic parallel tools and progress updates."""
    data_np_handle = executor.put_shared(data_np, name='tree_build_data')

    tasks = [TaskSpec(func=build_tree_mcf_backend,
                      kwargs={'data_np': data_np_handle,
                              'kw_tree': kw_tree,
                              'boot': boot,
                              'seedseq': child_seeds[boot],
                              'zero_tol': int_cfg.zero_tol,
                              },
                      name=f'tree_{boot}',)
             for boot in range(cf_cfg.boot)
             ]
    batch_size = batch_size_fct(no_batches=int_cfg.mp_batches,
                                no_workers=maxworkers, no_tasks=len(tasks),
                                )
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, gen_cfg.with_output and gen_cfg.verbose)

    pbar = (tqdm(total=cf_cfg.boot, desc='Building trees', leave=False, dynamic_ncols=True)
            if use_tqdm else None
            )
    jdx = 0
    try:
        for task_batch in batched_tasks(tasks, batch_size=batch_size):
            batch_results = executor.map(task_batch)

            for ret_all_i in batch_results:
                forest[jdx] = ret_all_i
                jdx += 1
                if pbar is not None:
                    pbar.update(1)
                mcf_gp.progress_clean_memory(output=output_clean,
                                             current_idx=jdx, total=cf_cfg.boot, clean_mem=True,
                                             )
    finally:
        if pbar is not None:
            pbar.close()

    return forest


def build_tree_mcf_backend(*, data_np: Any,
                           kw_tree: dict[str, Any],
                           boot: int,
                           seedseq: Any,
                           zero_tol: float,
                           ) -> Any:
    """Backend-agnostic tree builder for one subsampleing draw."""
    return build_tree_mcf(data_np, **kw_tree, boot=boot, seedseq=seedseq, zero_tol=zero_tol)


def _ray_build_tree_mcf_impl(data_np, *,
                             y_i, y_nn_i, x_i, d_i, d_grid_i, cl_i, w_i, x_type, x_values, x_ind,
                             x_ai_ind, gen_cfg, cf_cfg, ct_cfg, boot, pen_mult, cuda, cython,
                             bigdata_train, seedseq=None, zero_tol=1e-10,
                             ):
    """Prepare function for Ray (legacy version of ray only)."""
    return build_tree_mcf(data_np,
                          y_i=y_i, y_nn_i=y_nn_i, x_i=x_i, d_i=d_i, d_grid_i=d_grid_i, cl_i=cl_i,
                          w_i=w_i, x_type=x_type, x_values=x_values, x_ind=x_ind, x_ai_ind=x_ai_ind,
                          gen_cfg=gen_cfg, cf_cfg=cf_cfg, ct_cfg=ct_cfg, boot=boot,
                          pen_mult=pen_mult, cuda=cuda, cython=cython, bigdata_train=bigdata_train,
                          seedseq=seedseq, zero_tol=zero_tol,
                          )


ray_build_tree_mcf = ray.remote(_ray_build_tree_mcf_impl) if ray is not None else None


def build_tree_mcf(data_np: NDArray[np.floating], *,
                   y_i: int, y_nn_i: NDArray[np.intp],
                   x_i: NDArray[np.intp],
                   d_i: int, d_grid_i: int | None,
                   cl_i: int | None, w_i: int | None,
                   x_type: NDArray[np.int8],
                   x_values: list[list],
                   x_ind: NDArray[np.intp],
                   x_ai_ind: NDArray[np.intp] | list[int],
                   gen_cfg: 'GenCfg', cf_cfg: 'CfCfg', ct_cfg: 'CtGrid',
                   boot: int, pen_mult: float | np.floating, cuda: bool, cython: bool,
                   bigdata_train: bool = False,
                   seedseq: np.random.SeedSequence | int | None = None,
                   zero_tol: float = 1e-10,
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
    tree_empty = mcf_fo_asdict.make_default_tree_dict(np.min(cf_cfg.n_min_values), len(x_i),
                                                      indices, indices_oob,
                                                      bigdata_train=bigdata_train,
                                                      )
    # build trees for all m,n combinations
    grid_for_m = mcf_gp.check_if_iterable(cf_cfg.m_values)
    grid_for_n_min = mcf_gp.check_if_iterable(cf_cfg.n_min_values)
    grid_for_alpha_reg = mcf_gp.check_if_iterable(cf_cfg.alpha_reg_values)
    tree_all = [None for _ in range(len(grid_for_m) * len(grid_for_n_min) * len(grid_for_alpha_reg))
                ]
    j = 0
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                tree_all[j] = build_single_tree(data_np,
                                                y_i=y_i, y_nn_i=y_nn_i, d_i=d_i, d_grid_i=d_grid_i,
                                                x_i=x_i, w_i=w_i, x_type=x_type, x_values=x_values,
                                                x_ind=x_ind, x_ai_ind=x_ai_ind, cf_cfg=cf_cfg,
                                                gen_cfg=gen_cfg, ct_grid_nn_val=ct_cfg.grid_nn_val,
                                                mmm=m_idx, n_min=n_min, alpha_reg=alpha_reg,
                                                tree_dict_global=deepcopy(tree_empty),
                                                pen_mult=pen_mult, rng=rng, cuda=cuda,
                                                cython=cython, zero_tol=zero_tol,
                                                )
                j += 1

    return deepcopy(tree_all)


def build_single_tree(data: NDArray[np.float64], *,
                      y_i: int,
                      y_nn_i: NDArray[np.intp],
                      d_i: int,
                      d_grid_i: int | None,
                      x_i: NDArray[np.intp],
                      w_i: int | None,
                      x_type: NDArray[np.int8],
                      x_values: list[list[float | int | np.floating | np.integer]],
                      x_ind: NDArray[np.intp],
                      x_ai_ind: NDArray[np.intp] | list[int],
                      cf_cfg: 'CfCfg', gen_cfg: 'GenCfg',
                      ct_grid_nn_val: list,
                      mmm: int,
                      n_min: int,
                      alpha_reg: float,
                      tree_dict_global: dict,
                      pen_mult: float,
                      rng: np.random.Generator,
                      cuda: bool,
                      cython: bool,
                      zero_tol: float = 1e-10,
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
         split_n_oob_r, split_value) = mcf_fo_split.next_split(data_leaf, data_oob_leaf,
                                                               y_i=y_i, y_nn_i=y_nn_i, d_i=d_i,
                                                               d_grid_i=d_grid_i, x_i=x_i, w_i=w_i,
                                                               x_type=x_type, x_values=x_values,
                                                               x_ind=x_ind, x_ai_ind=x_ai_ind,
                                                               cf_cfg=cf_cfg, gen_cfg=gen_cfg,
                                                               ct_grid_nn_val=ct_grid_nn_val,
                                                               mmm=mmm, n_min=n_min,
                                                               alpha_reg=alpha_reg,
                                                               pen_mult=pen_mult, rng=rng,
                                                               cuda=cuda, cython=cython,
                                                               zero_tol=zero_tol,
                                                               )
        if terminal:
            leaf_id_daughters = None
        else:
            leaf_id_daughters = availabe_leaf_id_daughters[0:2]
            availabe_leaf_id_daughters = availabe_leaf_id_daughters[2:].copy()
            if leaf_id_daughters is None or len(leaf_id_daughters) < 2:
                raise RuntimeError('Not enough daughter leaves available for further splitting: '
                                   f'\n{leaf_id_daughters=:}'
                                   f'\n{availabe_leaf_id_daughters=:}'
                                   f'\n{leaf_ids=:}'
                                   f'\n{leaf_idx=:}'
                                   )
        tree_dict = mcf_fo_split.update_tree(data_oob_leaf, tree_dict,
                                             parent_idx=leaf_idx, split_var_i=split_var_i,
                                             split_type=split_type, split_value=split_value,
                                             split_n_l=split_n_l, split_n_r=split_n_r,
                                             split_leaf_l=split_leaf_l, split_leaf_r=split_leaf_r,
                                             split_leaf_oob_l=split_leaf_oob_l,
                                             split_leaf_oob_r=split_leaf_oob_r,
                                             split_n_oob_l=split_n_oob_l,
                                             split_n_oob_r=split_n_oob_r, terminal=terminal,
                                             leaf_id_daughters=leaf_id_daughters, d_i=d_i, w_i=w_i,
                                             d_grid_i=d_grid_i, y_nn_i=y_nn_i, y_i=y_i,
                                             ct_grid_nn_val=ct_grid_nn_val, gen_cfg=gen_cfg,
                                             cf_cfg=cf_cfg, rng=rng, cuda=cuda, cython=cython,
                                             zero_tol=zero_tol,
                                             )
    tree_dict = mcf_fo_asdict.cut_back_empty_cells_tree(tree_dict)

    return tree_dict


def best_m_n_min_alpha_reg(forest: list[dict], gen_cfg: 'GenCfg', cf_cfg: 'CfCfg', cython: bool
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
    dim_m_n_min_ar = len(grid_for_m) * len(grid_for_n_min) * len(grid_for_alpha_reg)
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
                    leaf_float_2_dim1 = isinstance(leaf_float_2, (float, np.float32, np.float64))
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
                                mse_mce_tree, obs_t_tree = mcf_fo_obj.add_rescale_mse_mce(
                                    leaf_float_2, leaf_int[9],
                                    mtot=cf_cfg.mtot, no_of_treat=no_of_treat,
                                    mse_mce_add_to=mse_mce_tree, obs_by_treat_add_to=obs_t_tree,
                                    compare_only_to_zero=cf_cfg.compare_only_to_zero,
                                    )
                if n_lost > 0:
                    if no_of_treat is None or leaf_float_2_dim1:
                        tree_mse = tree_mse * n_total / (n_total - n_lost)
                    else:
                        if (n_total - n_lost) < 1:
                            trees_without_oob[j] += 1
                if no_of_treat is not None and not leaf_float_2_dim1:
                    mse_mce_tree = mcf_fo_obj.get_avg_mse_mce(mse_mce_tree, obs_t_tree, cf_cfg.mtot,
                                                              no_of_treat,
                                                              cf_cfg.compare_only_to_zero,
                                                              )
                    if cython:
                        # raise ValueError('Cython currently not used.')
                        # tree_mse = mcf_cy.compute_mse_mce_cy(
                        #     mse_mce_tree, cf_cfg.mtot, no_of_treat,
                        #     cf_cfg.compare_only_to_zero)
                        pass
                    tree_mse = mcf_fo_obj.compute_mse_mce(mse_mce_tree, cf_cfg.mtot, no_of_treat,
                                                          cf_cfg.compare_only_to_zero
                                                          )
                mse_oob[j] += tree_mse     # Add MSE to MSE of forest j
        if np.any(trees_without_oob) > 0:
            for j, trees_without_oob_j in enumerate(trees_without_oob):
                if trees_without_oob_j > 0:
                    mse_oob[j] = mse_oob[j] * (cf_cfg.boot / (cf_cfg.boot - trees_without_oob_j))
        # Change value of mse to infinity if OOB-based mse could not be computed
        mse_oob = np.nan_to_num(mse_oob, np.inf)
        mse_oob[mse_oob < 0.01 * np.mean(mse_oob)] = np.inf
        min_i = np.argmin(mse_oob)
        mse_oob = mse_oob / cf_cfg.boot
        cf_cfg.n_min_values = mcf_gp.check_if_iterable(cf_cfg.n_min_values)
        cf_cfg.m_values = mcf_gp.check_if_iterable(cf_cfg.m_values)
        cf_cfg.alpha_reg_values = mcf_gp.check_if_iterable(cf_cfg.alpha_reg_values)
        if gen_cfg.with_output:
            txt = '\n' * 2 + '-' * 100
            txt += ('\nOOB MSE (without penalty) for M_try, minimum leafsize and alpha_reg '
                    'combinations'
                    '\nNumber of vars / min. leaf size / alpha reg. / OOB value. Trees without OOB'
                    '\n'
                    )
            j = 0
            for m_idx in cf_cfg.m_values:
                for n_min in cf_cfg.n_min_values:
                    for alpha_reg in cf_cfg.alpha_reg_values:
                        txt += (f'\n{m_idx:>12} {n_min:>12} {alpha_reg:15.3f} {mse_oob[j]:8.3f}'
                                f' {trees_without_oob[j]:4.0f}'
                                )
                        j += 1
            txt += (f'\nMinimum OOB MSE:     {mse_oob[min_i]:7.3f}'
                    f'\nNumber of variables: {m_n_min_ar_combi[min_i][0]}'
                    f'\nMinimum leafsize:    {m_n_min_ar_combi[min_i][1]}'
                    f'\nAlpha regularity:    {m_n_min_ar_combi[min_i][2]}'
                    )
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
                         '\nSecond round of estimation to get more efficient effects',
                         summary=summary
                         )
    # Switch the role of the subsamples
    fill_y_df, tree_df = tree_df, fill_y_df

    return tree_df, fill_y_df
