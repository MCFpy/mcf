"""
Created on Sat Jun 17 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the weights.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import ray
from scipy import sparse

from mcf import mcf_forest_add_functions as mcf_fo_add
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def get_weights_mp(mcf_: 'ModifiedCausalForest',
                   data_df: DataFrame,
                   forest_dic: dict,
                   reg_round: str = 'regular',
                   with_output: bool = True,
                   ) -> dict:
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    mcf_ : mcf-object.
    data_df : DataFrame: Prediction data.
    forstest_dic : Dict. Forest and Training data as DataFrame.
    reg_round : str. True if standard effects are estimated, False if
                         efficient effects are estimated.

    Returns
    -------
    weights_dic : Dict. Weights and training data as Numpy arrays.
    """
    gen_cfg, var_cfg, int_cfg = mcf_.gen_cfg, mcf_.var_cfg, mcf_.int_cfg
    cf_cfg, ct_cfg, p_cfg = mcf_.cf_cfg, mcf_.ct_cfg, mcf_.p_cfg
    print_yes = gen_cfg.with_output and gen_cfg.verbose and with_output

    if print_yes:
        print('\nObtaining weights from estimated forest')
    if int_cfg.weight_as_sparse_splits is None:
        if len(data_df) < 5000:
            int_cfg.weight_as_sparse_splits = 1
        else:
            base_dim = 25000
            int_cfg.weight_as_sparse_splits = round(
                len(forest_dic['y_train_df']) * len(data_df)
                / cf_cfg.folds / (base_dim ** 2)
                )
        int_cfg.weight_as_sparse_splits = max(int_cfg.weight_as_sparse_splits, 1
                                              )
    x_dat_all = mcf_gp.to_numpy_big_data(data_df[var_cfg.x_name],
                                         int_cfg.obs_bigdata
                                         )
    if int_cfg.weight_as_sparse_splits == 1 or not int_cfg.weight_as_sparse:
        # TODO Here, the variabes needed for bias_adjustment are extracted
        # from forest_dic
        # TODO Do these adjustments also in the cuda version
        (weights, y_dat, x_bala, cl_dat, w_dat, prog_dat, prop_dat, x_ba_dat
         ) = get_weights_mp_inner(
            forest_dic, x_dat_all, deepcopy(cf_cfg), ct_cfg, gen_cfg, int_cfg,
            p_cfg, with_output=with_output,
            bigdata_train=int_cfg.bigdata_train
            )
    else:
        unique_str = str(int(round(time() * 10000000)))
        temp_weight_name = 'temp_weights' + unique_str
        if gen_cfg.outpath is None or not gen_cfg.outpath.is_dir():
            base_name = Path(__file__).parent.absolute()
            base_file = base_name.with_name(f'{base_name.name}'
                                            f'{temp_weight_name}')
        else:
            base_file = gen_cfg.outpath / temp_weight_name
        x_dat_list = np.array_split(x_dat_all,
                                    int_cfg.weight_as_sparse_splits,
                                    axis=0
                                    )
        no_x_bala_return = True
        f_name = []
        reference_duration = +np.inf if int_cfg.bigdata_train else None
        for idx, x_dat in enumerate(x_dat_list):  # Das ist der Mastersplit
            time_start = time() if int_cfg.bigdata_train else None
            if print_yes:
                time1 = time()
                print('\n\nWeights computation with additional splitting. ',
                      f'Mastersplit {idx+1} ({len(x_dat_list)})')
                print('Computing weight matrix (inner loop).')
            if idx == len(x_dat_list) - 1:  # Last iteration, get x_bala
                no_x_bala_return = False

            (weights_i, y_dat, x_bala, cl_dat, w_dat, prog_dat, prop_dat,
             x_ba_dat
             ) = get_weights_mp_inner(
                forest_dic, x_dat, cf_cfg.copy(), ct_cfg, gen_cfg, int_cfg,
                p_cfg, no_x_bala_return=no_x_bala_return,
                with_output=with_output, bigdata_train=int_cfg.bigdata_train
                )
            if print_yes:
                print('\nComputing weight matrix (inner loop) took '
                      f'{int(round((time() - time1) / 60))} minutes.'
                      )
            f_name_i = []
            if int_cfg.weight_as_sparse_splits > 1:
                time1 = time()
                elem_per_chunk = len(weights_i)
                if print_yes:
                    print('Saving temporary weight matrix to harddrive.')
                for d_idx, weight_di in enumerate(weights_i):
                    f_name_d_idx = base_file.parent / (base_file.name
                                                       + f'{idx}_{d_idx}.npz')
                    mcf_sys.delete_file_if_exists(f_name_d_idx)
                    sparse.save_npz(f_name_d_idx, weight_di)
                    f_name_i.append(f_name_d_idx)
                f_name.append(f_name_i)
                del weights_i
                if print_yes:
                    print('Saving temporary weight matrix to harddrive took '
                          f'{int(round((time()-time1)/60))} minutes.')
            else:
                weights = weights_i

            if int_cfg.bigdata_train:
                duration = time() - time_start
                reference_duration, _ = mcf_sys.check_ray_shutdown(
                    gen_cfg, reference_duration, duration,
                    gen_cfg.mp_parallel,
                    max_multiplier=5,
                    with_output=(gen_cfg.with_output and gen_cfg.verbose
                                 and with_output),
                    err_txt='\nProblem in weight computation (outer function): '
                    )
            else:
                duration = None
        del x_dat_list, x_dat_all
        if int_cfg.weight_as_sparse_splits > 1:
            weights = []
            for d_idx in range(elem_per_chunk):
                w_list = []
                for idx in range(int_cfg.weight_as_sparse_splits):
                    file_to_load = f_name[idx][d_idx]
                    w_list.append(sparse.load_npz(file_to_load))
                    mcf_sys.delete_file_if_exists(file_to_load)
                w_all = sparse.vstack(w_list)
                weights.append(w_all)
            del w_all, w_list
    no_of_treat = (len(ct_cfg.grid_w_val)
                   if gen_cfg.d_type == 'continuous'
                   else gen_cfg.no_of_treat
                   )
    txt = mcf_sys.print_size_weight_matrix(
        weights, int_cfg.weight_as_sparse, no_of_treat)
    if isinstance(reg_round, bool):
        runde = 2 - int(reg_round)
    else:
        runde = 2 - int(reg_round == 'regular')

    mcf_ps.print_mcf(gen_cfg, f'Round {runde}. ' + txt, summary=False)

    weights_dic = {'weights': weights, 'y_dat_np': y_dat, 'x_bala_np': x_bala,
                   'cl_dat_np': cl_dat, 'w_dat_np': w_dat,
                   'prog_dat_np': prog_dat, 'prop_dat_np': prop_dat,
                   'x_ba_dat': x_ba_dat,
                   }
    if gen_cfg.with_output and gen_cfg.verbose:
        mcf_sys.print_mememory_statistics(
            gen_cfg, 'Prediction: End of weights estimation'
            )
    return weights_dic


def get_weights_mp_inner(
        forest_dic: dict,
        x_dat: NDArray[Any],
        cf_cfg: Any, ct_cfg: Any, gen_cfg: Any, int_cfg: Any, p_cfg: Any,
        no_x_bala_return: bool = False,
        with_output: bool = True,
        bigdata_train: bool = False,
        ) -> tuple[tuple[list],
                   NDArray[Any],
                   NDArray[Any] | None, NDArray[Any] | None,
                   NDArray[Any] | None, NDArray[Any] | None,
                   NDArray[Any] | None, NDArray[Any] | None
                   ]:
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    forest_dic : Dict. Contains data and node table of estimated forest.
    x_dat : Numpy array. X-data to make predictions for.
    cf_cfg :  CfCfg dataclass. Variables.
    ct_cfg :  Dataclass CtCfg. Variables.
    int_cfg :  Dataclass IntCfg. Variables.
    gen_cfg :  GenCfg Dataclass. Parameters.
    p_cfg : PCfg dataclass. Parameters.
    no_x_bala_return: Bool. Do not return X for balancing tests despite
                            p_cfg.bt_yes being True. Default is False.

    Returns
    -------
    weights : Tuple of lists (N_pred x 1 (x no_of_treat + 1).
    y_data : N_y x number of outcomes-Numpy array. Outcome variables.
    x_bala : N_y x number of balancing-vars-Numpy array. Balancing test vars.
    cl_dat : N_y x 1 Numpy array. Cluster number.
    w_dat : N_y x 1 Numpy array. Sampling weights (if used).
    prog_dat : N_y x number of treatments Numpy array. Prognostic score.
    prop_dat : N_y x number of treatments-1 Numpy array. Propensity score.
    x_ba_dat : N_y x number of variables in bias adjustment Numpy array.
    """
    print_yes = gen_cfg.with_output and gen_cfg.verbose and with_output
    y_dat = mcf_gp.to_numpy_big_data(forest_dic['y_train_df'],
                                     int_cfg.obs_bigdata)

    d_dat = np.int32(np.round(forest_dic['d_train_df'].to_numpy()))

    prog_dat = forest_dic['prog_dat_np']
    prop_dat = forest_dic['prop_dat_np']
    x_ba_dat = forest_dic['x_ba_dat_np']

    # This ensures that treatments are integers
    treat_is_integer = d_dat.min() >= 0
    # Check if treatment has values 0, 1, ..., number_of_treatments-1
    if gen_cfg.d_type != 'continuous':
        d_values_0_treatm1 = np.array_equal(
            np.sort(np.asarray(gen_cfg.d_values, dtype=np.intp)),
            np.arange(len(gen_cfg.d_values))
            )
    else:
        d_values_0_treatm1 = False

    n_x, n_y = len(x_dat), len(y_dat)
    cl_dat = (mcf_gp.to_numpy_big_data(forest_dic['cl_train_df'],
                                       int_cfg.obs_bigdata)
              if p_cfg.cluster_std else 0)
    w_dat = forest_dic['w_train_df'].to_numpy() if gen_cfg.weighted else 0
    x_bala = 0
    if p_cfg.bt_yes and not no_x_bala_return:
        x_bala = mcf_gp.to_numpy_big_data(forest_dic['x_bala_df'],
                                          int_cfg.obs_bigdata
                                          )
    empty_leaf_counter = merge_leaf_counter = 0
    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        if gen_cfg.mp_automatic:
            # This not really do anything ...
            maxworkers = mcf_sys.find_no_of_workers(gen_cfg.mp_parallel,
                                                    gen_cfg.sys_share,
                                                    zero_tol=int_cfg.zero_tol,
                                                    )
        else:
            maxworkers = gen_cfg.mp_parallel
    if gen_cfg.with_output and gen_cfg.verbose and with_output:
        mcf_ps.print_mcf(gen_cfg,
                         'Number of parallel processes (weight matrix): '
                         f'{maxworkers}',
                         summary=False)
    if maxworkers == 1:
        mp_over_boots = False
    else:
        mp_over_boots = bool(int_cfg.mp_weights_type == 2)
    no_of_treat = (len(ct_cfg.grid_w_val)
                   if gen_cfg.d_type == 'continuous'
                   else gen_cfg.no_of_treat
                   )
    if maxworkers == 1 or mp_over_boots:
        weights = initialise_weights(n_x, n_y, no_of_treat,
                                     int_cfg.weight_as_sparse)
        split_forest = False
        for idx in range(n_x):  # Iterate over i
            results_fut_idx = weights_obs_i(
                idx, n_y, forest_dic['forest'], x_dat, d_dat, cf_cfg, ct_cfg,
                gen_cfg, int_cfg, mp_over_boots, maxworkers, False,
                treat_is_integer=treat_is_integer,
                d_values_0_treatm1=d_values_0_treatm1,
                )
            if gen_cfg.with_output and gen_cfg.verbose:
                mcf_gp.share_completed(idx+1, n_x)
            if int_cfg.weight_as_sparse:
                for d_idx in range(no_of_treat):
                    indices = results_fut_idx[1][d_idx][0]
                    weights_obs = results_fut_idx[1][d_idx][1]
                    weights[d_idx][idx, indices] = weights_obs
            else:
                weights[idx] = results_fut_idx[1]
            empty_leaf_counter += results_fut_idx[2]
        if int_cfg.weight_as_sparse:
            weights = weights_to_csr(weights, no_of_treat)

    else:
        no_of_splits_i, max_size_i = maxworkers, 1000
        if n_x / no_of_splits_i > max_size_i:
            while True:
                no_of_splits_i += maxworkers
                if n_x / no_of_splits_i <= max_size_i:
                    break
        if print_yes:
            txt = ('\nOperational characteristics of weight estimation I'
                   f'\nNumber of workers: {maxworkers:2}'
                   f'\nNumber of observation chunks: {no_of_splits_i:5}'
                   '\nAverage # of observations per chunck:'
                   f' {n_x / no_of_splits_i:5.2f}')
            mcf_ps.print_mcf(gen_cfg, txt, summary=False)
        all_idx_split = np.array_split(range(n_x), no_of_splits_i)
        split_forest = False
        boot_indx_list = range(1)
        no_of_boot_splits = total_bootstraps = None

        if (not int_cfg.weight_as_sparse) and split_forest:
            if print_yes:
                txt = '\n' + 'XXXXXXXXDANGERXXXX' * 3
                txt += ('\nBootstrap splitting requires using sparse matrices.'
                        '\nProgramme continues without bootstrap splitting'
                        ' but may crash due to insufficient memory.')
                txt += '\n' + 'XXXXXXXX DANGER XXXXXXXX' * 3
                mcf_ps.print_mcf(gen_cfg, txt, summary=True)

        reference_duration = +np.inf if bigdata_train else None
        for b_i, boots_ind in enumerate(boot_indx_list):
            time_start = time() if bigdata_train else None
            weights = initialise_weights(n_x, n_y, no_of_treat,
                                         int_cfg.weight_as_sparse
                                         )
            if split_forest:
                # get subforest
                forest_temp = forest_dic['forest'][boots_ind[0]:
                                                   boots_ind[-1]+1]
                if print_yes and b_i == 0:
                    size = mcf_sys.total_size(forest_temp) / (1024*1024)
                    txt = f'\nSize of each submitted forest {size:6.2f} MB'
                    mcf_ps.print_mcf(gen_cfg, txt, summary=False)
                cf_cfg.boot = len(boots_ind)
                # weights über trees addieren
                if print_yes:
                    print(f'\nBoot Chunk {b_i+1:2} of {no_of_boot_splits:2}')
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    mcf_ps.print_mcf(gen_cfg, txt, summary=False)

            if not ray.is_initialized():
                _, no_of_workers = mcf_sys.init_ray_with_fallback(
                    maxworkers, int_cfg, gen_cfg,
                    mem_object_store=int_cfg.mem_object_store_3,
                    ray_err_txt='Ray does not start up in weight estimation 1'
                    )
            else:
                no_of_workers = maxworkers

            if (int_cfg.mem_object_store_3 is not None
                and gen_cfg.with_output and gen_cfg.verbose
                    and with_output):
                size = round(int_cfg.mem_object_store_3/(1024*1024))
                txt = f'Size of Ray Object Store: {size} MB'
                mcf_ps.print_mcf(gen_cfg, txt, summary=False)

            d_dat_ref = ray.put(d_dat)
            x_dat_ref = ray.put(x_dat)
            forest_ref = ray.put(forest_dic['forest'])
            still_running = [ray_weights_many_obs_i.remote(
                idx_list, n_y, forest_ref, x_dat_ref, d_dat_ref, cf_cfg,
                ct_cfg, gen_cfg, int_cfg, split_forest,
                treat_is_integer=treat_is_integer,
                d_values_0_treatm1=d_values_0_treatm1,
                )
                for idx_list in all_idx_split]

            jdx = 0
            time_start_ray = time() if bigdata_train else 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                # Perhaps avoid jamming the queue
                finished_res = ray.get(finished)

                for results_fut_idx in finished_res:
                    if print_yes:
                        if bigdata_train:
                            time_diff = (time() - time_start_ray) / (jdx+1)
                            td_str = ' (' + str(int(np.round(time_diff))) + 's)'
                        else:
                            td_str = ''

                        if jdx == 0:
                            print(f'\n   Obs chunk {jdx+1:2}'
                                  f' ({no_of_splits_i}){td_str}', end='')
                        else:
                            print(f' {jdx+1:2} ({no_of_splits_i}){td_str}',
                                  end='')
                        jdx += 1

                    for idx, val_list in enumerate(results_fut_idx[0]):
                        if int_cfg.weight_as_sparse:
                            for d_idx in range(no_of_treat):
                                indices = results_fut_idx[1][idx][d_idx][0]
                                weights_obs = (
                                    results_fut_idx[1][idx][d_idx][1])
                                idx_resized = np.resize(
                                    indices, (len(indices), 1))
                                weights[d_idx][val_list, idx_resized
                                               ] = weights_obs
                        else:
                            weights[val_list] = results_fut_idx[1][idx]
                    empty_leaf_counter += results_fut_idx[2]
            if 'refs' in int_cfg.mp_ray_del:
                del x_dat_ref, forest_ref, d_dat_ref
            if 'rest' in int_cfg.mp_ray_del:
                del finished_res, finished

            if int_cfg.weight_as_sparse:  # Transform to sparse matrix
                weights = weights_to_csr(weights, no_of_treat)
            if split_forest:
                if b_i == 0:  # only in case of splitted forests
                    weights_all = weights
                else:
                    for d_idx in range(no_of_treat):
                        weights_all[d_idx] += weights[d_idx]
                if print_yes:
                    print()
                    mcf_gp.print_size_weight_matrix(
                        weights_all, int_cfg.weight_as_sparse, no_of_treat)
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    mcf_ps.print_mcf(gen_cfg, ' ' + txt, summary=False)

        # Check if restart of ray is necessary
        if bigdata_train:
            duration = time() - time_start
            reference_duration, _ = mcf_sys.check_ray_shutdown(
                gen_cfg, reference_duration, duration, no_of_workers,
                max_multiplier=5,
                with_output=print_yes,
                err_txt='\nProblem in weight computation (inner function): '
                )
        else:
            duration = None

    if split_forest:
        cf_cfg.boot = total_bootstraps
        weights = normalize_weights(weights_all, no_of_treat,
                                    int_cfg.weight_as_sparse, n_x)
    weights = tuple(weights)
    if ((empty_leaf_counter > 0) or (merge_leaf_counter > 0)) and print_yes:
        txt = (f'\n\n{merge_leaf_counter:5} observations attributed in merged'
               f' leaves\n{empty_leaf_counter:5} observations attributed to'
               ' leaf w/o observations')
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return weights, y_dat, x_bala, cl_dat, w_dat, prog_dat, prop_dat, x_ba_dat


def normalize_weights(weights: list,
                      no_of_treat: int,
                      sparse_m: bool,
                      n_x: int
                      ) -> list:
    """Normalise weight matrix (needed when forest is split)."""
    for d_idx in range(no_of_treat):
        if sparse_m:
            weights[d_idx] = normal_weights_sparse(weights[d_idx], n_x)
        else:
            for i in range(n_x):
                weights[i][d_idx][1] = weights[i][d_idx][1] / np.sum(
                    weights[i][d_idx][1])
                weights[i][d_idx][1] = weights[i][d_idx][1].astype(np.float32)

    return weights


def normal_weights_sparse_loop(weights: sparse.csr_matrix,
                               n_x: int
                               ) -> sparse.csr_matrix:
    """Normalize sparse weight matrix (which is the heavy duty case."""
    row_sum = 1 / weights.sum(axis=1)
    for i in range(n_x):
        weights[i, :] = weights[i, :].multiply(
            row_sum[i])
    weights = weights.astype(np.float32, casting='same_kind')

    return weights


def normal_weights_sparse(weights: sparse.csr_matrix,
                          n_x: int
                          ) -> sparse.csr_matrix:
    """Normalize sparse weight matrix (which is the heavy duty case."""
    row_sum_inv = 1 / weights.sum(axis=1)
    # Create a diagonal matrix with the reciprocals of row sums
    diag_matrix = sparse.csr_array((row_sum_inv.toarray().ravel(),
                                   (range(n_x), range(n_x)))
                                   )
    # Multiply the original sparse matrix by the diagonal matrix
    normalized_weights = weights @ diag_matrix
    # Convert to float32
    normalized_weights = normalized_weights.astype(np.float32,
                                                   casting='same_kind'
                                                   )
    return normalized_weights


def initialise_weights(n_x: int,
                       n_y: int,
                       no_of_treat: int,
                       weight_as_sparse: bool,
                       ) -> list:
    """Initialise the weights matrix."""
    if weight_as_sparse:
        weights = []
        for _ in range(no_of_treat):
            weights.append(sparse.lil_array((n_x, n_y), dtype=np.float32))
    else:
        weights = [None for _ in range(n_x)]

    return weights


def weights_obs_i(idx: int,
                  n_y: int,
                  forest: list,
                  x_dat: NDArray[Any],
                  d_dat: NDArray[Any],
                  cf_cfg: Any,
                  ct_cfg: Any,
                  gen_cfg: Any,
                  int_cfg: Any,
                  mp_over_boots: bool = False,
                  maxworkers: int = 1,
                  split_forest: bool = False,
                  treat_is_integer: bool = True,
                  d_values_0_treatm1: bool = True,
                  ) -> tuple[int, list, int]:
    """
    Compute weight for single observation to predict.

    Parameters
    ----------
    idx : Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    cf_cfg : CfCfg dataclass. Parameters
    ct_cfg : CtCfg dataclass. Parameters
    gen_cfg : GenCfg dataclass. Parameters
    int_cfg : Instance of Dataclass. Parameters
    mp_over_boots : Bool. Multiprocessing at level of bootstraps.
                          Default is False.
    maxworkers: Int. Number of workers if MP.
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.

    """
    bigdata_train = int_cfg.bigdata_train
    empty_leaf_counter = 0

    if gen_cfg.d_type == 'continuous':
        no_of_treat = len(ct_cfg.grid_w_val)
        d_values = ct_cfg.grid_w_val
        continuous = True
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        continuous = False

    # Allocate weight matrix
    dtype = np.float32 if bigdata_train else np.float64
    weights_i_np = np.zeros((n_y, no_of_treat + 1), dtype=dtype)

    weights_i = [None for _ in range(no_of_treat)]  # 1 list for each treatment
    weights_i_np[:, 0] = np.arange(n_y)  # weight for index of outcomes
    x_dat_i = x_dat[idx, :]

    # weight_with_triplet should be much faster, but funtion for cont. treat
    # not yet adjusted
    weight_with_triplet = not continuous

    if not weight_with_triplet:
        # 24.10.2025; add buffer
        dtype = np.float32 if bigdata_train else np.float64
        buffer = np.zeros((n_y, no_of_treat), dtype=dtype)  # allocate once per
        #                                                     worker/loop

    if mp_over_boots:
        if not ray.is_initialized():
            mcf_sys.init_ray_with_fallback(
                maxworkers, int_cfg, gen_cfg,
                ray_err_txt='Ray does not start in weight estimation.'
                )
        d_dat_ref = ray.put(d_dat)

        if weight_with_triplet:  # Added Oct, 30, 2025
            still_running = [ray_weights_obs_i_inside_boot_triplet.remote(
                forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat_ref,
                continuous, bigdata_train,
                d_values_0_treatm1=d_values_0_treatm1,
                zero_tol=int_cfg.zero_tol,
                )
                for boot in range(cf_cfg.boot)]
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for result in finished_res:
                    if result[3]:
                        empty_leaf_counter += 1
                    else:
                        np.add.at(weights_i_np,
                                  (result[0], 1 + result[1]), result[2]
                                  )
        else:
            still_running = [ray_weights_obs_i_inside_boot.remote(
                forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat_ref,
                continuous, bigdata_train, treat_is_integer=treat_is_integer,
                buffer=buffer, d_values_0_treatm1=d_values_0_treatm1,
                zero_tol=int_cfg.zero_tol,
                )
                for boot in range(cf_cfg.boot)]
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for result in finished_res:
                    if result[1]:
                        empty_leaf_counter += 1
                    else:
                        weights_i_np[:, 1:] += result[0]

    else:
        # This is the default branch
        for boot in range(cf_cfg.boot):
            if weight_with_triplet:  # Added Oct, 30, 2025
                (rows, cols, vals, empty_leaf
                 ) = weights_obs_i_inside_boot_triplet(
                    forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat,
                    continuous, bigdata_train=bigdata_train,
                    d_values_0_treatm1=d_values_0_treatm1,
                    zero_tol=int_cfg.zero_tol,
                    )
                if empty_leaf:
                    empty_leaf_counter += 1
                else:
                    np.add.at(weights_i_np, (rows, 1 + cols), vals)
            else:
                weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
                    forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat,
                    continuous, bigdata_train=bigdata_train,
                    treat_is_integer=treat_is_integer, buffer=buffer,
                    d_values_0_treatm1=d_values_0_treatm1,
                    zero_tol=int_cfg.zero_tol,
                    )
                if empty_leaf:
                    empty_leaf_counter += 1
                else:
                    weights_i_np[:, 1:] += weights_ij_np

    obs_without_leaf = 1 if empty_leaf_counter == cf_cfg.boot else 0
    normalize = not split_forest

    weights_i = final_trans(weights_i_np, no_of_treat, normalize)

    return idx, weights_i, obs_without_leaf


@ray.remote
def ray_weights_obs_i_inside_boot(forest_b: dict,
                                  x_dat_i: NDArray[Any],
                                  n_y: int,
                                  no_of_treat: int,
                                  d_values: Sequence[int | float],
                                  d_dat: NDArray[Any],
                                  continuous: bool,
                                  bigdata_train: bool = False,
                                  treat_is_integer: bool = True,
                                  buffer: NDArray[np.floating] | None = None,
                                  d_values_0_treatm1: bool = True,
                                  zero_tol: float = 1e-15,
                                  ) -> tuple[NDArray[Any], bool]:
    """Allow for MP at bootstrap level (intermediate procedure) with ray."""
    weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, continuous,
        bigdata_train=bigdata_train, treat_is_integer=treat_is_integer,
        buffer=buffer, d_values_0_treatm1=d_values_0_treatm1,
        zero_tol=zero_tol,
        )
    return weights_ij_np, empty_leaf


@ray.remote
def ray_weights_obs_i_inside_boot_triplet(
        forest_b: dict,
        x_dat_i: NDArray[Any],
        n_y: int,
        no_of_treat: int,
        d_values: Sequence[int | float],
        d_dat: NDArray[Any],
        continuous: bool,
        bigdata_train: bool = False,
        d_values_0_treatm1: bool = True,
        zero_tol: float = 1e-15,
        ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.floating],
                   bool]:
    """Allow for MP at bootstrap level (intermediate procedure) with ray."""
    rows, cols, vals, empty_leaf = weights_obs_i_inside_boot_triplet(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, continuous,
        bigdata_train=bigdata_train,
        d_values_0_treatm1=d_values_0_treatm1,
        zero_tol=zero_tol,
        )
    return rows, cols, vals, empty_leaf


def weights_obs_i_inside_boot(forest_b: dict,
                              x_dat_i: NDArray[Any],
                              n_y: int,
                              no_of_treat: int,
                              d_values: Sequence[int | float],
                              d_dat: NDArray[Any],
                              continuous: bool,
                              treat_is_integer: bool = True,
                              bigdata_train: bool = False,
                              buffer: NDArray[np.floating] | None = None,
                              d_values_0_treatm1: bool = True,
                              zero_tol: float = 1e-15,
                              ) -> tuple[NDArray[Any], bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # This is the default branch
    if continuous:
        # Continuous treatments
        # buffer is ignored, might be added later after update of cont.treatm.
        return weights_obs_i_inside_boot_treat_cont(
            forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, bigdata_train,
            zero_tol=zero_tol,
            )
    # Discrete treatments
    return weights_obs_i_inside_boot_treat_disc(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat,
        bigdata_train=bigdata_train, treat_is_integer=treat_is_integer,
        out_buffer=buffer, d_values_0_treatm1=d_values_0_treatm1,
        zero_tol=zero_tol,
        )


def weights_obs_i_inside_boot_triplet(
        forest_b: dict,
        x_dat_i: NDArray[Any],
        n_y: int,
        no_of_treat: int,
        d_values: Sequence[int | float],
        d_dat: NDArray[Any],
        continuous: bool,
        bigdata_train: bool = False,
        d_values_0_treatm1: bool = True,
        zero_tol: float = 1e-15,
        ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.floating],
                   bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # This is the default branch
    if continuous:
        # Only 2 returns
        weights_ij_np, empty_leaf = weights_obs_i_inside_boot_treat_cont(
            forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, bigdata_train,
            zero_tol=zero_tol,
            )
         # Only for consistency, should not work with current vertsion
        return weights_ij_np, empty_leaf, None, None
    # Discrete treatments: 4 returns
    return weights_obs_i_inside_boot_treat_disc_triplets(
        forest_b, x_dat_i, no_of_treat, d_values, d_dat,
        bigdata_train=bigdata_train, d_values_0_treatm1=d_values_0_treatm1,
        zero_tol=zero_tol,
        )


def weights_obs_i_inside_boot_treat_cont(forest_b: dict,
                                         x_dat_i: NDArray[Any],
                                         n_y: int,
                                         no_of_treat: int,
                                         d_values: Sequence[int | float],
                                         d_dat: NDArray[Any],
                                         bigdata_train: bool = False,
                                         zero_tol: float = 1e-15,
                                         ) -> tuple[NDArray[Any], bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # here is the information for the different treatments needed
    # Problem ist dass man hier für die x alle potential outcomes berechnet
    # Bei continuous treatments gehören aber verschiedene leafs zu den gleichen
    # X. Ausserdem werden d=0 und die anderen Werte von d verschieden behandelt
    # x_dat_i ist eine design matrix, die dann auch die verschiedenen werte von
    # d enthalten sollte
    # d = 0 zu setzen scheint aber überflüssig bei der designmatrix bzgl. d
    fill_y_empty_leave = forest_b['fill_y_empty_leave']
    fill_y_indices_list = forest_b['fill_y_indices_list']

    leaf_id_list = []
    # We get a list of leafs that contains relevant elements, some of them
    # may be identical, but this does not matter
    for treat in d_values[1:]:
        x_dat_i_t = np.append(x_dat_i, treat)
        leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i_t,
                                                  zero_tol=zero_tol,
                                                  )
        leaf_id_list.append(leaf_id)
        if (fill_y_empty_leave[leaf_id] == 1
                or fill_y_indices_list[leaf_id] is None):
            empty_leaf, weights_ij_np = True, 0
            # If any of the subleaves is None, then stop ...
            break
        empty_leaf = False

    if not empty_leaf:
        fb_lid_14_list = [fill_y_indices_list[leaf_id]
                          for leaf_id in leaf_id_list
                          ]
        if bigdata_train:
            weights_ij_np = np.zeros((n_y, no_of_treat))
        else:
            weights_ij_np = np.zeros((n_y, no_of_treat), dtype=np.float32)

        # We need to collect information over various leafs for the 0
        # For the other leaves, the leaves are treatment specific
        leaf_0_complete, leaf_pos_complete = False, True
        # Zuerst 0 einsammeln
        for jdx, fb_lid_14 in enumerate(fb_lid_14_list):
            d_ib = d_dat[fb_lid_14].reshape(-1)  # view
            indices_ibj_0 = d_ib < zero_tol
            indices_ibj_pos = d_ib >= zero_tol
            if np.any(indices_ibj_0):  # any valid observations?
                fb_lid_14_indi_0 = fb_lid_14[indices_ibj_0]
                n_x_i_0 = len(fb_lid_14_indi_0)
                weights_ij_np[fb_lid_14_indi_0, 0] += 1 / n_x_i_0
                leaf_0_complete = True
            if np.any(indices_ibj_pos):  # any valid observations?
                fb_lid_14_indi_pos = fb_lid_14[indices_ibj_pos]
                n_x_i_pos = len(fb_lid_14_indi_pos)
                weights_ij_np[fb_lid_14_indi_pos, jdx+1] += (
                    1 / n_x_i_pos)
            else:
                leaf_pos_complete = False
                break
        leaf_complete = leaf_0_complete and leaf_pos_complete

        if not leaf_complete:
            empty_leaf, weights_ij_np = True, 0

    return weights_ij_np, empty_leaf


def weights_obs_i_inside_boot_treat_disc_triplets(
    forest_b: dict,
    x_dat_i: NDArray[Any],
    no_of_treat: int,
    d_values: Sequence[int | float],
    d_dat: NDArray[np.integer | np.floating],
    bigdata_train: bool = False,
    d_values_0_treatm1: bool = True,
    zero_tol: float = 1e-15,
        ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.floating],
                   bool
                   ]:
    """Allow for MP at bootstrap level (intermediate procedure).

    Like weights_obs_i_inside_boot_treat_disc but returns (rows, cols, vals,
    empty).
    Memory-light: only len(leaf_inds) elements are produced.
    """
    dtype_float = np.float32 if bigdata_train else np.float64
    if not d_values_0_treatm1:
        d_dat = encode_treatments(d_dat, d_values)
        d_values = np.arange(no_of_treat)        # now canonical

    empty_flags = forest_b['fill_y_empty_leave']
    inds_list = forest_b['fill_y_indices_list']

    leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i,
                                              zero_tol=zero_tol,
                                              )
    leaf_inds = inds_list[leaf_id]
    if empty_flags[leaf_id] == 1 or leaf_inds is None:
        return (np.empty(0, dtype=np.intp),
                np.empty(0, dtype=np.intp),
                np.empty(0, dtype=dtype_float),
                True,
                )
    leaf_inds = np.asarray(leaf_inds, dtype=np.intp)
    if leaf_inds.size == 0:
        return (leaf_inds,
                leaf_inds,
                np.empty(0, dtype=dtype_float),
                True,
                )
    # --- Build column indices d_leaf (0..no_of_treat-1) ---
    d_leaf = np.asarray(d_dat[leaf_inds], dtype=np.intp).ravel()

    # --- Completeness & per-observation weights ---
    counts = np.bincount(d_leaf, minlength=no_of_treat)

    if (counts[:no_of_treat] == 0).any():
        return (leaf_inds,
                d_leaf,
                np.empty(0, dtype=dtype_float),
                True
                )
    vals = (1.0 / counts[d_leaf]).astype(dtype_float, copy=False)

    return leaf_inds, d_leaf, vals, False  # Needs adapted caller


def weights_obs_i_inside_boot_treat_disc(
        forest_b: dict,
        x_dat_i: NDArray[Any],
        n_y: int,
        no_of_treat: int,
        d_values: Sequence[int | float],
        d_dat: NDArray[np.integer | np.floating],
        treat_is_integer: bool = True,
        bigdata_train: bool = False,
        out_buffer: NDArray[np.floating] | None = None,
        d_values_0_treatm1: bool = True,
        zero_tol: float = 1e-15,
        # Predefined out_buffer to reduce memory allocation time
        ) -> tuple[NDArray[np.floating] | int, bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # This function is used in the default computations.
    if not d_values_0_treatm1:
        d_dat = encode_treatments(d_dat, d_values)
        d_values = np.arange(no_of_treat)        # now canonical
        treat_is_integer = True                  # safe from now on
    empty_flags = forest_b['fill_y_empty_leave']
    inds_list = forest_b['fill_y_indices_list']

    leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i,
                                              zero_tol=zero_tol,
                                              )
    leaf_inds = inds_list[leaf_id]
    if empty_flags[leaf_id] == 1 or leaf_inds is None:
        return 0, True

    leaf_inds = np.asarray(leaf_inds, dtype=np.intp)  # 23.10.2025
    if leaf_inds.size == 0:                           # 23.10.2025
        return 0, True

    # 1) pull out the raw treatment slice
    if treat_is_integer:
        d_leaf = np.asarray(d_dat[leaf_inds], dtype=np.intp).ravel()
    else:
        raw = d_dat[leaf_inds]
        # 2) make sure we get a 1-D int array
        if raw.dtype == object:
            # each element is likely an array/sequence → concatenate
            d_leaf = np.concatenate([np.asarray(r).ravel() for r in raw])
        else:
            # a normal numeric slice → just flatten
            d_leaf = raw.ravel()
        # d_leaf = d_leaf.astype(int)  23.10.2025
        d_leaf = np.asarray(d_leaf, dtype=np.intp, copy=False)

    # 3) count and check completeness
    counts = np.bincount(d_leaf, minlength=no_of_treat)
    if (counts[:no_of_treat] == 0).any():
        return 0, True

    # 4) per-obs weight and scatter
    dtype = np.float32 if bigdata_train else np.float64
    if out_buffer is None:
        weights = np.zeros((n_y, no_of_treat), dtype=dtype)
    else:
        if out_buffer.shape != (n_y, no_of_treat):
            raise ValueError('out_buffer has wrong shape; expected '
                             '(n_y, no_of_treat)'
                             )
        if out_buffer.dtype != dtype:
            raise ValueError('out_buffer has wrong dtype; expected '
                             f'{dtype}, got {out_buffer.dtype}'
                             )
        weights = out_buffer
        weights.fill(0)
    # per_obs = 1.0 / counts[d_leaf]
    per_obs = (1.0 / counts[d_leaf]).astype(weights.dtype, copy=False)
    weights[leaf_inds, d_leaf] = per_obs

    return weights, False


def final_trans(weights_i_np: NDArray[Any],
                no_of_treat: int,
                normalize: bool = True
                ) -> list[NDArray[Any], NDArray[Any]]:
    """
    Compute last transformations of (positive only) weights.

    Returns a list of (indices, weights) pairs for each treatment.

    Parameters
    ----------
    weights_i_np : 2D Numpy array. Weights including zeros.
    no_of_treat : Int. Number of treatments.
    normalize : Bool. Normalize weights to row sum of 1. Default is True.

    Returns
    -------
    weights_i: List of arrays.
    """
    # 1) pull out the “global” row-indices (first column)
    idx_col = weights_i_np[:, 0].astype(np.int32)

    # 2) grab just the treatment weight columns
    w_cols = weights_i_np[:, 1: 1 + no_of_treat]

    weights_i: list[list[np.ndarray]] = []

    # 3) per-treatment processing (no_of_treat is small)
    for j in range(no_of_treat):
        col = w_cols[:, j]
        mask = col > 1e-14

        if mask.any():
            sel_idx = idx_col[mask]
            sel_w = col[mask]
            if normalize:
                total = sel_w.sum()
                if total > 0:
                    sel_w = sel_w / total
            sel_w = sel_w.astype(np.float32)
        else:
            sel_idx = np.empty(0, dtype=np.int32)
            sel_w = np.empty(0, dtype=np.float32)

        weights_i.append([sel_idx, sel_w])

    return weights_i


def weights_to_csr(weights: list[sparse.lil_matrix],
                   no_of_treat: int
                   ) -> sparse.csr_matrix:
    """Convert list of lil sparse matrices to csr format."""
    for d_idx in range(no_of_treat):
        weights[d_idx] = weights[d_idx].tocsr()

    return weights


@ray.remote
def ray_weights_many_obs_i(idx_list: list[int],
                           n_y: int,
                           forest: list[list],
                           x_dat: NDArray[Any],
                           d_dat: NDArray[Any],
                           cf_cfg: Any,
                           ct_cfg: Any,
                           gen_cfg: Any,
                           int_cfg: Any,
                           split_forest: bool = False,
                           treat_is_integer: bool = True,
                           d_values_0_treatm1: bool = True,
                           ) -> tuple[Any]:
    """Make function compatible with Ray."""
    return weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_cfg,
                              ct_cfg, gen_cfg, int_cfg, split_forest,
                              treat_is_integer=treat_is_integer,
                              d_values_0_treatm1=d_values_0_treatm1,
                              )


def weights_many_obs_i(idx_list: list[int],
                       n_y: int,
                       forest: list[list],
                       x_dat: NDArray[Any],
                       d_dat: NDArray[Any],
                       cf_cfg: Any,
                       ct_cfg: Any,
                       gen_cfg: Any,
                       int_cfg: Any,
                       split_forest=False,
                       treat_is_integer=True,
                       d_values_0_treatm1: bool = True,
                       ) -> tuple[list[int], list[list], int]:
    """
    Create chunks of task to be efficiently executed by MP.

    Parameters
    ----------
    idx_list : List of Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    cf_cfg, ct_cfg, gen_cfg : DataClasses. Parameters
    split_forest : Boolean. True if chunks of bootstraps. Default is False.
    ...

    Returns
    -------
    idx : List of Int. Int. Counter.
    weights : List of lists.
    empty_leaf_counter : Int.

    """
    weights = [None for _ in range(len(idx_list))]
    empty_leaf_counter = 0
    for idx, val in enumerate(idx_list):
        ret = weights_obs_i(val, n_y, forest, x_dat, d_dat, cf_cfg, ct_cfg,
                            gen_cfg, int_cfg, mp_over_boots=False, maxworkers=1,
                            split_forest=split_forest,
                            treat_is_integer=treat_is_integer,
                            d_values_0_treatm1=d_values_0_treatm1,
                            )
        weights[idx] = ret[1]
        empty_leaf_counter += ret[2]

    return idx_list, weights, empty_leaf_counter


def encode_treatments(d_dat: NDArray[np.floating | np.integer],
                      d_values: Sequence[int | float]
                      ) -> NDArray[np.intp]:
    """Map labels in d_dat to 0..T-1 according to the order in d_values.

    Raises if d_dat contains a label not present in d_values.
    Works for numeric d_dat (int/float). For object arrays, see note below.
    """
    dv = np.asarray(d_values)
    dvsize = dv.size

    # Fast path: already 0..T-1 in the right order
    if np.issubdtype(dv.dtype, np.integer) and np.array_equal(dv,
                                                              np.arange(dvsize)
                                                              ):
        return np.asarray(d_dat, dtype=np.intp)

    # General path: sort d_values and map via searchsorted (O(N log T))
    order = np.argsort(dv)
    dv_sorted = dv[order]

    raw = np.asarray(d_dat).ravel()
    pos = np.searchsorted(dv_sorted, raw)

    in_range = (pos < dvsize) & (dv_sorted[pos] == raw)
    if not in_range.all():
        bad = np.unique(raw[~in_range])
        raise ValueError(f'Unknown treatment label(s) in d_dat: {bad!r}')

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(dvsize)

    codes = inv_order[pos].astype(np.intp, copy=False)

    return codes.reshape(np.shape(d_dat))
