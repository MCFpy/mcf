"""
Created on Sat Jun 17 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the weights.

@author: MLechner
-*- coding: utf-8 -*-
"""
from time import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from pandas import DataFrame
from scipy import sparse
import ray

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
                   with_output: bool = True
                   ) -> dict:
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    mcf_ : mcf-object.
    data_df : DataFrame: Prediction data.
    forstest_dic : Dict. Forest and Training data as DataFrame.
    reg_round : Boolean. True if IATE is estimated, False if IATE_EFF is
              estimated

    Returns
    -------
    weights_dic : Dict. Weights and training data as Numpy arrays.
    """
    gen_dic, var_dic, int_dic = mcf_.gen_dict, mcf_.var_dict, mcf_.int_dict
    cf_dic, ct_dic, p_dic = mcf_.cf_dict, mcf_.ct_dict, mcf_.p_dict
    print_yes = int_dic['with_output'] and int_dic['verbose'] and with_output

    if print_yes:
        print('\nObtaining weights from estimated forest')
    if int_dic['weight_as_sparse_splits'] is None:
        if len(data_df) < 5000:
            int_dic['weight_as_sparse_splits'] = 1
        else:
            base_dim = 25000
            int_dic['weight_as_sparse_splits'] = round(
                len(forest_dic['y_train_df']) * len(data_df)
                / cf_dic['folds'] / (base_dim ** 2))
        if int_dic['weight_as_sparse_splits'] <= 1:
            int_dic['weight_as_sparse_splits'] = 1
    x_dat_all = mcf_gp.to_numpy_big_data(data_df[var_dic['x_name']],
                                         int_dic['obs_bigdata'])
    if int_dic['weight_as_sparse_splits'] == 1 or not int_dic[
            'weight_as_sparse']:
        weights, y_dat, x_bala, cl_dat, w_dat = get_weights_mp_inner(
            forest_dic, x_dat_all, cf_dic.copy(), ct_dic, gen_dic, int_dic,
            p_dic, with_output=with_output,
            bigdata_train=int_dic['bigdata_train']
            )
    else:
        unique_str = str(int(round(time() * 10000000)))
        temp_weight_name = 'temp_weights' + unique_str
        if gen_dic['outpath'] is None or not gen_dic['outpath'].is_dir():
            base_name = Path(__file__).parent.absolute()
            base_file = base_name.with_name(f'{base_name.name}'
                                            f'{temp_weight_name}')
        else:
            base_file = gen_dic['outpath'] / temp_weight_name
        x_dat_list = np.array_split(x_dat_all,
                                    int_dic['weight_as_sparse_splits'],
                                    axis=0
                                    )
        no_x_bala_return = True
        f_name = []
        reference_duration = +np.inf if int_dic['bigdata_train'] else None
        for idx, x_dat in enumerate(x_dat_list):  # Das ist der Mastersplit
            time_start = time() if int_dic['bigdata_train'] else None
            if print_yes:
                time1 = time()
                print('\n\nWeights computation with additional splitting. ',
                      f'Mastersplit {idx+1} ({len(x_dat_list)})')
                print('Computing weight matrix (inner loop).')
            if idx == len(x_dat_list) - 1:  # Last iteration, get x_bala
                no_x_bala_return = False

            weights_i, y_dat, x_bala, cl_dat, w_dat = get_weights_mp_inner(
                forest_dic, x_dat, cf_dic.copy(), ct_dic, gen_dic, int_dic,
                p_dic, no_x_bala_return=no_x_bala_return,
                with_output=with_output, bigdata_train=int_dic['bigdata_train']
                )
            if print_yes:
                print('\nComputing weight matrix (inner loop) took '
                      f'{int(round((time() - time1) / 60))} minutes.')
            f_name_i = []
            if int_dic['weight_as_sparse_splits'] > 1:
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

            if int_dic['bigdata_train']:
                duration = time() - time_start
                reference_duration, _ = mcf_sys.check_ray_shutdown(
                    gen_dic, reference_duration, duration,
                    gen_dic['mp_parallel'],
                    max_multiplier=5,
                    with_output=(int_dic['with_output'] and int_dic['verbose']
                                 and with_output),
                    err_txt='\nProblem in weight computation (outer function): '
                    )
            else:
                duration = None
        del x_dat_list, x_dat_all
        if int_dic['weight_as_sparse_splits'] > 1:
            weights = []
            for d_idx in range(elem_per_chunk):
                w_list = []
                for idx in range(int_dic['weight_as_sparse_splits']):
                    file_to_load = f_name[idx][d_idx]
                    w_list.append(sparse.load_npz(file_to_load))
                    mcf_sys.delete_file_if_exists(file_to_load)
                w_all = sparse.vstack(w_list)
                weights.append(w_all)
            del w_all, w_list
    if print_yes and reg_round:
        no_of_treat = (len(ct_dic['grid_w_val'])
                       if gen_dic['d_type'] == 'continuous'
                       else gen_dic['no_of_treat'])
        txt = mcf_sys.print_size_weight_matrix(
            weights, int_dic['weight_as_sparse'], no_of_treat)
        mcf_ps.print_mcf(gen_dic, ' ' + txt, summary=False)
    weights_dic = {'weights': weights, 'y_dat_np': y_dat, 'x_bala_np': x_bala,
                   'cl_dat_np': cl_dat, 'w_dat_np': w_dat}
    mcf_sys.print_mememory_statistics(gen_dic,
                                      'Prediction: End of weights estimation')
    return weights_dic


def get_weights_mp_inner(forest_dic: dict,
                         x_dat: np.ndarray,
                         cf_dic: dict, ct_dic: dict, gen_dic: dict,
                         int_dic: dict, p_dic: dict,
                         no_x_bala_return: bool = False,
                         with_output: bool = True,
                         bigdata_train: bool = False,
                         ) -> tuple[tuple[list], np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    forest_dic : Dict. Contains data and node table of estimated forest.
    x_dat : Numpy array. X-data to make predictions for.
    cf_dic :  Dict. Variables.
    ct_dic :  Dict. Variables.
    int_dic :  Dict. Variables.
    gen_dic :  Dict. Parameters.
    p_dic : Dict. Parameters.
    no_x_bala_return: Bool. Do not return X for balancing tests despite
                            p_dict['bt_yes'] being True. Default is False.

    Returns
    -------
    weights : Tuple of lists (N_pred x 1 (x no_of_treat + 1).
    y_data : N_y x number of outcomes-Numpy array. Outcome variables.
    x_bala : N_y x number of balancing-vars-Numpy array. Balancing test vars.
    cl_dat : N_y x 1 Numpy array. Cluster number.
    w_dat: N_y x 1 Numpy array. Sampling weights (if used).
    """
    print_yes = int_dic['with_output'] and int_dic['verbose'] and with_output
    y_dat = mcf_gp.to_numpy_big_data(forest_dic['y_train_df'],
                                     int_dic['obs_bigdata'])

    d_dat = np.int32(np.round(forest_dic['d_train_df'].to_numpy()))
    # This ensures that treatments are integers
    treat_is_integer = d_dat.min() >= 0

    n_x, n_y = len(x_dat), len(y_dat)
    cl_dat = (mcf_gp.to_numpy_big_data(forest_dic['cl_train_df'],
                                       int_dic['obs_bigdata'])
              if p_dic['cluster_std'] else 0)
    w_dat = forest_dic['w_train_df'].to_numpy() if gen_dic['weighted'] else 0
    x_bala = 0
    if p_dic['bt_yes'] and not no_x_bala_return:
        x_bala = mcf_gp.to_numpy_big_data(forest_dic['x_bala_df'],
                                          int_dic['obs_bigdata'])
    empty_leaf_counter = merge_leaf_counter = 0
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            # This not really do anything ...
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])

        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and int_dic['verbose'] and with_output:
        mcf_ps.print_mcf(gen_dic,
                         'Number of parallel processes (weight matrix): '
                         f'{maxworkers}',
                         summary=False)
    if maxworkers == 1:
        mp_over_boots = False
    else:
        mp_over_boots = bool(int_dic['mp_weights_type'] == 2)
    no_of_treat = (len(ct_dic['grid_w_val'])
                   if gen_dic['d_type'] == 'continuous'
                   else gen_dic['no_of_treat'])

    if maxworkers == 1 or mp_over_boots:
        weights = initialise_weights(n_x, n_y, no_of_treat,
                                     int_dic['weight_as_sparse'])
        split_forest = False
        for idx in range(n_x):  # Iterate over i
            results_fut_idx = weights_obs_i(
                idx, n_y, forest_dic['forest'], x_dat, d_dat, cf_dic, ct_dic,
                gen_dic, int_dic, mp_over_boots, maxworkers, False,
                treat_is_integer=treat_is_integer)
            if int_dic['with_output'] and int_dic['verbose']:
                mcf_gp.share_completed(idx+1, n_x)
            if int_dic['weight_as_sparse']:
                for d_idx in range(no_of_treat):
                    indices = results_fut_idx[1][d_idx][0]
                    weights_obs = results_fut_idx[1][d_idx][1]
                    weights[d_idx][idx, indices] = weights_obs
            else:
                weights[idx] = results_fut_idx[1]
            empty_leaf_counter += results_fut_idx[2]
        if int_dic['weight_as_sparse']:
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
            mcf_ps.print_mcf(gen_dic, txt, summary=False)
        all_idx_split = np.array_split(range(n_x), no_of_splits_i)
        split_forest = False
        boot_indx_list = range(1)
        no_of_boot_splits = total_bootstraps = None

        if (not int_dic['weight_as_sparse']) and split_forest:
            if print_yes:
                txt = '\n' + 'XXXXXXXXDANGERXXXX' * 3
                txt += ('\nBootstrap splitting requires using sparse matrices.'
                        '\nProgramme continues without bootstrap splitting'
                        ' but may crash due to insufficient memory.')
                txt += '\n' + 'XXXXXXXX DANGER XXXXXXXX' * 3
                mcf_ps.print_mcf(gen_dic, txt, summary=True)

        reference_duration = +np.inf if bigdata_train else None
        for b_i, boots_ind in enumerate(boot_indx_list):
            time_start = time() if bigdata_train else None
            weights = initialise_weights(n_x, n_y, no_of_treat,
                                         int_dic['weight_as_sparse'])
            if split_forest:
                # get subforest
                forest_temp = forest_dic['forest'][boots_ind[0]:
                                                   boots_ind[-1]+1]
                if print_yes and b_i == 0:
                    size = mcf_sys.total_size(forest_temp) / (1024*1024)
                    txt = f'\nSize of each submitted forest {size:6.2f} MB'
                    mcf_ps.print_mcf(gen_dic, txt, summary=False)
                cf_dic['boot'] = len(boots_ind)
                # weights über trees addieren
                if print_yes:
                    print(f'\nBoot Chunk {b_i+1:2} of {no_of_boot_splits:2}')
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    mcf_ps.print_mcf(gen_dic, txt, summary=False)

            if not ray.is_initialized():
                _, no_of_workers = mcf_sys.init_ray_with_fallback(
                    maxworkers, int_dic, gen_dic,
                    mem_object_store=int_dic['mem_object_store_3'],
                    ray_err_txt='Ray does not start up in weight estimation 1'
                    )
            else:
                no_of_workers = maxworkers

            if (int_dic['mem_object_store_3'] is not None
                and int_dic['with_output'] and int_dic['verbose']
                    and with_output):
                size = round(int_dic['mem_object_store_3']/(1024*1024))
                txt = f'Size of Ray Object Store: {size} MB'
                mcf_ps.print_mcf(gen_dic, txt, summary=False)
            x_dat_ref = ray.put(x_dat)
            forest_ref = ray.put(forest_dic['forest'])
            still_running = [ray_weights_many_obs_i.remote(
                idx_list, n_y, forest_ref, x_dat_ref, d_dat, cf_dic,
                ct_dic, gen_dic, int_dic, split_forest,
                treat_is_integer=treat_is_integer
                )
                for idx_list in all_idx_split]

            jdx = 0
            time_start_ray = time() if bigdata_train else 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
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
                        if int_dic['weight_as_sparse']:
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
            if 'refs' in int_dic['mp_ray_del']:
                del x_dat_ref, forest_ref
            if 'rest' in int_dic['mp_ray_del']:
                del finished_res, finished
            # if int_dic['mp_ray_shutdown']:
            #     ray.shutdown()
            #     mcf_ps.print_mcf(gen_dic, 'Ray is shuting down.',
            #                      summary=False)
            if int_dic['weight_as_sparse']:  # Transform to sparse matrix
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
                        weights_all, int_dic['weight_as_sparse'], no_of_treat)
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    mcf_ps.print_mcf(gen_dic, ' ' + txt, summary=False)

        # Check if restart of ray is necessary
        if bigdata_train:
            duration = time() - time_start
            reference_duration, _ = mcf_sys.check_ray_shutdown(
                gen_dic, reference_duration, duration, no_of_workers,
                max_multiplier=5,
                with_output=print_yes,
                err_txt='\nProblem in weight computation (inner function): '
                )
        else:
            duration = None

    if split_forest:
        cf_dic['boot'] = total_bootstraps
        weights = normalize_weights(weights_all, no_of_treat,
                                    int_dic['weight_as_sparse'], n_x)
    weights = tuple(weights)
    if ((empty_leaf_counter > 0) or (merge_leaf_counter > 0)) and print_yes:
        txt = (f'\n\n{merge_leaf_counter:5} observations attributed in merged'
               f' leaves\n{empty_leaf_counter:5} observations attributed to'
               ' leaf w/o observations')
        mcf_ps.print_mcf(gen_dic, txt, summary=True)
    return weights, y_dat, x_bala, cl_dat, w_dat


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


def normal_weights_sparse_loop(weights: sparse.csr, n_x: int) -> sparse.csr:
    """Normalize sparse weight matrix (which is the heavy duty case."""
    row_sum = 1 / weights.sum(axis=1)
    for i in range(n_x):
        weights[i, :] = weights[i, :].multiply(
            row_sum[i])
    weights = weights.astype(np.float32, casting='same_kind')

    return weights


def normal_weights_sparse(weights: sparse.csr, n_x: int) -> sparse.csr:
    """Normalize sparse weight matrix (which is the heavy duty case."""
    row_sum_inv = 1 / weights.sum(axis=1)
    # Create a diagonal matrix with the reciprocals of row sums
    diag_matrix = sparse.csr_array((row_sum_inv.toarray().ravel(),
                                   (range(n_x), range(n_x))))
    # Multiply the original sparse matrix by the diagonal matrix
    normalized_weights = weights @ diag_matrix
    # Convert to float32
    normalized_weights = normalized_weights.astype(np.float32,
                                                   casting='same_kind')

    return normalized_weights


def initialise_weights(n_x: int, n_y: int, no_of_treat: int,
                       weight_as_sparse: bool
                       ) -> list:
    """Initialise the weights matrix."""
    if weight_as_sparse:
        weights = []
        for _ in range(no_of_treat):
            weights.append(sparse.lil_array((n_x, n_y), dtype=np.float32))
    else:
        weights = [None for _ in range(n_x)]

    return weights


def weights_obs_i(idx: int, n_y: int,
                  forest: list,
                  x_dat: np.ndarray, d_dat: np.ndarray,
                  cf_dic: dict, ct_dic: dict, gen_dic: dict, int_dic: dict,
                  mp_over_boots: bool = False, maxworkers: int = 1,
                  split_forest: bool = False,
                  treat_is_integer: bool = True,
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
    cf_dic : Dict. Parameters
    ct_dic : Dict. Parameters
    gen_dic : Dict. Parameters
    int_dic : Dict. Parameters
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
    bigdata_train = int_dic['bigdata_train']
    empty_leaf_counter = 0
    if gen_dic['d_type'] == 'continuous':
        no_of_treat = len(ct_dic['grid_w_val'])
        d_values = ct_dic['grid_w_val']
        continuous = True
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        continuous = False

    # Allocate weight matrix
    dtype = np.float32 if bigdata_train else np.float64
    weights_i_np = np.zeros((n_y, no_of_treat + 1), dtype=dtype)

    weights_i = [None for _ in range(no_of_treat)]  # 1 list for each treatment
    weights_i_np[:, 0] = np.arange(n_y)  # weight for index of outcomes
    x_dat_i = x_dat[idx, :]
    if mp_over_boots:
        if not ray.is_initialized():
            mcf_sys.init_ray_with_fallback(
                maxworkers, int_dic, gen_dic,
                ray_err_txt='Ray does not start in weight estimation.'
                )
        still_running = [ray_weights_obs_i_inside_boot.remote(
            forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat,
            continuous, bigdata_train, treat_is_integer=treat_is_integer)
            for boot in range(cf_dic['boot'])]
        while len(still_running) > 0:
            finished, still_running = ray.wait(still_running)
            finished_res = ray.get(finished)
            for result in finished_res:
                if result[1]:
                    empty_leaf_counter += 1
                else:
                    weights_i_np[:, 1:] += result[0]
    else:
        # This is the default branch
        for boot in range(cf_dic['boot']):
            weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
                forest[boot], x_dat_i, n_y, no_of_treat, d_values, d_dat,
                continuous, bigdata_train=bigdata_train,
                treat_is_integer=treat_is_integer
                )
            if empty_leaf:
                empty_leaf_counter += 1
            else:
                weights_i_np[:, 1:] += weights_ij_np
    obs_without_leaf = 1 if empty_leaf_counter == cf_dic['boot'] else 0
    normalize = not split_forest

    weights_i = final_trans(weights_i_np, no_of_treat, normalize)

    return idx, weights_i, obs_without_leaf


@ray.remote
def ray_weights_obs_i_inside_boot(forest_b, x_dat_i, n_y, no_of_treat,
                                  d_values, d_dat, continuous, bigdata_train,
                                  treat_is_integer=True):
    """Allow for MP at bootstrap level (intermediate procedure) with ray."""
    weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, continuous,
        bigdata_train=bigdata_train, treat_is_integer=treat_is_integer)
    return weights_ij_np, empty_leaf


# Old version of June, 6, 2025
# def weights_obs_i_inside_boot(forest_b: dict,
#                               x_dat_i: np.ndarray,
#                               n_y: int,
#                               no_of_treat: int,
#                               d_values: int,
#                               d_dat: np.ndarray,
#                               continuous: bool,
#                               bigdata_train: bool = False
#                               ) -> tuple[np.ndarray, bool]:
#     """Allow for MP at bootstrap level (intermediate procedure)."""
#     # here is the information for the different treatments needed
#     # Problem ist dass man hier für die x alle potential outcomes berechnet
#     # Bei continuous treatments gehören aber verschiedene leafs zu dengleichen
#     # X. Ausserdem werden d=0 und die anderen Werte von d verschiedenbehandelt
#     # x_dat_i ist eine design matrix, die dann auch die verschiedenenwerte von
#     # d enthalten sollte
#     # d = 0 zu setzen scheint aber überflüssig bei der designmatrix bzgl. d
#     fill_y_empty_leave = forest_b['fill_y_empty_leave']
#     fill_y_indices_list = forest_b['fill_y_indices_list']

#     if continuous:  # Not final, inefficient check when it becomes relevant
#         leaf_id_list = []
#         # We get a list of leafs that contains relevant elements, some of them
#         # may be identical, but this does not matter
#         for treat in d_values[1:]:
#             x_dat_i_t = np.append(x_dat_i, treat)
#             leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i_t)
#             leaf_id_list.append(leaf_id)
#             if (fill_y_empty_leave[leaf_id] == 1
#                     or fill_y_indices_list[leaf_id] is None):
#                 empty_leaf, weights_ij_np = True, 0
#                 # If any of the subleaves is None, then stop ...
#                 break
#             empty_leaf = False

#     else:
#         leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i)
#         if (fill_y_empty_leave[leaf_id] == 1
#                 or fill_y_indices_list[leaf_id] is None):
#             empty_leaf, weights_ij_np = True, 0
#         else:
#             empty_leaf = False

#     if not empty_leaf:
#         if continuous:
#             fb_lid_14_list = [fill_y_indices_list[leaf_id]
#                               for leaf_id in leaf_id_list]
#         else:
#             fb_lid_14 = fill_y_indices_list[leaf_id]
#         if bigdata_train:
#             weights_ij_np = np.zeros((n_y, no_of_treat))
#         else:
#             weights_ij_np = np.zeros((n_y, no_of_treat), dtype=np.float32)
#         if continuous:
#             # We need to collect information over various leafs for the 0
#             # For the other leaves, the leaves are treatment specific
#             leaf_0_complete, leaf_pos_complete = False, True
#             # Zuerst 0 einsammeln
#             for jdx, fb_lid_14 in enumerate(fb_lid_14_list):
#                 d_ib = d_dat[fb_lid_14].reshape(-1)  # view
#                 indices_ibj_0 = d_ib < 1e-15
#                 indices_ibj_pos = d_ib >= 1e-15
#                 if np.any(indices_ibj_0):  # any valid observations?
#                     fb_lid_14_indi_0 = fb_lid_14[indices_ibj_0]
#                     n_x_i_0 = len(fb_lid_14_indi_0)
#                     weights_ij_np[fb_lid_14_indi_0, 0] += 1 / n_x_i_0
#                     leaf_0_complete = True
#                 if np.any(indices_ibj_pos):  # any valid observations?
#                     fb_lid_14_indi_pos = fb_lid_14[indices_ibj_pos]
#                     n_x_i_pos = len(fb_lid_14_indi_pos)
#                     weights_ij_np[fb_lid_14_indi_pos, jdx+1] += (
#                         1 / n_x_i_pos)
#                 else:
#                     leaf_pos_complete = False
#                     break
#             leaf_complete = leaf_0_complete and leaf_pos_complete
#         else:
#             d_ib = d_dat[fb_lid_14].reshape(-1)  # view
#             leaf_complete = True
#             for jdx, treat in enumerate(d_values):
#                 indices_ibj = d_ib == treat
#                 if np.any(indices_ibj):  # any valid observations?
#                     fb_lid_14_indi = fb_lid_14[indices_ibj]
#                     n_x_i = len(fb_lid_14_indi)
#                     weights_ij_np[fb_lid_14_indi, jdx] += 1 / n_x_i
#                 else:
#                     leaf_complete = False
#                     break
#         if not leaf_complete:
#             empty_leaf, weights_ij_np = True, 0

#     return weights_ij_np, empty_leaf


def weights_obs_i_inside_boot(forest_b: dict,
                              x_dat_i: np.ndarray,
                              n_y: int,
                              no_of_treat: int,
                              d_values: Sequence[int | float],
                              d_dat: np.ndarray,
                              continuous: bool,
                              treat_is_integer: bool = True,
                              bigdata_train: bool = False
                              ) -> tuple[np.ndarray, bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # This is the default branch
    if continuous:
        # Continuous treatments
        return weights_obs_i_inside_boot_treat_cont(
            forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, bigdata_train
            )
    # Discrete treatments
    return weights_obs_i_inside_boot_treat_disc(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat,
        bigdata_train=bigdata_train, treat_is_integer=treat_is_integer
        )


def weights_obs_i_inside_boot_treat_cont(forest_b: dict,
                                         x_dat_i: np.ndarray,
                                         n_y: int,
                                         no_of_treat: int,
                                         d_values: Sequence[int | float],
                                         d_dat: np.ndarray,
                                         bigdata_train: bool = False
                                         ) -> tuple[np.ndarray, bool]:
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
        leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i_t)
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
            indices_ibj_0 = d_ib < 1e-15
            indices_ibj_pos = d_ib >= 1e-15
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


def weights_obs_i_inside_boot_treat_disc(forest_b: dict,
                                         x_dat_i: np.ndarray,
                                         n_y: int,
                                         no_of_treat: int,
                                         d_values: Sequence[int | float],
                                         d_dat: np.ndarray,
                                         treat_is_integer: bool = True,
                                         bigdata_train: bool = False
                                         ) -> tuple[np.ndarray, bool]:
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # This function is used in the default computations.
    _OPTIMIZED = True   # Optimized version of July, 8, 2025

    if _OPTIMIZED:
        empty_flags = forest_b['fill_y_empty_leave']
        inds_list = forest_b['fill_y_indices_list']
        leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i)

        leaf_inds = inds_list[leaf_id]
        if empty_flags[leaf_id] == 1 or leaf_inds is None:
            return 0, True

        # 1) pull out the raw treatment slice
        if treat_is_integer:
            d_leaf = d_dat[leaf_inds].ravel()
        else:
            raw = d_dat[leaf_inds]
            # 2) make sure we get a 1-D int array
            if raw.dtype == object:
                # each element is likely an array/sequence → concatenate
                d_leaf = np.concatenate([np.asarray(r).ravel() for r in raw])
            else:
                # a normal numeric slice → just flatten
                d_leaf = raw.ravel()
            d_leaf = d_leaf.astype(int)

        # 3) count and check completeness
        counts = np.bincount(d_leaf, minlength=no_of_treat)
        if (counts == 0).any():
            return 0, True

        # 4) per-obs weight and scatter
        per_obs = 1.0 / counts[d_leaf]
        dtype = np.float32 if bigdata_train else np.float64
        weights = np.zeros((n_y, no_of_treat), dtype=dtype)
        weights[leaf_inds, d_leaf] = per_obs

        return weights, False

    fill_y_empty_leave = forest_b['fill_y_empty_leave']
    fill_y_indices_list = forest_b['fill_y_indices_list']

    leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i)

    leaf_inds = fill_y_indices_list[leaf_id]
    if fill_y_empty_leave[leaf_id] == 1 or leaf_inds is None:
        return 0, True     # weights_ij_np, empty_leaf

    # Allocate weight matrix
    dtype = np.float32 if bigdata_train else np.float64
    weights_ij_np = np.zeros((n_y, no_of_treat), dtype=dtype)

    empty_leaf = False

    fb_lid_14 = leaf_inds
    d_ib = d_dat[fb_lid_14].reshape(-1)  # view
    leaf_complete = True
    for jdx, treat in enumerate(d_values):
        indices_ibj = d_ib == treat
        if np.any(indices_ibj):  # any valid observations?
            fb_lid_14_indi = fb_lid_14[indices_ibj]
            n_x_i = len(fb_lid_14_indi)
            weights_ij_np[fb_lid_14_indi, jdx] += 1 / n_x_i
        else:
            leaf_complete = False
            break

    if not leaf_complete:
        empty_leaf, weights_ij_np = True, 0

    return weights_ij_np, empty_leaf


def final_trans_old(weights_i_np, no_of_treat, normalize=True):
    """
    Compute last transformations of (positive only) weights.

    Parameters
    ----------
    weights_i_np : 2D Numpy array. Weights including zeros.
    no_of_treat : Int. Number of treatments.
    normalize : Bool. Normalize weights to row sum of 1. Default is True.

    Returns
    -------
    weights_i: List of lists (of different lengths).

    """
    weights_i = [None for _ in range(no_of_treat)]
    for jdx in range(no_of_treat):
        weights_t = weights_i_np[weights_i_np[:, jdx+1] > 1e-14]
        weights_ti = np.int32(weights_t[:, 0])  # Indices
        if normalize:
            weights_tw = weights_t[:, jdx+1] / np.sum(weights_t[:, jdx+1])
        else:
            weights_tw = weights_t[:, jdx+1].copy()
        weights_i[jdx] = [weights_ti, weights_tw.astype(np.float32)]
    return weights_i


def final_trans(weights_i_np: np.ndarray,
                no_of_treat: int,
                normalize: bool = True
                ) -> list[np.ndarray, np.ndarray]:
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
    # This function is used in the main branch
    _OPTIMIZED = True   # Optimized version of July, 8, 2025

    if _OPTIMIZED:
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

    else:  # OLd version that worked fine, but maybe slower
        weights_i = [None for _ in range(no_of_treat)]
        for jdx in range(no_of_treat):
            mask = weights_i_np[:, jdx + 1] > 1e-14
            weights_t = weights_i_np[mask]
            weights_ti = np.int32(weights_t[:, 0])  # Indices
            weights_tw = weights_t[:, jdx + 1]
            if normalize:
                weights_tw /= np.sum(weights_tw)
            weights_i[jdx] = [weights_ti, weights_tw.astype(np.float32)]

    return weights_i


def weights_to_csr(weights, no_of_treat):
    """Convert list of lil sparse matrices to csr format."""
    for d_idx in range(no_of_treat):
        weights[d_idx] = weights[d_idx].tocsr()

    return weights


@ray.remote
def ray_weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
                           gen_dic, int_dic, split_forest=False,
                           treat_is_integer=True):
    """Make function compatible with Ray."""
    return weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic,
                              ct_dic, gen_dic, int_dic, split_forest,
                              treat_is_integer=treat_is_integer)


def weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
                       gen_dic, int_dic, split_forest=False,
                       treat_is_integer=True
                       ):
    """
    Create chunks of task to be efficiently executed by MP.

    Parameters
    ----------
    idx_list : List of Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    cf_dic, ct_dic, gen_dic : Dict. Parameters
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : List of Int. Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.
    merge_leaf_counter : Int.

    """
    weights = [None for _ in range(len(idx_list))]
    empty_leaf_counter = 0
    for idx, val in enumerate(idx_list):
        ret = weights_obs_i(val, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
                            gen_dic, int_dic, mp_over_boots=False, maxworkers=1,
                            split_forest=split_forest,
                            treat_is_integer=treat_is_integer
                            )
        weights[idx] = ret[1]
        empty_leaf_counter += ret[2]
    return idx_list, weights, empty_leaf_counter
