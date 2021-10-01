"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the weight computation of MCF
@author: MLechner
-*- coding: utf-8 -*-
"""
from concurrent import futures
import numpy as np
import pandas as pd
from scipy import sparse
import ray
from mcf import general_purpose as gp
from mcf import general_purpose_system_files as gp_sys
from mcf import general_purpose_mcf as gp_mcf
from mcf import mcf_forest_functions as mcf_forest


def get_weights_mp(forest, x_file, y_file, v_dict, c_dict, x_name,
                   regrf=False):
    """Get weights for obs in pred_data & outcome and cluster from y_data.

    Parameters
    ----------
    forest : Tuple of lists. Node_table.
    x_file : String. csv-file with data to make predictions for.
    y_file : String. csv-file with outcome data.
    v_dict :  Dict. Variables.
    c_dict :  Dict. Parameters.
    x_name : List of str.
    regrf : Bool. Honest regression Random Forest. Default is False.

    Returns
    -------
    weights : Tuple of lists (N_pred x 1 (x no_of_treat + 1).
    y_data : N_y x number of outcomes-Numpy array. Outcome variables.
    x_bala : N_y x number of balancing-vars-Numpy array. Balancing test vars.
    cl_dat : N_y x 1 Numpy array. Cluster number.
    w_dat: N_y x 1 Numpy array. Sampling weights (if used).
    """
    gp_sys.clean_futures()
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nObtaining weights from estimated forest')
    data_x = pd.read_csv(x_file)
    x_dat = data_x[x_name].to_numpy()
    del data_x
    data_y = pd.read_csv(y_file)
    y_dat = data_y[v_dict['y_name']].to_numpy()
    if regrf:
        d_dat = None
    else:
        d_dat = data_y[v_dict['d_name']].to_numpy()
    if not regrf:
        d_dat = np.int16(np.round(d_dat))
    n_x = len(x_dat)
    n_y = len(y_dat)
    if c_dict['cluster_std']:
        cl_dat = data_y[v_dict['cluster_name']].to_numpy()
    else:
        cl_dat = 0
    if c_dict['w_yes']:
        w_dat = data_y[v_dict['w_name']].to_numpy()
    else:
        w_dat = 0
    x_bala = 0
    if not regrf:
        if c_dict['balancing_test_w']:
            x_bala = data_y[v_dict['x_balance_name']].to_numpy()
    del data_y
    empty_leaf_counter = 0
    merge_leaf_counter = 0
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        if c_dict['mp_automatic']:
            maxworkers = gp_mcf.find_no_of_workers(c_dict['no_parallel'],
                                                   c_dict['sys_share'])
        else:
            maxworkers = c_dict['no_parallel']
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1 or c_dict['mp_with_ray']:
        mp_over_boots = False
    else:
        if regrf:
            mp_over_boots = False
        else:
            mp_over_boots = bool(c_dict['mp_type_weights'] == 2)
    if maxworkers == 1 or mp_over_boots:
        weights = initialise_weights(c_dict, n_x, n_y, regrf)
        split_forest = False
        for idx in range(n_x):
            results_fut_idx = weights_obs_i(
                idx, n_y, forest, x_dat, d_dat, c_dict, regrf,
                mp_over_boots, maxworkers, False)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, n_x)
            if c_dict['weight_as_sparse']:
                for d_idx in range(c_dict['no_of_treat']):
                    indices = results_fut_idx[1][d_idx][0]
                    weights_obs = results_fut_idx[1][d_idx][1]
                    weights[d_idx][idx, indices] = weights_obs
                    if regrf:
                        break
            else:
                weights[idx] = results_fut_idx[1]
            empty_leaf_counter += results_fut_idx[2]
            merge_leaf_counter += results_fut_idx[3]
        if c_dict['weight_as_sparse']:
            weights = weights_to_csr(weights, c_dict['no_of_treat'], regrf)
    else:
        no_of_splits_i = maxworkers
        max_size_i = 1000
        if n_x / no_of_splits_i > max_size_i:
            while True:
                no_of_splits_i += maxworkers
                if n_x / no_of_splits_i <= max_size_i:
                    break
        if c_dict['with_output'] and c_dict['verbose']:
            print()
            print('Operational characteristics of weight estimation I:')
            print('Number of workers {:2}'.format(maxworkers))
            print('Number of observation chunks: {:5}'.format(no_of_splits_i))
            print('Average # of observations per chunck: {:5.2f}'.format(
                n_x / no_of_splits_i))
        all_idx_split = np.array_split(range(n_x), no_of_splits_i)
        split_forest = False
        if not c_dict['mp_with_ray']:
            if c_dict['mp_weights_tree_batch'] > 1:  # User def. # of batches
                no_of_boot_splits = c_dict['mp_weights_tree_batch']
                split_forest = True
                if c_dict['with_output'] and c_dict['verbose']:
                    print()
                    print('User determined number of tree batches')
            elif c_dict['mp_weights_tree_batch'] == 0:  # Automatic # of batch
                size_of_forest_mb = gp_sys.total_size(forest) / (1024 * 1024)
                no_of_boot_splits = gp_mcf.no_of_boot_splits_fct(
                    size_of_forest_mb, maxworkers, c_dict['with_output'])
                if no_of_boot_splits > 1:
                    split_forest = True
                else:
                    if c_dict['with_output']:
                        print('No tree batching')
        if split_forest:
            boot_indx_list = np.array_split(range(c_dict['boot']),
                                            no_of_boot_splits)
            total_bootstraps = c_dict['boot']
            if c_dict['with_output'] and c_dict['verbose']:
                print('Number of bootstrap chunks: {:5}'.format(
                    no_of_boot_splits))
                print('Average # of bootstraps per chunck: {:5.2f}'.format(
                    c_dict['boot']/no_of_boot_splits))
        else:
            boot_indx_list = range(1)
        if (not c_dict['weight_as_sparse']) and split_forest:
            if c_dict['with_output'] and c_dict['verbose']:
                print('XXXXXXXXDANGERXXXX' * 3)
                print('Bootstrap splitting requires using sparse',
                      ' matrices.')
                print('Programme continues without bootstrap splitting',
                      ' but may crash due to insufficient memory.')
                print('XXXXXXXXDANGERXXXX' * 3)
        for b_i, boots_ind in enumerate(boot_indx_list):
            weights = initialise_weights(c_dict, n_x, n_y, regrf)
            if split_forest:
                # get subforest
                # forest_temp = [forest[boot] for boot in boots_ind]
                forest_temp = forest[boots_ind[0]:boots_ind[-1]+1]
                if c_dict['with_output'] and c_dict['verbose'] and b_i == 0:
                    print('Size of each submitted forest {:6.2f} MB'
                          .format(gp_sys.total_size(forest_temp)/(1024*1024)))
                c_dict['boot'] = len(boots_ind)
                # weights über trees addieren
                if c_dict['with_output'] and c_dict['verbose']:
                    print()
                    print('Boot Chunk {:2} of {:2}'.format(
                        b_i+1, no_of_boot_splits))
                    gp_sys.memory_statistics()
            else:
                if not c_dict['mp_with_ray']:
                    forest_temp = forest
            if c_dict['mp_with_ray']:
                if c_dict['mem_object_store_3'] is None:
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=c_dict['mem_object_store_3'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ",
                              round(c_dict['mem_object_store_3']/(1024*1024)),
                              " MB")
                x_dat_ref = ray.put(x_dat)
                forest_ref = ray.put(forest)
                tasks = [ray_weights_many_obs_i.remote(
                    idx_list, n_y, forest_ref, x_dat_ref, d_dat, c_dict, regrf,
                    split_forest) for idx_list in all_idx_split]
                still_running = list(tasks)
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        if c_dict['with_output'] and c_dict['verbose']:
                            if jdx == 0:
                                print()
                                print('   Obs chunk {:2} ({:2})'.format(
                                    jdx+1, no_of_splits_i), end='')
                            else:
                                print(' {:2} ({:2})'.format(
                                    jdx+1, no_of_splits_i), end='')
                            jdx += 1
                        for idx, val_list in enumerate(results_fut_idx[0]):
                            if c_dict['weight_as_sparse']:
                                for d_idx in range(c_dict['no_of_treat']):
                                    indices = results_fut_idx[1][idx][d_idx][0]
                                    weights_obs = results_fut_idx[1][idx][d_idx
                                                                          ][1]
                                    idx_resized = np.resize(indices,
                                                            (len(indices), 1))
                                    weights[d_idx][val_list, idx_resized
                                                   ] = weights_obs
                                    if regrf:
                                        break
                            else:
                                weights[val_list] = results_fut_idx[1][idx]
                        empty_leaf_counter += results_fut_idx[2]
                        merge_leaf_counter += results_fut_idx[3]
                del x_dat_ref, forest_ref, finished, still_running, tasks
                ray.shutdown()
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        weights_many_obs_i, idx_list, n_y, forest_temp, x_dat,
                        d_dat, c_dict, regrf, split_forest):
                        idx_list for idx_list in all_idx_split}
                    for jdx, fur in enumerate(futures.as_completed(ret_fut)):
                        results_fut_idx = fur.result()
                        del ret_fut[fur]
                        del fur
                        if c_dict['with_output'] and c_dict['verbose']:
                            if jdx == 0:
                                print()
                                print('   Obs chunk {:2} ({:2})'.format(
                                    jdx+1, no_of_splits_i), end='')
                            else:
                                print(' {:2} ({:2})'.format(
                                    jdx+1, no_of_splits_i), end='')
                        idx_list = results_fut_idx[0]
                        for idx, val_list in enumerate(idx_list):
                            if c_dict['weight_as_sparse']:
                                for d_idx in range(c_dict['no_of_treat']):
                                    indices = results_fut_idx[1][idx][d_idx][0]
                                    weights_obs = results_fut_idx[1][idx][d_idx
                                                                          ][1]
                                    weights[d_idx][val_list, indices
                                                   ] = weights_obs
                                    if regrf:
                                        break
                            else:
                                weights[val_list] = results_fut_idx[1][idx]
                        empty_leaf_counter += results_fut_idx[2]
                        merge_leaf_counter += results_fut_idx[3]
            if c_dict['weight_as_sparse']:
                weights = weights_to_csr(weights, c_dict['no_of_treat'],
                                         regrf)
            if split_forest:
                if b_i == 0:  # only in case of splitted forests
                    weights_all = weights
                else:
                    for d_idx in range(c_dict['no_of_treat']):
                        weights_all[d_idx] += weights[d_idx]
                        if regrf:
                            break
                if c_dict['with_output'] and c_dict['verbose']:
                    print()
                    gp_mcf.print_size_weight_matrix(
                        weights_all, c_dict['weight_as_sparse'],
                        c_dict['no_of_treat'])
                    gp_sys.memory_statistics()
    if split_forest:
        c_dict['boot'] = total_bootstraps
        weights = normalize_weights(weights_all, c_dict['no_of_treat'], regrf,
                                    c_dict['weight_as_sparse'], n_x)
    weights = tuple(weights)
    if (((empty_leaf_counter > 0) or (merge_leaf_counter > 0))
            and c_dict['with_output']) and c_dict['verbose']:
        print('\n')
        print('{:5} observations attributed in merged leaves'.format(
            merge_leaf_counter))
        print('{:5} observations attributed to leaf w/o observations'.format(
            empty_leaf_counter))
    return weights, y_dat, x_bala, cl_dat, w_dat


def normalize_weights(weights, no_of_treat, regrf, sparse_m, n_x):
    """Normalise weight matrix (needed when forest is split)."""
    for d_idx in range(no_of_treat):
        if sparse_m:
            row_sum = 1 / weights[d_idx].sum(axis=1)
            for i in range(n_x):
                weights[d_idx][i, :] = weights[d_idx][i, :].multiply(
                    row_sum[i])
            weights[d_idx] = weights[d_idx].astype(np.float32,
                                                   casting='same_kind')
        else:
            for i in range(n_x):
                weights[i][d_idx][1] = weights[i][d_idx][1] / np.sum(
                    weights[i][d_idx][1])
                weights[i][d_idx][1] = weights[i][d_idx][1].astype(np.float32)
        if regrf:
            break
    return weights


def weights_to_csr(weights, no_of_treat, regrf):
    """Convert list of lil sparse matrices to csr format."""
    for d_idx in range(no_of_treat):
        weights[d_idx] = weights[d_idx].tocsr()
        if regrf:
            break
    return weights


def initialise_weights(c_dict, n_x, n_y, regrf):
    """Initialise the weights matrix."""
    if c_dict['weight_as_sparse']:
        weights = []
        for _ in range(c_dict['no_of_treat']):
            weights.append(sparse.lil_matrix((n_x, n_y), dtype=np.float32))
            if regrf:
                break
        # weights = [sparse.lil_matrix((n_x, n_y), dtype=np.float32)
        #            ] * c_dict['no_of_treat']
    else:
        weights = [None] * n_x
    return weights


@ray.remote
def ray_weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                           regrf=False, split_forest=False):
    """Make function compatible with Ray."""
    return weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                              regrf, split_forest)


def weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, c_dict,
                       regrf=False, split_forest=False):
    """
    Create chunks of task to be efficiently executed by MP.

    Parameters
    ----------
    idx_list : List of Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    c_dict : Dict. Parameters
    regrf: Bool. Regression random forest. Default is False.
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : List of Int. Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.
    merge_leaf_counter : Int.

    """
    weights = [None] * len(idx_list)
    empty_leaf_counter = 0
    merge_leaf_counter = 0
    for idx, val in enumerate(idx_list):
        results_fut_idx = weights_obs_i(
            val, n_y, forest, x_dat, d_dat, c_dict, regrf, mp_over_boots=False,
            maxworkers=1, split_forest=split_forest)
        weights[idx] = results_fut_idx[1]
        empty_leaf_counter += results_fut_idx[2]
        merge_leaf_counter += results_fut_idx[3]
    return idx_list, weights, empty_leaf_counter, merge_leaf_counter


def weights_obs_i(idx, n_y, forest, x_dat, d_dat, c_dict, regrf=False,
                  mp_over_boots=False, maxworkers=1, split_forest=False):
    """
    Compute weight for single observation to predict.

    Parameters
    ----------
    idx : Int. Counter.
    n_y: Int. Length of training data.
    forest : List of Lists.
    x_dat : Numpy array. Prediction sample.
    d_dat: Numpy array. Training sample.
    c_dict : Dict. Parameters
    regrf: Bool. Regression random forest. Default is False.
    mp_over_boots : Bool. Multiprocessing at level of bootstraps.
                          Default is False.
    maxworkers: Int. Number of workers if MP.
    split_forest : Boolean. True if chunks of bootstraps. Default is False.

    Returns
    -------
    idx : Int. Counter.
    weights_i : List of lists.
    empty_leaf_counter : Int.
    merge_leaf_counter : Int.

    """
    empty_leaf_counter = 0
    merge_leaf_counter = 0
    if regrf:
        weights_i_np = np.zeros((n_y, 2))
        weights_i = [None]
    else:
        weights_i_np = np.zeros((n_y, c_dict['no_of_treat'] + 1))
        weights_i = [None] * c_dict['no_of_treat']  # 1 list for each treatment
    weights_i_np[:, 0] = np.arange(n_y)  # weight for index of outcomes
    x_dat_i = x_dat[idx, :]
    if mp_over_boots:
        with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
            ret_fut = {fpp.submit(
                weights_obs_i_inside_boot, forest[boot], x_dat_i, regrf, n_y,
                c_dict['no_of_treat'], c_dict['d_values'], d_dat):
                boot for boot in range(c_dict['boot'])}
            for fur in futures.as_completed(ret_fut):
                result = fur.result()
                del ret_fut[fur]
                del fur
                if result[1]:
                    empty_leaf_counter += 1
                else:
                    weights_i_np[:, 1:] += result[0]
    else:
        for boot in range(c_dict['boot']):
            weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
                forest[boot], x_dat_i, regrf, n_y, c_dict['no_of_treat'],
                c_dict['d_values'], d_dat)
            if empty_leaf:
                empty_leaf_counter += 1
            else:
                weights_i_np[:, 1:] += weights_ij_np
    if empty_leaf_counter == c_dict['boot']:
        obs_without_leaf = 1
    else:
        obs_without_leaf = 0
    if split_forest:
        normalize = False
    else:
        normalize = True
    if regrf:
        weights_i = final_trans(weights_i_np, None, regrf, normalize)
    else:
        weights_i = final_trans(weights_i_np, c_dict['no_of_treat'], regrf,
                                normalize)
    return idx, weights_i, obs_without_leaf, merge_leaf_counter


def weights_obs_i_inside_boot(forest_b, x_dat_i, regrf, n_y, no_of_treat,
                              d_values, d_dat):
    """Allow for MP at bootstrap level (intermediate procedure)."""
    leaf_id = mcf_forest.get_terminal_leaf_no(forest_b, x_dat_i)
    if forest_b[leaf_id][14] is None:  # Leave will ignored
        # if int(leaf_id) % 2 == 0:  # even leaf id
        #     leaf_id -= 1
        # else:                      # odd leaf id
        #     leaf_id += 1
        empty_leaf = True
        weights_ij_np = 0
    else:
        empty_leaf = False
        fb_lid_14 = forest_b[leaf_id][14]
        if regrf:
            weights_ij_np = np.zeros((n_y, 1))
            # n_x_i = len(forest_b[leaf_id][14])
            # weights_ij_np[forest_b[leaf_id][14], 0] += 1 / n_x_i
            n_x_i = len(fb_lid_14)
            weights_ij_np[fb_lid_14, 0] += 1 / n_x_i
        else:
            weights_ij_np = np.zeros((n_y, no_of_treat))
            # d_ib = d_dat[forest_b[leaf_id][14]].reshape(-1)  # view
            d_ib = d_dat[fb_lid_14].reshape(-1)  # view
            leaf_complete = True
            for jdx, treat in enumerate(d_values):
                indices_ibj = d_ib == treat
                if np.any(indices_ibj):  # any valid observations?
                    # n_x_i = len(forest_b[leaf_id][14][indices_ibj])
                    # weights_ij_np[forest_b[leaf_id][14][indices_ibj],
                    #               jdx] += 1 / n_x_i
                    fb_lid_14_indi = fb_lid_14[indices_ibj]
                    n_x_i = len(fb_lid_14_indi)
                    weights_ij_np[fb_lid_14_indi, jdx] += 1 / n_x_i
                else:
                    leaf_complete = False
                    break
            if not leaf_complete:
                weights_ij_np = 0
                empty_leaf = True
    return weights_ij_np, empty_leaf


def final_trans(weights_i_np, no_of_treat, regrf, normalize=True):
    """
    Compute last transformations of (positive only) weights.

    Parameters
    ----------
    weights_i_np : 2D Numpy array. Weights including zeros.
    no_of_treat : Int. Number of treatments.
    regrf : Bool.
    normalize : Bool. Normalize weights to row sum of 1. Default is True.

    Returns
    -------
    weights_i: List of lists.

    """
    if regrf:
        iterator = 1
        weights_i = [None]
    else:
        iterator = no_of_treat
        weights_i = [None] * no_of_treat
    for jdx in range(iterator):
        weights_t = weights_i_np[weights_i_np[:, jdx+1] > 1e-14]
        weights_ti = np.int32(weights_t[:, 0])  # Indices
        if normalize:
            weights_tw = weights_t[:, jdx+1] / np.sum(weights_t[:, jdx+1])
        else:
            weights_tw = weights_t[:, jdx+1].copy()
        weights_tw = weights_tw.astype(np.float32)
        weights_i[jdx] = [weights_ti, weights_tw]
        if regrf:
            break
    return weights_i
