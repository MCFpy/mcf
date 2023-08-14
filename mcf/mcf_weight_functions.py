"""
Created on Sat Jun 17 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the weights.

@author: MLechner
-*- coding: utf-8 -*-
"""
import numpy as np
from scipy import sparse
import ray

from mcf import mcf_forest_add_functions as mcf_fo_add
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def get_weights_mp(mcf_, data_df, forest_dic, reg_round, with_output=True):
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
    if int_dic['with_output'] and int_dic['verbose'] and with_output:
        print('\nObtaining weights from estimated forest')
    if int_dic['weight_as_sparse_splits'] is None:
        if len(data_df) < 5000:
            int_dic['weight_as_sparse_splits'] = 1
        else:
            int_dic['weight_as_sparse_splits'] = round(
                len(forest_dic['y_train_df'])
                * len(data_df / cf_dic['folds']) / (20000 * 20000))
        if int_dic['weight_as_sparse_splits'] < 1:
            int_dic['weight_as_sparse_splits'] = 1
    x_dat_all = data_df[var_dic['x_name']].to_numpy()
    if int_dic['weight_as_sparse_splits'] == 1 or not int_dic[
            'weight_as_sparse']:
        weights, y_dat, x_bala, cl_dat, w_dat = get_weights_mp_inner(
            forest_dic, x_dat_all, cf_dic.copy(), ct_dic, gen_dic, int_dic,
            p_dic, with_output=with_output)
    else:
        base_file = gen_dic['outpath'] + '/' + 'temp_weights'
        x_dat_list = np.array_split(x_dat_all,
                                    int_dic['weight_as_sparse_splits'], axis=0)
        no_x_bala_return = True
        f_name = []
        for idx, x_dat in enumerate(x_dat_list):
            if int_dic['with_output'] and int_dic['verbose'] and with_output:
                print('\nWeights computation with additional splitting. ',
                      f'Mastersplit {idx} ({len(x_dat_list)})')
            if idx == len(x_dat_list) - 1:  # Last iteration, get x_bala
                no_x_bala_return = False
            weights_i, y_dat, x_bala, cl_dat, w_dat = get_weights_mp_inner(
                forest_dic, x_dat, cf_dic.copy(), ct_dic, gen_dic, int_dic,
                p_dic, no_x_bala_return=no_x_bala_return,
                with_output=with_output)
            f_name_i = []
            if int_dic['weight_as_sparse_splits'] > 1:
                elem_per_chunk = len(weights_i)
                for d_idx, weight_di in enumerate(weights_i):
                    f_name_d_idx = (base_file + str(idx) + '_' + str(d_idx)
                                    + '.npz')
                    mcf_sys.delete_file_if_exists(f_name_d_idx)
                    sparse.save_npz(f_name_d_idx, weight_di)
                    f_name_i.append(f_name_d_idx)
                f_name.append(f_name_i)
                del weights_i
            else:
                weights = weights_i
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
    if (int_dic['with_output'] and int_dic['verbose']
            and reg_round and with_output):
        no_of_treat = (len(ct_dic['grid_w_val'])
                       if gen_dic['d_type'] == 'continuous'
                       else gen_dic['no_of_treat'])
        txt = mcf_sys.print_size_weight_matrix(
            weights, int_dic['weight_as_sparse'], no_of_treat)
        ps.print_mcf(gen_dic, ' ' + txt, summary=False)
    weights_dic = {'weights': weights, 'y_dat_np': y_dat, 'x_bala_np': x_bala,
                   'cl_dat_np': cl_dat, 'w_dat_np': w_dat}
    return weights_dic


def get_weights_mp_inner(forest_dic, x_dat, cf_dic, ct_dic, gen_dic, int_dic,
                         p_dic, no_x_bala_return=False, with_output=True):
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
    y_dat = forest_dic['y_train_df'].to_numpy()
    d_dat = np.int16(np.round(forest_dic['d_train_df'].to_numpy()))
    n_x, n_y = len(x_dat), len(y_dat)
    cl_dat = (forest_dic['cl_train_df'].to_numpy()
              if p_dic['cluster_std'] else 0)
    w_dat = forest_dic['w_train_df'].to_numpy() if gen_dic['weighted'] else 0
    x_bala = 0
    if p_dic['bt_yes'] and not no_x_bala_return:
        x_bala = forest_dic['x_bala_df'].to_numpy()
    empty_leaf_counter = merge_leaf_counter = 0
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])
        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and int_dic['verbose'] and with_output:
        ps.print_mcf(gen_dic, f'Number of parallel processes: {maxworkers}',
                     summary=False)
    if maxworkers == 1 or int_dic['ray_or_dask'] == 'ray':
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
        for idx in range(n_x):
            results_fut_idx = weights_obs_i(
                idx, n_y, forest_dic['forest'], x_dat, d_dat, cf_dic, ct_dic,
                gen_dic, mp_over_boots, maxworkers, False)
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
            merge_leaf_counter += results_fut_idx[3]
        if int_dic['weight_as_sparse']:
            weights = weights_to_csr(weights, no_of_treat)
    else:
        no_of_splits_i, max_size_i = maxworkers, 1000
        if n_x / no_of_splits_i > max_size_i:
            while True:
                no_of_splits_i += maxworkers
                if n_x / no_of_splits_i <= max_size_i:
                    break
        if int_dic['with_output'] and int_dic['verbose'] and with_output:
            txt = ('\nOperational characteristics of weight estimation I'
                   f'\nNumber of workers: {maxworkers:2}'
                   f'\nNumber of observation chunks: {no_of_splits_i:5}'
                   '\nAverage # of observations per chunck:'
                   f' {n_x / no_of_splits_i:5.2f}')
            ps.print_mcf(gen_dic, txt, summary=False)
        all_idx_split = np.array_split(range(n_x), no_of_splits_i)
        split_forest = False
        if int_dic['ray_or_dask'] != 'ray':
            if int_dic['mp_weights_tree_batch'] > 1:  # User def. # of batches
                no_of_boot_splits = int_dic['mp_weights_tree_batch']
                split_forest = True
                if (int_dic['with_output'] and int_dic['verbose']
                        and with_output):
                    txt = '\nUser determined number of tree batches'
                    ps.print_mcf(gen_dic, txt, summary=False)
            elif int_dic['mp_weights_tree_batch'] == 0:  # Automatic # of batch
                size_of_forest_mb = mcf_sys.total_size(forest_dic['forest']
                                                       / (1024 * 1024))
                no_of_boot_splits, txt = mcf_sys.no_of_boot_splits_fct(
                    size_of_forest_mb, maxworkers)
                if (int_dic['with_output'] and int_dic['verbose']
                        and with_output):
                    ps.print_mcf(gen_dic, txt, summary=False)
                if no_of_boot_splits > 1:
                    split_forest = True
                else:
                    if int_dic['with_output'] and with_output:
                        ps.print_mcf(gen_dic, 'No tree batching',
                                     summary=False)
        if split_forest:
            boot_indx_list = np.array_split(range(cf_dic['boot']),
                                            no_of_boot_splits)
            total_bootstraps = cf_dic['boot']
            if int_dic['with_output'] and int_dic['verbose'] and with_output:
                txt = (f'\nNumber of bootstrap chunks: {no_of_boot_splits:5}'
                       '\nAverage # of bootstraps per chunck:'
                       f' {int_dic["boot"]/no_of_boot_splits:5.2f}')
                ps.print_mcf(gen_dic, txt, summary=False)
        else:
            boot_indx_list = range(1)
        if (not int_dic['weight_as_sparse']) and split_forest:
            if int_dic['with_output'] and int_dic['verbose'] and with_output:
                txt = '\n' + 'XXXXXXXXDANGERXXXX' * 3
                txt += ('\nBootstrap splitting requires using sparse matrices.'
                        '\nProgramme continues without bootstrap splitting'
                        ' but may crash due to insufficient memory.')
                txt += '\n' + 'XXXXXXXXDANGERXXXX' * 3
                ps.print_mcf(gen_dic, txt, summary=True)
        for b_i, boots_ind in enumerate(boot_indx_list):
            weights = initialise_weights(n_x, n_y, no_of_treat,
                                         int_dic['weight_as_sparse'])
            if split_forest:
                # get subforest
                forest_temp = forest_dic['forest'][boots_ind[0]:
                                                   boots_ind[-1]+1]
                if (int_dic['with_output'] and int_dic['verbose'] and b_i == 0
                        and with_output):
                    size = mcf_sys.total_size(forest_temp) / (1024*1024)
                    txt = f'\nSize of each submitted forest {size:6.2f} MB'
                    ps.print_mcf(gen_dic, txt, summary=False)
                cf_dic['boot'] = len(boots_ind)
                # weights über trees addieren
                if (int_dic['with_output'] and int_dic['verbose']
                        and with_output):
                    print(f'\nBoot Chunk {b_i+1:2} of {no_of_boot_splits:2}')
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    ps.print_mcf(gen_dic, txt, summary=False)
            else:
                if int_dic['ray_or_dask'] != 'ray':
                    forest_temp = forest_dic['forest']
            if int_dic['ray_or_dask'] == 'ray':
                if int_dic['mem_object_store_3'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=int_dic['mem_object_store_3'])
                    if (int_dic['with_output'] and int_dic['verbose']
                            and with_output):
                        size = round(int_dic['mem_object_store_3']/(1024*1024))
                        txt = f'Size of Ray Object Store: {size} MB'
                        ps.print_mcf(gen_dic, txt, summary=False)
                x_dat_ref = ray.put(x_dat)
                forest_ref = ray.put(forest_dic['forest'])
                still_running = [ray_weights_many_obs_i.remote(
                    idx_list, n_y, forest_ref, x_dat_ref, d_dat, cf_dic,
                    ct_dic, gen_dic, split_forest)
                    for idx_list in all_idx_split]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for results_fut_idx in finished_res:
                        if (int_dic['with_output'] and int_dic['verbose']
                                and with_output):
                            if jdx == 0:
                                print(f'\n   Obs chunk {jdx+1:2}'
                                      f' ({no_of_splits_i})', end='')
                            else:
                                print(f' {jdx+1:2} ({no_of_splits_i})',
                                      end='')
                            jdx += 1
                        for idx, val_list in enumerate(results_fut_idx[0]):
                            if int_dic['weight_as_sparse']:
                                for d_idx in range(no_of_treat):
                                    indices = results_fut_idx[1][idx][d_idx][0]
                                    weights_obs = results_fut_idx[1][idx][d_idx
                                                                          ][1]
                                    idx_resized = np.resize(indices,
                                                            (len(indices), 1))
                                    weights[d_idx][val_list, idx_resized
                                                   ] = weights_obs
                            else:
                                weights[val_list] = results_fut_idx[1][idx]
                        empty_leaf_counter += results_fut_idx[2]
                        merge_leaf_counter += results_fut_idx[3]
                if 'refs' in int_dic['mp_ray_del']:
                    del x_dat_ref, forest_ref
                if 'rest' in int_dic['mp_ray_del']:
                    del finished_res, finished
                if int_dic['mp_ray_shutdown']:
                    ray.shutdown()
            if int_dic['weight_as_sparse']:
                weights = weights_to_csr(weights, no_of_treat)
            if split_forest:
                if b_i == 0:  # only in case of splitted forests
                    weights_all = weights
                else:
                    for d_idx in range(no_of_treat):
                        weights_all[d_idx] += weights[d_idx]
                if (int_dic['with_output'] and int_dic['verbose']
                        and with_output):
                    print()
                    mcf_gp.print_size_weight_matrix(
                        weights_all, int_dic['weight_as_sparse'], no_of_treat)
                    _, _, _, _, txt = mcf_sys.memory_statistics()
                    ps.print_mcf(gen_dic, ' ' + txt, summary=False)
    if split_forest:
        cf_dic['boot'] = total_bootstraps
        weights = normalize_weights(weights_all, no_of_treat,
                                    int_dic['weight_as_sparse'], n_x)
    weights = tuple(weights)
    if (((empty_leaf_counter > 0) or (merge_leaf_counter > 0))
            and int_dic['with_output']) and int_dic['verbose'] and with_output:
        txt = (f'\n\n{merge_leaf_counter:5} observations attributed in merged'
               f' leaves\n{empty_leaf_counter:5} observations attributed to'
               ' leaf w/o observations')
        ps.print_mcf(gen_dic, txt, summary=True)
    return weights, y_dat, x_bala, cl_dat, w_dat


def normalize_weights(weights, no_of_treat, sparse_m, n_x):
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
    return weights


def initialise_weights(n_x, n_y, no_of_treat, weight_as_sparse):
    """Initialise the weights matrix."""
    if weight_as_sparse:
        weights = []
        for _ in range(no_of_treat):
            weights.append(sparse.lil_matrix((n_x, n_y), dtype=np.float32))
    else:
        weights = [None] * n_x
    return weights


def weights_obs_i(idx, n_y, forest, x_dat, d_dat, cf_dic, ct_dic, gen_dic,
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
    cf_dic : Dict. Parameters
    ct_dic : Dict. Parameters
    gen_dic : Dict. Parameters
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
    empty_leaf_counter = merge_leaf_counter = 0
    if gen_dic['d_type'] == 'continuous':
        no_of_treat = len(ct_dic['grid_w_val'])
        d_values = ct_dic['grid_w_val']
        continuous = True
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
        continuous = False
    weights_i_np = np.zeros((n_y, no_of_treat + 1))
    weights_i = [None] * no_of_treat  # 1 list for each treatment
    weights_i_np[:, 0] = np.arange(n_y)  # weight for index of outcomes
    x_dat_i = x_dat[idx, :]
    if mp_over_boots:
        if not ray.is_initialized():
            ray.init(num_cpus=maxworkers, include_dashboard=False)
            still_running = [ray_weights_obs_i_inside_boot.remote(
                weights_obs_i_inside_boot, forest[boot], x_dat_i, n_y,
                no_of_treat, d_values, d_dat, continuous)
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
        for boot in range(cf_dic['boot']):
            weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
                forest[boot], x_dat_i, n_y, no_of_treat, d_values,
                d_dat, continuous)
            if empty_leaf:
                empty_leaf_counter += 1
            else:
                weights_i_np[:, 1:] += weights_ij_np
    obs_without_leaf = 1 if empty_leaf_counter == cf_dic['boot'] else 0
    normalize = not split_forest
    weights_i = final_trans(weights_i_np, no_of_treat, normalize)
    return idx, weights_i, obs_without_leaf, merge_leaf_counter


@ray.remote
def ray_weights_obs_i_inside_boot(forest_b, x_dat_i, n_y, no_of_treat,
                                  d_values, d_dat, continuous):
    """Allow for MP at bootstrap level (intermediate procedure) with ray."""
    weights_ij_np, empty_leaf = weights_obs_i_inside_boot(
        forest_b, x_dat_i, n_y, no_of_treat, d_values, d_dat, continuous)
    return weights_ij_np, empty_leaf


def weights_obs_i_inside_boot(forest_b, x_dat_i, n_y, no_of_treat,
                              d_values, d_dat, continuous):
    """Allow for MP at bootstrap level (intermediate procedure)."""
    # here is the information for the different treatments needed
    # Problem ist dass man hier für die x alle potential outcomes berechnet
    # Bei continuous treatments gehören aber verschiedene leafs zu den gleichen
    # X. Ausserdem werden d=0 und die anderen Werte von d verschieden behandelt
    # x_dat_i ist eine design matrix, die dann auch die verschiedenen werte von
    # d enthalten sollte
    # d = 0 zu setzen scheint aber überflüssig bei der designmatrix bzgl. d
    if continuous:
        leaf_id_list = []
        # We get a list of leafs that contains relevant elements, some of them
        # may be identical, but this does not matter
        for treat in d_values[1:]:
            x_dat_i_t = np.append(x_dat_i, treat)
            leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i_t)
            leaf_id_list.append(leaf_id)
            if forest_b[leaf_id][14] is None:  # Leave will be ignored
                empty_leaf, weights_ij_np = True, 0
                # If any of the subleaves is None, then stop ...
                break
            empty_leaf = False

    else:
        leaf_id = mcf_fo_add.get_terminal_leaf_no(forest_b, x_dat_i)
        if forest_b[leaf_id][14] is None:  # Leave will be ignored
            empty_leaf, weights_ij_np = True, 0
        else:
            empty_leaf = False
    if not empty_leaf:
        if continuous:
            fb_lid_14_list = [forest_b[leaf_id][14]
                              for leaf_id in leaf_id_list]
        else:
            fb_lid_14 = forest_b[leaf_id][14]
        weights_ij_np = np.zeros((n_y, no_of_treat))
        if continuous:
            # We need to collect information over various leafs for the 0
            # For the other leaves, the leaves are treatment specific
            leaf_0_complete, leaf_pos_complete = False, True
            # Zuerst 0 einsammeln
            for jdx, fb_lid_14 in enumerate(fb_lid_14_list):
                # fb_lid_14 = fb_lid_14_list[jdx]
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
        else:
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


def final_trans(weights_i_np, no_of_treat, normalize=True):
    """
    Compute last transformations of (positive only) weights.

    Parameters
    ----------
    weights_i_np : 2D Numpy array. Weights including zeros.
    no_of_treat : Int. Number of treatments.
    normalize : Bool. Normalize weights to row sum of 1. Default is True.

    Returns
    -------
    weights_i: List of lists.

    """
    iterator, weights_i = no_of_treat, [None] * no_of_treat
    for jdx in range(iterator):
        weights_t = weights_i_np[weights_i_np[:, jdx+1] > 1e-14]
        weights_ti = np.int32(weights_t[:, 0])  # Indices
        weights_tw = (weights_t[:, jdx+1] / np.sum(weights_t[:, jdx+1])
                      if normalize else weights_t[:, jdx+1].copy())
        weights_tw = weights_tw.astype(np.float32)
        weights_i[jdx] = [weights_ti, weights_tw]
    return weights_i


def weights_to_csr(weights, no_of_treat):
    """Convert list of lil sparse matrices to csr format."""
    for d_idx in range(no_of_treat):
        weights[d_idx] = weights[d_idx].tocsr()
    return weights


@ray.remote
def ray_weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
                           gen_dic, split_forest=False):
    """Make function compatible with Ray."""
    return weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic,
                              ct_dic, gen_dic, split_forest)


def weights_many_obs_i(idx_list, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
                       gen_dic, split_forest=False):
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
    weights = [None] * len(idx_list)
    empty_leaf_counter = merge_leaf_counter = 0
    for idx, val in enumerate(idx_list):
        results_fut_idx = weights_obs_i(
            val, n_y, forest, x_dat, d_dat, cf_dic, ct_dic,
            gen_dic, mp_over_boots=False,
            maxworkers=1, split_forest=split_forest)
        weights[idx] = results_fut_idx[1]
        empty_leaf_counter += results_fut_idx[2]
        merge_leaf_counter += results_fut_idx[3]
    return idx_list, weights, empty_leaf_counter, merge_leaf_counter
