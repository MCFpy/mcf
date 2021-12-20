"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for the tree and forest computations of MCF
@author: MLechner
-*- coding: utf-8 -*-
"""
from concurrent import futures
import copy
import random
from numba import njit
import numpy as np
import ray
from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import general_purpose_system_files as gp_sys
from mcf import general_purpose_mcf as gp_mcf
from mcf import mcf_data_functions as mcf_data


def fill_trees_with_y_indices_mp(forest, indatei, v_dict, v_x_type, v_x_values,
                                 c_dictin, x_name_mcf, regrf=False):
    """Fill trees with indices of outcomes, MP.

    Parameters
    ----------
    forest : Tuple of lists. Node_table.
    indatei : String. csv-file with data.
    v_dict : Dict. Variables.
    v_x_type : Dict. Name and type of covariates.
    v_x_values : Dict. Name and values of covariates.
    c_dictin : Dict. Parameters.
    x_name_mcf : List of str.
    regrf : Bool. Regression or MCF. Default is False.

    Returns
    -------
    forest_with_y : List of lists. Updated Node_table.
    terminal_nodes: Tuple of np.arrays. No of final node.
    no_of_avg_nodes: INT. Average no of unfilled leafs.

    """
    if c_dictin['with_output'] and c_dictin['verbose']:
        print("\nFilling trees with indicies of outcomes")
    (x_name, _, _, c_dict, _, data_np, _, _, x_i, _, _, d_i, _, _
     ) = mcf_data.prepare_data_for_forest(
         indatei, v_dict, v_x_type, v_x_values, c_dictin, True, regrf=regrf)
    if x_name_mcf != x_name:
        raise Exception('Wrong order of variable names', x_name, x_name_mcf)
    x_dat = data_np[:, x_i]
    d_dat = np.int16(np.round(data_np[:, d_i]))
    obs = len(x_dat)
    # indices = np.arange(obs)
    terminal_nodes = [None] * c_dict['boot']
    nodes_empty = np.zeros(c_dict['boot'])
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
    if maxworkers == 1:
        for idx in range(c_dict['boot']):
            (_, forest[idx], terminal_nodes[idx], nodes_empty[idx]
             ) = fill_mp(forest[idx], obs, d_dat, x_dat, idx, c_dict, regrf)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, c_dict['boot'])
    else:
        if c_dict['mp_with_ray']:
            if c_dict['mem_object_store_2'] is None:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
            else:
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False,
                             object_store_memory=c_dict['mem_object_store_2'])
                if c_dict['with_output'] and c_dict['verbose']:
                    print("Size of Ray Object Store: ", round(
                        c_dict['mem_object_store_2']/(1024*1024)), " MB")
            x_dat_ref = ray.put(x_dat)
            still_running = [ray_fill_mp.remote(
                forest[idx], obs, d_dat, x_dat_ref, idx, c_dict, regrf)
                for idx in range(c_dict['boot'])]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for ret_all_i in finished_res:
                    iix = ret_all_i[0]
                    forest[iix] = ret_all_i[1]
                    terminal_nodes[iix] = ret_all_i[2]
                    nodes_empty[iix] = ret_all_i[3]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, c_dict['boot'])
                    jdx += 1
            if 'refs' in c_dict['_mp_ray_del']:
                del x_dat_ref
            if 'rest' in c_dict['_mp_ray_del']:
                del finished_res, finished
            if c_dict['_mp_ray_shutdown']:
                ray.shutdown()
        else:
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
                ret_fut = {fpp.submit(fill_mp, forest[idx], obs, d_dat, x_dat,
                                      idx, c_dict, regrf):
                           idx for idx in range(c_dict['boot'])}
                for jdx, frv in enumerate(futures.as_completed(ret_fut)):
                    ret_all_i = frv.result()
                    del ret_fut[frv]
                    del frv
                    iix = ret_all_i[0]
                    forest[iix] = ret_all_i[1]
                    terminal_nodes[iix] = ret_all_i[2]
                    nodes_empty[iix] = ret_all_i[3]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, c_dict['boot'])
    no_of_avg_enodes = np.mean(nodes_empty)
    if c_dict['with_output'] and c_dict['verbose']:
        print('\nNumber of leaves w/o all treatments per tree',
              ' in %: {:8.3f}'.format(no_of_avg_enodes*100))
        if no_of_avg_enodes > 0:
            print('Incomplete leafs will not be considered for weight',
                  'computation.')
    return forest, terminal_nodes, no_of_avg_enodes


@ray.remote
def ray_fill_mp(node_table, obs, d_dat, x_dat, b_idx, c_dict, regrf=False):
    """Make it work under Ray."""
    return fill_mp(node_table, obs, d_dat, x_dat, b_idx, c_dict, regrf)


def fill_mp(node_table, obs, d_dat, x_dat, b_idx, c_dict, regrf=False):
    """Compute new node_table and list of final leaves.

    Parameters
    ----------
    node_table : List of lists.
    obs : Int. Sample size.
    d_dat : Numpy array. Treatment.
    x_dat : Numpy array. Features.
    b_idx : Int. Tree number.
    c_dict : Dict. Controls.
    regrf: Bool. Regression or MCF. Default is False.

    Returns
    -------
    node_table : List of lists.
    unique_leafs : List.
    b_idx : Int. Tree number.

    """
    if not regrf:
        subsam = bool(c_dict['subsam_share_eval'] < 1)
    else:
        subsam = False
    indices = np.arange(obs)
    if subsam:
        obs = round(obs * c_dict['subsam_share_eval'])
        rng = np.random.default_rng((10+b_idx)**2+121)
        indices = rng.choice(indices, size=obs, replace=False)
        obs_in_leaf = np.zeros((obs, 1), dtype=np.uint32)
        for i, idx in enumerate(indices):
            obs_in_leaf[i] = get_terminal_leaf_no(node_table, x_dat[idx, :])
        unique_leafs = np.unique(obs_in_leaf)
        unique_leafs = unique_leafs[1:]  # remove first index: obs not used
        d_dat = d_dat[indices]
    else:
        obs_in_leaf = np.zeros((obs, 1), dtype=np.uint32)
        for idx in indices:
            obs_in_leaf[idx] = get_terminal_leaf_no(node_table, x_dat[idx, :])
        unique_leafs = np.unique(obs_in_leaf)
    nodes_empty = 0
    for leaf_id in unique_leafs:
        sel_ind = obs_in_leaf.reshape(-1) == leaf_id
        node_table[leaf_id][14] = indices[sel_ind]
        if regrf:
            empty_leaf = len(sel_ind) < 1
        else:
            empty_leaf = len(np.unique(d_dat[sel_ind])) < c_dict['no_of_treat']
        if empty_leaf:
            node_table[leaf_id][16] = 1   # Leaf to be ignored
            nodes_empty += 1
    return b_idx, node_table, unique_leafs, nodes_empty/len(unique_leafs)


def remove_oob_from_leaf0(forest):
    """Save memory by removing OOB indices.

    Parameters
    ----------
    forest : List of list. Node_tables.

    Returns
    -------
    forest_out : List of list. Node_tables.
    """
    for idx, _ in enumerate(forest):
        forest[idx][0][16] = 0
    return forest


def fs_adjust_vars(vi_i, vi_g, vi_ag, v_dict, v_x_type, v_x_values, x_name,
                   c_dict, regrf=False):
    """Deselect variables that have a too low variable importance.

    Parameters
    ----------
    vi_i : Tuple (List relative OOB, indices). Least important variables last.
    vi_g : Tuple (List relative OOB, indices). Least important group last.
    vi_ag : Tuple (List relative OOB, indices). Least import. accu. group last.
    v : Dict. Variables.
    v_x_type : Dict. Type of variable.
    v_x_values : Dict. Possible values of variables.
    x_name: List of strings. Names of covariates.
    c_dict : Dict. Parameters.
    regrf : Bool. Honest regression forest. Default is False.

    Returns
    -------
    var : Dict. Variables.
    var_x_type : Dict. Type of variable.
    var_x_values : Dict. Possible values of variables.

    """
    ind_i = np.array(vi_i[1], copy=True, dtype=object)
    ind_g = np.array(vi_g[1], copy=True, dtype=object)
    if vi_ag is None:
        ind_ag = ind_g
    else:
        ind_ag = np.array(vi_ag[1], copy=True, dtype=object)
    below_i = vi_i[0] <= (100 + c_dict['fs_rf_threshold'])
    below_g = vi_g[0] <= (100 + c_dict['fs_rf_threshold'])
    if vi_ag is None:
        below_ag = below_g
    else:
        below_ag = vi_ag[0] <= (100 + c_dict['fs_rf_threshold'])
    nothing_removed = True
    if ((np.count_nonzero(below_i) > 0) and (np.count_nonzero(below_g) > 0)
            and (np.count_nonzero(below_ag) > 0)):   # necessary conditions met
        ind_i = set(ind_i[below_i])
        indi_g_flat = set(gp_est.flatten_list(list(ind_g[below_g])))
        indi_ag_flat = set(gp_est.flatten_list(list(ind_ag[below_ag])))
        remove_ind = ind_i & indi_g_flat & indi_ag_flat
        if remove_ind:           # If list is empty, this will be False
            names_to_remove1 = []
            for i in remove_ind:
                names_to_remove1.append(x_name[i])
            if regrf:
                forbidden_vars = (v_dict['x_name_always_in']
                                  + v_dict['x_name_remain'])
            else:
                forbidden_vars = (v_dict['x_name_always_in']
                                  + v_dict['x_name_remain']
                                  + v_dict['z_name'] + v_dict['z_name_list']
                                  + v_dict['z_name_mgate']
                                  + v_dict['z_name_amgate'])
            names_to_remove2 = []
            for name in names_to_remove1:
                if name not in forbidden_vars:
                    names_to_remove2.append(name)
            if names_to_remove2:
                nothing_removed = False
                for name_weg in names_to_remove2:
                    v_x_type.pop(name_weg)
                    v_x_values.pop(name_weg)
                    v_dict['x_name'].remove(name_weg)
                if c_dict['with_output']:
                    print('\nVariables deleted: ', names_to_remove2)
                    print('\nVariables kept: ', v_dict['x_name'])
    if nothing_removed:
        if c_dict['with_output']:
            print('\n', 'No variables removed in feature selection')
    return v_dict, v_x_type, v_x_values


def oob_in_tree(obs_in_leaf, y_dat, y_nn, d_dat, w_dat, mtot, no_of_treat,
                treat_values, w_yes, regrf=False):
    """Compute OOB values for a tree.

    Parameters
    ----------
    obs_in_leaf : List of int. Terminal leaf no of observation
    y : Numpy array.
    y_nn : Numpy array.
    d : Numpy array.
    w : Numpy array.
    mtot : INT. Method used.
    no_of_treat : INT.
    treat_values : INT.
    w_yes : INT.

    Returns
    -------
    oob_tree : INT. OOB value of the MSE of the tree

    """
    leaf_no = np.unique(obs_in_leaf[:, 1])
    oob_tree = n_lost = n_total = 0
    if not regrf:
        mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
        obs_t_tree = np.zeros(no_of_treat)
    for leaf in leaf_no:
        in_leaf = obs_in_leaf[:, 1] == leaf
        if w_yes:
            w_l = w_dat[in_leaf]
        else:
            w_l = 0
        n_l = np.count_nonzero(in_leaf)
        if regrf:
            if n_l > 1:
                mse_oob = regrf_mse(y_dat[in_leaf],  w_l, n_l, w_yes)
                oob_tree += mse_oob * n_l
        else:
            d_dat_in_leaf = d_dat[in_leaf]  # makes a copy
            if n_l < no_of_treat:
                enough_data_in_leaf = False
            else:
                enough_data_in_leaf = True
                if n_l < 40:          # this is done for efficiency reasons
                    if set(d_dat_in_leaf.reshape(-1)) != set(treat_values):
                        enough_data_in_leaf = False
                else:
                    if len(np.unique(d_dat_in_leaf)) < no_of_treat:  # No MSE
                        enough_data_in_leaf = False
            if enough_data_in_leaf:
                mse_mce_leaf, _, obs_by_treat_leaf = mcf_mse(
                    y_dat[in_leaf], y_nn[in_leaf], d_dat_in_leaf, w_l, n_l,
                    mtot, no_of_treat, treat_values, w_yes)
                mse_mce_tree, obs_t_tree = add_rescale_mse_mce(
                    mse_mce_leaf, obs_by_treat_leaf, mtot, no_of_treat,
                    mse_mce_tree, obs_t_tree)
            else:
                n_lost += n_l
            n_total += n_l
    if not regrf:
        mse_mce_tree = get_avg_mse_mce(mse_mce_tree, obs_t_tree, mtot,
                                       no_of_treat)
        oob_tree = compute_mse_mce(mse_mce_tree, mtot, no_of_treat)
    return oob_tree


def get_terminal_leaf_no(node_table, x_dat):
    """Get the leaf number of the terminal node for single observation.

    Parameters
    ----------
    node_table : List of list. Single tree.
    x_dat : Numpy array. Data.

    Raises
    ------
    Exception
        If leaf turns out to be neither terminal nor inactive.

    Returns
    -------
    leaf_no : INT. Number of terminal leaf the observation belongs to.

    Note: This only works if nodes are ordered subsequently. Do not remove
          leafs when pruning. Only changes their activity status.

    """
    not_terminal = True
    leaf_id = 0
    while not_terminal:
        leaf = node_table[leaf_id]
        if leaf[4] == 1:             # Terminal leaf
            not_terminal = False
            leaf_no = leaf[0]
        elif leaf[4] == 0:          # Intermediate leaf
            if leaf[10] == 0:        # Continuous variable
                if (x_dat[leaf[8]] - 1e-15) <= leaf[9]:
                    leaf_id = leaf[2]
                else:
                    leaf_id = leaf[3]
            else:                   # Categorical variable
                prime_factors = gp.primes_reverse(leaf[9], False)
                if int(np.round(x_dat[leaf[8]])) in prime_factors:
                    leaf_id = leaf[2]
                else:
                    leaf_id = leaf[3]
        else:
            raise Exception('Leaf is still active.{}'.format(leaf[4]))
    return leaf_no


def get_tree_infos(forest):
    """Obtain some basic information about estimated trees.

    Parameters
    ----------
    forest : List of lists. Collection of node_tables.

    Returns
    -------
    leaf_info : List. Some information about tree.

    """
    leaf_info_tmp = np.zeros([len(forest), 6])
    for boot, tree in enumerate(forest):
        for leaf in tree:
            if leaf[4] == 1:   # Terminal leafs only
                leaf_info_tmp[boot, 0] += 1  # Number of leaves
        leaf_info_tree = np.zeros(int(leaf_info_tmp[boot, 0]))
        j = 0
        for leaf in tree:
            if leaf[4] == 1:
                leaf_info_tree[j] = leaf[5]
                j += 1
        leaf_info_tmp[boot, 1] = np.mean(leaf_info_tree)
        leaf_info_tmp[boot, 2] = np.median(leaf_info_tree)
        leaf_info_tmp[boot, 3] = np.min(leaf_info_tree)
        leaf_info_tmp[boot, 4] = np.max(leaf_info_tree)
        leaf_info_tmp[boot, 5] = np.sum(leaf_info_tree)
    leaf_info = np.empty(6)
    list_of_ind = [0, 1, 5]  # Average #, size of leaves, # of obs in leaves
    leaf_info[list_of_ind] = np.mean(leaf_info_tmp[:, list_of_ind], axis=0)
    leaf_info[2] = np.median(leaf_info_tmp[:, 2])   # Min size of leaves
    leaf_info[3] = np.min(leaf_info_tmp[:, 3])      # Min size of leaves
    leaf_info[4] = np.max(leaf_info_tmp[:, 4])      # Min size of leaves
    return leaf_info


def describe_forest(forest, m_n_min_ar, v_dict, c_dict, pen_mult=0,
                    regrf=False):
    """Describe estimated forest by collecting information in trees.

    Parameters
    ----------
    forest : List of List. Each forest consist of one node_table.
    m_n_min : List of INT. Number of variables and minimum leaf size
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    None.

    """
    print('\n')
    print('-' * 80)
    print('Parameters of estimation to build random forest')
    print('Outcome variable used to build forest:  ', *v_dict['y_tree_name'])
    print('Features used to build forest:          ', *v_dict['x_name'])
    print('Variables always included in splitting: ',
          *v_dict['x_name_always_in'])
    print('Number of replications:     {0:<4}'.format(c_dict['boot']))
    if not regrf:
        if c_dict['mtot'] == 3:
            splitting_rule = 'MSEs of regressions only considered'
        elif c_dict['mtot'] == 1:
            splitting_rule = 'MSE+MCE criterion'
        elif c_dict['mtot'] == 2:
            splitting_rule = '-Var(effect)'
        elif c_dict['mtot'] == 4:
            splitting_rule = 'Random switching'
        print('Splitting rule used:        {0:<4}'.format(splitting_rule))
        if c_dict['mtot_p_diff_penalty'] > 0:
            print('Penalty used in splitting: ', pen_mult)
    print('Share of data in subsample for forest buildung: {0:<4}'.format(
        c_dict['subsam_share_forest']))
    print('Share of data in subsample for forest evaluation: {0:<4}'.format(
        c_dict['subsam_share_eval']))
    print('Total number of variables available for splitting: {0:<4}'.format(
        len(v_dict['x_name'])))
    print('# of variables (M) used for split: {0:<4}'.format(m_n_min_ar[0]))
    if c_dict['m_random_poisson']:
        print('           (# of variables drawn from 1+Poisson(M-1))')
    print('Minimum leaf size:                 {0:<4}'.format(m_n_min_ar[1]))
    print('Alpha regularity:                 {:5.3f}'.format(m_n_min_ar[2]))
    print('------------------- Estimated tree -------------------------------')
    leaf_info = get_tree_infos(forest)
    print('Average # of leaves:      {0:4.1f}'.format(leaf_info[0]))
    print('Average size of leaves:   {0:4.1f}'.format(leaf_info[1]))
    print('Median size of leaves:    {:4.1f}'.format(leaf_info[2]))
    print('Min size of leaves:       {:4.0f}'.format(leaf_info[3]))
    print('Max size of leaves:       {:4.0f}'.format(leaf_info[4]))
    print('Total # of obs in leaves: {:4.0f}'.format(leaf_info[5]))
    print('-' * 80)


def best_m_n_min_alpha_reg(forest, c_dict):
    """Get best forest for the tuning parameters m_try, n_min, alpha_reg.

    Parameters
    ----------
    forest : List of list of lists... Estimated forests.
    c : Dict. Parameters.

    Returns
    -------
    forest_final : List of lists. OOB-optimal forest.
    m_n_final : List. Optimal values of m and n_min.

    """
    m_n_min_ar_combi = []
    if np.size(c_dict['grid_m']) == 1:
        grid_for_m = [c_dict['grid_m']]
    else:
        grid_for_m = c_dict['grid_m']
    if np.size(c_dict['grid_n_min']) == 1:
        grid_for_n_min = [c_dict['grid_n_min']]
    else:
        grid_for_n_min = c_dict['grid_n_min']
    if np.size(c_dict['grid_alpha_reg']) == 1:
        grid_for_alpha_reg = [c_dict['grid_alpha_reg']]
    else:
        grid_for_alpha_reg = c_dict['grid_alpha_reg']
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                m_n_min_ar_combi.append([m_idx, n_min, alpha_reg])
    dim_m_n_min_ar = np.size(c_dict['grid_m']) * np.size(
        c_dict['grid_n_min']) * np.size(c_dict['grid_alpha_reg'])
    if (dim_m_n_min_ar) > 1:       # Find best of trees
        mse_oob = np.zeros(dim_m_n_min_ar)
        trees_without_oob = np.zeros(dim_m_n_min_ar)
        for trees_m_n_min_ar in forest:                  # different forests
            for j, tree in enumerate(trees_m_n_min_ar):  # trees within forest
                n_lost = n_total = 0
                if c_dict['no_of_treat'] is not None:
                    mse_mce_tree = np.zeros((c_dict['no_of_treat'],
                                             c_dict['no_of_treat']))
                    obs_t_tree = np.zeros(c_dict['no_of_treat'])
                tree_mse = 0
                for leaf in tree:                        # leaves within tree
                    if leaf[4] == 1:   # Terminal leafs only
                        n_total += np.sum(leaf[6])
                        if leaf[7] is None:
                            if c_dict['no_of_treat'] is None:
                                n_lost += leaf[6]
                            else:
                                n_lost += np.sum(leaf[6])  # [6]: Leaf size
                        else:
                            if c_dict['no_of_treat'] is None:  # [7]: leaf_mse
                                tree_mse += leaf[6] * leaf[7]
                            else:
                                mse_mce_tree, obs_t_tree = add_rescale_mse_mce(
                                    leaf[7], leaf[6], c_dict['mtot'],
                                    c_dict['no_of_treat'], mse_mce_tree,
                                    obs_t_tree)
                if n_lost > 0:
                    if c_dict['no_of_treat'] is None:
                        tree_mse = tree_mse * n_total / (n_total - n_lost)
                    else:
                        if (n_total - n_lost) < 1:
                            trees_without_oob[j] += 1
                if c_dict['no_of_treat'] is not None:
                    mse_mce_tree = get_avg_mse_mce(
                        mse_mce_tree, obs_t_tree, c_dict['mtot'],
                        c_dict['no_of_treat'])
                    tree_mse = compute_mse_mce(mse_mce_tree, c_dict['mtot'],
                                               c_dict['no_of_treat'])
                mse_oob[j] += tree_mse     # Add MSE to MSE of forest j
        if np.any(trees_without_oob) > 0:
            for j, _ in enumerate(trees_without_oob):
                if trees_without_oob[j] > 0:
                    mse_oob[j] = mse_oob[j] * (
                        c_dict['boot'] / (c_dict['boot']
                                          - trees_without_oob[j]))
        min_i = np.argmin(mse_oob)
        mse_oob = mse_oob / c_dict['boot']
        if not isinstance(c_dict['grid_n_min'], (list, tuple, np.ndarray)):
            c_dict['grid_n_min'] = [c_dict['grid_n_min']]
        if not isinstance(c_dict['grid_m'], (list, tuple, np.ndarray)):
            c_dict['grid_m'] = [c_dict['grid_m']]
        if not isinstance(c_dict['grid_alpha_reg'], (list, tuple, np.ndarray)):
            c_dict['grid_alpha_reg'] = [c_dict['grid_alpha_reg']]
        if c_dict['with_output']:
            print('\n')
            print('-' * 80,
                  '\nOOB MSE (without penalty) for M_try, minimum leafsize',
                  ' and alpha_reg combinations', '\n')
            j = 0
            print('\nNumber of vars / min. leaf size / alpha reg. / OOB value',
                  'Trees without OOB')
            for m_idx in c_dict['grid_m']:
                for n_min in c_dict['grid_n_min']:
                    for alpha_reg in c_dict['grid_alpha_reg']:
                        print('{0:>12}'.format(m_idx), '{0:>12}'.format(n_min),
                              ' {:15.3f}'.format(alpha_reg),
                              ' {:8.3f}'.format(mse_oob[j]),
                              ' {:4.0f}'.format(trees_without_oob[j]))
                        j = j + 1
            print('Minimum OOB MSE:      {:8.3f}'.format(mse_oob[min_i]))
            print('Number of variables: ', m_n_min_ar_combi[min_i][0])
            print('Minimum leafsize:    ', m_n_min_ar_combi[min_i][1])
            print('Alpha regularity:    ', m_n_min_ar_combi[min_i][2])
            print('-' * 80)
        forest_final = []
        for trees_m_n_min in forest:
            forest_final.append(trees_m_n_min[min_i])
        m_n_min_ar_opt = m_n_min_ar_combi[min_i]
    else:       # Find best of trees
        forest_final = []
        for trees_m_n_min_ar in forest:
            forest_final.append(trees_m_n_min_ar[0])
        m_n_min_ar_opt = m_n_min_ar_combi[0]
    return forest_final, m_n_min_ar_opt


def regrf_mse(y_dat, w_dat, obs, w_yes):
    """Compute average mse for the data passed. Regression Forest.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    w_dat : Numpy Nx1 vector. Weights.
    obs : INT. Leaf size for this split.
    w_yes: Boolean. Weighted estimation.

    Returns
    -------
    mse : Mean squared error.

    """
    if w_yes:
        y_mean = np.average(y_dat, weights=w_dat, axis=0)
        mse = np.average(np.square(y_dat - y_mean), weights=w_dat, axis=0)
    else:
        y_dat = y_dat.reshape(-1)
        mse = np.inner(y_dat, y_dat) / obs - np.mean(y_dat)**2
    return mse


def mcf_mse(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
            treat_values, w_yes, splitting=False):
    """Compute average mse for the data passed. Based on different methods.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    if w_yes:
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_not_numba(
            y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
            treat_values, w_yes, splitting)
    else:
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_numba(
            y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat,
            np.array(treat_values, dtype=np.int8), w_yes)
    return mse_mce, treat_shares, no_of_obs_by_treat


def mcf_mse_not_numba(y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
                      treat_values, w_yes, splitting=False):
    """Compute average mse for the data passed. Based on different methods.

    CURRENTLY ONLY USED FOR WEIGHTED.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.
    splitting: Boolean. Default is False.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    if mtot in (1, 4):
        treat_shares = np.empty(no_of_treat)
    else:
        treat_shares = 0
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = len(y_dat[d_m])
        no_of_obs_by_treat[m_idx] = n_m
        if w_yes:
            w_m = w_dat[d_m]
            y_m_mean = np.average(y_dat[d_m], weights=w_m, axis=0)
            mse_m = np.average(np.square(y_dat[d_m] - y_m_mean),
                               weights=w_m, axis=0)
        else:
            y_m_mean = np.average(y_dat[d_m], axis=0)
            mse_m = np.dot(y_dat[d_m], y_dat[d_m]) / n_m - (y_m_mean**2)
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs
            mse_mce[m_idx, m_idx] = mse_m
        elif mtot == 3:
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:  # Variance of effects mtot = 2
                    d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                    if w_yes:
                        y_l_mean = np.average(y_dat[d_l], weights=w_dat[d_l],
                                              axis=0)
                    else:
                        y_l_mean = np.average(y_dat[d_l], axis=0)
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    d_ml = d_ml[:, 0]
                    y_nn_m = y_nn[d_ml, m_idx]
                    y_nn_l = y_nn[d_ml, v_idx]
                    if w_yes:
                        w_ml = w_dat[d_ml].reshape(-1)
                        if splitting and (no_of_treat == 2):
                            mce_ml = ((np.average(y_nn_m, weights=w_ml,
                                                  axis=0)) *
                                      (np.average(y_nn_l, weights=w_ml,
                                                  axis=0)) * (-1))
                        else:
                            mce_ml = np.average(
                                (y_nn_m - np.average(y_nn_m, weights=w_ml,
                                                     axis=0)) *
                                (y_nn_l - np.average(y_nn_l, weights=w_ml,
                                                     axis=0)),
                                weights=w_ml, axis=0)
                    else:
                        aaa = np.average(y_nn_m, axis=0) * np.average(y_nn_l,
                                                                      axis=0)
                        bbb = np.dot(y_nn_m, y_nn_l) / len(y_nn_m)
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


@njit
def mcf_mse_numba(y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat,
                  treat_values, w_yes):
    """Compute average mse for the data passed. Based on different methods.

       WEIGHTED VERSION DOES NOT YET WORK. TRY with next Numba version.
       Need to change list format soon.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    n : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : 1D Numpy array of INT. Treatment values.
    w_yes: Boolean. Weighted estimation.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.
    """
    obs = len(y_dat)
    if mtot in (1, 4):
        treat_shares = np.zeros(no_of_treat)
    else:
        treat_shares = np.zeros(1)
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    no_of_obs_by_treat = np.zeros(no_of_treat)
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # d_m is Boolean
        n_m = np.sum(d_m)
        no_of_obs_by_treat[m_idx] = n_m
        y_m = np.empty(n_m)
        j = 0
        for i in range(obs):
            if d_m[i]:
                y_m[j] = y_dat[i, 0]
                j += 1
        if w_yes:
            raise Exception('not yet implemented for weighting with Numba')
        y_m_mean = np.sum(y_m) / n_m
        mse_m = np.dot(y_m, y_m) / n_m - (y_m_mean**2)
        if mtot in (1, 3, 4):
            treat_shares[m_idx] = n_m / n_obs
            mse_mce[m_idx, m_idx] = mse_m
        if mtot != 3:
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                n_l = np.sum(d_l)
                if mtot == 2:  # Variance of effects mtot = 2
                    y_l = np.empty(n_l)
                    j = 0
                    for i in range(obs):
                        if d_l[i]:
                            y_l[j] = y_dat[i, 0]
                            j += 1
                    if w_yes:
                        pass
                    else:
                        y_l_mean = np.sum(y_l) / n_l
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    n_ml = np.sum(d_ml)
                    y_nn_l = np.empty(n_ml)
                    y_nn_m = np.empty_like(y_nn_l)
                    j = 0
                    for i in range(obs):
                        if d_ml[i]:
                            y_nn_l[j] = y_nn[i, v_idx]
                            y_nn_m[j] = y_nn[i, m_idx]
                            j += 1
                    if w_yes:
                        raise Exception('Numba version not implemented for' +
                                        ' weighted estimation.')
                    else:
                        aaa = np.sum(y_nn_m) / n_ml * np.sum(y_nn_l) / n_ml
                        bbb = np.dot(y_nn_m, y_nn_l) / n_ml
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml
    return mse_mce, treat_shares, no_of_obs_by_treat


def add_mse_mce_split(mse_mce_l, mse_mce_r, obs_by_treat_l, obs_by_treat_r,
                      mtot, no_of_treat):
    """Sum up MSE parts of use in splitting rule."""
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat = np.empty(no_of_treat)
    for m_idx in range(no_of_treat):
        obs_by_treat[m_idx] = obs_by_treat_l[m_idx] + obs_by_treat_r[m_idx]
        mse_mce[m_idx, m_idx] = (
            mse_mce_l[m_idx, m_idx] * obs_by_treat_l[m_idx]
            + mse_mce_r[m_idx, m_idx] * obs_by_treat_r[m_idx]
            ) / obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                n_ml_l = obs_by_treat_l[m_idx] + obs_by_treat_l[v_idx]
                n_ml_r = obs_by_treat_r[m_idx] + obs_by_treat_r[v_idx]
                mse_mce[m_idx, v_idx] = (mse_mce_l[m_idx, v_idx] * n_ml_l
                                         + mse_mce_r[m_idx, v_idx] * n_ml_r
                                         ) / (n_ml_l + n_ml_r)
    return mse_mce


def add_rescale_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat,
                        mse_mce_add_to, obs_by_treat_add_to):
    """Rescale MSE_MCE matrix and update observation count."""
    mse_mce_sc = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat_new = obs_by_treat + obs_by_treat_add_to
    for m_idx in range(no_of_treat):
        mse_mce_sc[m_idx, m_idx] = mse_mce[m_idx, m_idx] * obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                mse_mce_sc[m_idx, v_idx] = mse_mce[m_idx, v_idx] * (
                    obs_by_treat[m_idx] + obs_by_treat[v_idx])
    mse_mce_new = mse_mce_add_to + mse_mce_sc
    return mse_mce_new, obs_by_treat_new


def get_avg_mse_mce(mse_mce, obs_by_treat, mtot, no_of_treat):
    """Bring MSE_MCE matrix in average form."""
    mse_mce_avg = mse_mce.copy()
    for m_idx in range(no_of_treat):
        mse_mce_avg[m_idx, m_idx] = mse_mce[m_idx, m_idx] / obs_by_treat[m_idx]
        if mtot != 3:
            for v_idx in range(m_idx+1, no_of_treat):
                mse_mce_avg[m_idx, v_idx] = mse_mce[m_idx, v_idx] / (
                    obs_by_treat[m_idx] + obs_by_treat[v_idx])
    return mse_mce_avg


def compute_mse_mce(mse_mce, mtot, no_of_treat):
    """Sum up MSE parts for use in splitting rule and else."""
    if no_of_treat > 4:
        if mtot in (1, 4):
            mse = no_of_treat * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 2:
            mse = 2 * np.trace(mse_mce) - mse_mce.sum()
        elif mtot == 3:
            mse = np.trace(mse_mce)
    else:
        mse = mce = 0
        for m_idx in range(no_of_treat):
            if mtot in (1, 4):
                mse_a = (no_of_treat - 1) * mse_mce[m_idx, m_idx]
            else:
                mse_a = mse_mce[m_idx, m_idx]
            mse += mse_a
            if mtot != 3:
                for v_idx in range(m_idx+1, no_of_treat):
                    mce += mse_mce[m_idx, v_idx]
        mse -= 2 * mce
    return mse


def term_or_data(data_tr_ns, data_oob_ns, y_i, d_i, x_i_ind_split,
                 no_of_treat, regrf=False):
    """Check if terminal leaf. If not, provide data.

    Parameters
    ----------
    data_tr_ns : Numpy array. Data used for splitting.
    data_oob_ns : Numpy array. OOB Data.
    y_i : List of INT. Indices of y in data.
    d_i : List of INT. Indices of d in data.
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
    terminal2 : Boolean. Try new variables.

    """
    terminal = terminal_x = False
    y_oob = d_dat = d_oob = x_dat = x_oob = None
    x_no_variation = []
    y_dat = data_tr_ns[:, y_i]
    if np.all(np.isclose(y_dat, y_dat[0])):    # all elements are equal
        terminal = True
    else:
        y_oob = data_oob_ns[:, y_i]
        d_dat = data_tr_ns[:, d_i]
        if not regrf:
            if len(np.unique(d_dat)) < no_of_treat:
                terminal = True
        if not terminal:
            if not regrf:
                d_oob = data_oob_ns[:, d_i]
            x_dat = data_tr_ns[:, x_i_ind_split]
            x_no_variation = []       # List of Booleans
            for cols in range(len(x_i_ind_split)):
                x_no_variation.append(
                    np.all(np.isclose(x_dat[:, cols], x_dat[0, cols])))
            if np.all(x_no_variation):
                terminal_x = True
            else:
                x_oob = data_oob_ns[:, x_i_ind_split]
    return (y_dat, y_oob, d_dat, d_oob, x_dat, x_oob, terminal, terminal_x,
            x_no_variation)


def next_split(current_node, data_tr, data_oob, y_i, y_nn_i, d_i, x_i, w_i,
               x_type, x_values, x_ind, x_ai_ind, c_dict, mmm, n_min,
               alpha_reg, pen_mult, trl, regrf=False):
    """Find best next split of leaf (or terminate splitting for this leaf).

    Parameters
    ----------
    current_node : List of list: Information about leaf to split.
    data_tr: Numpy array. All training data.
    data_oob : Numpy array: All OOB data.
    y_i : INT. Location of Y in data matrix.
    y_nn_i :  List of INT. Location of Y_NN in data matrix.
    d_i : INT. Location of D in data matrix.
    x_i : List of INT. Location of X in data matrix.
    x_type : List of INT (0,1,2). Type of X.
    x_ind : List INT. Location of X in X matrix.
    x_ai_ind : List of INT. Location of X_always in X matrix.
    c_dict : DICT. Parameters.
    mmm : INT. Number of X-variables to choose for splitting.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. Alpha regularity.
    pen_mult: Float. Penalty multiplier.
    trl: Int. Tree length.
    regrf: Boolean. Regression Random Forest. Default is False.

    Returns
    -------
    left : List of lists. Information about left leaf.
    right : List of lists. Information about right leaf.
    current : List of lists. Updated information about this leaf.
    terminal : INT. 1: No splits for this leaf. 0: Leaf splitted

    """
    data_tr_ns = data_tr[current_node[11], :]   # Train. data of node
    data_oob_ns = data_oob[current_node[12], :]   # OOB data of  node
    terminal = split_done = False
    p_all_x = len(x_ind)
    if current_node[5] < (2 * n_min):
        terminal = True
    else:
        if not regrf:
            if current_node[5] < 100:  # Otherwise, it might be too slow
                ret = np.unique(data_tr_ns[:, d_i], return_counts=True)
                if np.any(ret[1] < 2):  # Cannot split with too few treated
                    terminal = True
    if not terminal:
        obs_min = max([round(current_node[5] * alpha_reg), n_min])
        p_x = None
        for trials in range(c_dict['stop_empty']):
            terminal = False
            best_mse = 1e100000   # Initialisation: Infinity as default values
            for _ in range(2):  # avoid to get same variables
                x_ind_split = rnd_variable_for_split(x_ind, x_ai_ind, c_dict,
                                                     mmm)
                if ((p_all_x - mmm) > 5) and (p_all_x > 10):
                    break
                if trials == 0:  # Avoid drawing same vars again
                    if c_dict['stop_empty'] > 1:
                        x_ind_split_prev = x_ind_split[:]
                    break
                if p_x == len(x_ind):
                    x_ind_split_prev = x_ind_split[:]
                    break
                if not p_x == len(x_ind_split):
                    x_ind_split_prev = x_ind_split[:]
                    break
                if not set(x_ind_split_prev) == set(x_ind_split):
                    x_ind_split_prev = x_ind_split[:]
                    break
            p_x = len(x_ind_split)   # indices refer to order of x in data_*
            x_type_split = copy.copy(x_type[x_ind_split])
            x_values_split = []
            for v_idx in x_ind_split:
                x_values_split.append(x_values[v_idx].copy())
            # Check if split is possible ... sequatial order to minimize costs
            # Check if enough variation in the data to do splitting (costly)
            if regrf:
                (y_dat, _, d_dat, _, x_dat, x_oob, terminal, terminal_x,
                 x_no_varia) = term_or_data(data_tr_ns, data_oob_ns, y_i, d_i,
                                            x_i[x_ind_split], None, regrf)
            else:
                (y_dat, _, d_dat, _, x_dat, x_oob, terminal, terminal_x,
                 x_no_varia) = term_or_data(data_tr_ns, data_oob_ns, y_i, d_i,
                                            x_i[x_ind_split],
                                            c_dict['no_of_treat'], regrf)
            if terminal:
                break
            if terminal_x:
                terminal = True
                continue   # Jump to start of loop and try another value of x
            if not regrf:
                if (c_dict['mtot'] == 1) or (c_dict['mtot'] == 4):
                    y_nn = data_tr_ns[:, y_nn_i]
                else:
                    y_nn = y_nn_l = y_nn_r = 0
            if c_dict['w_yes']:
                w_dat = data_tr_ns[:, [w_i]]
            else:
                w_dat = [1]
            for j in range(p_x):  # Loops over the variables
                if not x_no_varia[j]:  # No variation of this x -> no split
                    x_j = np.copy(x_dat[:, j])   # This variable to be invest.
                    x_oob_j = np.copy(x_oob[:, j])
                    if x_type_split[j] > 0:
                        x_j = x_j.astype(np.int32)
                        x_oob_j = x_oob_j.astype(np.int32)
                    split_values = get_split_values(
                        y_dat, w_dat, x_j, x_type_split[j], x_values_split[j],
                        current_node[5], c_dict)
                    split_values_unord_j = []
                    for val in split_values:  # Loops over values of variables
                        if x_type_split[j] == 0:
                            leaf_l = (x_j - 1e-15) <= val  # because of float
                        else:                          # ordered with few vals.
                            # Categorial variable: Either in group or not
                            split_values_unord_j.append(val)
                            leaf_l = np.isin(x_j, split_values_unord_j)
                        n_l = np.count_nonzero(leaf_l)
                        n_r = current_node[5] - n_l
                        # Check if enough observations available
                        if (n_l < obs_min) or (n_r < obs_min):
                            continue
                        if x_type_split[j] == 0:
                            leaf_oob_l = (x_oob_j - 1e-15) <= val
                        else:
                            leaf_oob_l = np.isin(x_oob_j, split_values_unord_j)
                        n_oob_l = np.count_nonzero(leaf_oob_l)
                        n_oob_r = current_node[6] - n_oob_l
                        # Next we check if any obs in each treatment
                        if not regrf:
                            if len(np.unique(
                                    d_dat[leaf_l])) < c_dict['no_of_treat']:
                                continue
                        leaf_r = np.invert(leaf_l)  # Reverses True to False
                        if not regrf:
                            if len(np.unique(
                                    d_dat[leaf_r])) < c_dict['no_of_treat']:
                                continue   # Splits possible?
                        leaf_oob_r = np.invert(leaf_oob_l)
                        if not regrf:
                            if (c_dict['mtot'] == 1) or (c_dict['mtot'] == 4):
                                y_nn_l = y_nn[leaf_l, :]
                                y_nn_r = y_nn[leaf_r, :]
                            else:
                                y_nn_l = y_nn_r = 0
                        if c_dict['w_yes']:
                            w_l = w_dat[leaf_l]
                            w_r = w_dat[leaf_r]
                        else:
                            w_l = w_r = 0
                        # compute objective functions given particular method
                        if regrf:
                            mse_l = regrf_mse(y_dat[leaf_l],  w_l, n_l,
                                              c_dict['w_yes'])
                            mse_r = regrf_mse(y_dat[leaf_r],  w_r, n_r,
                                              c_dict['w_yes'])
                            mse_split = (mse_l * n_l + mse_r * n_r) / (
                                n_l + n_r)
                        else:
                            mse_mce_l, shares_l, obs_by_treat_l = mcf_mse(
                                y_dat[leaf_l], y_nn_l, d_dat[leaf_l], w_l,
                                n_l, c_dict['mtot'], c_dict['no_of_treat'],
                                c_dict['d_values'], c_dict['w_yes'])
                            mse_mce_r, shares_r, obs_by_treat_r = mcf_mse(
                                y_dat[leaf_r], y_nn_r, d_dat[leaf_r], w_r,
                                n_r, c_dict['mtot'], c_dict['no_of_treat'],
                                c_dict['d_values'], c_dict['w_yes'])
                            mse_mce = add_mse_mce_split(
                                mse_mce_l, mse_mce_r, obs_by_treat_l,
                                obs_by_treat_r, c_dict['mtot'],
                                c_dict['no_of_treat'])
                            mse_split = compute_mse_mce(
                                mse_mce, c_dict['mtot'], c_dict['no_of_treat'])
                        # add penalty for this split
                        if not regrf:
                            if (c_dict['mtot'] == 1) or (
                                    (c_dict['mtot'] == 4) and
                                    (random.random() > 0.5)):
                                penalty = mcf_penalty(shares_l, shares_r)
                                mse_split = mse_split + pen_mult * penalty
                        if mse_split < best_mse:
                            split_done = True
                            best_mse = mse_split
                            best_var_i = copy.copy(x_ind_split[j])
                            best_type = copy.copy(x_type_split[j])
                            best_n_l = n_l
                            best_n_r = n_r
                            best_leaf_l = np.copy(leaf_l)  # ML 30.11.2020
                            best_leaf_r = np.copy(leaf_r)
                            best_leaf_oob_l = np.copy(leaf_oob_l)
                            best_leaf_oob_r = np.copy(leaf_oob_r)
                            best_n_oob_l = n_oob_l
                            best_n_oob_r = n_oob_r
                            if best_type == 0:
                                best_value = copy.copy(val)
                            else:
                                best_value = split_values_unord_j[:]  # left
            if split_done:
                break   # No need to try different covariates
    if not split_done:
        terminal = True
    if terminal:
        terminal = True
        current_node[4] = 1  # terminal
        if c_dict['w_yes']:
            w_oob = data_oob_ns[:, [w_i]]
        else:
            w_oob = 0
        n_oob = np.copy(current_node[6])
        if regrf:
            if n_oob > 1:
                current_node[7] = regrf_mse(data_oob_ns[:, y_i],  w_oob, n_oob,
                                            c_dict['w_yes'])
            elif n_oob == 1:
                current_node[7] = 0
            else:
                current_node[7] = None      # MSE cannot be computed
        else:
            if len(np.unique(data_oob_ns[:, d_i])) < c_dict['no_of_treat']:
                current_node[7] = None      # MSE cannot be computed
            else:
                current_node[7], shares_r, current_node[6] = mcf_mse(
                    data_oob_ns[:, y_i], data_oob_ns[:, y_nn_i],
                    data_oob_ns[:, d_i], w_oob, n_oob, c_dict['mtot'],
                    c_dict['no_of_treat'], c_dict['d_values'], c_dict['w_yes'])
        current_node[11] = 0    # Data, no longer needed
        current_node[12] = 0    # OOB Data, no longer needed
        newleaf_l = []
        newleaf_r = []
    else:
        newleaf_l = copy.deepcopy(current_node)
        newleaf_r = copy.deepcopy(current_node)
        newleaf_l[0] = trl               # Tree length, Index fängt bei null an
        newleaf_r[0] = trl + 1
        newleaf_l[1] = copy.deepcopy(current_node[0])  # Parent nodes
        newleaf_r[1] = copy.deepcopy(current_node[0])
        newleaf_l[2] = newleaf_r[2] = None             # Following splits l
        newleaf_l[3] = newleaf_r[3] = None             # Following splits r
        newleaf_l[4] = newleaf_r[4] = 2                # Node is active
        newleaf_l[5] = best_n_l         # Leaf size training
        newleaf_r[5] = best_n_r
        newleaf_l[6] = best_n_oob_l     # Leaf size OOB
        newleaf_r[6] = best_n_oob_r
        newleaf_l[7] = newleaf_r[7] = None         # OOB MSE without penalty
        newleaf_l[8] = newleaf_r[8] = None         # Variable for next split
        newleaf_l[9] = newleaf_r[9] = newleaf_l[10] = newleaf_r[10] = None
        train_list = np.array(current_node[11], copy=True)
        oob_list = np.array(current_node[12], copy=True)
        newleaf_l[11] = train_list[best_leaf_l].tolist()
        newleaf_r[11] = train_list[best_leaf_r].tolist()
        newleaf_l[12] = oob_list[best_leaf_oob_l].tolist()
        newleaf_r[12] = oob_list[best_leaf_oob_r].tolist()
        newleaf_l[13] = newleaf_r[13] = newleaf_l[14] = newleaf_r[14] = None
        newleaf_l[15] = newleaf_r[15] = None
        current_node[2] = copy.copy(newleaf_l[0])  # ID of daughter leaf
        current_node[3] = copy.copy(newleaf_r[0])
        current_node[4] = 0     # not active, not terminal - intermediate
        current_node[8] = copy.copy(best_var_i)
        if best_type > 0:  # Save as product of primes
            best_value = gp.list_product(best_value)   # int
        current_node[9] = copy.copy(best_value)    # <= -> left
        current_node[10] = copy.copy(best_type)
        current_node[11] = current_node[12] = 0   # Data, no longer needed
        if current_node[0] != 0:
            current_node[16] = 0
        else:    # Need to keep OOB data in first leaf for VIB, Feature select
            if (not c_dict['var_import_oob']) and (not c_dict['fs_yes']):
                current_node[16] = 0    # Data, no longer needed, saves memory
    return newleaf_l, newleaf_r, current_node, terminal


def rnd_variable_for_split(x_ind_pos, x_ai_ind_pos, c_dict, mmm):
    """Generate variables to be used for split.

    Parameters
    ----------
    x_ind_pos : List. Indices of all x-variables.
    x_ai_ind : List. Indices of all x-variables always used for splitting.
    c_dict : Dict. Parameters
    mmm : Number of variables to draw.

    Returns
    -------
    x_i_for_split : List of indices in x of splitting variables.

    """
    rng = np.random.default_rng()
    qqq = len(x_ind_pos)
    if c_dict['m_random_poisson']:
        m_l = 1 + rng.poisson(lam=mmm-1, size=1)
        if m_l < 1:
            m_l = 1
        elif m_l > qqq:
            m_l = qqq
    else:
        m_l = mmm
    if x_ai_ind_pos == []:
        x_i_for_split = rng.choice(x_ind_pos, m_l, replace=False)
        x_i_for_split_list = x_i_for_split.tolist()
    else:
        if m_l > len(x_ai_ind_pos):
            x_i_for_split = rng.choice(x_ind_pos, m_l-len(x_ai_ind_pos),
                                       replace=False)
            x_i_for_split = np.concatenate((x_i_for_split, x_ai_ind_pos))
            x_i_for_split = np.unique(x_i_for_split)
            x_i_for_split_list = x_i_for_split.tolist()
        else:
            x_i_for_split_list = x_ai_ind_pos[:]
    return x_i_for_split_list


def build_single_tree(data, data_oob, y_i, y_nn_i, d_i, x_i, w_i, x_type,
                      x_values, x_ind, x_ai_ind, c_dict, mmm, n_min, alpha_reg,
                      node_table, pen_mult, regrf=False):
    """Build single tree given random sample split.

    Parameters
    ----------
    data : Nympy array. Training data
    data_oob : Numpy array. OOB data
    y_i : List. Position of y in numpy array.
    y_nn_i : List. Position of y_nn in numpy array.
    d_i : List. Position of d in numpy array..
    x_i : List. Position of x in numpy array.
    x_type : List of INT. Type of covariate (0,1,2).
    x_values: List of lists. Values of covariate (if not too many)
    x_ind : List. Postion of covariate in x for easy reference.
    x_ai_ind : List. Postion of covariate always-in in x for easy reference.
    c_dict : Dict. Parameters.
    m : INT. Number of covariates to be included.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. alpha regularity.
    node_table : List of list of lists. Initial tree (basic leaf)
    pen_mult: Float. Multiplier of penalty.
    regrf: Boolean. Regression Random Forest. Default is False.

    Returns
    -------
    node_table : List of list of lists. Final tree.
    """
    continue_to_split = True
    while continue_to_split:
        len_table = len(node_table)
        active_knots = 0
        for node_i in range(len_table):
            if node_table[node_i][4] == 2:
                current = copy.deepcopy(node_table[node_i])
                left, right, current, terminal = next_split(
                    current, data, data_oob, y_i, y_nn_i, d_i, x_i, w_i,
                    x_type, x_values, x_ind, x_ai_ind, c_dict, mmm, n_min,
                    alpha_reg, pen_mult, len(node_table), regrf)
                node_table[node_i] = copy.deepcopy(current)
                if not terminal:
                    active_knots += 1
                    node_table.append(copy.deepcopy(left))
                    node_table.append(copy.deepcopy(right))
        if active_knots == 0:
            continue_to_split = False  # Tree completed
    return node_table


def structure_of_node_tabl():
    """Info about content of NODE_TABLE.

    Returns
    -------
    decription : STR. Information on node table with inital node

    """
    description = """Trees are fully saved in Node_Table (list of lists)
    Structure of node_table
      - Each knot is one list that contains further lists
    This is the position and information for a given node
    The following items will be filled in the first sample
    0: Node identifier (INT: 0-...)
    1: Parent kno
    2: Child node left
    3: Child node right
    4: Type of node (2: Active -> will be further splitted or made terminal
                    1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: Leafsize Training (later on used for pruning)
    6: Leafsize OOB sample
    7: OOB value of objective function (if node size <= n_min_max, or
                                         terminal node)
    8: INT: Index of variable used for decision of next split
    9: If x_type = 0: Cut-off value (larger goes to right daughter,
                                    equal and smaller to left daughter)
        (ID of right dauhgter equals ID of left daughter + 1)
    9:  If x_type = 1,2: Product of primes that goes to left daughter
    10: x_type of variable
    11: Numpy arrays: Training  data
        -either list with data or indices
    12: Numpy array: OOB data
        -either list with data or indices
    The following items will be filled in second sample
    13: List of potential outcomes for all treatments
    14: List of indices of variables used to compute predictions
    15: Number of obs (2nd sample) in terminal leaf
    16: Indices of OOB observations in total sample (only in leaf 0)
        In second part used to indicate need for pruning (1: prune, 0: ok,
        used only in terminal leaf)
    """
    print("\n", description)


def init_node_table(n_tr, n_oob, indices_oob):
    """Initialise Node table for first leaf.

    Parameters
    ----------
    n_tr : INT. Number of observation in training subsample.
    n_oob : INT. Number of observation in OOB subsample.

    Returns
    -------
    node_table : List of lists. First init_node_table

    """
    node_table = []
    id_node_0 = 0
    id_parent_1 = id_child_left_2 = id_child_right_3 = None
    active_4 = 2
    leaf_size_tr_5 = n_tr
    leaf_size_oob_6 = n_oob
    objective_fct_value_oob_7 = next_split_i_8 = cut_off_prime_l_9 = None
    x_type_10 = None
    data_tr_indi_11 = list(range(n_tr))
    data_oob_indi_12 = list(range(n_oob))
    pot_outcomes_13 = pot_variables_used_indi_14 = leaf_size_pot_15 = None
    indices_oob_16 = indices_oob
    node_table.append(id_node_0)
    node_table.append(id_parent_1)
    node_table.append(id_child_left_2)
    node_table.append(id_child_right_3)
    node_table.append(active_4)
    node_table.append(leaf_size_tr_5)
    node_table.append(leaf_size_oob_6)
    node_table.append(objective_fct_value_oob_7)
    node_table.append(next_split_i_8)
    node_table.append(cut_off_prime_l_9)
    node_table.append(x_type_10)
    node_table.append(data_tr_indi_11)
    node_table.append(data_oob_indi_12)
    node_table.append(pot_outcomes_13)
    node_table.append(pot_variables_used_indi_14)
    node_table.append(leaf_size_pot_15)
    node_table.append(indices_oob_16)
    return [node_table]


@ray.remote
def ray_build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                             x_values, x_ind, x_ai_ind, c_dict, boot_indices,
                             pen_mult, regrf=False):
    """Prepare function for Ray."""
    return build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                                x_values, x_ind, x_ai_ind, c_dict,
                                boot_indices, pen_mult, regrf)


def build_many_trees_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                         x_values, x_ind, x_ai_ind, c_dict, boot_indices,
                         pen_mult, regrf=False):
    """Build larger pieces of the forest (for MP)."""
    little_forest = []
    for boot in boot_indices:
        tree = build_tree_mcf(
            data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type, x_values, x_ind,
            x_ai_ind, c_dict, boot, pen_mult, regrf)
        little_forest.append(tree)
    return little_forest


@ray.remote
def ray_build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                       x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult,
                       regrf=False):
    """Prepare function for Ray."""
    return build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                          x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult,
                          regrf)


def build_tree_mcf(data, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type, x_values,
                   x_ind, x_ai_ind, c_dict, boot, pen_mult, regrf=False):
    """Build single trees for all values of tuning parameters.

    Parameters
    ----------
    y_i : Position of Outcome in DATA.
    y_nn_i: Position of Matched outcomes.
    x_i : Position of Covariates.
    d_i : Position of Treatment.
    cl_i : Position of Cluster variable.
    x_type : List of INT. Type of variable: 0,1,2
    x_values: List of lists. Values of variable (if few or categorical)
    x_ind : List of INT. Identifier of variables
    x_ai_ind : List of INT. 1 if variable is included in every split
    c_dict : Dict. Control parameters
    boot : INT. Counter for bootstrap replication (currently not used)
    regrf: Boolean. Regression Random Forest. Default=False.

    Returns
    -------
    tree_all : LIST (m_grid x N_min_grid x alpha_grid) with trees for all
               values of tuning parameters

    """
    # split data into OOB and tree data
    n_obs = data.shape[0]
    # Random number initialisation.Hope this seeds rnd generator within process
    random.seed((10+boot)**2+121)
    np.random.seed((10+boot)**2+121)
    if c_dict['panel_in_rf']:
        cl_unique = np.unique(data[:, cl_i])
        n_cl = cl_unique.shape[0]
        n_train = round(n_cl * c_dict['subsam_share_forest'])
        indices_cl = random.sample(range(n_cl), n_train)  # Returns list
        indices = []
        for i in range(n_obs):
            if data[i, cl_i] in indices_cl:
                indices.append(i)
        data = data[:, :-1]                 # CL_ind is at last position
    else:
        n_train = round(n_obs * c_dict['subsam_share_forest'])
        indices = random.sample(range(n_obs), n_train)
    data_tr = data[indices]
    data_oob = np.delete(data, indices, axis=0)
    n_tr = data_tr.shape[0]
    n_oob = data_oob.shape[0]
    node_t_init = init_node_table(n_tr, n_oob,
                                  np.delete(range(n_obs), indices, axis=0))
    # build trees for all m,n combinations
    if np.size(c_dict['grid_m']) == 1:
        grid_for_m = [c_dict['grid_m']]
    else:
        grid_for_m = c_dict['grid_m']
    if np.size(c_dict['grid_n_min']) == 1:
        grid_for_n_min = [c_dict['grid_n_min']]
    else:
        grid_for_n_min = c_dict['grid_n_min']
    if np.size(c_dict['grid_alpha_reg']) == 1:
        grid_for_alpha_reg = [c_dict['grid_alpha_reg']]
    else:
        grid_for_alpha_reg = c_dict['grid_alpha_reg']
    tree_all = [None] * len(grid_for_m) * len(grid_for_n_min) * len(
        grid_for_alpha_reg)
    j = 0
    for m_idx in grid_for_m:
        for n_min in grid_for_n_min:
            for alpha_reg in grid_for_alpha_reg:
                node_table_0 = copy.deepcopy(node_t_init)  # emty table
                tree_all[j] = build_single_tree(
                    data_tr, data_oob, y_i, y_nn_i, d_i, x_i, w_i, x_type,
                    x_values, x_ind, x_ai_ind, c_dict, m_idx, n_min, alpha_reg,
                    node_table_0, pen_mult, regrf)
                j += 1
    return tree_all


def build_forest(indatei, v_dict, v_x_type, v_x_values, c_dict, regrf=False):
    """Build MCF (not yet populated by w and outcomes).

    Parameters
    ----------
    datendatei : string. Data  contained in csv file
    v : Dictionary. Variable names.
    v_x_type : Dictionary. Key: Variable name. Values: 0,1,2
    v_x_values: Dictionary. Key: Variable name. Values: List with INT/Float
    c : Dictionary. Control parameters
    regrf: Boolean. True if regression random forest. Default=False.

    Returns
    -------
    forest_final : Dictionary. All info needed for the forest estimated
    x_name: List. Order of x_name as used by tree building

    """
    old_mp_with_ray = c_dict['mp_with_ray']
    if c_dict['no_ray_in_forest_building'] and c_dict['mp_with_ray']:
        if c_dict['with_output'] and c_dict['verbose']:
            print('No use of ray in forest building.')
        c_dict['mp_with_ray'] = False
    (x_name, x_type, x_values, c_dict, pen_mult, data_np, y_i, y_nn_i, x_i,
     x_ind, x_ai_ind, d_i, w_i, cl_i) = mcf_data.prepare_data_for_forest(
         indatei, v_dict, v_x_type, v_x_values, c_dict, regrf=regrf)
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
    if maxworkers == 1:
        forest = [None] * c_dict['boot']
        for idx in range(c_dict['boot']):
            forest[idx] = build_tree_mcf(
                data_np, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type, x_values,
                x_ind, x_ai_ind, c_dict, idx, pen_mult, regrf)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, c_dict['boot'])
    else:
        if c_dict['mem_object_store_1'] is not None and c_dict['mp_with_ray']:
            boot_by_boot = 1
        else:
            boot_by_boot = c_dict['boot_by_boot']
        forest = []
        if boot_by_boot == 1:
            if c_dict['mp_with_ray']:
                if c_dict['mem_object_store_1'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_1'])
                    if c_dict['with_output'] and c_dict['verbose']:
                        print("Size of Ray Object Store: ", round(
                            c_dict['mem_object_store_1']/(1024*1024)), " MB")
                data_np_ref = ray.put(data_np)
                still_running = [ray_build_tree_mcf.remote(
                    data_np_ref, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                    x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult, regrf)
                    for boot in range(c_dict['boot'])]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        forest.append(ret_all_i)
                        # del ret_all_i
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, c_dict['boot'])
                        jdx += 1
                    if jdx % 50 == 0:   # every 50'th tree
                        gp_sys.auto_garbage_collect(50)  # do if half mem full
                if 'refs' in c_dict['_mp_ray_del']:
                    del data_np_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        build_tree_mcf, data_np, y_i, y_nn_i, x_i, d_i,
                        cl_i, w_i, x_type, x_values, x_ind, x_ai_ind, c_dict,
                        boot, pen_mult, regrf):
                            boot for boot in range(c_dict['boot'])}
                    for jdx, frx in enumerate(futures.as_completed(ret_fut)):
                        forest.append(frx.result())
                        del ret_fut[frx]
                        del frx
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, c_dict['boot'])
        else:
            if (c_dict['boot'] / maxworkers) > c_dict['boot_by_boot']:
                no_of_split = round(c_dict['boot'] / c_dict['boot_by_boot'])
            else:
                no_of_split = maxworkers
            boot_indx_list = np.array_split(range(c_dict['boot']), no_of_split)
            if c_dict['with_output'] and c_dict['verbose']:
                print('Avg. number of bootstraps per process: {}'.format(
                    round(c_dict['boot'] / no_of_split, 2)))
            if c_dict['mp_with_ray']:
                if c_dict['mem_object_store_1'] is None:
                    if not ray.is_initialized():
                        ray.init(num_cpus=maxworkers, include_dashboard=False)
                else:
                    if not ray.is_initialized():
                        ray.init(
                            num_cpus=maxworkers, include_dashboard=False,
                            object_store_memory=c_dict['mem_object_store_1'])
                data_np_ref = ray.put(data_np)
                still_running = [ray_build_many_trees_mcf.remote(
                    data_np_ref, y_i, y_nn_i, x_i, d_i, cl_i, w_i, x_type,
                    x_values, x_ind, x_ai_ind, c_dict, boot, pen_mult, regrf)
                    for boot in boot_indx_list]
                jdx = 0
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for ret_all_i in finished_res:
                        forest.extend(ret_all_i)
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, len(boot_indx_list))
                        jdx += 1
                    if jdx % 50 == 0:   # every 50'th tree
                        gp_sys.auto_garbage_collect(50)  # do if 0.5 mem. full
                if 'refs' in c_dict['_mp_ray_del']:
                    del data_np_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        build_many_trees_mcf, data_np, y_i, y_nn_i, x_i, d_i,
                        cl_i, w_i, x_type, x_values, x_ind, x_ai_ind, c_dict,
                        boot, pen_mult, regrf):
                            boot for boot in boot_indx_list}
                    for jdx, frx in enumerate(futures.as_completed(ret_fut)):
                        forest.extend(frx.result())
                        del ret_fut[frx]
                        del frx
                        if c_dict['with_output'] and c_dict['verbose']:
                            gp.share_completed(jdx+1, len(boot_indx_list))
        if len(forest) != c_dict['boot']:
            raise Exception('Forest has wrong size: ', len(forest),
                            'Bug in Multiprocessing.')
    # find best forest given the saved oob values
    forest_final, m_n_final = best_m_n_min_alpha_reg(forest, c_dict)
    del forest    # Free memory
    # Describe final tree
    if c_dict['with_output']:
        describe_forest(forest_final, m_n_final, v_dict, c_dict, pen_mult,
                        regrf)
    if c_dict['no_ray_in_forest_building'] and old_mp_with_ray:
        c_dict['mp_with_ray'] = old_mp_with_ray
    return forest_final, x_name


@njit
def mcf_penalty(shares_l, shares_r):
    """Generate the (unscaled) penalty.

    Parameters
    ----------
    shares_l : Numpy array. Treatment shares left.
    shares_r : Numpy array. Treatment shares right.

    Returns
    -------
    penalty : Numpy INT. Penalty of split.

    """
    diff = (shares_l - shares_r) ** 2
    penalty = 1 - (np.sum(diff) / len(shares_l))
    return penalty


def get_split_values(y_dat, w_dat, x_dat, x_type, x_values, leaf_size, c_dict):
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

    Returns
    -------
    splits : List. Splitting values to use.

    """
    rng = np.random.default_rng()
    if x_type == 0:
        if bool(x_values):  # Limited number of values in x_value
            min_x = np.amin(x_dat)
            max_x = np.amax(x_dat)
            del_values = []
            for j, val in enumerate(x_values):
                if (val < (min_x - 1e-15)) or (val > (max_x + 1e-15)):
                    del_values.append(j)
            if del_values:  # List is not empty
                splits = [x for x in x_values if x not in del_values]
            else:
                splits = x_values[:]
            if len(splits) > 1:
                splits = splits[:-1]
                if 0 < c_dict['random_thresholds'] < len(splits):
                    splits = np.unique(random.sample(
                        splits, c_dict['random_thresholds']))
        else:  # Continoues variable with very many values; x_values empty
            if 0 < c_dict['random_thresholds'] < (leaf_size - 1):
                x_vals_np = rng.choice(
                    x_dat, c_dict['random_thresholds'], replace=False)
                x_vals_np = np.unique(x_vals_np)
                splits = x_vals_np.tolist()
            else:
                x_vals_np = np.unique(x_dat)
                splits = x_vals_np.tolist()
                if len(splits) > 1:
                    splits = splits[:-1]
    else:
        y_mean_by_cat = np.empty(len(x_values))  # x_vals comes as list
        x_vals_np = np.array(x_values, dtype=np.int32, copy=True)
        used_values = []
        for v_idx, val in enumerate(x_vals_np):
            value_equal = np.isclose(x_dat, val)
            if np.any(value_equal):  # Position of empty cells do not matter
                if c_dict['w_yes']:
                    y_mean_by_cat[v_idx] = np.average(
                        y_dat[value_equal], weights=w_dat[value_equal], axis=0)
                else:
                    y_mean_by_cat[v_idx] = np.average(
                        y_dat[value_equal], axis=0)
                used_values.append(v_idx)
        x_vals_np = x_vals_np[used_values]
        sort_ind = np.argsort(y_mean_by_cat[used_values])
        x_vals_np = x_vals_np[sort_ind]
        splits = x_vals_np.tolist()
        splits = splits[:-1]  # Last category not needed
    return splits
