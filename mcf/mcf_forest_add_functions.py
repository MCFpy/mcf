"""Created on Thu Feb 24 10:17:14 2022.

Contains the functions needed for the tree and forest computations of MCF
@author: MLechner
-*- coding: utf-8 -*-
"""
from concurrent import futures
from numba import njit
from dask.distributed import Client, as_completed

import numpy as np
import ray

from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_general_purpose as mcf_gp


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
    (x_name, _, _, c_dict, _, data_np, _, _, x_i, _, _, d_i, _, _, _
     ) = mcf_data.prepare_data_for_forest(
         indatei, v_dict, v_x_type, v_x_values, c_dictin, True, regrf=regrf)
    err_txt = 'Wrong order of variables' + str(x_name) + ': ' + str(x_name_mcf)
    assert x_name_mcf == x_name, err_txt
    if c_dict['d_type'] == 'continuous':
        d_dat = data_np[:, d_i]
        # substitute those d used for splitting only that have a zero with
        # random element from the positive treatment levels
        d_pos = d_dat[d_dat > 1e-15]
        rng = np.random.default_rng(12366456)
        d_values = rng.choice(d_pos, size=len(d_dat)-len(d_pos), replace=False)
        d_dat_for_x = np.copy(d_dat)
        j = 0
        for i, d_i in enumerate(d_dat):
            if d_i < 1e-15:
                d_dat_for_x[i, 0] = d_values[j]
                j += 1
        x_dat = np.concatenate((data_np[:, x_i], d_dat_for_x), axis=1)
    else:
        x_dat = data_np[:, x_i]
        d_dat = np.int16(np.round(data_np[:, d_i]))
    obs = len(x_dat)
    terminal_nodes = [None] * c_dict['boot']
    nodes_empty = np.zeros(c_dict['boot'])
    if c_dict['no_parallel'] < 1.5:
        maxworkers = 1
    else:
        maxworkers = (mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                c_dict['sys_share'])
                      if c_dict['mp_automatic'] else c_dict['no_parallel'])
    if c_dict['with_output'] and c_dict['verbose']:
        print('Number of parallel processes: ', maxworkers)
    if maxworkers == 1:
        for idx in range(c_dict['boot']):
            (_, forest[idx], terminal_nodes[idx], nodes_empty[idx]
             ) = fill_mp(forest[idx], obs, d_dat, x_dat, idx, c_dict, regrf)
            if c_dict['with_output'] and c_dict['verbose']:
                gp.share_completed(idx+1, c_dict['boot'])
    else:
        if c_dict['_ray_or_dask'] == 'ray':
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
        elif c_dict['_ray_or_dask'] == 'dask':
            with Client(n_workers=maxworkers) as clt:
                x_dat_ref, d_dat_ref = clt.scatter(x_dat), clt.scatter(d_dat)
                ret_fut = {clt.submit(fill_mp, forest[idx], obs, d_dat_ref,
                                      x_dat_ref, idx, c_dict, regrf):
                           idx for idx in range(c_dict['boot'])}
                jdx = 0
                for _, res in as_completed(ret_fut, with_results=True):
                    jdx += 1
                    iix = res[0]
                    forest[iix] = res[1]
                    terminal_nodes[iix] = res[2]
                    nodes_empty[iix] = res[3]
                    if c_dict['with_output'] and c_dict['verbose']:
                        gp.share_completed(jdx+1, c_dict['boot'])
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
              f' in %: {no_of_avg_enodes*100:8.3f}')
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
    subsam = c_dict['subsam_share_eval'] < 1 if not regrf else False
    indices = np.arange(obs)
    if subsam:
        obs = round(obs * c_dict['subsam_share_eval'])
        rng = np.random.default_rng((10+b_idx)**2+121)
        indices = rng.choice(indices, size=obs, replace=False)
    obs_in_leaf = np.zeros((obs, 1), dtype=np.uint32)
    for i, idx in enumerate(indices):
        obs_in_leaf[i] = get_terminal_leaf_no(node_table, x_dat[idx, :])
    unique_leafs = np.unique(obs_in_leaf)
    if subsam:
        unique_leafs = unique_leafs[1:]  # remove first index: obs not used
        d_dat = d_dat[indices]
    nodes_empty = 0
    no_of_treat = (2 if c_dict['d_type'] == 'continuous'
                   else c_dict['no_of_treat'])
    for leaf_id in unique_leafs:
        sel_ind = obs_in_leaf.reshape(-1) == leaf_id
        node_table[leaf_id][14] = indices[sel_ind]
        empty_leaf = (len(sel_ind) < 1 if regrf else len(np.unique(
            d_dat[sel_ind])) < no_of_treat)
        if empty_leaf:
            node_table[leaf_id][16] = 1   # Leaf to be ignored
            nodes_empty += 1
    return b_idx, node_table, unique_leafs, nodes_empty/len(unique_leafs)


def get_terminal_leaf_no(node_table, x_dat):
    """Get the leaf number of the terminal node for single observation.

    Parameters
    ----------
    node_table : List of list. Single tree.
    x_dat : Numpy array. Data.

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
        assert leaf[4] == 1 or leaf[4] == 0, f'Leaf is still active. {leaf[4]}'
        if leaf[4] == 1:             # Terminal leaf
            not_terminal = False
            leaf_no = leaf[0]
        elif leaf[4] == 0:          # Intermediate leaf
            if leaf[10] == 0:        # Continuous variable
                leaf_id = (leaf[2] if (x_dat[leaf[8]] - 1e-15) <= leaf[9]
                           else leaf[3])
            else:                   # Categorical variable
                prime_factors = gp.primes_reverse(leaf[9], False)
                leaf_id = (leaf[2]
                           if int(np.round(x_dat[leaf[8]])) in prime_factors
                           else leaf[3])
    return leaf_no


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
    print(f'Number of replications:     {c_dict["boot"]:<4}')
    if not regrf:
        if c_dict['mtot'] == 3:
            splitting_rule = 'MSEs of regressions only considered'
        elif c_dict['mtot'] == 1:
            splitting_rule = 'MSE+MCE criterion'
        elif c_dict['mtot'] == 2:
            splitting_rule = '-Var(effect)'
        elif c_dict['mtot'] == 4:
            splitting_rule = 'Random switching'
        print(f'Splitting rule used:        {splitting_rule:<4}')
        if c_dict['mtot_p_diff_penalty'] > 0:
            print('Penalty used in splitting: ', pen_mult)
    print('Share of data in subsample for forest buildung:',
          f' {c_dict["subsam_share_forest"]:<4}')
    print('Share of data in subsample for forest evaluation:',
          f' {c_dict["subsam_share_eval"]:<4}')
    print('Total number of variables available for splitting:',
          f' {len(v_dict["x_name"]):<4}')
    print(f'# of variables (M) used for split: {m_n_min_ar[0]:<4}')
    if c_dict['m_random_poisson']:
        print('           (# of variables drawn from 1+Poisson(M-1))')
    print(f'Minimum leaf size:                 {m_n_min_ar[1]:<4}')
    print(f'Alpha regularity:                  {m_n_min_ar[2]:5.3f}')
    print('------------------- Estimated tree -------------------------------')
    leaf_info = get_tree_infos(forest)
    print(f'Average # of leaves:      {leaf_info[0]:4.1f}')
    print(f'Average size of leaves:   {leaf_info[1]:4.1f}')
    print(f'Median size of leaves:    {leaf_info[2]:4.1f}')
    print(f'Min size of leaves:       {leaf_info[3]:4.0f}')
    print(f'Max size of leaves:       {leaf_info[4]:4.0f}')
    print(f'Total # of obs in leaves: {leaf_info[5]:4.0f}')
    print('-' * 80)


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
    ind_ag = (ind_g if vi_ag is None
              else np.array(vi_ag[1], copy=True, dtype=object))
    below_i = vi_i[0] <= (100 + c_dict['fs_rf_threshold'])
    below_g = vi_g[0] <= (100 + c_dict['fs_rf_threshold'])
    below_ag = (below_g if vi_ag is None
                else vi_ag[0] <= (100 + c_dict['fs_rf_threshold']))
    nothing_removed = True
    if ((np.count_nonzero(below_i) > 0) and (np.count_nonzero(below_g) > 0)
            and (np.count_nonzero(below_ag) > 0)):   # necessary conditions met
        ind_i = set(ind_i[below_i])
        indi_g_flat = set(gp_est.flatten_list(list(ind_g[below_g])))
        indi_ag_flat = set(gp_est.flatten_list(list(ind_ag[below_ag])))
        remove_ind = ind_i & indi_g_flat & indi_ag_flat
        if remove_ind:           # If list is empty, this will be False
            names_to_remove1 = [x_name[i] for i in remove_ind]
            if regrf:
                forbidden_vars = (v_dict['x_name_always_in']
                                  + v_dict['x_name_remain'])
            else:
                forbidden_vars = (v_dict['x_name_always_in']
                                  + v_dict['x_name_remain']
                                  + v_dict['z_name'] + v_dict['z_name_list']
                                  + v_dict['z_name_mgate']
                                  + v_dict['z_name_amgate'])
            names_to_remove2 = [name for name in names_to_remove1
                                if name not in forbidden_vars]
            if names_to_remove2:
                nothing_removed = False
                for name_weg in names_to_remove2:
                    v_x_type.pop(name_weg)
                    v_x_values.pop(name_weg)
                    v_dict['x_name'].remove(name_weg)
                if c_dict['with_output']:
                    print('\nVariables deleted: ', *names_to_remove2)
                    print('\nVariables kept: ', *v_dict['x_name'])
                    gp.print_f(c_dict['outfilesummary'],
                               'Variables deleted: ', *names_to_remove2,
                               '\nVariables kept: ', *v_dict['x_name'])
    if nothing_removed:
        if c_dict['with_output']:
            print('\n', 'No variables removed in feature selection')
            gp.print_f(c_dict['outfilesummary'],
                       'No variables removed in feature selection')
    return v_dict, v_x_type, v_x_values


@njit
def select_one_row_element(data, indices):
    """Randomly find one element per row."""
    data_selected = np.empty((data.shape[0], 1))
    for idx, val in enumerate(indices):
        data_selected[idx, 0] = data[idx, val]
    return data_selected


def match_cont(d_grid, y_nn, grid_values, rng):
    """
    Select suitable match in case of continuous treatment.

    Parameters
    ----------
    d_grid : Numpy array.
        Discretised treatment.
    y_nn : Numpy array.
        Neighbours.
    leaf_l : Numpy array.
        Observations going to left leaf. (d < larger splitting value).
    leaf_r : Numpy array.
        Observations going to right leaf (d >= larger splitting value).
    grid_values : Numpy array.
        Values (midpoints) used to generate discretised treatment
    rng : Default random number generator object

    Returns
    -------
    y_nn_cont: N x 2 Numpy array.
        Selected neighbours to the left.
    """
    grid = grid_values[1:]    # Controls are not relevant
    min_d_grid, max_d_grid = np.min(d_grid), np.max(d_grid)
    col_no_min = np.argmin(np.abs(grid-min_d_grid))
    col_no_max = np.argmin(np.abs(grid-max_d_grid))
    indices = np.arange(col_no_min, col_no_max + 1)
    nn_indices = rng.choice(indices, size=d_grid.shape[0])
    y_nn_sel = select_one_row_element(y_nn[:, 1:], nn_indices)
    y_nn_red = np.concatenate((y_nn[:, 0].reshape(-1, 1), y_nn_sel), axis=1)
    return y_nn_red
