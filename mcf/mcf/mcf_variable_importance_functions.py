"""
Contains functions for common support adjustments.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from concurrent import futures
from math import floor

import numpy as np
import pandas as pd
import ray

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mcf import mcf_forest_data_functions as mcf_data
from mcf import mcf_forest_functions as mcf_fo
from mcf import mcf_forest_add_functions as mcf_fo_add
from mcf import mcf_general as gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def variable_importance(mcf_, data_df, forest, x_name_mcf):
    """Compute variable importance measure.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame.
    forest : Estimated forest.
    x_name_mcf : List. Variable names from MCF procedure


    Returns
    -------
    vim : Dictionary. Variable importance measures and names of variable

    Procedure:
    a) Predict Total of OOB of estimated forest_est (should already be there)
    b) For each variable, randomize one or groups of covariates
    c) recompute OOB-MSE with the ys of these more noisy variables

    Use multiprocessing in new oob prediction in the same way as in forest
    Building
    Outer loop over variables and group of variables
        Inner loop: Loop over trees
            Initially take out the indices of leaf 0 [16]--> OOB data
            For every observation determine it terminal leaf
                - save index together with number of terminal leaf
            For all terminal leaves compute OOB prediction

    """
    int_dic, gen_dic = mcf_.int_dict, mcf_.gen_dict
    if int_dic['with_output'] and int_dic['verbose']:
        txt = '\nVariable importance measures (OOB data)\nSingle variables'
        ps.print_mcf(gen_dic, txt, summary=True)
    (x_name, _, _, cf_dic, _, data_np, y_i, y_nn_i, x_i, _, _, d_i, w_i, _, _
     ) = mcf_data.prepare_data_for_forest(mcf_, data_df)
    no_of_vars = len(x_name)
    partner_k = determine_partner_k(x_name)
    # Loop over all variables to get respective OOB values of MSE
    if x_name != x_name_mcf:
        raise ValueError('Wrong order of variable names', x_name, x_name_mcf)
    number_of_oobs = 1 + no_of_vars
    oob_values = [None] * number_of_oobs
    if gen_dic['mp_parallel'] < 1.5:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])
        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and int_dic['verbose']:
        txt = f'\nNumber of parallel processes: {maxworkers}'
        ps.print_mcf(gen_dic, txt, summary=False)
    if maxworkers > 1 and int_dic['ray_or_dask'] == 'ray':
        if int_dic['mem_object_store_2'] is None:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False)
        else:
            if not ray.is_initialized():
                ray.init(num_cpus=maxworkers, include_dashboard=False,
                         object_store_memory=int_dic['mem_object_store_2'])
            if int_dic['with_output'] and int_dic['verbose']:
                txt = '\nSize of Ray Object Store: '
                txt += f'{round(int_dic["mem_object_store_2"]/(1024*1024))} MB'
                ps.print_mcf(gen_dic, txt, summary=False)
        data_np_ref = ray.put(data_np)
        forest_ref = ray.put(forest)
    if (int_dic['mp_vim_type'] == 2 and int_dic['ray_or_dask'] != 'ray') or (
            maxworkers == 1):
        for jdx in range(number_of_oobs):
            oob_values[jdx], _ = get_oob_mcf(
                data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic, cf_dic,
                jdx, True, [], forest, False, partner_k[jdx])
            if int_dic['with_output'] and int_dic['verbose']:
                gp.share_completed(jdx+1, number_of_oobs)
    else:  # Fast but needs a lot of memory because it copied a lot
        maxworkers = min(maxworkers, number_of_oobs)
        if int_dic['ray_or_dask'] == 'ray':
            still_running = [ray_get_oob_mcf.remote(
                data_np_ref, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                cf_dic, idx, True, [], forest_ref, True, partner_k[idx])
                for idx in range(number_of_oobs)]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for res in finished_res:
                    iix = res[1]
                    oob_values[iix] = res[0]
                    if int_dic['with_output'] and int_dic['verbose']:
                        gp.share_completed(jdx+1, number_of_oobs)
                        jdx += 1
    oob_values = np.array(oob_values)
    oob_values = oob_values.reshape(-1)
    mse_ref = oob_values[0]   # reference value
    vim, txt = vim_print(mse_ref, oob_values[1:], x_name, 0,
                         int_dic['with_output'], True, partner_k)
    if int_dic['with_output']:
        ps.print_mcf(gen_dic, txt, summary=True)
    # Variables are grouped
    no_g, no_m_g = number_of_groups_vi(no_of_vars)
    partner_k = None
    if no_g > 0:
        if int_dic['with_output']:
            print('\nGroups of variables')
        ind_groups = vim_grouping(vim, no_g)
        n_g = len(ind_groups)
        oob_values = [None] * n_g
        if maxworkers > 1 and int_dic['ray_or_dask'] == 'ray':
            still_running = [ray_get_oob_mcf.remote(
                data_np_ref, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                cf_dic, idx, False, ind_groups, forest_ref, True, partner_k)
                for idx in range(n_g)]
            idx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for res in finished_res:
                    iix = res[1]
                    oob_values[iix] = res[0]
                    if int_dic['with_output'] and int_dic['verbose']:
                        gp.share_completed(idx+1, n_g)
                        idx += 1
        else:
            for idx in range(n_g):
                oob_values[idx], _ = get_oob_mcf(
                    data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                    cf_dic, idx, False, ind_groups, forest, False, partner_k)
                if int_dic['with_output'] and int_dic['verbose']:
                    gp.share_completed(idx+1, n_g)
        vim_g, txt = vim_print(mse_ref, np.array(oob_values), x_name,
                               ind_groups, int_dic['with_output'], False)
        if int_dic['with_output']:
            ps.print_mcf(gen_dic, txt, summary=True)
    else:
        vim_g = None
    # Groups are accumulated from worst to best
    if no_m_g > 0:
        if int_dic['with_output']:
            ps.print_mcf(gen_dic, '\nMerged groups of variables', summary=True)
        ind_groups = vim_grouping(vim_g, no_m_g, True)
        n_g = len(ind_groups)
        oob_values = [None] * n_g
        if maxworkers > 1 and int_dic['ray_or_dask'] == 'ray':
            still_running = [ray_get_oob_mcf.remote(
                data_np_ref, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                cf_dic, idx, False, ind_groups, forest_ref, True, partner_k)
                for idx in range(n_g)]
            idx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for res in finished_res:
                    iix = res[1]
                    oob_values[iix] = res[0]
                    if int_dic['with_output'] and int_dic['verbose']:
                        gp.share_completed(idx+1, n_g)
                        idx += 1
        else:
            for idx in range(n_g):
                oob_values[idx], _ = get_oob_mcf(
                    data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                    cf_dic, idx, False, ind_groups, forest, False, partner_k)
                if int_dic['with_output'] and int_dic['verbose']:
                    gp.share_completed(idx+1, n_g)
        vim_mg, txt = vim_print(mse_ref, np.array(oob_values), x_name,
                                ind_groups, int_dic['with_output'], False)
        if int_dic['with_output']:
            ps.print_mcf(gen_dic, txt, summary=True)
    else:
        vim_mg = None
    if int_dic['ray_or_dask'] == 'ray' and maxworkers > 1:
        if 'refs' in int_dic['mp_ray_del']:
            del data_np_ref, forest_ref
        if 'rest' in int_dic['mp_ray_del']:
            del finished_res, finished
        if int_dic['mp_ray_shutdown']:
            ray.shutdown()
    return vim, vim_g, vim_mg, x_name


def vim_grouping(vim, no_groups, accu=False):
    """Group variables according to their variable importance measure.

    Parameters
    ----------
    vim : Tuple (Numpy array list of INT). Relative vim and index.
    no_g : INT. No of groups.
    accu : Bool. Build groups by accumulation. Default = False.

    Returns
    -------
    ind_groups : List of list of INT. Grouping of indices.

    """
    indices = vim[1]
    no_ind = len(indices)
    if not accu:
        group_size = no_ind / no_groups
        group_size_int = floor(group_size)
        one_more, start_i, ind_groups = 0, 0, []
    for idx in range(no_groups):
        if accu:
            if idx < 2:
                ind_groups = [indices[0], indices[0] + indices[idx]]
            else:
                new_group = ind_groups[idx-1] + indices[idx]
                ind_groups.append(new_group)
        else:
            if idx == (no_groups - 1):
                end_i = no_ind - 1
            else:
                end_i = start_i + group_size_int - 1
                one_more += group_size - group_size_int
                if one_more >= 1:
                    one_more -= 1
                    end_i += 1
            ind_groups.append(indices[start_i:end_i+1])
            start_i = end_i + 1
    return ind_groups


def number_of_groups_vi(no_x_names):
    """Determine no of groups for groupwise variable importance measure.

    Parameters
    ----------
    no_x_names :INT. No of variables considered in analysis.

    Returns
    -------
    groups : INT.
    merged_groups : INT.

    """
    if no_x_names >= 100:
        groups, merged_groups = 20, 19
    elif 20 <= no_x_names < 100:
        groups, merged_groups = 10, 9
    elif 10 <= no_x_names < 20:
        groups, merged_groups = 5, 4
    elif 4 <= no_x_names < 10:
        groups, merged_groups = 2, 0
    else:
        groups = merged_groups = 0
    return groups, merged_groups


def vim_print(mse_ref, mse_values, x_name, ind_list=0, with_output=True,
              single=True, partner_k=None):
    """Print Variable importance measure and create sorted output.

    Parameters
    ----------
    mse_ref : Numpy Float. Reference value of non-randomized x.
    mse_values : Numpy array. MSE's for randomly permuted x.
    x_name : List of strings. Variable names.
    ind_list : List of INT, optional. Variable positions. Default is 0.
    with_output : Boolean, optional. Default is True.
    single : Boolean, optional. The default is True.
    partner_k : List of None and Int or None. Index of variables that were
                jointly randomized. Default is None.

    Returns
    -------
    vim: Tuple of Numpy array and list of lists. MSE sorted and sort index.

    """
    if partner_k is not None:
        for idx, val in enumerate(partner_k):
            if val is not None:
                if (idx > (val-1)) and (idx > 0):
                    mse_values[idx-1] = mse_values[val-1]
    mse = mse_values / np.array(mse_ref) * 100
    var_indices = np.argsort(mse)
    var_indices = np.flip(var_indices)
    vim_sorted = mse[var_indices]
    if single:
        x_names_sorted = np.array(x_name, copy=True)
        x_names_sorted = x_names_sorted[var_indices]
        ind_sorted = list(var_indices)
    else:
        var_indices = list(var_indices)
        ind_sorted = []
        x_names_sorted = []
        for i in var_indices:
            ind_i = ind_list[i]
            ind_sorted.append(ind_i)
            x_name_i = [x_name[j] for j in ind_i]
            x_names_sorted.append(x_name_i)
    txt = ''
    if with_output:
        txt += '\n' * 2 + '-' * 100 + f'\nOut of bag MSE: {mse_ref:8.3f}'
        txt += '\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
        txt += '\nVariable importance statistics in %-lost of base value'
        for idx, vim in enumerate(vim_sorted):
            if single:
                txt += f'\n{x_names_sorted[idx]:<50}: {vim-100:>7.4f}%'
            else:
                txt += '\n' + ' '.join(x_names_sorted[idx])
                txt += f'\n{" ":<50}: {vim-100:>7.4f}%'
        txt += '\n' + '-' * 100
        txt += ('\nComputed as share of OOB MSE of estimated forest relative'
                '\nto OOB MSE of variable (or group of variables)'
                '\nwith randomized covariate values in %.')
    ind_sorted.reverse()
    vim_sorted = np.flip(vim_sorted)
    vim = (vim_sorted, ind_sorted)
    if with_output:
        first_time = True
        if partner_k is not None:
            for idx, val in enumerate(partner_k):
                if val is not None:
                    if first_time:
                        txt += ('\nThe following variables are jointly'
                                ' analysed: ')
                        first_time = False
                    if idx < val:
                        txt += x_name[idx-1] + ' ' + x_name[val-1] + ' / '
        txt += '\n' + '-' * 100
    return vim, txt


@ray.remote
def ray_get_oob_mcf(data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                    cf_dic, k, single, group_ind_list, forest, no_mp=False,
                    partner_k=None):
    """Make function usable for Ray."""
    return get_oob_mcf(data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic,
                       cf_dic, k, single, group_ind_list, forest, no_mp,
                       partner_k)


def get_oob_mcf(data_np, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, int_dic, cf_dic,
                k, single, group_ind_list, forest, no_mp=False,
                partner_k=None):
    """Get the OOB value of a forest.

    Parameters
    ----------
    data_np : Numpy array. Data
    y_i : INT. Position in data_np.
    y_nn_i : Numpy array.
    x_i : Numpy array.
    d_i : INT.
    w_i : INT.
    gen_dic, int_dic, cf_dic : Dict.
    k: INT. Number of groups/variables.
    single : Bool. Single variable.
    group_ind_list : List of Lists of Int.
    forest : List of lists. Estimated forest in table form.
    no_mp : Bool. No multiprocessing.  Default is False.
    partner_k : For single variables only: Allows to jointly randomize another
                single variables that strongly covaries with variable k.

    Returns
    -------
    oob_value : Float. MSE based on out-of-bag observations.
    k: INT. Number of groups/variables.

    """
    oob_value = 0
    if gen_dic['mp_parallel'] < 1.5 or no_mp:
        maxworkers = 1
    else:
        if gen_dic['mp_automatic']:
            maxworkers = mcf_sys.find_no_of_workers(gen_dic['mp_parallel'],
                                                    gen_dic['sys_share'])
        else:
            maxworkers = gen_dic['mp_parallel']
    if int_dic['with_output'] and not no_mp and int_dic['verbose']:
        ps.print_mcf(gen_dic, f'Number of parallel processes: {maxworkers}',
                     summary=False)
    if (maxworkers == 1) or no_mp:
        for idx in range(cf_dic['boot']):
            oob_tree = get_oob_mcf_b(
                data_np[forest[idx][0][16]], y_i, y_nn_i, x_i, d_i, w_i,
                gen_dic, cf_dic, k, single, group_ind_list, forest[idx],
                partner_k=partner_k)
            oob_value += oob_tree
    else:
        if int_dic['mp_weights_tree_batch'] > 1:  # User defined # of batches
            no_of_boot_splits = int_dic['mp_weights_tree_batch']
            split_forest = True
        elif int_dic['mp_weights_tree_batch'] == 0:  # Automatic # of batches
            size_of_forest_mb = mcf_sys.total_size(forest) / (1024 * 1024)
            no_of_boot_splits, txt = mcf_sys.no_of_boot_splits_fct(
                size_of_forest_mb, maxworkers)
            if int_dic['with_output'] and int_dic['verbose']:
                ps.print_mcf(gen_dic, txt, summary=False)
            split_forest = bool(no_of_boot_splits < cf_dic['boot'])
        else:
            split_forest = False
        if split_forest and int_dic['with_output'] and int_dic['verbose']:
            txt = f'Number of tree chuncks: {no_of_boot_splits:5d}'
            ps.print_mcf(gen_dic, txt, summary=False)
        if split_forest:
            b_ind_list = np.array_split(range(cf_dic['boot']),
                                        no_of_boot_splits)
            with futures.ProcessPoolExecutor(max_workers=maxworkers) as fpp:
                ret_fut = {}
                for idx, b_ind in enumerate(b_ind_list):
                    forest_temp = forest[b_ind[0]:b_ind[-1]+1]
                    ret_fut_t = {fpp.submit(
                        get_oob_mcf_chuncks, data_np, y_i, y_nn_i, x_i, d_i,
                        w_i, gen_dic, cf_dic, k, single, group_ind_list,
                        forest_temp, b_ind, partner_k): idx}
                    ret_fut.update(ret_fut_t)
                for frv in futures.as_completed(ret_fut):
                    oob_value += frv.result()
                    del ret_fut[frv]
                    del frv
    oob_value = oob_value / cf_dic['boot']
    return oob_value, k


def get_oob_mcf_chuncks(data, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, cf_dic, k,
                        single, group_ind_list, node_table, index_list,
                        partner_k=None):
    """Compute OOB value in chuncks."""
    oob_value = 0
    for idx, _ in enumerate(index_list):
        oob_value += get_oob_mcf_b(
            data[node_table[idx][0][16]], y_i, y_nn_i, x_i, d_i, w_i, gen_dic,
            cf_dic, k, single, group_ind_list, node_table[idx], partner_k)
    return oob_value


def get_oob_mcf_b(data, y_i, y_nn_i, x_i, d_i, w_i, gen_dic, cf_dic, k, single,
                  group_ind_list, node_table, partner_k=None):
    """Generate OOB contribution for single bootstrap."""
    x_dat, y_dat = data[:, x_i], data[:, y_i]
    y_nn = data[:, y_nn_i]
    d_dat = np.int16(np.round(data[:, d_i]))
    w_dat = data[:, [w_i]] if gen_dic['weighted'] else None
    obs = len(y_dat)
    rng = np.random.default_rng(55436356)
    if not (single and (k == 0)):
        if single:
            rng.shuffle(x_dat[:, k-1])
            if partner_k is not None:   # Randomises variable related to k-1
                rng.shuffle(x_dat[:, partner_k-1])
        else:
            rand_ind = np.arange(obs)
            rng.shuffle(rand_ind)
            for i in group_ind_list[k]:
                x_dat[:, i] = x_dat[rand_ind, i]
    obs_in_leaf = np.empty((obs, 2), dtype=np.uint32)
    for idx in range(obs):
        leaf_no = mcf_fo_add.get_terminal_leaf_no(node_table, x_dat[idx, :])
        obs_in_leaf[idx, 0], obs_in_leaf[idx, 1] = idx, leaf_no
    oob_tree = mcf_fo.oob_in_tree(
        obs_in_leaf, y_dat, y_nn, d_dat, w_dat, cf_dic['mtot'],
        gen_dic['no_of_treat'], gen_dic['d_values'], gen_dic['weighted'],
        cont=gen_dic['d_type'] == 'continuous')
    return oob_tree


def determine_partner_k(x_name):
    """Find variable that is descretized equivalent to other variable.

    Parameters
    ----------
    x_name : List of str. Variable names to check.

    Returns
    -------
    partner_k : List of int. Contains either index (+1) of partner or None.

    """
    no_of_vars = len(x_name)
    partner_k = [None] * (no_of_vars + 1)
    x_partner_name = [None] * (no_of_vars + 1)
    for idx, val in enumerate(x_name):  # check if it ends with CATV & remove
        if len(val) > 4:
            if val.endswith('CATV'):
                x_partner_name[idx+1] = val[:-4]
    for idx, val in enumerate(x_name):
        for jdx, jval in enumerate(x_partner_name):
            if (val == jval) and (partner_k[idx+1] is None):
                partner_k[idx+1] = jdx
                partner_k[jdx] = idx + 1
                break
    return partner_k


def print_variable_importance(clas_obj, x_df, d_df, x_name, names_uo,
                              unordered_dummy_names, gen_dic, summary=False):
    """Compute and print variable importance by permutation for CS."""
    x_train_df, x_test_df, d_train_df, d_test_df = train_test_split(
        x_df, d_df, test_size=0.25, random_state=42)
    x_train_np = x_train_df.to_numpy()
    d_train_np = d_train_df.to_numpy().ravel()
    clas_obj.fit(x_train_np, d_train_np)
    x_test_np = x_test_df.to_numpy()
    d_test_np = d_test_df.to_numpy().ravel()
    score_d_full = accuracy_score(d_test_np, clas_obj.predict(x_test_np),
                                  normalize=True)
    vi_information = pd.DataFrame(columns=x_name, index=['score_w/o_x_d',
                                                         'rel_diff_d_%'])
    for name in x_name:
        if name in names_uo:
            names_to_shuffle = unordered_dummy_names[name]
        else:
            names_to_shuffle = name
        x_all_rnd_df = x_test_df.copy().reset_index(drop=True)
        x_rnd_df = x_test_df[names_to_shuffle].sample(frac=1, random_state=42)
        x_all_rnd_df[names_to_shuffle] = x_rnd_df.reset_index(drop=True)
        d_pred_rnd = clas_obj.predict(x_all_rnd_df.to_numpy())
        d_score = accuracy_score(d_test_np, d_pred_rnd, normalize=True)
        d_rel_diff = (score_d_full - d_score) / score_d_full
        vi_information[name] = [d_score, d_rel_diff*100]
    ps.print_mcf(gen_dic, f'Full score d: {score_d_full:6.3f} ',
                 summary=summary)
    with pd.option_context('display.max_rows', None,
                           'display.expand_frame_repr', True,
                           'chop_threshold', 1e-13):
        ps.print_mcf(gen_dic, vi_information.transpose().sort_values(
            by=['rel_diff_d_%'], ascending=False), summary=False)
