"""
Created on Wed April  4 15:20:07 2022.

Optimal Policy Trees: Tree Output Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
from copy import deepcopy
import random

import pandas as pd
import numpy as np

from mcf import optp_tree_functions as optp_t
from mcf import optp_print as optp_p
from mcf import general_purpose as gp


def combinations_categorical(single_x_np, ps_np_diff, c_dict, ft_yes=True):
    """
    Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    single_x_np : 1D Numpy array. Features.
    ps_np_diff : 2D Numpy array. Policy scores as difference.
    c_dict : Dict. Controls.

    Returns
    -------
    combinations : List of tuples with values for each split.

    """
    values = np.unique(single_x_np)
    no_of_values = len(values)
    no_of_combinations = gp.total_sample_splits_categorical(no_of_values)
    if no_of_combinations < c_dict['ft_no_of_evalupoints']:
        combinations = gp.all_combinations_no_complements(list(values))
    else:
        values_sorted, no_of_ps = optp_t.get_values_ordered(
            single_x_np, ps_np_diff, values, no_of_values,
            with_numba=c_dict['with_numba'])
        combinations_t = sorted_values_into_combinations(
            values_sorted, no_of_ps, no_of_values)
        if (len(combinations_t) > c_dict['ft_no_of_evalupoints']) and ft_yes:
            combinations_t = random.sample(
                combinations_t, c_dict['ft_no_of_evalupoints'])
        combinations, _ = gp.drop_complements(combinations_t, list(values))
    return combinations


def sorted_values_into_combinations(values_sorted, no_of_ps, no_of_values):
    """
    Transfrom sorted values into unique combinations of values.

    Parameters
    ----------
    values_sorted : 2D numpy array. Sorted values for each policy score
    no_of_ps : Int. Number of policy scores.
    no_of_values : Int. Number of values.

    Returns
    -------
    unique_combinations : Unique Tuples to be used for sample splitting.

    """
    unique_combinations = []
    value_idx = np.arange(no_of_values-1)
    for j in range(no_of_ps):
        for i in value_idx:
            next_combi = tuple(values_sorted[value_idx[:i+1], j])
            if next_combi not in unique_combinations:
                unique_combinations.append(next_combi)
    return unique_combinations


def prepare_data_for_tree_builddata(datafile_name, c_dict, v_dict, x_type,
                                    x_value):
    """Prepare data for tree building."""
    if c_dict['only_if_sig_better_vs_0']:
        data_ps, data_df = optp_t.adjust_policy_score(datafile_name, c_dict,
                                                      v_dict)
    else:
        data_df = pd.read_csv(datafile_name)
        data_ps = data_df[v_dict['polscore_name']].to_numpy()
    data_ps_diff = data_ps[:, 1:] - data_ps[:, 0, np.newaxis]
    no_of_x = len(x_type)
    name_x = [None] * no_of_x
    type_x, values_x = [None] * no_of_x, [None] * no_of_x
    for j, key in enumerate(x_type.keys()):
        name_x[j], type_x[j] = key, x_type[key]
        values_x[j] = (sorted(x_value[key])
                       if x_value[key] is not None else None)
    data_x = data_df[name_x].to_numpy()
    del data_df
    if c_dict['x_unord_flag']:
        for m_i in range(no_of_x):
            if type_x[m_i] == 'unord':
                data_x[:, m_i] = np.round(data_x[:, m_i])
                values_x[m_i] = combinations_categorical(
                    data_x[:, m_i], data_ps_diff, c_dict)
    return data_x, data_ps, data_ps_diff, name_x, type_x, values_x


def automatic_cost(datafile_name, v_dict, c_dict):
    """Compute costs that fulfill constraints.

    Parameters
    ----------
    data_file : string. Input data
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.

    Returns
    -------
    c_dict : Dict. Parameters with modified cost variable.

    """
    if c_dict['with_output']:
        print('\nSearching cost values that fulfill constraints')
    if c_dict['only_if_sig_better_vs_0']:
        data_ps, data_df = optp_t.adjust_policy_score(datafile_name, c_dict,
                                                      v_dict)
    else:
        data_df = pd.read_csv(datafile_name)
        data_ps = data_df[v_dict['polscore_name']].to_numpy()
    obs = len(data_ps)
    max_by_treat = np.array(c_dict['max_by_treat'])
    costs_of_treat = np.zeros(c_dict['no_of_treatments'])
    std_ps = np.std(data_ps.reshape(-1))
    step_size = 0.02
    while True:
        treatments = np.argmax(data_ps - costs_of_treat, axis=1)
        values, count = np.unique(treatments, return_counts=True)
        if len(count) == c_dict['no_of_treatments']:
            alloc = count
        else:
            alloc = np.zeros(c_dict['no_of_treatments'])
            for i, j in enumerate(values):
                alloc[j] = count[i]
        diff = alloc - max_by_treat
        diff[diff < 0] = 0
        if not np.any(diff > 0):
            break
        costs_of_treat += diff / obs * std_ps * step_size
    alloc = np.uint32(alloc)
    costs_of_treat_neu = costs_of_treat * c_dict['costs_mult']
    if c_dict['with_output']:
        print('Constraints (share): ', end='')
        for j in c_dict['max_by_treat']:
            print(f'{j / obs:7.2%} ', end='')
        print('\n' + '- ' * 40)
        print('Cost values determined by unconstrained optimization: ', end='')
        for j in costs_of_treat:
            print(f'{j:8.3f}', end='')
        print('\nMultiplier to be used: ', c_dict["costs_mult"])
        print('- ' * 40)
        print('Adjusted Cost values and allocation in unconstrained',
              ' optimization')
        for i, j in enumerate(costs_of_treat_neu):
            print(f'{j:8.3f}  {alloc[i]:6d}  {alloc[i] / obs:6.2%}')
        print('-' * 80)
    return costs_of_treat_neu


def two_leafs_info(tree, polscore_df, x_df, leaf, polscore_is_index=False):
    """Compute the information contained in two adjacent leaves.

    Parameters
    ----------
    tree : List of lists.
    polscore_df : Dataframe. Policyscore or index.
    x_df : Dataframe. Policy variables.
    leaf : List. Terminal leaf under investigation.

    Raises
    ------
    Exception : If inconsistent leaf numbers.

    Returns
    -------
    leaf_splits : Tuple of List of dict. All splits that lead to left leaf.
    score : Tuple of Float. Final value of left leaf. (left, right)
    obs : Tuple of Int. Number of observations in leaf.

    """
    # Collect decision path of the two final leaves
    leaf_splits_pre = []
    parent_nr, current_nr = leaf[1], leaf[0]
    if len(tree) > 1:
        while parent_nr is not None:
            for leaf_i in tree:
                if leaf_i[0] == parent_nr:
                    if current_nr == leaf_i[2]:
                        left_right = 'left'
                    elif current_nr == leaf_i[3]:
                        left_right = 'right'
                    else:
                        raise Exception('Leaf numbers are inconsistent.')
                    new_dic = optp_t.final_leaf_dict(leaf_i, left_right)
                    leaf_splits_pre.append(new_dic.copy())
                    try:
                        parent_nr = leaf_i[1].copy()
                    except AttributeError:
                        parent_nr = leaf_i[1]
                    try:
                        current_nr = leaf_i[0].copy()
                    except AttributeError:
                        current_nr = leaf_i[0]
        leaf_splits_pre = list(reversed(leaf_splits_pre))  # Starting 1st split
        # compute policy score
        for split in leaf_splits_pre:
            polscore_df, x_df = subsample_leaf(polscore_df, x_df, split)
    final_dict_l = optp_t.final_leaf_dict(leaf, 'left')
    final_dict_r = optp_t.final_leaf_dict(leaf, 'right')
    polscore_df_l = subsample_leaf(polscore_df, x_df, final_dict_l)[0]
    polscore_df_r = subsample_leaf(polscore_df, x_df, final_dict_r)[0]
    obs = (polscore_df_l.shape[0], polscore_df_r.shape[0])
    if polscore_is_index:
        # Policy score contains index of observation
        score = (0, 0)
    else:
        polscore_df_l = polscore_df_l.iloc[:, leaf[8][0]]
        polscore_df_r = polscore_df_r.iloc[:, leaf[8][1]]
        score = (polscore_df_l.sum(axis=0), polscore_df_r.sum(axis=0))
    leaf_splits_r = deepcopy(leaf_splits_pre)
    leaf_splits_l = leaf_splits_pre   # one copy is enough
    leaf_splits_l.append(final_dict_l)
    leaf_splits_r.append(final_dict_r)
    leaf_splits = (leaf_splits_l, leaf_splits_r)
    polscore_df_lr = (polscore_df_l, polscore_df_r)
    # leaf 8 contains treatment information in final leaf
    return leaf_splits, score, obs, tuple(leaf[8]), polscore_df_lr


def pred_policy_allocation(tree, x_name, v_dict, c_dict, no_of_treat,
                           train_data=None):
    """Describe optimal policy tree in prediction sample and get predictions.

    Content of tree for each node:
    0: Node identifier (INT: 0-...)
    1: Parent knot
    2: Child node left
    3: Child node right
    4: Type of node (2: Active -> will be further splitted or made terminal
                    1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: String: Name of variable used for decision of next split
    6: x_type of variable (policy categorisation, maybe different from MCF)
    7: If x_type = 'unordered': Set of values that goes to left daughter
    7: If x_type = 0: Cut-off value (larger goes to right daughter)
    8: List of Treatment state for both daughters [left, right]

    Parameters
    ----------
    datafile_name: String.
    tree : List of lists.
    x_type : Dict. Type information of variables.
    v_dict : Dict. Variables.
    c_dict : Dict. Controls.
    no_of_treat: Int. Number of treated.

    Returns
    -------
    None.

    """
    def print_stat(obs, score, diff, txt):
        """Print stats."""
        print(txt)
        print(f'Obs: {obs:8} mean pred. outcome: {score[0]:8.4f} ',
              f'mean obs.  outcome {score[1]:8.4f}',
              f' difference {diff:8.4f}')
          
    def evaluate_score(score_all, treat_act, treat_pred):
        """Find score that fits to treatment and compute means."""
        obs = len(score_all)
        score = np.empty((obs, 2))
        for i, score_all_i in enumerate(score_all):
            score[i, 0] = score_all_i[treat_pred[i]]
            score[i, 1] = score_all_i[treat_act[i]]
        score_mean = np.mean(score, axis=0)
        switch = treat_act != treat_pred
        obs_sw = np.sum(switch)
        if obs_sw > 0:
            score_sw = score[switch, :]
            score_mean_sw = np.mean(score_sw, axis=0)
        else:
            score_mean_sw = 0
        return score_mean, score_mean_sw, obs, obs_sw        
        
    def pred_treat_fct(treat, indx_in_leaf, total_obs):
        """Collect data and bring in the same as original data."""
        pred_treat = np.zeros((total_obs, 2), dtype=np.int64)
        idx_start = 0
        for idx, idx_leaf in enumerate(indx_in_leaf):
            idx_treat = treat[idx]
            idx_end = idx_start + len(idx_leaf[0])
            pred_treat[idx_start:idx_end, 0] = idx_leaf[0].to_numpy().flatten()
            pred_treat[idx_start:idx_end, 1] = idx_treat[0]
            idx_start = idx_end
            idx_end = idx_start + len(idx_leaf[1])
            pred_treat[idx_start:idx_end, 0] = idx_leaf[1].to_numpy().flatten()
            pred_treat[idx_start:idx_end, 1] = idx_treat[1]
            idx_start = idx_end
        pred_treat = pred_treat[pred_treat[:, 0].argsort()]  # sort on indices
        pred_treat = pred_treat[:, 1]
        return pred_treat

    if train_data is None:
        data_df = pd.read_csv(c_dict['preddata'])
    else:
        data_df = pd.read_csv(train_data)
    x_name = [x.upper() for x in x_name]
    data_df.columns = [x.upper() for x in data_df.columns]
    data_df_x = data_df[x_name]   # pylint: disable=E1136
    total_obs = len(data_df_x)
    x_indx = pd.DataFrame(data=range(len(data_df_x)), columns=('Sorter',))
    if tree is None:
        raise Exception('Not enough evaluation points for current depth.' +
                        'Reduce depth or increase variable and / or ' +
                        'evaluation points for continuous variables.')
    length = len(tree)
    ids = [None] * length
    terminal_leafs = []
    for leaf_i in range(length):
        ids[leaf_i] = tree[leaf_i][0]
        if tree[leaf_i][4] == 1:
            terminal_leafs.append(tree[leaf_i])
    assert len(set(ids)) == len(ids), ('Some leafs IDs are identical.' +
                                       'Rerun programme.')
    if c_dict['with_output']:
        print('\n' + ('=' * 80))
        if train_data is None:
            print('Descriptive statistic of estimated policy tree: prediction',
                  ' sample')
        else:
            print('Analysis of changers in training sample')
    splits_seq = [None] * len(terminal_leafs)
    obs = [None] * len(terminal_leafs)
    treat = [None] * len(terminal_leafs)
    indx_in_leaf = [None] * len(terminal_leafs)
    for i, leaf in enumerate(terminal_leafs):
        splits_seq[i], _, obs[i], treat[i], indx_in_leaf[i] = two_leafs_info(
            tree, x_indx, data_df_x, leaf, polscore_is_index=True)
    predicted_treatment = pred_treat_fct(treat, indx_in_leaf, total_obs)
    if train_data is None:
        if c_dict['save_pred_to_file']:
            pd_df = pd.DataFrame(data=predicted_treatment,
                                 columns=('Alloctreat',))
            datanew = pd.concat([pd_df, data_df], axis=1)
            gp.delete_file_if_exists(c_dict['pred_save_file'])
            datanew.to_csv(c_dict['pred_save_file'], index=False)
        if c_dict['with_output']:
            total_obs_by_treat = np.zeros(no_of_treat)
            total_obs_temp = 0
            for i, obs_i in enumerate(obs):
                total_obs_temp += obs_i[0] + obs_i[1]
                total_obs_by_treat[treat[i][0]] += obs_i[0]
                total_obs_by_treat[treat[i][1]] += obs_i[1]
            check_total_obs_temp_no_match(total_obs_temp, total_obs)
            optp_p.print_allocation(None, total_obs, v_dict, c_dict,
                                    total_obs_by_treat, prediction=True)
            optp_p.print_split_info(splits_seq, treat, c_dict)
        if v_dict['d_name'] is not None:
            treat_in_pred = v_dict['d_name'][0] in data_df.columns
            if v_dict['polscore_name'] is not None:
                pols_in_pred = v_dict['polscore_name'][0] in data_df.columns
            else:
                pols_in_pred = False
        else:
            treat_in_pred = pols_in_pred = False
        if treat_in_pred and pols_in_pred and c_dict['with_output']:
            actual_treatment = data_df[v_dict['d_name']].to_numpy(
                dtype=np.int64).flatten()
            polscores = data_df[v_dict['polscore_name']].to_numpy()
            score, score_switch, obs, obs_switch = evaluate_score(
                polscores, actual_treatment, predicted_treatment)
            diff = score[0] - score[1]
            print_stat(obs, score, diff, 'All obs. in prediction sample')
            if obs_switch > 0:
                diff_switch = score_switch[0] - score_switch[1]
                print_stat(obs_switch, score_switch, diff_switch,
                           'Switchers in prediction sample')
                print('-' * 80)
        return predicted_treatment
    actual_treatment = data_df[v_dict['d_name']].to_numpy(dtype=np.int64)
    return predicted_treatment, actual_treatment.flatten()


def structure_of_node_tabl_poltree():
    """Info about content of NODE_TABLE.

    Returns
    -------
    decription : STR. Information on node table with inital node.

    """
    description = """Trees are fully saved in Node_Table (list of lists)
    Structure des Node_table
      - Each knot is one list that contains further lists
    This is the position and information for a given node
    The following items will be filled in the first sample
    0: Node identifier (INT: 0-...)
    1: Parent knot
    2: Child node left
    3: Child node right
    4: Type of node (2: Active -> will be further splitted or made terminal
                    1: Terminal node, no further splits
                    0: previous node that lead already to further splits)
    5: String: Name of variable used for decision of next split
    6: x_type of variable (policy categorisation, maybe different from MCF)
    7: If x_type = 'unordered': Set of values that goes to left daughter
    8: If x_type = 0: Cut-off value (larger goes to right daughter,
                                    equal and smaller to left daughter)

    """
    print("\n", description)


def subsample_leaf(any_df, x_df, split):
    """Reduces dataframes to data in leaf.

    Parameters
    ----------
    any_df : Dataframe. Policyscores, indices or any other of same length as x.
    x_df : Dataframe. Policy variables.
    split : dict. Split information.

    Returns
    -------
    polscore_df_red : Dataframe. Reduced.
    x_df_red : Dataframe. Reduced

    """
    if split['x_type'] == 'unord':
        if split['left or right'] == 'left':  # x in set
            condition = x_df[split['x_name']].isin(split['cut-off or set'])
        else:
            condition = ~x_df[split['x_name']].isin(split['cut-off or set'])
    else:  # x not in set
        if split['left or right'] == 'left':
            condition = x_df[split['x_name']] <= split['cut-off or set']
        else:
            condition = x_df[split['x_name']] > split['cut-off or set']
    any_df_red = any_df[condition]
    x_df_red = x_df[condition]
    return any_df_red, x_df_red


def classify_var_for_pol_tree(datafile_name, v_dict, c_dict, all_var_names):
    """Classify variables as most convenient for policy trees building.

    Parameters
    ----------
    datafile_name : String.
    v_dict : Dict. Variables.
    c_dict : Dict. Parameters.
    all_var_names : List of strings. All variables available.

    Returns
    -------
    x_type_dict : Dict. Key is variable name. Value is type of variable.
    x_value_dict : Dict. Key is variable name. Value is list of values.

    """
    assert all_var_names, 'No variables left.'
    data = pd.read_csv(datafile_name)
    x_continuous = x_ordered = x_unordered = False
    x_type_dict = {}
    x_value_dict = {}
    for var in all_var_names:
        values = np.unique(data[var].to_numpy())  # Sorted values
        if var in v_dict['x_ord_name']:
            if len(values) > c_dict['ft_no_of_evalupoints']:
                x_type_dict.update({var: 'cont'})
                x_value_dict.update({var: None})
                x_continuous = True
            else:
                x_type_dict.update({var: 'disc'})
                x_value_dict.update({var: values.tolist()})
                x_ordered = True
        elif var in v_dict['x_unord_name']:
            values_round = np.round(values)
            a_str = 'Categorical variables must be coded as integers.'
            assert np.sum(np.abs(values-values_round)) <= 1e-10, a_str
            x_type_dict.update({var: 'unord'})
            x_value_dict.update({var: values.tolist()})
            x_unordered = True
        else:
            raise Exception(var + 'is neither ordered nor unordered.')
    c_dict.update({'x_cont_flag': x_continuous,
                   'x_ord_flag': x_ordered,
                   'x_unord_flag': x_unordered})
    return x_type_dict, x_value_dict, c_dict


def check_total_obs_temp_no_match(obs_temp, obs):
    """Check if dimensions fit."""
    if int(obs_temp) != obs:
        print('Total observations in x:                       ', obs)
        print('Total observations with treatment allocated:', obs_temp)
        raise Exception('Some observations did not get an allocation.')
