"""
Created on Mon Oct 30 13:55:51 2023.

Contains the functions for saving trees and travelling inside them.

@author: Michael Lechner
# -*- coding: utf-8 -*-
"""
import numpy as np


def make_default_tree_dict(n_leaf_min: int,
                           number_of_features: int,
                           indices_train: list[int],
                           indices_oob: list[int],
                           bigdata_train: bool = False
                           ) -> dict:
    """
    Define a dict. containing the full information of the causal mcf tree.

    Parameters
    ----------
    n_leaf_min : Int.
        Minimum number of observations per lead.
    ...

    Returns
    -------
    Causal_tree : Dict.
        All information about the causal tree.
    """
    leaves_max = int(np.ceil(len(indices_train) / n_leaf_min) * 2)
    max_cols_int, max_cols_float = 10, 3
    if (leaves_max < 127 and number_of_features < 127
            and len(indices_train) < 127 and len(indices_oob) < 127):
        leaf_info_int = -np.ones((leaves_max, max_cols_int), dtype=np.int8)
        leaf_ids = np.arange(leaves_max, dtype=np.int8)
    elif (leaves_max < 32767 and number_of_features < 32767
          and len(indices_train) < 32767 and len(indices_oob) < 32767):
        leaf_info_int = -np.ones((leaves_max, max_cols_int), dtype=np.int16)
        leaf_ids = np.arange(leaves_max, dtype=np.int16)
    elif (leaves_max < 2147483647 and number_of_features < 2147483647
          and len(indices_train) < 2147483647
          and len(indices_oob) < 2147483647):
        leaf_info_int = -np.ones((leaves_max, max_cols_int), dtype=np.int32)
        leaf_ids = np.arange(leaves_max, dtype=np.int32)
    else:
        leaf_info_int = -np.ones((leaves_max, max_cols_int), dtype=np.int64)
        leaf_ids = np.arange(leaves_max, dtype=np.int64)
    leaf_info_int[:, 0] = leaf_ids
    leaf_info_int[0, 1] = -1
    leaf_info_int[0, 6] = 0
    leaf_info_int[0, 7] = 2
    leaf_info_int[0, 8] = len(indices_train)
    leaf_info_int[0, 9] = len(indices_oob)
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
    if bigdata_train:
        leaf_info_float = -np.ones((leaves_max, max_cols_float),
                                   dtype=np.float32)
    else:
        leaf_info_float = -np.ones((leaves_max, max_cols_float),
                                   dtype=np.float64)
    leaf_info_float[:, 0] = leaf_ids
    # leaf_info_float 0: ID of leaf
    #                 1: Cut-of value of ordered variables
    #                 2: OOB value of leaf

    # Primes need to stored separately as np.int64 may not be sufficient to
    # store extra large values
    cats_prime = [0] * leaves_max   # List, otherwise could not hold long int
    indices_train_all = [None for _ in range(leaves_max)]
    indices_oob_all = [None for _ in range(leaves_max)]
    indices_train_all[0] = indices_train
    indices_oob_all[0] = indices_oob
    causal_tree_empty_dic = {
        'leaf_info_int': leaf_info_int,
        # 2dnarray with leaf info, integers
        'leaf_info_float': leaf_info_float,
        # 2dnarray, with leaf info for floats
        'cats_prime': cats_prime,
        # 2dnarray, prime value for categorical features
        'oob_indices': indices_oob,
        # 1D ndarray, indices of tree-specific OOB data
        'train_data_list': indices_train_all,
        # Indices of data needed during tree building, will be removed after use
        'oob_data_list': indices_oob_all,
        # Indices of oob data needed during tree building, will be removed after
        # use
        'fill_y_indices_list': None,
        # list (dim: # of leaves) of leaf-specific indices (to be filled after
        # forest building) - for terminal leaves only
        'fill_y_empty_leave': None,
        # list (dim: # of leaves) of leaf-specific indices (to be filled after
        # forest building) - for terminal leaves only
        }
    return causal_tree_empty_dic


def cut_back_empty_cells_tree(tree: dict) -> dict:
    """Cut back matrices and vector to max element."""
    no_leaves = 0
    for active in tree['leaf_info_int'][:, 7]:
        if active == -1:
            break
        no_leaves += 1
    tree['leaf_info_int'] = tree['leaf_info_int'][:no_leaves, :]
    tree['leaf_info_float'] = tree['leaf_info_float'][:no_leaves, :]
    tree['cats_prime'] = tree['cats_prime'][:no_leaves]
    tree['train_data_list'] = tree['train_data_list'][:no_leaves]
    tree['oob_data_list'] = tree['oob_data_list'][:no_leaves]
    tree['fill_y_indices_list'] = [None for _ in range(no_leaves)]

    return tree


def delete_training_data_forest(forest: list) -> list:
    """Delete training data from forest."""
    for tree in forest:
        tree['train_data_list'] = None
    return forest


def delete_data_from_forest(forest: list) -> list:
    """Delete oob data from forest."""
    for tree in forest:
        tree['oob_indices'] = None
        tree['oob_data_list'] = None
        tree['train_data_list'] = None
    return forest
