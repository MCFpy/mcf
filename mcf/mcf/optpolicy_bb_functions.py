"""
Provide functions for Black-Box allocations.

Created on Sun Jul 16 14:03:58 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from math import inf

import numpy as np
import pandas as pd


def black_box_allocation(optp_, data_df, bb_rest_variable):
    """Compute various Black-Box allocations and return to main programme."""
    rng = np.random.default_rng(234356)
    var_dic, ot_dic = optp_.var_dict, optp_.other_dict
    po_np = data_df[var_dic['polscore_name']].to_numpy()
    no_obs, no_treat = po_np.shape
    largest_gain = largest_gain_fct(po_np)
    allocations_df = pd.DataFrame(data=largest_gain,
                                  columns=('bb_unrestricted',))
    if ot_dic['restricted']:
        max_by_cat = np.int64(
            np.floor(no_obs * np.array(ot_dic['max_shares'])))
        random_rest = random_rest_fct(no_treat, no_obs, max_by_cat, rng)
        allocations_df['random_rest'] = random_rest
        largest_gain_rest = largest_gain_rest_fct(
            po_np, no_treat, no_obs, max_by_cat, largest_gain)
        allocations_df['largest_gain_rest'] = largest_gain_rest
        largest_gain_rest_random_order = largest_gain_rest_random_order_fct(
            po_np, no_treat, no_obs, max_by_cat, largest_gain, rng)
        allocations_df['largest_gain_rest_random_order'
                       ] = largest_gain_rest_random_order
        if bb_rest_variable:
            largest_gain_rest_other_var = largest_gain_rest_other_var_fct(
                po_np, no_treat, max_by_cat, var_dic, largest_gain, data_df)
            name = 'largest_gain_rest_' + '_'.join(var_dic['bb_restrict_name'])
            allocations_df[name] = largest_gain_rest_other_var
    return allocations_df


def largest_gain_fct(po_np):
    """Compute allocation with largest gains."""
    return np.argmax(po_np, axis=1)


def random_rest_fct(no_treat, no_obs, max_by_cat, rng):
    """Compute random allocation under restrictions."""
    alloc = np.zeros(no_obs)
    so_far_by_cat = np.zeros_like(max_by_cat)
    for idx in range(no_obs):
        for _ in range(10):
            draw = rng.integers(0, high=no_treat, size=1)
            max_by_cat_draw = max_by_cat[draw]  # pylint: disable=E1136
            if so_far_by_cat[draw] <= max_by_cat_draw:
                so_far_by_cat[draw] += 1
                alloc[idx] = draw
                break
    return alloc


def largest_gain_rest_fct(po_np, no_treat, no_obs, max_by_cat, largest_gain):
    """Compute allocation based on largest gains, under restrictions."""
    alloc, val_best_treat = np.zeros(no_obs), np.empty(no_obs)
    for i in range(no_obs):
        val_best_treat[i] = (po_np[i, largest_gain[i]] - po_np[i, 0])
    order_best_treat = np.flip(np.argsort(val_best_treat))
    alloc = largest_gain_rest_idx_fct(order_best_treat, largest_gain,
                                      max_by_cat, po_np.copy(), no_treat)
    return alloc


def largest_gain_rest_random_order_fct(po_np, no_treat, no_obs, max_by_cat,
                                       largest_gain, rng):
    """Compute allocation based on first come first served, restricted."""
    order_random = np.arange(no_obs)
    rng.shuffle(order_random)
    alloc = largest_gain_rest_idx_fct(order_random, largest_gain, max_by_cat,
                                      po_np.copy(), no_treat)
    return alloc


def largest_gain_rest_other_var_fct(po_np, no_treat, max_by_cat,
                                    var_dic, largest_gain, data_df):
    """Compute allocation of largest gain under restriction, order by var."""
    order_other_var = np.flip(
        np.argsort(data_df[var_dic['bb_restrict_name']].to_numpy(), axis=0))
    order_other_var = [x[0] for x in order_other_var]
    alloc = largest_gain_rest_idx_fct(
        order_other_var, largest_gain, max_by_cat, po_np.copy(), no_treat)
    return alloc


def largest_gain_rest_idx_fct(order_treat, largest_gain_alloc, max_by_cat,
                              po_np, no_treat):
    """Get index of largest gain under restr. for each obs with given order."""
    def helper_largest_gain(best_last, po_np_i, so_far_by_cat, max_by_cat):
        po_np_i[best_last] = -inf
        best = np.argmax(po_np_i)
        if so_far_by_cat[best] <= max_by_cat[best]:
            so_far_by_cat[best] += 1
            success = True
        else:
            success = False
        # otherwise it remains at the zero default
        return so_far_by_cat, best, success

    so_far_by_cat = np.zeros_like(max_by_cat)
    largest_gain_rest = np.zeros_like(largest_gain_alloc)
    for i in order_treat:
        best_1 = largest_gain_alloc[i]
        if so_far_by_cat[best_1] <= max_by_cat[best_1]:
            so_far_by_cat[best_1] += 1
            largest_gain_rest[i] = best_1
        else:
            if no_treat > 2:
                so_far_by_cat, best_2, success = helper_largest_gain(
                    best_1, po_np[i], so_far_by_cat, max_by_cat)
                if success:
                    largest_gain_rest[i] = best_2
                else:
                    if no_treat > 3:
                        so_far_by_cat, best_3, success = helper_largest_gain(
                             best_2, po_np[i], so_far_by_cat, max_by_cat)
                        if success:
                            largest_gain_rest[i] = best_3
    return largest_gain_rest
