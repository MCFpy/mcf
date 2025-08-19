"""
Provide functions for Black-Box allocations.

Created on Sun Jul 16 14:03:58 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
import numpy as np
import pandas as pd


def black_box_allocation(optp_, data_df, bb_rest_variable, seed=234356):
    """Compute various Black-Box allocations and return to main programme."""
    rng = np.random.default_rng(seed)
    var_dic, ot_dic = optp_.var_dict, optp_.other_dict
    po_np = data_df[var_dic['polscore_name']].to_numpy()
    no_obs, no_treat = po_np.shape
    largest_gain = largest_gain_fct(po_np)
    allocations_df = pd.DataFrame(data=largest_gain,
                                  columns=('bb',))
    if ot_dic['restricted']:
        max_by_cat = np.int64(
            np.floor(no_obs * np.array(ot_dic['max_shares'])))
        random_rest = random_rest_fct(no_treat, no_obs, max_by_cat, rng)
        allocations_df['bb_rest_rnd'] = random_rest
        largest_gain_rest = largest_gain_rest_fct(
            po_np, no_treat, no_obs, max_by_cat, largest_gain)
        allocations_df['bb_rest_maxgain'] = largest_gain_rest
        largest_gain_rest_random_order = largest_gain_rest_random_order_fct(
            po_np, no_treat, no_obs, max_by_cat, largest_gain, rng)
        allocations_df['bb_rest_maxgain_rndorder'
                       ] = largest_gain_rest_random_order
        if bb_rest_variable:
            largest_gain_rest_other_var = largest_gain_rest_other_var_fct(
                po_np, no_treat, max_by_cat, var_dic, largest_gain, data_df)
            name = 'bb_rest_maxgain_' + '_'.join(
                var_dic['bb_restrict_name'])
            allocations_df[name] = largest_gain_rest_other_var
    return allocations_df


def largest_gain_fct(po_np):
    """Compute allocation with largest gains."""
    return np.argmax(po_np, axis=1)


def random_rest_fct(no_treat, no_obs, max_by_cat, rng, max_attempts=10):
    """Compute random allocation under restrictions."""
    allocations = np.zeros(no_obs, dtype=int)
    so_far_by_cat = np.zeros_like(max_by_cat, dtype=int)

    for i in range(no_obs):
        for _ in range(max_attempts):
            draw = rng.integers(0, high=no_treat, size=1)
            max_by_cat_draw = max_by_cat[draw]  # pylint: disable=E1136
            if so_far_by_cat[draw] <= max_by_cat_draw:
                so_far_by_cat[draw] += 1
                allocations[i] = draw[0]
                break

    return allocations


def largest_gain_rest_fct(po_np, no_treat, no_obs, max_by_cat, largest_gain):
    """Compute allocation based on largest gains, under restrictions."""
    val_best_treat = np.empty(no_obs)
    for i in range(no_obs):
        val_best_treat[i] = (po_np[i, largest_gain[i]] - po_np[i, 0])
    order_best_treat = np.flip(np.argsort(val_best_treat))
    allocations = largest_gain_rest_idx_fct(order_best_treat, largest_gain,
                                            max_by_cat, po_np.copy(), no_treat)

    return allocations


def largest_gain_rest_random_order_fct(po_np, no_treat, no_obs, max_by_cat,
                                       largest_gain, rng):
    """Compute allocation based on first come first served, restricted."""
    order_random = np.arange(no_obs)
    rng.shuffle(order_random)
    allocations = largest_gain_rest_idx_fct(
        order_random, largest_gain, max_by_cat, po_np.copy(), no_treat)

    return allocations


def largest_gain_rest_other_var_fct(po_np, no_treat, max_by_cat,
                                    var_dic, largest_gain, data_df):
    """Compute allocation of largest gain under restriction, order by var."""
    order_other_var = np.flip(
        np.argsort(data_df[var_dic['bb_restrict_name']].to_numpy(), axis=0))
    order_other_var = [x[0] for x in order_other_var]
    allocations = largest_gain_rest_idx_fct(
        order_other_var, largest_gain, max_by_cat, po_np.copy(), no_treat)

    return allocations


# def largest_gain_rest_idx_fct(order_treat, largest_gain_alloc, max_by_cat,
#                               po_np, no_treat):
#     """Get index of largest gain under restr. for each obs with given order."""
#     def helper_largest_gain(best_last, po_np_i, so_far_by_cat, max_by_cat):
#         po_np_i[best_last] = float('-inf')
#         best_candidates = np.where(po_np_i == np.max(po_np_i))[0]
#         for best in best_candidates:
#             if so_far_by_cat[best] < max_by_cat[best]:
#                 so_far_by_cat[best] += 1
#                 success = True
#                 break
#         else:
#             success, best = False, best_last
#         return so_far_by_cat, best, success

#     so_far_by_cat = np.zeros_like(max_by_cat)
#     largest_gain_rest = np.zeros_like(largest_gain_alloc)
#     for i in order_treat:
#         best_1 = largest_gain_alloc[i]
#         if so_far_by_cat[best_1] <= max_by_cat[best_1]:
#             so_far_by_cat[best_1] += 1
#             largest_gain_rest[i] = best_1
#         else:
#             if no_treat > 2:
#                 so_far_by_cat, best_2, success = helper_largest_gain(
#                     best_1, po_np[i], so_far_by_cat, max_by_cat)
#                 if success:
#                     largest_gain_rest[i] = best_2
#                 else:
#                     if no_treat > 3:
#                         so_far_by_cat, best_3, success = helper_largest_gain(
#                              best_2, po_np[i], so_far_by_cat, max_by_cat)
#                         if success:
#                             largest_gain_rest[i] = best_3
#     return largest_gain_rest


def largest_gain_rest_idx_fct(order_treat, largest_gain_alloc, max_by_cat,
                              po_np, no_treat):
    """
    Get index of largest gain under restrictions for each observation (obs).

    The function assigns each obs to its best category (largest gain). If that
    category is at capacity, it attempts to find the next best category by
    temporarily marking the current best as -∞ and taking the new argmax.
    This repeats until either an available category is found or all categories
    are exhausted.

    Parameters
    ----------
    order_treat : array-like of int
        The order in which observations are processed. Each element is an index
        into largest_gain_alloc (and po_np rows).
    largest_gain_alloc : np.ndarray
        1D array, length = number of observations, containing the "best"
        treatment index for each observation as initially estimated.
    max_by_cat : np.ndarray
        1D array specifying the maximum allowed allocations for each treatment.
    po_np : np.ndarray
        2D array of potential outcomes, shape = (number_of_observations,
                                                 no_treat).
    no_treat : int
        Number of treatments (columns in po_np).

    Returns
    -------
    largest_gain_rest : np.ndarray
        1D array (same length as largest_gain_alloc), storing the final
        allocated treatment for each observation respecting capacity
        constraints.
    """
    # Track how many observations have been allocated to each category
    so_far_by_cat = np.zeros_like(max_by_cat, dtype=int)
    # Final allocations (treatments) after restrictions
    largest_gain_rest = np.zeros_like(largest_gain_alloc, dtype=int)

    for obs_idx in order_treat:
        # Start with the initially "best" category
        best_cat = largest_gain_alloc[obs_idx]

        # If the best category still has capacity, assign and move on
        if so_far_by_cat[best_cat] < max_by_cat[best_cat]:
            so_far_by_cat[best_cat] += 1
            largest_gain_rest[obs_idx] = best_cat
        else:
            # Otherwise, repeatedly look for the next-best category
            row = po_np[obs_idx].copy()  # Copy to avoid modifying original arr.
            # "Remove" the already-full best_cat from consideration
            row[best_cat] = float('-inf')

            assigned_cat = None
            # Try up to (no_treat - 1) times to find a fallback category
            for _ in range(no_treat - 1):
                next_best_cat = np.argmax(row)
                if so_far_by_cat[next_best_cat] < max_by_cat[next_best_cat]:
                    so_far_by_cat[next_best_cat] += 1
                    largest_gain_rest[obs_idx] = next_best_cat
                    assigned_cat = next_best_cat
                    break
                # Mark this category as unavailable and continue
                row[next_best_cat] = float('-inf')

            if assigned_cat is None:
                print('All treatments at capacity.')

    return largest_gain_rest
