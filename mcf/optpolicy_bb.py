"""
Provide functions for Black-Box allocations.

Created on Sun Jul 16 14:03:58 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.optpolicy_init import VarCfg


def black_box_allocation(optp_: 'OptimalPolicy',
                         data_df: pd.DataFrame,
                         bb_rest_variable: bool,
                         seed: int = 234356
                         ) -> pd.DataFrame:
    """Compute various Black-Box allocations and return to main programme."""
    rng = np.random.default_rng(seed=seed)
    var_cfg, other_cfg = optp_.var_cfg, optp_.other_cfg
    po_np = data_df[var_cfg.polscore_name].to_numpy(copy=True)
    no_obs, no_treat = po_np.shape
    largest_gain = largest_gain_fct(po_np)
    allocations_df = pd.DataFrame(data=largest_gain, columns=('bb',))
    if other_cfg.restricted:
        max_by_cat = np.int64(np.floor(no_obs * np.array(other_cfg.max_shares)))
        random_rest = random_rest_fct(no_treat, no_obs, max_by_cat, rng)
        allocations_df['bb_rest_rnd'] = random_rest
        largest_gain_rest = largest_gain_rest_fct(po_np, no_treat, no_obs, max_by_cat, largest_gain)
        allocations_df['bb_rest_maxgain'] = largest_gain_rest
        largest_gain_rest_random_order = largest_gain_rest_random_order_fct(po_np,
                                                                            no_treat=no_treat,
                                                                            no_obs=no_obs,
                                                                            max_by_cat=max_by_cat,
                                                                            largest_gain=
                                                                            largest_gain, rng=rng,
                                                                            )
        allocations_df['bb_rest_maxgain_rndorder'] = largest_gain_rest_random_order
        if bb_rest_variable:
            largest_gain_rest_other_var = largest_gain_rest_other_var_fct(po_np,
                                                                          no_treat=no_treat,
                                                                          max_by_cat=max_by_cat,
                                                                          var_cfg=var_cfg,
                                                                          largest_gain=largest_gain,
                                                                          data_df=data_df,
                                                                          )
            name = 'bb_rest_maxgain_' + '_'.join(var_cfg.bb_restrict_name)
            allocations_df[name] = largest_gain_rest_other_var

    return allocations_df


def largest_gain_fct(po_np: NDArray[Any]) -> NDArray[Any]:
    """Compute allocation with largest gains."""
    return np.argmax(po_np, axis=1)


def random_rest_fct(no_treat: int,
                    no_obs: int,
                    max_by_cat: NDArray[np.integer],
                    rng: np.random.Generator,
                    max_attempts: int = 10,
                    ) -> NDArray[np.integer]:
    """Compute random allocation under restrictions."""
    allocations = np.empty(no_obs, dtype=int)
    so_far_by_cat = np.zeros_like(max_by_cat, dtype=int)

    for i in range(no_obs):
        feasible = np.flatnonzero(so_far_by_cat < max_by_cat)
        if feasible.size == 0:
            raise ValueError('Restricted random allocation infeasible: all capacities exhausted.')

        for _ in range(max_attempts):
            draw = int(rng.integers(0, high=no_treat))

            if so_far_by_cat[draw] < max_by_cat[draw]:
                so_far_by_cat[draw] += 1
                allocations[i] = draw
                break
        else:
            # Avoid silently leaving an invalid default allocation after repeated full draws.
            draw = int(rng.choice(feasible))
            so_far_by_cat[draw] += 1
            allocations[i] = draw

    return allocations


def largest_gain_rest_fct(po_np: NDArray[Any],
                          no_treat: int,
                          no_obs: int,
                          max_by_cat: list[float],
                          largest_gain: NDArray[Any],
                          ) -> NDArray[Any]:
    """Compute allocation based on largest gains, under restrictions."""
    val_best_treat = np.empty(no_obs)
    for i in range(no_obs):
        val_best_treat[i] = (po_np[i, largest_gain[i]] - po_np[i, 0])
    order_best_treat = np.flip(np.argsort(val_best_treat))
    allocations = largest_gain_rest_idx_fct(order_best_treat, largest_gain,
                                            max_by_cat, po_np.copy(), no_treat
                                            )
    return allocations


def largest_gain_rest_random_order_fct(po_np: NDArray[Any], *,
                                       no_treat: int,
                                       no_obs: int,
                                       max_by_cat: list[float],
                                       largest_gain: NDArray[Any],
                                       rng: np.random.Generator,
                                       ) -> NDArray[Any]:
    """Compute allocation based on first come first served, restricted."""
    order_random = np.arange(no_obs)
    rng.shuffle(order_random)
    allocations = largest_gain_rest_idx_fct(order_random, largest_gain, max_by_cat, po_np.copy(),
                                            no_treat
                                            )
    return allocations


def largest_gain_rest_other_var_fct(po_np: NDArray[Any], *,
                                    no_treat: int,
                                    max_by_cat: list[float],
                                    var_cfg: 'VarCfg',
                                    largest_gain: NDArray[Any],
                                    data_df) -> NDArray[Any]:
    """Compute allocation of largest gain under restriction, order by var."""
    order_other_var = np.flip(np.argsort(data_df[var_cfg.bb_restrict_name].to_numpy(), axis=0))
    order_other_var = [x[0] for x in order_other_var]
    allocations = largest_gain_rest_idx_fct(order_other_var, largest_gain, max_by_cat, po_np.copy(),
                                            no_treat
                                            )
    return allocations


def largest_gain_rest_idx_fct(order_treat: NDArray[Any],
                              largest_gain_alloc: NDArray[Any],
                              max_by_cat: NDArray[Any],
                              po_np: NDArray[Any],
                              no_treat: int
                              ) -> NDArray[Any]:
    """
    Get index of largest gain under restrictions for each observation (obs).

    The function assigns each obs to its best category (largest gain). If that
    category is at capacity, it attempts to find the next best category by
    temporarily marking the current best as -∞ and taking the new argmax.
    This repeats until either an available category is found or all categories are exhausted.

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
        2D array of potential outcomes, shape = (number_of_observations, no_treat).
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
