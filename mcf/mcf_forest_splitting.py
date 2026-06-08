"""
Created on Thu Mar 26 06:49:30 2026.

@author: MLechner

# -*- coding: utf-8 -*-

Contain functions for tree splitting.
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mcf import mcf_forest_add as mcf_fo_add
from mcf import mcf_forest_objfct as mcf_fo_obj
from mcf.mcf_general import list_product

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg
    from mcf.mcf_init_train import CfCfg


def get_split_values(y_dat: NDArray[np.float64 | np.intp],
                     w_dat: NDArray[np.float64 | np.intp] | None,
                     x_dat: NDArray[np.float64 | np.intp], *,
                     x_type: np.int8 | int,
                     x_values: list[float | int | np.floating | np.integer],
                     leaf_size: int,
                     random_thresholds: int,
                     w_yes: bool,
                     rng: np.random.Generator | None = None,
                     ) -> list:
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
    rng : Random Number Generator object

    Returns
    -------
    splits : List. Splitting values to use.
    """
    if rng is None:
        rng = np.random.default_rng()
        print("Warning: unseeded random numbers used - impossible to replicate")

    if x_type == 0:   # Continuous variable
        if bool(x_values):  # Continuous var with limited number of values
            min_x, max_x = np.amin(x_dat), np.amax(x_dat)
            xv = np.asarray(x_values)
            mask = (xv >= min_x) & (xv <= max_x)
            splits_x = xv[mask].tolist()

            if len(splits_x) == 1:
                return splits_x

            if len(splits_x) == 0:  # Super unlikely to happen
                return x_values[:]

            # More than 1 splitting value
            splits_x = splits_x[:-1]
            if 0 < random_thresholds < len(splits_x):
                splits = np.unique(rng.choice(splits_x,
                                              size=random_thresholds, replace=False, shuffle=False,
                                              )
                                   )
            else:
                splits = splits_x

            return splits

        # Continoues variable with very many values; x_values empty
        if 0 < random_thresholds < (leaf_size - 1):
            x_vals_np = rng.choice(x_dat, size=random_thresholds, replace=False, shuffle=False)
            splits = np.unique(x_vals_np).tolist()
        else:
            splits = np.unique(x_dat).tolist()
            if len(splits) > 1:
                splits = splits[:-1]

        return splits

    # Discrete variable
    y_mean_by_cat = np.empty(len(x_values))  # x_vals comes as list
    x_vals_np = np.array(x_values, dtype=np.int32, copy=True)
    used_values = []
    for v_idx, val in enumerate(x_vals_np):
        value_equal = np.isclose(x_dat, val)
        if np.any(value_equal):  # Position of empty cells do not matter
            if w_yes:
                y_mean_by_cat[v_idx] = np.average(y_dat[value_equal].reshape(-1),
                                                  weights=w_dat[value_equal].reshape(-1)
                                                  )
            else:
                y_mean_by_cat[v_idx] = np.average(y_dat[value_equal])
            used_values.append(v_idx)
    x_vals_np = x_vals_np[used_values]
    sort_ind = np.argsort(y_mean_by_cat[used_values])
    x_vals_np = x_vals_np[sort_ind]
    splits = x_vals_np.tolist()
    splits = splits[:-1]  # Last category not needed

    return splits


def term_or_data(data_tr_ns: NDArray[np.float64],
                 data_oob_ns: NDArray[np.float64], *,
                 y_i: int,
                 d_i: int,
                 d_grid_i: int | list[int] | None,
                 x_i_ind_split: NDArray[np.intp],
                 no_of_treat: int,
                 with_d_oob: bool = True,
                 cont_treat: bool = False,
                 zero_tol: float = 1e-10,
                 ) -> tuple[NDArray[np.float64],         # y_dat
                            NDArray[np.float64] | None,  # y_oob
                            NDArray[np.float64] | None,  # d_dat
                            NDArray[np.float64] | None,  # d_oob
                            NDArray[Any] | None,         # d_grid_dat
                            NDArray[Any] | None,         # d_grid_oob
                            NDArray[np.float64] | None,  # x_dat
                            NDArray[np.float64] | None,  # x_oob
                            bool,                        # terminal
                            bool,                        # terminal_x
                            list[bool]                   # x_no_variation
                            ]:
    """Check if terminal leaf. If not, provide data.

    Parameters
    ----------
    data_tr_ns : Numpy array. Data used for splitting.
    data_oob_ns : Numpy array. OOB Data.
    y_i : INT. Index of y in data.
    d_i : INT. Index of d in data.
    d_grid_i : List of INT. Indices of d_grid in data.
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
    terminal_x : Boolean. No variation in X. Try new variables.
    ...
    """
    y_dat = data_tr_ns[:, y_i]
    if np.allclose(y_dat, y_dat[0]):    # all elements are equal
        return y_dat, None, None, None, None, None, None, None, True, False, []
    d_grid_dat = None
    y_oob = data_oob_ns[:, y_i]
    d_dat = data_tr_ns[:, d_i]
    if d_grid_i is not None:
        d_grid_dat = data_tr_ns[:, d_grid_i]

    if cont_treat:
        # treat as binary indicator > 0 with a tiny tolerance
        d_bin = d_dat > zero_tol
        terminal = (np.all(d_bin) or np.all(~d_bin))
    else:
        # discrete labels 0..no_of_treat-1
        d_int = d_dat.astype(np.int64, copy=False)
        counts = np.bincount(d_int.ravel(), minlength=no_of_treat)
        terminal = np.any(counts[:no_of_treat] == 0)

    if terminal:
        return y_dat, y_oob, d_dat, None, d_grid_dat, None, None, None, True, False, []

    d_oob = d_grid_oob = x_oob = None
    x_no_variation = []
    if with_d_oob:
        d_oob = data_oob_ns[:, d_i]
        if d_grid_i is not None:
            d_grid_oob = data_oob_ns[:, d_grid_i]

    x_dat = data_tr_ns[:, x_i_ind_split]

    x_first = x_dat[0, :]
    if np.issubdtype(x_dat.dtype, np.integer):
        x_no_variation_arr = np.all(x_dat == x_first, axis=0)
    else:
        x_no_variation_arr = np.all(np.isclose(x_dat, x_first, rtol=1e-05, atol=1e-08), axis=0)
    x_no_variation = x_no_variation_arr.tolist()

    terminal_x = bool(np.all(x_no_variation_arr))
    if not terminal_x:
        x_oob = data_oob_ns[:, x_i_ind_split]

    return (y_dat, y_oob, d_dat, d_oob, d_grid_dat, d_grid_oob, x_dat, x_oob, False, terminal_x,
            x_no_variation,
            )


def next_split(data_train: NDArray[np.floating],
               data_oob: NDArray[np.floating], *,
               y_i: int,
               y_nn_i: NDArray[np.intp],
               d_i: int,
               d_grid_i: int | None,
               x_i: NDArray[np.intp],
               w_i: int | None,
               x_type: NDArray[np.intp],
               x_values: list[list[float]],
               x_ind: NDArray[np.intp],
               x_ai_ind: NDArray[np.intp] | list[int],
               cf_cfg: 'CfCfg', gen_cfg: 'GenCfg',
               ct_grid_nn_val: dict | None,
               mmm: int,
               n_min: int,
               alpha_reg: float,
               pen_mult: np.floating,
               rng: np.random.Generator,
               cuda: bool,
               cython: bool,
               zero_tol: float = 1e-10,
               ) -> tuple[bool,
                          int | None,
                          int | None,
                          int | None,
                          int | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          NDArray[bool] | None,
                          int | None,
                          int | None,
                          float | list[int] | None,
                          ]:
    """Find best next split of leaf (or terminate splitting for this leaf).

    Parameters
    ----------
    data_train : Numpy array. Training data of leaf.
    data_oob : Numpy array: OOB data of leaf.
    y_i : int. Location of Y in data matrix.
    y_nn_i :  Numpy array. Location of Y_NN in data matrix.
    d_i : INT. Location of D in data matrix.
    d_grid_i : INT. Location of D_grid in data matrix.
    x_i : Numpy array. Location of X in data matrix.
    x_type : Numpy array.(0,1,2). Type of X.
    x_ind : Numpy array. Location of X in X matrix.
    x_ai_ind : Numpy array. Location of X_always in X matrix.
    ... : DICT. Parameters.
    mmm : INT. Number of X-variables to choose for splitting.
    n_min : Int. Minimum leaf size.
    alpha_reg : Float. Alpha regularity.
    pen_mult : Float. Penalty multiplier.
    rng : Numpy default random number generator object.
    cuda : Boolean. Use cuda if True.
    cython : Boolean. Use cython for faster usage.

    Returns
    -------
    left : List of lists. Information about left leaf.
    right : List of lists. Information about right leaf.
    current : List of lists. Updated information about this leaf.
    terminal : INT. 1: No splits for this leaf. 0: Leaf splitted
    ...
    """
    # cache config & funcs once
    compare_only_to_zero = cf_cfg.compare_only_to_zero
    n_min_treat = cf_cfg.n_min_treat
    weighted = gen_cfg.weighted
    penalty_type = cf_cfg.penalty_type
    mtot = cf_cfg.mtot
    random_thresholds = cf_cfg.random_thresholds

    count_nonzero = np.count_nonzero
    mcf_mse_fn = mcf_fo_obj.mcf_mse
    add_mse_mce_split_fn = mcf_fo_obj.add_mse_mce_split
    compute_mse_mce_fn = mcf_fo_obj.compute_mse_mce
    mcf_penalty_fn = mcf_fo_obj.mcf_penalty
    not_enough_treated_fn = mcf_fo_add.not_enough_treated
    match_cont_fn = mcf_fo_add.match_cont

    terminal = split_done = False
    leaf_size_train = data_train.shape[0]
    leaf_size_oob = data_oob.shape[0]

    pen_mult_d = pen_mult if (penalty_type == 'mse_d' and pen_mult > 0) else 0
    if leaf_size_train < (2 * n_min):
        terminal = True
    elif np.all(np.isclose(data_train[:, d_i], data_train[0, d_i], atol=zero_tol, rtol=zero_tol)):
        terminal = True
    else:
        if leaf_size_train < 200:  # Otherwise, too slow:
            if gen_cfg.d_type == 'continuous':
                terminal = not 2 <= np.sum(np.isclose(data_train[:, d_i], 0,
                                                      atol=zero_tol, rtol=zero_tol
                                                      )
                                           ) <= leaf_size_train - 2
            else:
                ret = np.unique(data_train[:, d_i], return_counts=True)
                terminal = len(ret[0]) < gen_cfg.no_of_treat or np.any(ret[1] < 2 * n_min_treat)
    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
        d_split_in_x_ind = np.max(x_ind) + 1
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        d_bin_dat, continuous, d_split_in_x_ind = None, False, None
    if not terminal:
        obs_min = max(round(leaf_size_train * alpha_reg), n_min)
        best_mse = np.inf
        total_attempts = 2   # New in version 0.9.0; it was 5 before
        all_vars = False
        x_already_used: set[int] = set()   # For faster look-up below
        # Find valid splitting values, try again if failed
        for idx in range(total_attempts):
            if idx == 0:
                x_ind_idx = x_ind.copy()
            if idx == total_attempts - 1:  # last iteration
                all_vars = True
            x_ind_split = mcf_fo_add.rnd_variable_for_split(x_ind_idx, x_ai_ind, cf_cfg, mmm, rng,
                                                            all_vars=all_vars,
                                                            first_attempt=idx == 0,
                                                            )
            x_type_split = x_type[x_ind_split].copy()
            x_values_split = [x_values[v_idx].copy() for v_idx in x_ind_split]
            # Check if split is possible ... sequential order to minimize costs
            # Check if enough variation in the data to do splitting (costly)
            with_d_oob = continuous
            (y_dat, _, d_dat, d_oob, d_grid_dat, _, x_dat, x_oob, terminal, terminal_x, x_no_varia
             ) = term_or_data(data_train, data_oob,
                              y_i=y_i, d_i=d_i, d_grid_i=d_grid_i, x_i_ind_split=x_i[x_ind_split],
                              no_of_treat=no_of_treat, with_d_oob=with_d_oob, cont_treat=continuous,
                              zero_tol=zero_tol,
                              )
            if terminal:  # No variation in d or/and y, game over.
                break
            if not terminal_x:  # Variation in x, find splits
                break

            x_already_used.update(x_ind_split)

            if len(x_already_used) >= len(x_ind):
                break

            x_ind_idx = [ind for ind in x_ind_idx if ind not in x_already_used]

        if terminal_x:
            terminal = True  # No variation in X. Splitting stops.

    best_var_i = best_type = best_n_l = best_n_r = best_leaf_l = None
    best_leaf_r = best_leaf_oob_l = best_leaf_oob_r = best_n_oob_l = None
    best_n_oob_r = best_value = None
    if not terminal:
        # Added Oct., 22, 2025
        best = {'j': None, 'type': None, 'value': None, 'mse': np.inf, 'd_cont_split': False,
                'n_l': 0, 'n_r': 0, 'n_oob_l': 0, 'n_oob_r': 0,
                # for continuous-D only:
                'zeros_l': None, 'zeros_l_oob': None
                }
        if mtot in (1, 4):
            y_nn = data_train[:, y_nn_i]
        else:
            y_nn = y_nn_l = y_nn_r = 0

        if weighted:
            if w_i is None:
                raise ValueError('weighted=True but w_i is None')
            w_dat = data_train[:, [w_i]]
        else:
            w_dat = [1]

        if continuous:
            d_bin_dat = d_dat > zero_tol   # Binary-boolean treatment indicator
            x_no_varia.append(np.all(d_bin_dat == d_bin_dat[0]))
            x_ind_split.append(d_split_in_x_ind)
            x_type_split = np.append(x_type_split, np.int8(0))
        p_x = len(x_ind_split)  # indices refer to order of x in data_*
        d_cont_split = False
        # x_dat_copy, x_oob_copy = np.copy(x_dat), np.copy(x_oob) NOT NEEDED
        for j in range(p_x):  # Loops over the variables
            if not x_no_varia[j]:  # No variation of this x -> no split
                d_cont_split = continuous and (j == p_x - 1)
                if d_cont_split:
                    x_j, x_oob_j = d_dat, d_oob
                    x_j_pos = x_j[x_j > zero_tol]  # Positive treatment values
                    nr_pos, nr_all = x_j_pos.size,  x_j.size
                    nr_0 = nr_all - nr_pos
                    nr_all_oob = x_oob_j.shape[0]
                    if nr_0 < 2 or nr_pos < 2:  # Too few controls
                        continue
                    split_values = np.unique(x_j_pos).tolist()
                    if len(split_values) > 1:
                        split_values = split_values[:-1]  # max not included

                    # Randomly allocate half the controls to left leaf
                    # rnd_in = rng.choice([True, False], size=(nr_all, 1))
                    # # Somewhat inefficient as it is also applied to treated
                    # treat_0 = (x_j - zero_tol) <= 0
                    # zeros_l = treat_0 & rnd_in
                    # rnd_in_oob = rng.choice([True, False], size=(nr_all_oob,
                    #                                              1))
                    # # Somewhat inefficient as it is also applied to treated
                    # treat_0_oob = (x_oob_j - zero_tol) <= 0
                    # zeros_l_oob = treat_0_oob & rnd_in_oob
                    # 1-D masks; no .flatten() needed later
                    treat_0 = x_j <= zero_tol
                    rnd_in = rng.random(nr_all) < 0.5
                    zeros_l = treat_0 & rnd_in

                    treat_0_oob = x_oob_j <= zero_tol
                    rnd_in_oob = rng.random(nr_all_oob) < 0.5
                    zeros_l_oob = treat_0_oob & rnd_in_oob

                else:
                    x_j, x_oob_j = x_dat[:, j], x_oob[:, j]
                    if x_type_split[j] > 0:
                        x_j = x_j.astype(np.int32, copy=False)
                        x_oob_j = x_oob_j.astype(np.int32, copy=False)
                    split_values = get_split_values(y_dat, w_dat, x_j,
                                                    x_type=x_type_split[j],
                                                    x_values=x_values_split[j],
                                                    leaf_size=leaf_size_train,
                                                    random_thresholds= random_thresholds,
                                                    w_yes=weighted, rng=rng
                                                    )
                split_values_unord_j = []
                cat_mask, cat_mask_oob = None, None
                # Loops over values of variables
                for val in split_values:
                    if x_type_split[j] == 0:
                        val_eps = np.nextafter(val, np.inf)
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l = ~treat_0 & (x_j <= val_eps)
                            leaf_l = treated_l | zeros_l
                        else:
                            leaf_l = x_j <= val_eps
                            # because of float
                    else:                      # ordered with few vals.
                        # Categorial variable: Either in group or not
                        split_values_unord_j.append(val)
                        if cat_mask is None:
                            cat_mask = x_j == val
                        else:
                            cat_mask |= (x_j == val)

                        leaf_l = cat_mask

                    n_l = count_nonzero(leaf_l)
                    n_r = leaf_size_train - n_l
                    # Check if enough observations available
                    if (n_l < obs_min) or (n_r < obs_min):
                        continue
                    # Next we check if any obs in each treatment
                    d_dat_l = (d_bin_dat[leaf_l] if continuous else d_dat[leaf_l])
                    if not_enough_treated_fn(continuous, n_min_treat, d_dat_l, no_of_treat):
                        continue
                    leaf_r = ~leaf_l  # Reverses True to False
                    d_dat_r = d_bin_dat[leaf_r] if continuous else d_dat[leaf_r]
                    if not_enough_treated_fn(continuous, n_min_treat, d_dat_r, no_of_treat):
                        continue

                    if x_type_split[j] == 0:
                        if d_cont_split:   # Treated and selected non-treated
                            treated_l_oob = ~treat_0_oob & (x_oob_j <= val_eps)
                            # leaf_oob_l = treated_l_oob | zeros_l_oob).flatten()
                            leaf_oob_l = treated_l_oob | zeros_l_oob
                        else:
                            leaf_oob_l = x_oob_j <= val_eps
                    else:
                        if cat_mask_oob is None:
                            cat_mask_oob = x_oob_j == val
                        else:
                            cat_mask_oob |= (x_oob_j == val)

                        leaf_oob_l = cat_mask_oob

                    n_oob_l = count_nonzero(leaf_oob_l)
                    n_oob_r = leaf_size_oob - n_oob_l

                    # leaf_oob_r = ~leaf_oob_l
                    if mtot in (1, 4):
                        if continuous:
                            y_nn_l = match_cont_fn(d_grid_dat[leaf_l], y_nn[leaf_l, :],
                                                   ct_grid_nn_val, rng
                                                   )
                            y_nn_r = match_cont_fn(d_grid_dat[leaf_r], y_nn[leaf_r, :],
                                                   ct_grid_nn_val, rng
                                                   )
                        else:
                            y_nn_l, y_nn_r = y_nn[leaf_l, :], y_nn[leaf_r, :]
                    else:
                        y_nn_l = y_nn_r = 0
                    if weighted:
                        w_l, w_r = w_dat[leaf_l], w_dat[leaf_r]
                    else:
                        w_l = w_r = 0
                    # compute objective functions given particular method
                    mse_mce_l, shares_l, obs_by_treat_l = mcf_mse_fn(
                        y_dat[leaf_l], y_nn_l, d_dat_l, w_l,
                        n_obs=n_l, mtot=mtot, no_of_treat=no_of_treat, treat_values=d_values,
                        w_yes=weighted, splitting=False, cuda=cuda, cython=cython,
                        compare_only_to_zero=compare_only_to_zero, pen_mult=pen_mult_d,
                        zero_tol=zero_tol,
                        )
                    mse_mce_r, shares_r, obs_by_treat_r = mcf_mse_fn(
                        y_dat[leaf_r], y_nn_r, d_dat_r, w_r,
                        n_obs=n_r, mtot=mtot, no_of_treat=no_of_treat, treat_values=d_values,
                        w_yes=weighted, splitting=False, cuda=cuda, cython=cython,
                        compare_only_to_zero=compare_only_to_zero, pen_mult=pen_mult_d,
                        zero_tol=zero_tol,
                        )
                    mse_mce = add_mse_mce_split_fn(mse_mce_l=mse_mce_l, mse_mce_r=mse_mce_r,
                                                   obs_by_treat_l=obs_by_treat_l,
                                                   obs_by_treat_r=obs_by_treat_r, mtot=mtot,
                                                   no_of_treat=no_of_treat,
                                                   compare_only_to_zero=compare_only_to_zero,
                                                   )
                    if cython:
                        # raise ValueError('Cython currently not used.')
                        # mse_split = mcf_cy.compute_mse_mce_cy(
                        #     mse_mce, mtot, no_of_treat,
                        #     compare_only_to_zero)
                        pass

                    mse_split = compute_mse_mce_fn(mse_mce, mtot, no_of_treat, compare_only_to_zero)
                    # add penalty for this split if 'diff_d'
                    if (penalty_type == 'diff_d'
                            and ((mtot == 1) or ((mtot == 4) and (rng.random() > 0.5)))):
                        penalty = mcf_penalty_fn(shares_l, shares_r)
                        mse_split = mse_split + pen_mult * penalty

                    if mse_split < best_mse:
                        split_done = True
                        best_mse = mse_split
                        # changed Oct., 22, 2025 to reduce copying
                        best['j'] = int(j)
                        best['type'] = int(x_type_split[j])
                        best['d_cont_split'] = bool(d_cont_split)
                        best['n_l'], best['n_r'] = n_l, n_r
                        best['n_oob_l'], best['n_oob_r'] = n_oob_l, n_oob_r
                        if x_type_split[j] == 0:
                            best['value'] = float(val)
                            if d_cont_split:
                                # persist control assignments; they are part of the split definition
                                best['zeros_l'] = zeros_l.copy()
                                best['zeros_l_oob'] = zeros_l_oob.copy()
                        else:
                            # keep the current category set without
                            # recomputing from scratch
                            best['value'] = split_values_unord_j[:]
                            # list of included categories

        if not split_done:
            return True, None, None, None, None, None, None, None, None, None, None, None

        j = best['j']
        if best['type'] == 0:
            val_eps = np.nextafter(best['value'], np.inf)
            if best['d_cont_split']:
                x_j = d_dat
                x_oob_j = d_oob
                treated_l = (~(x_j <= zero_tol)) & (x_j <= val_eps)
                leaf_l = treated_l | best['zeros_l']
                treated_l_oob = (~(x_oob_j <= zero_tol)) & (x_oob_j <= val_eps)
                leaf_oob_l = treated_l_oob | best['zeros_l_oob']
            else:
                x_j = x_dat[:, j]
                x_oob_j = x_oob[:, j]
                leaf_l = x_j <= val_eps
                leaf_oob_l = x_oob_j <= val_eps
        else:
            cats = best['value']
            x_j = x_dat[:, j]
            x_oob_j = x_oob[:, j]
            leaf_l = np.isin(x_j, cats)
            leaf_oob_l = np.isin(x_oob_j, cats)

        leaf_r = ~leaf_l
        leaf_oob_r = ~leaf_oob_l

        # assign to variables to be returned
        best_var_i = x_ind_split[j]
        best_type = best['type']
        best_n_l = best['n_l']
        best_n_r = best['n_r']
        best_leaf_l = leaf_l
        best_leaf_r = leaf_r
        best_leaf_oob_l = leaf_oob_l
        best_leaf_oob_r = leaf_oob_r
        best_n_oob_l = best['n_oob_l']
        best_n_oob_r = best['n_oob_r']
        best_value = best['value']

    return (terminal, best_var_i, best_type, best_n_l, best_n_r, best_leaf_l, best_leaf_r,
            best_leaf_oob_l, best_leaf_oob_r, best_n_oob_l, best_n_oob_r, best_value
            )


def update_tree(data_oob_parent: NDArray[np.float64],
                tree_dict_global: dict, *,
                parent_idx: int,
                split_var_i: int | None,
                split_type: int | None,
                split_value: int | float | np.floating | np.integer | None,
                split_n_l: int | None,
                split_n_r: int | None,
                split_leaf_l: NDArray[bool] | None,
                split_leaf_r: NDArray[bool] | None,
                split_leaf_oob_l: NDArray[bool] | None,
                split_leaf_oob_r: NDArray[bool] | None,
                split_n_oob_l: int | None,
                split_n_oob_r: int | None,
                terminal: bool,
                leaf_id_daughters: NDArray[np.integer] | None,
                d_i: int,
                w_i: int | None,
                d_grid_i: int | None,
                y_nn_i: NDArray[np.intp],
                y_i: int,
                ct_grid_nn_val: list[Any] | None,
                gen_cfg: 'GenCfg', cf_cfg: 'CfCfg',
                rng: np.random.Generator,
                cuda: bool,
                cython: bool,
                zero_tol: float = 1e-10,
                ) -> dict:
    """Assign values obtained from splitting to parent & daughter leaves."""
    tree = deepcopy(tree_dict_global)
    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values, continuous = 2, [0, 1], True
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        continuous = False
    if terminal:
        tree['leaf_info_int'][parent_idx, 7] = 1  # Terminal
        tree['leaf_info_int'][parent_idx, 6] = 1  # Terminal
        w_oob = data_oob_parent[:, [w_i]] if gen_cfg.weighted else 0
        n_oob = data_oob_parent.shape[0]
        d_oob = data_oob_parent[:, d_i] > zero_tol if continuous else data_oob_parent[:, d_i]

        if len(np.unique(d_oob)) < no_of_treat:
            obj_fct_oob, obs_oob = 0, n_oob      # MSE cannot be computed
        else:
            if continuous:
                y_nn = mcf_fo_add.match_cont(data_oob_parent[:, d_grid_i],
                                             data_oob_parent[:, y_nn_i],
                                             ct_grid_nn_val, rng
                                             )
            else:
                y_nn = data_oob_parent[:, y_nn_i]
            mse_mce, _, obs_oob_list = mcf_fo_obj.mcf_mse(data_oob_parent[:, y_i], y_nn, d_oob,
                                                          w_oob,
                                                          n_obs=n_oob, mtot=cf_cfg.mtot,
                                                          no_of_treat=no_of_treat,
                                                          treat_values=d_values,
                                                          w_yes=gen_cfg.weighted,
                                                          splitting=False, cuda=cuda, cython=cython,
                                                          compare_only_to_zero=
                                                              cf_cfg.compare_only_to_zero,
                                                          zero_tol=zero_tol,
                                                          )
            if cython:
                # raise ValueError('Cython currently not used.')
                pass
                # obj_fct_oob = mcf_cy.compute_mse_mce_cy(
                #     mse_mce, cf_cfg.mtot, no_of_treat,
                #     cf_cfg.compare_only_to_zero)

            obj_fct_oob = mcf_fo_obj.compute_mse_mce(mse_mce, cf_cfg.mtot, no_of_treat,
                                                     cf_cfg.compare_only_to_zero
                                                     )
            obs_oob = np.sum(obs_oob_list)
        tree['leaf_info_float'][parent_idx, 2] = obj_fct_oob
        tree['leaf_info_int'][parent_idx, 9] = obs_oob
        tree['train_data_list'][parent_idx] = None
        if not cf_cfg.vi_oob_yes:
            tree['train_data_list'][parent_idx] = None
    else:
        train_list = np.array(tree['train_data_list'][parent_idx], copy=True)
        oob_list = np.array(tree['oob_data_list'][parent_idx], copy=True)
        # Change information in parent leave
        #    Assign IDs
        tree['leaf_info_int'][parent_idx, 2:4] = leaf_id_daughters
        #    Assign status and splitting information
        tree['leaf_info_int'][parent_idx, 7] = 0
        #                             not active, not terminal - intermediate
        tree['leaf_info_int'][parent_idx, 4] = split_var_i
        if split_type > 0:  # Save as product of primes
            tree['cats_prime'][parent_idx] = list_product(split_value)
        else:
            tree['leaf_info_float'][parent_idx, 1] = split_value
        tree['leaf_info_int'][parent_idx, 5] = split_type

        # Initialise daughter leaves
        id_l, id_r = leaf_id_daughters
        # Assign IDs
        tree['leaf_info_int'][id_l, 1] = tree['leaf_info_int'][parent_idx, 0]
        tree['leaf_info_int'][id_r, 1] = tree['leaf_info_int'][parent_idx, 0]
        tree['leaf_info_int'][id_l, 6] = tree['leaf_info_int'][id_r, 6] = 0
        tree['leaf_info_int'][id_l, 7] = tree['leaf_info_int'][id_r, 7] = 2
        tree['leaf_info_int'][id_l, 8] = split_n_l
        tree['leaf_info_int'][id_r, 8] = split_n_r
        tree['leaf_info_int'][id_l, 9] = split_n_oob_l
        tree['leaf_info_int'][id_r, 9] = split_n_oob_r

        tree['train_data_list'][id_l] = train_list[split_leaf_l]
        tree['train_data_list'][id_r] = train_list[split_leaf_r]
        tree['oob_data_list'][id_l] = oob_list[split_leaf_oob_l]
        tree['oob_data_list'][id_r] = oob_list[split_leaf_oob_r]

        tree['train_data_list'][parent_idx] = None
        tree['oob_data_list'][parent_idx] = None

    return tree


def oob_in_tree(obs_in_leaf,
                y_dat: NDArray[np.floating | np.integer],
                y_nn: NDArray[np.floating | np.integer],
                d_dat: NDArray[np.floating | np.integer],
                w_dat: NDArray[np.floating | np.integer] | None, *,
                mtot: int,
                no_of_treat: int,
                treat_values: list[int] | NDArray[np.floating | np.integer],
                w_yes: bool,
                cont: bool = False,
                cuda: bool = False,
                cython: bool = True,
                compare_only_to_zero: bool = False,
                zero_tol: float = 1e-10,
                ) -> np.floating:
    """Compute OOB values for a tree.

    Parameters
    ----------
    obs_in_leaf : List of int. Terminal leaf no of observation
    y : Numpy array.
    y_nn : Numpy array.
    d : Numpy array.
    w : Numpy array.
    mtot : Integer. Method used.
    no_of_treat : Integer.
    treat_values : List of integer.
    w_yes : Boolean.
    cont : Boolean. Default is False.
    cuda : Boolean. MSE computation with Cuda if True. Default is False.
    # cython : Boolean. MSE computation with Cython if True. Default is True.
    compare_only_to_zero : Boolean. Reduced covariance matrix.

    Returns
    -------
    oob_tree : INT. OOB value of the MSE of the tree

    """
    leaf_no = np.unique(obs_in_leaf[:, 1])
    oob_tree = n_lost = n_total = 0
    mse_mce_tree = np.zeros((no_of_treat, no_of_treat))
    obs_t_tree = np.zeros(no_of_treat)
    for leaf in leaf_no:
        in_leaf = obs_in_leaf[:, 1] == leaf
        w_l = w_dat[in_leaf] if w_yes else 0
        n_l = np.count_nonzero(in_leaf)
        d_dat_in_leaf = d_dat[in_leaf]  # makes a copy
        if n_l < no_of_treat:
            enough_data_in_leaf = False
        else:
            enough_data_in_leaf = True
            if n_l < 40:          # this is done for efficiency reasons
                if set(d_dat_in_leaf.reshape(-1)) < set(treat_values):
                    enough_data_in_leaf = False
            else:
                if len(np.unique(d_dat_in_leaf)) < no_of_treat:  # No MSE
                    enough_data_in_leaf = False
        if enough_data_in_leaf:
            mse_mce_leaf, _, obs_by_treat_leaf = mcf_fo_obj.mcf_mse(
                y_dat[in_leaf], y_nn[in_leaf], d_dat_in_leaf, w_l,
                n_obs=n_l, mtot=mtot, no_of_treat=no_of_treat, treat_values=treat_values,
                w_yes=w_yes, splitting=cont, cuda=cuda, cython=cython,
                compare_only_to_zero=compare_only_to_zero, zero_tol=zero_tol,
                )
            mse_mce_tree, obs_t_tree = mcf_fo_obj.add_rescale_mse_mce(
                mse_mce_leaf, obs_by_treat_leaf,
                mtot=mtot, no_of_treat=no_of_treat, mse_mce_add_to=mse_mce_tree,
                obs_by_treat_add_to=obs_t_tree, compare_only_to_zero=compare_only_to_zero,
                )
        else:
            n_lost += n_l
        n_total += n_l
    mse_mce_tree = mcf_fo_obj.get_avg_mse_mce(mse_mce_tree, obs_t_tree, mtot, no_of_treat,
                                              compare_only_to_zero
                                              )
    if cython:
        pass
        # raise ValueError('Cython currently not used.')
        # oob_tree = mcf_cy.compute_mse_mce_cy(mse_mce_tree, mtot, no_of_treat,compare_only_to_zero)
    oob_tree = mcf_fo_obj.compute_mse_mce(mse_mce_tree, mtot, no_of_treat, compare_only_to_zero)

    return oob_tree
