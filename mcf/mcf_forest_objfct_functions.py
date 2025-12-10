"""
Contains functions to compute objective functions of mcf.

Created on Mon Oct 30 13:09:25 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from typing import Any

from numba import njit
import numpy as np
from numpy.typing import NDArray

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

from mcf import mcf_cuda_functions as mcf_c
# from mcf import mcf_forest_cy as mcf_cy


def mcf_mse(y_dat: NDArray[np.float64],
            y_nn: NDArray[np.float64],
            d_dat: NDArray[np.integer | np.floating],
            w_dat: NDArray[np.integer | np.floating] | int | None,
            n_obs: int,
            mtot: int,
            no_of_treat: int,
            treat_values: list[int],
            w_yes: bool = False,
            splitting: bool = False,
            cuda: bool = False,
            cython: bool = True,
            compare_only_to_zero: bool = False,
            pen_mult: float = 0
            ) -> tuple[NDArray[np.float64],
                       NDArray[np.float64],
                       NDArray[np.float64]
                       ]:
    """Compute average mse for the data passed. Based on different methods.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    w_dat : Numpy Nx1 vector. Weights (or 0)
    n_obs : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : List of INT. Treatment values.
    w_yes : Boolean. Weighted estimation.
    splitting : Boolean. Default is False.
    cuda : Boolean. Use cuda if True.
    compare_only_to_zero : Boolean. Use reduced MSE matrix.
    pen_mult : Multiplier of penalty related to MSE of treatment prediction.

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.

    """
    min_obs_for_cuda = 10000000000  # No gain in speed compared to Numba

    # CUDA is currently not used here (ineffective when used from many
    # processes simultaneously. Too much traffic.)
    if cuda and len(y_dat) > min_obs_for_cuda:
        mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_cuda(
            y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
            treat_values, w_yes, splitting, compare_only_to_zero, pen_mult)
    else:
        if w_yes or mtot in (2, 3):
            mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_not_numba(
                y_dat, y_nn, d_dat, w_dat, n_obs, mtot, no_of_treat,
                treat_values, w_yes, splitting, compare_only_to_zero, pen_mult
                )
        else:
            if cython:
                raise ValueError('Cython currently not used.')

            mse_mce, treat_shares, no_of_obs_by_treat = mcf_mse_numba(
                y_dat, y_nn, d_dat, n_obs, mtot, no_of_treat,
                np.array(treat_values, dtype=np.int8), compare_only_to_zero,
                pen_mult, pen_mult > 0
                )
    return mse_mce, treat_shares, no_of_obs_by_treat


def mcf_mse_cuda(y_dat_np: NDArray[np.float64],
                 y_nn_np: NDArray[np.float64],
                 d_dat_np: NDArray[np.integer | np.floating],
                 w_dat_np: NDArray[np.integer | np.floating] | int | None,
                 n_obs: int,
                 mtot: int,
                 no_of_treat: int,
                 treat_values_list: list[int],
                 w_yes: bool,
                 splitting: bool = False,
                 compare_only_to_zero: bool = False,
                 pen_mult: float | int = 0,
                 ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute average MSE / MCE for the data passed (CUDA)."""
    precision = 32

    # Move inputs to GPU
    y_dat, y_nn, d_dat, w_dat, treat_values = tensors_to_gpu(
        y_dat_np, y_nn_np, d_dat_np, w_dat_np,
        treat_values_list,
        w_yes,
        )
    device = y_dat.device
    float_dtype = mcf_c.tdtype('float', precision)
    int_dtype = mcf_c.tdtype('int', precision)

    if w_yes and w_dat is None:
        msg = 'w_yes=True requires w_dat_np to be provided.'
        raise ValueError(msg)

    # Treatment shares only relevant for some mtot, but always allocate tensor
    treat_shares = torch.zeros(no_of_treat, device=device, dtype=float_dtype,)

    mse_mce = torch.zeros((no_of_treat, no_of_treat),
                          device=device, dtype=float_dtype,
                          )
    no_of_obs_by_treat = torch.zeros(no_of_treat,
                                     device=device, dtype=int_dtype,
                                     )
    for m_idx in range(no_of_treat):
        d_m = d_dat == treat_values[m_idx]   # boolean mask
        y_m = y_dat[d_m]
        n_m = int(y_m.shape[0])
        no_of_obs_by_treat[m_idx] = n_m

        if n_m == 0:
            # No observations in this treatment cell
            continue

        if w_yes:
            w_m = w_dat[d_m]
            y_m_mean = mean_t_w(y_m, w_m, dim=0)
            mse_m = mean_t_w((y_m - y_m_mean) ** 2, w_m, dim=0)

            if pen_mult > 0:
                d_mf = d_m.to(dtype=float_dtype)
                d_m_mean = mean_t_w(d_mf, w_dat, dim=0)
                mse_m = mse_m + pen_mult * mean_t_w((d_mf - d_m_mean) ** 2,
                                                    w_dat, dim=0,
                                                    )
        else:
            y_m_mean = torch.mean(y_m, dim=0)
            mse_m = torch.sum(y_m ** 2) / n_m - (y_m_mean ** 2)

            if pen_mult > 0:
                d_mf = d_m.to(dtype=float_dtype)
                d_m_mean = torch.mean(d_mf, dim=0)
                mse_m = mse_m + pen_mult * (torch.sum(d_mf ** 2)
                                            / n_obs - (d_m_mean ** 2)
                                            )
        # Treatment shares only used for some mtot values
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs

        # Diagonal entries
        if mtot in (1, 3, 4):
            mse_mce[m_idx, m_idx] = mse_m

        # Off-diagonal entries (cross terms)
        if (mtot != 3) and (m_idx == 0 or not compare_only_to_zero):
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:
                    # Variance of effects
                    d_l = d_dat == treat_values[v_idx]
                    y_l = y_dat[d_l]
                    if y_l.shape[0] == 0:
                        continue

                    if w_yes:
                        y_l_mean = mean_t_w(y_l, w_dat[d_l], dim=0)
                    else:
                        y_l_mean = torch.mean(y_l, dim=0)

                    mce_ml = (y_m_mean - y_l_mean) ** 2
                else:
                    # Correlation-type terms based on matched outcomes
                    d_ml = ((d_dat == treat_values[v_idx])
                            | (d_dat == treat_values[m_idx])
                            )

                    y_nn_m = y_nn[d_ml, m_idx]
                    y_nn_l = y_nn[d_ml, v_idx]

                    if y_nn_m.shape[0] == 0:
                        continue

                    if w_yes:
                        w_ml = w_dat[d_ml].reshape(-1)
                        if splitting and (no_of_treat == 2):
                            mce_ml = (mean_t_w(y_nn_m, w_ml, dim=0)
                                      * mean_t_w(y_nn_l, w_ml, dim=0)
                                      * (-1)
                                      )
                        else:
                            mce_ml = mean_t_w(
                                (y_nn_m - mean_t_w(y_nn_m, w_ml, dim=0))
                                * (y_nn_l - mean_t_w(y_nn_l, w_ml, dim=0)),
                                w_ml, dim=0,
                                )
                    else:
                        mean_m = torch.mean(y_nn_m, dim=0)
                        mean_l = torch.mean(y_nn_l, dim=0)
                        aaa = mean_m * mean_l
                        bbb = torch.sum(y_nn_m * y_nn_l) / y_nn_m.shape[0]
                        mce_ml = bbb - aaa

                mse_mce[m_idx, v_idx] = mce_ml

    mse_mce_np, treat_shares_np, no_of_obs_by_treat_np, = tensors_to_np_to_cpu(
        mse_mce,
        treat_shares, no_of_obs_by_treat,
        )
    return mse_mce_np, treat_shares_np, no_of_obs_by_treat_np


def mean_t_w(tensor: torch.Tensor,
             weights: torch.Tensor,
             dim: int | None = None
             ) -> torch.Tensor:
    """
    Compute the weighted mean of a tensor along a specified dimension.

    Parameters
    ----------
    tensor : torch.Tensor. Input tensor.
    weights : torch.Tensor: Tensor of weights.
    dim : int, optional: Dimension along which to compute the mean.

    Returns
    -------
    weighted_mean : torch.Tensor: Weighted mean.

    """
    weighted_sum = torch.sum(tensor * weights, dim=dim)
    sum_of_weights = torch.sum(weights, dim=dim)
    weighted_mean = weighted_sum / sum_of_weights
    return weighted_mean


def tensors_to_np_to_cpu(mse_mce: torch.Tensor,
                         treat_shares: torch.Tensor,
                         no_of_obs_by_treat: torch.Tensor
                         ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Copy tensors to cpu and convert to numpy arrays."""
    mse_mce = mse_mce.to('cpu')
    treat_shares = treat_shares.to('cpu')
    no_of_obs_by_treat = no_of_obs_by_treat.to('cpu')

    mse_mce_np = mse_mce.numpy()
    treat_shares_np = treat_shares.numpy()
    no_of_obs_by_treat_np = no_of_obs_by_treat.numpy()

    return mse_mce_np, treat_shares_np, no_of_obs_by_treat_np


def tensors_to_gpu(y_dat_np: NDArray[np.integer | np.floating],
                   y_nn_np: NDArray[Any],
                   d_dat_np: NDArray[np.integer | np.floating],
                   w_dat_np: NDArray[np.integer | np.floating] | int | None,
                   treat_values_list: list[int],
                   w_yes: bool,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor,
                              ]:
    """Get the tensors and copy them to gpu."""
    y_dat = torch.from_numpy(y_dat_np)
    y_nn = torch.from_numpy(y_nn_np)
    d_dat = torch.from_numpy(d_dat_np)
    w_dat = torch.from_numpy(w_dat_np) if w_yes else None
    treat_values = torch.tensor(treat_values_list)

    y_dat = y_dat.to("cuda")
    y_nn = y_nn.to("cuda")
    d_dat = d_dat.to("cuda")
    if w_yes:
        w_dat = w_dat.to("cuda")
    treat_values = treat_values.to("cuda")

    return y_dat, y_nn, d_dat, w_dat, treat_values


def mcf_mse_not_numba(y_dat: NDArray[np.float64],
                      y_nn: NDArray[np.float64],
                      d_dat: NDArray[np.integer | np.floating],
                      w_dat: NDArray[np.integer | np.floating] | int | None,
                      n_obs: int,
                      mtot: int,
                      no_of_treat: int,
                      treat_values: list[int],
                      w_yes: bool,
                      splitting: bool = False,
                      compare_only_to_zero: bool = False,
                      pen_mult: float | int | None = 0,
                      ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute average mse for the data passed. Based on different methods.

    CURRENTLY ONLY USED FOR WEIGHTED.

    Parameters
    ----------
    y_dat : Numpy Nx1 vector. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_all : Numpy Nx1 vector. Treatment.
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
    no_of_obs_by_treat : 1D Numpy array. Observations by treatment.

    """
    treat_shares = np.empty(no_of_treat) if mtot in (1, 4) else 0
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
            if pen_mult > 0:
                d_m_mean = np.average(d_m, weights=w_dat, axis=0)
                mse_m += pen_mult * np.average(np.square(d_m - d_m_mean),
                                               weights=w_dat, axis=0)
        else:
            y_m_mean = np.average(y_dat[d_m], axis=0)
            mse_m = np.dot(y_dat[d_m], y_dat[d_m]) / n_m - (y_m_mean**2)
            if pen_mult > 0:
                d_m_mean = np.average(d_m, axis=0)
                mse_m += np.dot(d_m, d_m) / n_obs - (d_m_mean**2)
        if mtot in (1, 4):
            treat_shares[m_idx] = n_m / n_obs
        if mtot in (1, 3, 4):
            mse_mce[m_idx, m_idx] = mse_m
        if (mtot != 3) and (m_idx == 0 or not compare_only_to_zero):
            mce_ml = 0
            for v_idx in range(m_idx + 1, no_of_treat):
                if mtot == 2:  # Variance of effects mtot = 2
                    d_l = d_dat == treat_values[v_idx]   # d_l is Boolean
                    if w_yes:
                        y_l_mean = np.average(y_dat[d_l],
                                              weights=w_dat[d_l], axis=0)
                    else:
                        y_l_mean = np.average(y_dat[d_l], axis=0)
                    mce_ml = (y_m_mean - y_l_mean)**2
                else:
                    d_ml = (d_dat == treat_values[v_idx]) | (
                        d_dat == treat_values[m_idx])
                    d_ml = d_ml[:, 0]
                    y_nn_m, y_nn_l = y_nn[d_ml, m_idx], y_nn[d_ml, v_idx]
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
                        aaa = (np.average(y_nn_m, axis=0)
                               * np.average(y_nn_l, axis=0))
                        bbb = np.dot(y_nn_m, y_nn_l) / len(y_nn_m)
                        mce_ml = bbb - aaa
                mse_mce[m_idx, v_idx] = mce_ml

    return mse_mce, treat_shares, no_of_obs_by_treat


# Faster version as implemented in June, 6, 2024, 4.7.2025
@njit
def mcf_mse_numba(y_dat: NDArray[np.float64],
                  y_nn: NDArray[np.float64],
                  d_dat: NDArray[np.integer | np.floating],
                  n_obs: int,
                  mtot: int,
                  no_of_treat: int,
                  treat_values: NDArray[Any],
                  compare_only_to_zero: bool,
                  pen_mult: float,
                  penalty: bool,
                  ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute average mse for the data passed. Based on different methods.

    WEIGHTED VERSION DOES NOT YET WORK. TRY with next Numba version.
    Need to change list format soon.

    Parameters
    ----------
    y_dat : Numpy vector with N rows. Outcome variable of observation.
    y_nn : Numpy N x no_of_treatments array. Matched outcomes.
    d_dat : Numpy Nx1 vector. Treatment.
    n_obs : INT. Leaf size.
    mtot : INT. Method.
    no_of_treat : INT. Number of treated.
    treat_values : 1D Numpy array of INT. Treatment values.
    compare_only_to_zero : BOOLEAN. Compare only to zero.
    ...

    Returns
    -------
    mse : Mean squared error (average not acccount of number of obs).
    treat_share: 1D Numpy array. Treatment shares.
    no_of_obs_by_treat : 1D Numpy array. Observations by treatment.
    """
    mtot_in_134 = mtot in (1, 3, 4)
    mtot_in_14 = mtot in (1, 4)
    mtot_not_3 = mtot != 3
    mtot_is_2 = mtot == 2

    y_col = y_dat[:, 0] if y_dat.ndim == 2 else y_dat

    treat_shares = np.zeros(no_of_treat, dtype=np.float64)
    mse_mce = np.zeros((no_of_treat, no_of_treat), dtype=np.float64)
    no_of_obs_by_treat = np.zeros(no_of_treat, dtype=np.float64)

    for m_idx in range(no_of_treat):
        tv = treat_values[m_idx]
        # accumulate for group m
        n_m = 0
        sum_y = 0.0
        sum_y2 = 0.0
        for i in range(n_obs):
            if d_dat[i] == tv:
                v = y_col[i]
                n_m += 1
                sum_y += v
                sum_y2 += v * v

        no_of_obs_by_treat[m_idx] = n_m

        if n_m > 0:
            # y_m_mean = np.sum(y_m) / n_m
            # mse_m = np.dot(y_m, y_m) / n_m - (y_m_mean**2)
            y_mean_m = sum_y / n_m
            mse_m = (sum_y2 / n_m) - (y_mean_m * y_mean_m)

            if penalty:
                s = n_m / n_obs
                mse_m += pen_mult * (s - s * s)

            if mtot_in_134:
                treat_shares[m_idx] = n_m / n_obs
                mse_mce[m_idx, m_idx] = mse_m
        else:
            y_mean_m = 0
        # Get off-diagonal elements
        if mtot_not_3 and (m_idx == 0 or not compare_only_to_zero):
            for v_idx in range(m_idx + 1, no_of_treat):
                tv2 = treat_values[v_idx]  # second treatment value
                mce_ml = 0.0
                if mtot_is_2:  # Variance of effects mtot = 2
                    # mean for group v
                    n_v = 0
                    sum_y_v = 0.0
                    for i in range(n_obs):
                        if d_dat[i] == tv2:
                            sum_y_v += y_col[i]
                            n_v += 1
                    if (no_of_obs_by_treat[m_idx] > 0) and (n_v > 0):
                        y_mean_v = sum_y_v / n_v
                        diff = y_mean_m - y_mean_v
                        mce_ml = diff * diff

                elif mtot_in_14:
                    # covariance of matched outcomes for m and v over d∈{m,v}
                    n_ml = 0
                    sum_m = 0.0
                    sum_l = 0.0
                    sum_ml = 0.0
                    for i in range(n_obs):
                        di = d_dat[i]
                        if (di == tv) or (di == tv2):
                            ym = y_nn[i, m_idx]
                            yl = y_nn[i, v_idx]
                            n_ml += 1
                            sum_m += ym
                            sum_l += yl
                            sum_ml += ym * yl
                    if n_ml > 0:
                        mce_ml = ((sum_ml / n_ml)
                                  - (sum_m / n_ml) * (sum_l / n_ml)
                                  )
                mse_mce[m_idx, v_idx] = mce_ml

    return (mse_mce, treat_shares if mtot_in_134 else np.zeros(1),
            no_of_obs_by_treat,
            )


# Faster version as implemented in June, 6, 2024, und July, 4, 2025
@njit
def compute_mse_mce(mse_mce: NDArray[np.float64],
                    mtot: int,
                    no_of_treat: int,
                    compare_only_to_zero: bool
                    ) -> np.floating:
    """Sum up MSE parts for use in splitting rule and else."""
    mtot_in_14 = mtot in (1, 4)
    if no_of_treat > 4 and not compare_only_to_zero:
        trace_mse_mce = np.trace(mse_mce)
        sum_mse_mce = mse_mce.sum()
        if mtot_in_14:
            mse = no_of_treat * trace_mse_mce - sum_mse_mce
        elif mtot == 2:
            mse = 2 * trace_mse_mce - sum_mse_mce
        elif mtot == 3:
            mse = trace_mse_mce
    else:
        mse = 0.0
        mce = 0.0
        for m_idx in range(no_of_treat):
            if mtot_in_14:
                mse_a = (no_of_treat - 1) * mse_mce[m_idx, m_idx]
            else:
                mse_a = mse_mce[m_idx, m_idx]

            mse += mse_a

            if (mtot != 3) and (m_idx == 0 or not compare_only_to_zero):
                for v_idx in range(m_idx + 1, no_of_treat):
                    mce += mse_mce[m_idx, v_idx]

        mse -= 2 * mce

    return mse


# Optimized version of June, 6, 2024
@njit
def mcf_penalty(shares_l: NDArray[Any],
                shares_r: NDArray[Any]
                ) -> np.floating[Any]:
    """Generate the (unscaled) penalty.

    Parameters
    ----------
    shares_l : Numpy array. Treatment shares left.
    shares_r : Numpy array. Treatment shares right.

    Returns
    -------
    penalty : Numpy float. Penalty of split.

    """
    diff = shares_l - shares_r
    sum_diff_sq = np.dot(diff, diff)  # Efficient sum of squared differences
    penalty = 1 - (sum_diff_sq / shares_l.size)  # Use size attribute directly

    return penalty


# Optimized version of 4.6.2024, no change July, 7, 2025
@njit
def get_avg_mse_mce(mse_mce: NDArray[Any],
                    obs_by_treat: NDArray[Any],
                    mtot: int,
                    no_of_treat: int,
                    compare_only_to_zero: bool
                    ) -> np.floating:
    """Bring MSE_MCE matrix in average form."""
    mse_mce_avg = mse_mce.copy()

    for m_idx in range(no_of_treat):
        obs_m_idx = obs_by_treat[m_idx]
        if obs_m_idx > 0:
            mse_mce_avg[m_idx, m_idx] /= obs_m_idx

        if mtot != 3 and (m_idx == 0 or not compare_only_to_zero):
            for v_idx in range(m_idx + 1, no_of_treat):
                obs_v_idx = obs_by_treat[v_idx]
                total_obs = obs_m_idx + obs_v_idx
                if total_obs > 0:
                    mse_mce_avg[m_idx, v_idx] /= total_obs

    return mse_mce_avg


# Optimized code of 4.6.2024
@njit
def add_rescale_mse_mce(mse_mce: NDArray[Any],
                        obs_by_treat: NDArray[Any],
                        mtot: int,
                        no_of_treat: int,
                        mse_mce_add_to: NDArray[Any],
                        obs_by_treat_add_to: NDArray[Any],
                        compare_only_to_zero: bool
                        ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Rescale MSE_MCE matrix and update observation count."""
    mse_mce_sc = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat_new = obs_by_treat + obs_by_treat_add_to

    for m_idx in range(no_of_treat):
        obs_m_idx = obs_by_treat[m_idx]
        mse_mce_sc[m_idx, m_idx] = mse_mce[m_idx, m_idx] * obs_m_idx

        if mtot != 3 and (m_idx == 0 or not compare_only_to_zero):
            for v_idx in range(m_idx + 1, no_of_treat):
                obs_v_idx = obs_by_treat[v_idx]
                total_obs = obs_m_idx + obs_v_idx
                mse_mce_sc[m_idx, v_idx] = mse_mce[m_idx, v_idx] * total_obs

    mse_mce_new = mse_mce_add_to + mse_mce_sc

    return mse_mce_new, obs_by_treat_new


# Optimized version of 4.6.2024, 4.7.2025, 22.10.2025
@njit
def add_mse_mce_split(mse_mce_l: NDArray[Any],
                      mse_mce_r: NDArray[Any],
                      obs_by_treat_l: NDArray[Any],
                      obs_by_treat_r: NDArray[Any],
                      mtot: int,
                      no_of_treat: int,
                      compare_only_to_zero: bool
                      ) -> NDArray[Any]:
    """Sum up MSE parts of use in splitting rule."""
    mse_mce = np.zeros((no_of_treat, no_of_treat))
    obs_by_treat = obs_by_treat_l + obs_by_treat_r
    mtot_is_not_3 = mtot != 3
    for m_idx in range(no_of_treat):
        obs_m_idx = obs_by_treat[m_idx]
        if obs_m_idx > 0:
            mse_mce[m_idx, m_idx] = (
                (mse_mce_l[m_idx, m_idx] * obs_by_treat_l[m_idx]
                 + mse_mce_r[m_idx, m_idx] * obs_by_treat_r[m_idx])
                / obs_m_idx)

        if mtot_is_not_3 and (m_idx == 0 or not compare_only_to_zero):
            for v_idx in range(m_idx + 1, no_of_treat):
                n_ml_l = obs_by_treat_l[m_idx] + obs_by_treat_l[v_idx]
                n_ml_r = obs_by_treat_r[m_idx] + obs_by_treat_r[v_idx]
                total_obs = n_ml_l + n_ml_r
                if total_obs > 0:
                    mse_mce[m_idx, v_idx] = (
                        (mse_mce_l[m_idx, v_idx] * n_ml_l
                         + mse_mce_r[m_idx, v_idx] * n_ml_r)
                        / total_obs
                        )
    return mse_mce
