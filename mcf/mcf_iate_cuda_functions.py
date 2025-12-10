"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the IATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from math import ceil
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

from mcf import mcf_cuda_functions as mcf_cuda
from mcf import mcf_estimation_cuda_functions as mcf_est_cuda
from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_general_sys as mcf_sys


def iate_cuda(weights_list: list[NDArray[Any] | Sequence[Any]],
              cl_dat_np: NDArray[Any],
              no_of_cluster: int,
              w_dat_np: NDArray[Any],
              w_ate_np: NDArray[Any] | None,
              y_dat_np: NDArray[Any],
              no_of_out: int,
              n_y: int,
              ct_cfg: Any,
              int_cfg: Any,
              gen_cfg: Any,
              p_cfg: Any,
              p_ba_cfg: Any,
              iate_se_flag: bool,
              se_boot_iate: bool,
              iate_m_ate_flag: bool,
              n_x: int,
              no_of_treat_dr: int,
              ) -> tuple[NDArray[Any],
                         NDArray[Any] | None,
                         NDArray[Any] | None,
                         NDArray[Any] | None,
                         list[list[NDArray[Any] | str]],
                         NDArray[Any],
                         ]:
    """Compute IATE and SE of IATE using GPU."""
    precision = 32

    # Move inputs to GPU (and cast dtype) using your helper
    weights, cl_dat, w_dat, w_ate, y_dat = tensor_to_gpu(
        weights_list,
        cl_dat_np, w_dat_np, w_ate_np, y_dat_np,
        int_cfg.weight_as_sparse, iate_m_ate_flag, precision,
        )
    device = y_dat.device
    float_dtype = mcf_cuda.tdtype('float', precision)

    # Containers for potential outcomes etc. on GPU
    pot_y = torch.empty((n_x, no_of_treat_dr, no_of_out),
                        device=device, dtype=float_dtype,
                        )
    pot_y_m_ate = torch.empty_like(pot_y) if iate_m_ate_flag else None
    pot_y_var = torch.empty_like(pot_y) if iate_se_flag else None
    pot_y_m_ate_var = (torch.empty_like(pot_y)
                       if (iate_se_flag and iate_m_ate_flag)
                       else None
                       )
    # Share of censored weights (per treatment)
    share_censored = torch.zeros(no_of_treat_dr,
                                 device=device, dtype=float_dtype,
                                 )
    # Per-x diagnostics from analyse_weights_cuda
    l1_to_9: list[list[torch.Tensor | str] | None] = [None for _ in range(n_x)]

    # Treatment grid (continuous vs discrete)
    if gen_cfg.d_type == 'continuous':
        d_values_dr = torch.from_numpy(ct_cfg.d_values_dr_np,).to(device)
    else:
        d_values_dr = torch.tensor(gen_cfg.d_values,
                                   device=device,
                                   dtype=mcf_cuda.tdtype('int', precision),
                                   )
    # Main loop over x-points (operates entirely on GPU)
    for idx in range(n_x):
        if int_cfg.weight_as_sparse:
            # weights: list over treatments; each element is a sparse tensor
            weights_idx = [weights_tx[idx] for weights_tx in weights]
        else:
            # weights: dense tensor of shape (n_x, ...)
            weights_idx = weights[idx]

        ret_all_i = iate_func1_for_cuda(
            idx, weights_idx,
            cl_dat, no_of_cluster, w_dat, w_ate, y_dat,
            no_of_out, n_y,
            ct_cfg, int_cfg, gen_cfg, p_cfg, p_ba_cfg,
            iate_se_flag, se_boot_iate, iate_m_ate_flag,
            d_values_dr,
            precision=precision,
            zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
            )
        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
         l1_to_9, share_censored,
         ) = assign_ret_all_i_cuda(pot_y, pot_y_var, pot_y_m_ate,
                                   pot_y_m_ate_var,
                                   l1_to_9, share_censored,
                                   ret_all_i, n_x, idx,
                                   )
        if gen_cfg.with_output and gen_cfg.verbose:
            mcf_gp.share_completed(idx + 1, n_x)

    (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np,
     share_censored_np, l1_to_9_np,) = gpu_to_numpy(
         pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
        share_censored, l1_to_9,
        )
    return (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np,
            l1_to_9_np, share_censored_np,
            )


def gpu_to_numpy(pot_y: torch.Tensor,
                 pot_y_var: torch.Tensor | None,
                 pot_y_m_ate: torch.Tensor | None,
                 pot_y_m_ate_var: torch.Tensor | None,
                 share_censored: torch.Tensor,
                 l1_to_9: list[list[torch.Tensor]],
                 ) -> tuple[NDArray[Any],
                            NDArray[Any] | None,
                            NDArray[Any] | None,
                            NDArray[Any] | None,
                            NDArray[Any],
                            list[list[NDArray[Any]]],
                            ]:
    """Move tensors back to CPU and convert to NumPy arrays."""

    def to_numpy(x: torch.Tensor | None):
        if isinstance(x, torch.Tensor):
            # detach() to be safe if tensors still track gradients
            return x.detach().cpu().numpy()

        return x

    pot_y_np = to_numpy(pot_y)
    pot_y_var_np = to_numpy(pot_y_var)
    pot_y_m_ate_np = to_numpy(pot_y_m_ate)
    pot_y_m_ate_var_np = to_numpy(pot_y_m_ate_var)
    share_censored_np = to_numpy(share_censored)

    # Nested structure directly via comprehension; no preallocation needed
    l1_to_9_list = [[to_numpy(var) for var in group] for group in l1_to_9]

    return (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np,
            share_censored_np, l1_to_9_list,
            )


def assign_ret_all_i_cuda(pot_y: torch.Tensor,
                          pot_y_var: torch.Tensor | None,
                          pot_y_m_ate: torch.Tensor | None,
                          pot_y_m_ate_var: torch.Tensor | None,
                          l1_to_9: list[tuple[torch.Tensor | str, ...] | None],
                          share_censored: torch.Tensor,
                          ret_all_i: Any,
                          n_x: int,
                          idx: int | None = None,
                          ) -> tuple[torch.Tensor,
                                     torch.Tensor | None,
                                     torch.Tensor | None,
                                     torch.Tensor | None,
                                     (list[tuple[torch.Tensor | str, ...]
                                           | None]),
                                     torch.Tensor,
                                     ]:
    """Use to avoid duplicate code when collecting CUDA results."""

    if idx is None:
        idx = int(ret_all_i[0])

    pot_y_i = ret_all_i[1]
    pot_y_var_i = ret_all_i[2]
    pot_y_m_ate_i = ret_all_i[3]
    pot_y_m_ate_var_i = ret_all_i[4]
    l1_to_9_i = ret_all_i[5]
    share_i = ret_all_i[6]

    pot_y[idx, :, :] = pot_y_i

    if pot_y_var is not None and pot_y_var_i is not None:
        pot_y_var[idx, :, :] = pot_y_var_i

    if pot_y_m_ate is not None and pot_y_m_ate_i is not None:
        pot_y_m_ate[idx, :, :] = pot_y_m_ate_i

    if pot_y_m_ate_var is not None and pot_y_m_ate_var_i is not None:
        pot_y_m_ate_var[idx, :, :] = pot_y_m_ate_var_i

    l1_to_9[idx] = l1_to_9_i

    share_censored = share_censored + share_i / n_x

    return (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
            l1_to_9, share_censored,
            )


def iate_func1_for_cuda(idx: int,
                        weights_i: list | torch.Tensor,
                        cl_dat: torch.Tensor | None,
                        no_of_cluster: int | None,
                        w_dat: torch.Tensor | None,
                        w_ate: torch.Tensor | None,
                        y_dat: torch.Tensor,
                        no_of_out: int,
                        n_y: int,
                        ct_cfg: Any,
                        int_cfg: Any,
                        gen_cfg: Any,
                        p_cfg: Any,
                        p_ba_cfg: Any,
                        iate_se_flag: bool,
                        se_boot_iate: bool,
                        iate_m_ate_flag: bool,
                        d_values_dr: torch.Tensor,
                        precision: int = 32,
                        iv: bool = False,
                        sum_tol: float = 1e-12,
                        zero_tol: float = 1e-15,
                        ) -> tuple[int,
                                   torch.Tensor,
                                   torch.Tensor | None,
                                   torch.Tensor | None,
                                   torch.Tensor | None,
                                   tuple[torch.Tensor, ...],
                                   torch.Tensor,
                                   ]:
    """Compute function to be looped over observations for CUDA."""
    def get_walli(w_index: torch.Tensor,
                  n_y: int,
                  w_i: torch.Tensor,
                  w_i_unc: torch.Tensor,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        w_all_i = torch.zeros(n_y, device=device, dtype=dtype)
        w_all_i[w_index] = w_i
        w_all_i_unc = torch.zeros_like(w_all_i)
        w_all_i_unc[w_index] = w_i_unc
        return w_all_i, w_all_i_unc

    device = y_dat.device
    dtype = mcf_cuda.tdtype('float', precision)
    ret_ma = None
    # --- continuous vs discrete treatment setup -------------------------

    if gen_cfg.d_type == 'continuous':
        print('WARNING: IATE computation for continuous treatment on cuda not '
              'yet tested at all. Most likely it will break.',
              )
        continuous = True
        no_of_treat = ct_cfg.grid_w
        i_w01 = ct_cfg.w_to_dr_int_w01
        i_w10 = ct_cfg.w_to_dr_int_w10
        index_full = ct_cfg.w_to_dr_index_full
        no_of_treat_dr = len(d_values_dr)
    else:
        continuous = False
        no_of_treat = gen_cfg.no_of_treat
        no_of_treat_dr = gen_cfg.no_of_treat
        i_w01 = None
        i_w10 = None
        index_full = None

    # --- output containers ---------------------------------------------

    pot_y_i = torch.empty((no_of_treat_dr, no_of_out),
                          device=device, dtype=dtype,
                          )
    if iate_m_ate_flag:
        pot_y_m_ate_i = torch.empty_like(pot_y_i)
    else:
        pot_y_m_ate_i = None

    share_i = torch.zeros(no_of_treat_dr, device=device, dtype=dtype,)

    if iate_se_flag:
        pot_y_var_i = torch.empty_like(pot_y_i)
        if iate_m_ate_flag:
            pot_y_m_ate_var_i = torch.empty_like(pot_y_i)
        else:
            pot_y_m_ate_var_i = None
        cluster_std = p_cfg.cluster_std
    else:
        pot_y_var_i = None
        pot_y_m_ate_var_i = None
        cluster_std = False

    if cluster_std:
        if no_of_cluster is None:
            msg = 'no_of_cluster must be provided when cluster_std is True.'
            raise ValueError(msg)
        w_add = torch.zeros((no_of_treat_dr, no_of_cluster),
                            device=device, dtype=dtype,
                            )
    else:
        w_add = torch.zeros((no_of_treat_dr, n_y),
                            device=device, dtype=dtype,
                            )
    w_add_unc = torch.zeros((no_of_treat_dr, n_y),
                            device=device, dtype=dtype,
                            )
    # --- main loop over treatment values -------------------------------

    cl_i_both: torch.Tensor | None = None

    for t_idx in range(no_of_treat):
        extra_weight_p1 = continuous and (t_idx < no_of_treat - 1)

        # weights for this treatment
        if int_cfg.weight_as_sparse:
            coalesced_tensor = weights_i[t_idx].coalesce()
            w_index = coalesced_tensor.indices().reshape(-1)
            w_i_t = coalesced_tensor.values()
        else:
            w_index = weights_i[t_idx][0]
            w_i_t = weights_i[t_idx][1].clone()

        # optional combination with t+1 (continuous case)
        if extra_weight_p1:
            if int_cfg.weight_as_sparse:
                coalesced_tensor_p1 = weights_i[t_idx + 1].coalesce()
                w_index_p1 = coalesced_tensor_p1.indices().reshape(-1)
            else:
                w_index_p1 = weights_i[t_idx + 1][0]

            w_index_both = torch.unique(torch.cat((w_index, w_index_p1)))

            w_i = torch.zeros(n_y, device=device, dtype=dtype)
            w_i[w_index] = w_i_t

            w_i_p1 = torch.zeros_like(w_i)
            if int_cfg.weight_as_sparse:
                w_i_p1[w_index_p1] = coalesced_tensor_p1.values()
            else:
                w_i_p1[w_index_p1] = weights_i[t_idx + 1][1].clone()

            w_i = w_i[w_index_both]
            w_i_p1 = w_i_p1[w_index_both]
        else:
            w_index_both = w_index
            w_i = w_i_t

        # global sampling weights, if any
        if gen_cfg.weighted and w_dat is not None:
            w_t = w_dat[w_index].reshape(-1)
            w_i = w_i * w_t
            if extra_weight_p1:
                w_t_p1 = w_dat[w_index_p1].reshape(-1)
                w_i_p1 = w_i_p1 * w_t_p1
        else:
            w_t = None
            if extra_weight_p1:
                w_t_p1 = None  # noqa: F841

        # normalize local weights if they should approximately sum to 1
        w_i_sum = w_i.sum()
        w_i_sum_val = float(w_i_sum)
        if (not (1.0 - sum_tol < w_i_sum_val < 1.0 + sum_tol)
                and not (continuous or iv)):
            w_i = w_i / w_i_sum

        w_i_unc = w_i.clone()

        # bound weights if needed
        if p_cfg.max_weight_share < 1 and not continuous:
            if iv:
                (w_i, _, share_i[t_idx],
                 ) = mcf_gp.bound_norm_weights_not_one_cuda(
                     w_i,
                     p_cfg.max_weight_share,
                     zero_tol=int_cfg.zero_tol,
                     sum_tol=int_cfg.sum_tol,)
            else:
                (w_i, _, share_i[t_idx]) = mcf_gp.bound_norm_weights_cuda(
                    w_i,
                    p_cfg.max_weight_share,
                    zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                    negative_weights_possible=p_ba_cfg.yes,
                    )
        if extra_weight_p1:
            w_i_unc_p1 = w_i_p1.clone()

        # cluster-level full weight vectors if needed
        if cluster_std:
            w_all_i, w_all_i_unc = get_walli(w_index, n_y, w_i, w_i_unc)
            cl_i = cl_dat[w_index] if cl_dat is not None else None
            if extra_weight_p1:
                w_all_i_p1, w_all_i_unc_p1 = get_walli(
                    w_index_p1, n_y, w_i_p1, w_i_unc_p1
                    )
                cl_i_both = (cl_dat[w_index_both] if cl_dat is not None
                             else None
                             )
        else:
            cl_i, cl_i_both = None, None

        # --- loop over outcomes ----------------------------------------

        for o_idx in range(no_of_out):
            if continuous:
                # continuous treatment grid
                y_dat_cont = y_dat[w_index_both, o_idx]

                for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                    if extra_weight_p1:
                        w_i_cont = w10 * w_i + w01 * w_i_p1
                        w_i_unc_cont = w10 * w_i_unc + w01 * w_i_unc_p1
                        if w_t is None:
                            w_t_cont = None
                        else:
                            w_t_cont = w10 * w_t + w01 * w_t_p1
                        cl_i_cont = cl_i_both
                    else:
                        w_i_cont = w_i
                        w_i_unc_cont = w_i_unc
                        w_t_cont = w_t
                        cl_i_cont = cl_i_both

                    # renormalize combined local weights
                    w_i_cont = w_i_cont / w_i_cont.sum()
                    if w_t_cont is not None:
                        w_t_cont = w_t_cont / w_t_cont.sum()
                    w_i_unc_cont = w_i_unc_cont / w_i_unc_cont.sum()

                    # optional bounding
                    if p_cfg.max_weight_share < 1:
                        if iv:
                            (w_i_cont, _, share_cont
                             ) = mcf_gp.bound_norm_weights_not_one_cuda(
                                 w_i_cont,
                                 p_cfg.max_weight_share,
                                 zero_tol=int_cfg.zero_tol,
                                 sum_tol=int_cfg.sum_tol,
                                 )
                        else:
                            (w_i_cont, _, share_cont,
                             ) = mcf_gp.bound_norm_weights_cuda(
                                 w_i_cont,
                                 p_cfg.max_weight_share,
                                 zero_tol=int_cfg.zero_tol,
                                 sum_tol=int_cfg.sum_tol,
                                 negative_weights_possible=p_ba_cfg.yes,
                                 )
                        if i == 0:
                            share_i[t_idx] = share_cont

                    # main IATE estimate
                    ret = mcf_est_cuda.weight_var_cuda(
                        w_i_cont,
                        y_dat_cont,
                        cl_i_cont,
                        gen_cfg,
                        p_cfg,
                        weights=w_t_cont,
                        se_yes=iate_se_flag,
                        bootstrap=se_boot_iate,
                        keep_all=int_cfg.keep_w0,
                        precision=precision,
                        normalize=not iv,
                        zero_tol=int_cfg.zero_tol,
                        sum_tol=int_cfg.sum_tol,
                        )
                    ti_idx = index_full[t_idx, i]

                    pot_y_i[ti_idx, o_idx] = ret[0]
                    if iate_se_flag and ret[1] is not None:
                        pot_y_var_i[ti_idx, o_idx] = ret[1]

                    # extra cluster-based bookkeeping
                    if cluster_std:
                        if extra_weight_p1:
                            w_cont = w10 * w_all_i + w01 * w_all_i_p1
                        else:
                            w_cont = w_all_i

                        ret2 = mcf_est_cuda.aggregate_cluster_pos_w_cuda(
                            cl_dat, w_cont, y_dat[:, o_idx],
                            sweights=w_dat,
                            precision=precision,
                            zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                            )

                        if o_idx == 0:
                            w_add[ti_idx, :] = ret2[0].clone()
                            if w_ate is None:
                                if extra_weight_p1:
                                    w_diff = (w10 * w_all_i_unc
                                              + w01 * w_all_i_unc_p1
                                              )
                                else:
                                    w_diff = w_all_i_unc
                            else:
                                if extra_weight_p1:
                                    w_ate_cont = (w10 * w_ate[t_idx, :]
                                                  + w01 * w_ate[t_idx + 1, :]
                                                  )
                                    w_ate_cont = w_ate_cont / w_ate_cont.sum()
                                    w_diff = w_all_i_unc - w_ate_cont
                                else:
                                    w_diff = w_all_i_unc - w_ate[t_idx, :]

                        if iate_m_ate_flag:
                            ret_ma = mcf_est_cuda.weight_var_cuda(
                                w_diff, y_dat[:, o_idx], cl_dat,
                                gen_cfg, p_cfg,
                                normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                keep_all=int_cfg.keep_w0,
                                precision=precision,
                                zero_tol=int_cfg.zero_tol,
                                sum_tol=int_cfg.sum_tol,
                                )
                    else:
                        if o_idx == 0:
                            w_add[ti_idx, w_index_both] = ret[2]
                            w_i_unc_sum = w_i_unc_cont.sum()
                            w_i_unc_sum_val = float(w_i_unc_sum)
                            if (not(1.0 - sum_tol < w_i_unc_sum_val
                                    < 1.0 + sum_tol
                                    ) and not iv):
                                w_add_unc[ti_idx, w_index_both] = (
                                    w_i_unc_cont / w_i_unc_sum
                                    )
                            else:
                                w_add_unc[ti_idx, w_index_both] = (
                                    w_i_unc_cont
                                    )
                            if w_ate is None:
                                w_diff = w_add_unc[ti_idx, :]
                            else:
                                if extra_weight_p1:
                                    w_ate_cont = (w10 * w_ate[t_idx, :]
                                                  + w01 * w_ate[t_idx + 1, :]
                                                  )
                                    w_ate_cont = w_ate_cont / w_ate_cont.sum()
                                    w_diff = ( w_add_unc[ti_idx, :]
                                              - w_ate_cont
                                              )
                                else:
                                    w_diff = (w_add_unc[ti_idx, :]
                                              - w_ate[t_idx, :]
                                              )
                        if iate_m_ate_flag:
                            ret_ma = mcf_est_cuda.weight_var_cuda(
                                w_diff, y_dat[:, o_idx], None,
                                gen_cfg, p_cfg,
                                normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate,
                                se_yes=iate_se_flag,
                                keep_all=int_cfg.keep_w0, precision=precision,
                                zero_tol=int_cfg.zero_tol,
                                sum_tol=int_cfg.sum_tol,
                                )
                    if iate_m_ate_flag:
                        pot_y_m_ate_i[ti_idx, o_idx] = ret_ma[0]
                    if iate_se_flag and iate_m_ate_flag:
                        if ret_ma[1] is not None:
                            pot_y_m_ate_var_i[ti_idx, o_idx] = ret_ma[1]

                    if not extra_weight_p1:
                        break

            else:
                # discrete treatment
                ret = mcf_est_cuda.weight_var_cuda(
                    w_i, y_dat[w_index, o_idx], cl_i,
                    gen_cfg, p_cfg,
                    weights=w_t, se_yes=iate_se_flag, bootstrap=se_boot_iate,
                    keep_all=int_cfg.keep_w0, precision=precision,
                    normalize=not iv,
                    zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                    )
                pot_y_i[t_idx, o_idx] = ret[0]
                if iate_se_flag and ret[1] is not None:
                    pot_y_var_i[t_idx, o_idx] = ret[1]

                if cluster_std:
                    ret2 = mcf_est_cuda.aggregate_cluster_pos_w_cuda(
                        cl_dat, w_all_i, y_dat[:, o_idx],
                        sweights=w_dat, precision=precision,
                        zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                        )
                    if o_idx == 0:
                        w_add[t_idx, :] = ret2[0].clone()
                        if w_ate is None:
                            w_diff = w_all_i_unc
                        else:
                            w_diff = w_all_i_unc - w_ate[t_idx, :]

                    if iate_m_ate_flag:
                        ret_ma = mcf_est_cuda.weight_var_cuda(
                            w_diff, y_dat[:, o_idx], cl_dat,
                            gen_cfg, p_cfg,
                            normalize=False, weights=w_dat,
                            bootstrap=se_boot_iate, se_yes=iate_se_flag,
                            keep_all=int_cfg.keep_w0, precision=precision,
                            zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                            )
                else:
                    if o_idx == 0:
                        w_add[t_idx, w_index] = ret[2]
                        w_i_unc_sum = w_i_unc.sum()
                        w_i_unc_sum_val = float(w_i_unc_sum)

                        if (not (1.0 - sum_tol < w_i_unc_sum_val
                                 < 1.0 + sum_tol)
                            and not iv):
                            w_add_unc[t_idx, w_index] = w_i_unc / w_i_unc_sum
                        else:
                            w_add_unc[t_idx, w_index] = w_i_unc

                        if w_ate is None:
                            w_diff = w_add_unc[t_idx, :]
                        else:
                            w_diff = w_add_unc[t_idx, :] - w_ate[t_idx, :]

                    if iate_m_ate_flag:
                        ret_ma = mcf_est_cuda.weight_var_cuda(
                            w_diff, y_dat[:, o_idx], None,
                            gen_cfg, p_cfg,
                            normalize=False, weights=w_dat,
                            bootstrap=se_boot_iate, se_yes=iate_se_flag,
                            keep_all=int_cfg.keep_w0, precision=precision,
                            zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                            )
                if iate_m_ate_flag:
                    pot_y_m_ate_i[t_idx, o_idx] = ret_ma[0]
                if iate_m_ate_flag and iate_se_flag:
                    if ret_ma[1] is not None:
                        pot_y_m_ate_var_i[t_idx, o_idx] = ret_ma[1]

    # --- final weight diagnostics --------------------------------------

    l1_to_9 = mcf_est_cuda.analyse_weights_cuda(
        w_add, None,
        gen_cfg, p_cfg,
        ate=False, continuous=continuous,
        no_of_treat_cont=no_of_treat_dr, d_values_cont=d_values_dr,
        precision=precision, zero_tol=zero_tol, sum_tol=sum_tol,
        )
    return (idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
            l1_to_9, share_i,
            )


def tensor_to_gpu(weights_l_np: NDArray[Any] | Sequence[Any],
                  cl_dat_np: NDArray[Any] | None,
                  w_dat_np: NDArray[Any] | None,
                  w_ate_np: NDArray[Any] | None,
                  y_dat_np: NDArray[Any],
                  weight_as_sparse: bool,
                  iate_m_ate_flag: bool,
                  precision: int = 32,
                  device: torch.device | str = 'cuda',
                  ) -> tuple[torch.Tensor | list[torch.Tensor],
                             torch.Tensor | None,
                             torch.Tensor | None,
                             torch.Tensor | None,
                             torch.Tensor,
                             ]:
    """Convert numpy arrays to torch tensors and move them to GPU."""
    def _np_to_gpu_optional(arr_np: NDArray[Any] | None,
                            ) -> torch.Tensor | None:
        if arr_np is None:
            return None

        return torch.from_numpy(arr_np).to(device=device, dtype=dtype)

    dtype = mcf_cuda.tdtype('float', precision)
    device = torch.device(device)

    # Weights
    if weight_as_sparse:
        # assuming scipy_sparse_to_torch_sparse returns a CPU sparse tensor
        weights = [scipy_sparse_to_torch_sparse(weight, precision,
                                                device=device).to(device)
                   for weight in weights_l_np
                   ]
    else:
        weights = torch.from_numpy(weights_l_np).to(device=device, dtype=dtype)

    # Cluster data (can be None)
    cl_dat = _np_to_gpu_optional(cl_dat_np)

    # Weights for data (can be None)
    w_dat = _np_to_gpu_optional(w_dat_np)

    # Outcome data (required)
    y_dat = torch.from_numpy(y_dat_np).to(device=device, dtype=dtype)

    # ATE weights (optional)
    if iate_m_ate_flag and w_ate_np is not None:
        w_ate = torch.from_numpy(w_ate_np).to(device=device, dtype=dtype)
    else:
        w_ate = None

    return weights, cl_dat, w_dat, w_ate, y_dat


def scipy_sparse_to_torch_sparse(scipy_sparse_matrix: spmatrix,
                                 precision: int = 32,
                                 device: torch.device | str = 'cuda',
                                 ) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a torch COO sparse tensor on `device`."""

    dtype = mcf_cuda.tdtype('float', precision)
    device = torch.device(device)

    # Ensure COO format (covers csr, csc, etc.)
    coo = scipy_sparse_matrix.tocoo()

    # Build indices (2 x nnz) as int64, without unnecessary copies
    indices_np = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    indices = torch.from_numpy(indices_np)

    # Values
    values = torch.from_numpy(coo.data)

    # Create sparse COO tensor and move to device
    sparse_tensor = torch.sparse_coo_tensor(indices, values,
                                            size=coo.shape,
                                            dtype=dtype, device=device,
                                            )
    return sparse_tensor


def max_batch_size(n_x: int,
                   weights: Any,
                   max_share_weights: float,
                   weight_as_sparse: bool,
                   gen_cfg: Any,
                   ) -> int:
    """Find maximum batch size for CUDA computations."""

    # in gigabytes
    in_gb = 1024 ** -3

    device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory * in_gb

    allocated_memory = torch.cuda.memory_allocated(device=device) * in_gb
    cached_memory = torch.cuda.memory_reserved(device=device) * in_gb

    # Treat cached memory as free
    free_memory = total_memory - (allocated_memory - cached_memory)

    txt = '\n' + f'Total GPU memory:        {total_memory:.3f} GB'
    txt += '\n' + f'Allocated GPU memory:    {allocated_memory:.3f} GB'
    txt += '\n' + f'Cached GPU memory:       {cached_memory:.3f} GB'
    txt += '\n' + f'Free GPU memory:         {free_memory:.3f} GB'

    weight_bytes = mcf_sys.print_size_weight_matrix(
        weights,
        weight_as_sparse,
        gen_cfg.no_of_treat,
        no_text=True,
        )
    weight_memory = weight_bytes * in_gb

    txt += f'\nGPU memory needed for weight matrix: {weight_memory:.3f} GB'

    # Guard against degenerate cases
    usable_mem = free_memory * max(max_share_weights, 1e-6)
    if usable_mem <= 0 or weight_memory <= 0:
        no_batches = 1
    else:
        no_batches = max(1, ceil(weight_memory / usable_mem))

    batch_max_size = max(1, ceil(n_x / no_batches))

    txt += '\n' + f'# of batches: {no_batches}'
    txt += ('\nGPU memory needed for one batch of weight matrix: '
            f'{weight_memory / no_batches:.2f} GB'
            )
    mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return batch_max_size
