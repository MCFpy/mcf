"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the IATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from __future__ import annotations

from math import ceil
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from mcf import mcf_cuda
from mcf import mcf_estimation_cuda as mcf_est_cuda
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats as mcf_ps
from mcf.mcf_versions_cuda import version_wregr_cuda

if TYPE_CHECKING:
    from mcf.mcf_init import GenCfg, IntCfg, CtGrid, GenTvCfg
    from mcf.mcf_init_predict import PCfg
    from torch import Tensor
    from torch import dtype as TorchDType
else:
    Tensor = Any
    TorchDType = Any

type ArrayLike = NDArray[Any] | None
type TensorLike = Tensor | None


def iate_cuda(weights_list: list[NDArray[Any] | Sequence[Any]], *,
              cl_dat_np: NDArray[Any],
              no_of_cluster: int,
              w_dat_np: NDArray[Any],
              w_ate_np: ArrayLike,
              y_dat_np: NDArray[Any],
              no_of_out: int,
              n_y: int,
              ct_cfg: 'CtGrid',
              int_cfg: 'IntCfg',
              gen_cfg: 'GenCfg',
              p_cfg: 'PCfg',
              # ba_data: Any, p_ba_cfg: PBaCfg, not yet implemented
              d_tv_train_np: NDArray[np.integer] | None,
              x_tv_train_np: ArrayLike,
              x_tv_pred_np: ArrayLike,
              tv_scores_train_np: list[NDArray[Any] | None] | None = None,
              tv_scores_pred_np: list[NDArray[Any] | None] | None = None,
              gen_tv_cfg: 'GenTvCfg',
              iate_se_flag: bool,
              se_boot_iate: bool,
              iate_m_ate_flag: bool,
              n_x: int,
              no_of_treat_dr: int,
              iv: bool = False,
              ) -> tuple[NDArray[Any],
                         ArrayLike,
                         ArrayLike,
                         ArrayLike,
                         list[list[NDArray[Any] | str]],
                         NDArray[Any],
                         ]:
    """Compute IATE and SE of IATE using GPU."""
    precision = 32
    gen_tv = gen_tv_cfg.yes
    tv_estimator = gen_tv_cfg.estimator == 'ridge'
    tv_min_subtreat = gen_tv_cfg.tv_min_subtreat
    tv_cv_k = gen_tv_cfg.cv_k

    (weights, cl_dat, w_dat, w_ate, y_dat,
     d_tv_train, x_tv_train, x_tv_pred, tv_sub_treat, tv_d_all_values,
     tv_scores_train, tv_scores_pred,
     ) = tensor_to_gpu(weights_list,
                       cl_dat_np=cl_dat_np,
                       w_dat_np=w_dat_np,
                       w_ate_np=w_ate_np,
                       y_dat_np=y_dat_np,
                       weight_as_sparse=int_cfg.weight_as_sparse,
                       iate_m_ate_flag=iate_m_ate_flag,
                       d_tv_train_np=d_tv_train_np,
                       x_tv_train_np=x_tv_train_np,
                       x_tv_pred_np=x_tv_pred_np,
                       tv_sub_treat_np=(gen_tv_cfg.d_dict['sub_treat']
                                        if d_tv_train_np is not None else None),
                       tv_d_all_values_np=gen_tv_cfg.d_all_values,
                       tv_scores_train_np=tv_scores_train_np,
                       tv_scores_pred_np=tv_scores_pred_np,
                       precision=precision,
                       )
    device = y_dat.device
    float_dtype = mcf_cuda.tdtype('float', precision)

    pot_y = torch.empty((n_x, no_of_treat_dr, no_of_out),
                        device=device,
                        dtype=float_dtype,
                        )
    if gen_tv:
        treat_main_tv = torch.tensor(gen_tv_cfg.no_of_treat_per_main,
                                     device=pot_y.device,
                                     dtype=torch.int32,
                                     )
        pot_y = pot_y.repeat_interleave(treat_main_tv, dim=1)
    else:
        treat_main_tv = None

    pot_y_m_ate = torch.empty_like(pot_y) if iate_m_ate_flag else None
    pot_y_var = torch.empty_like(pot_y) if iate_se_flag else None
    pot_y_m_ate_var = torch.empty_like(pot_y) if (iate_se_flag and iate_m_ate_flag) else None
    share_censored = torch.zeros(pot_y.shape[1], device=device, dtype=float_dtype)
    l1_to_9: list[list[torch.Tensor | str] | None] = [None for _ in range(n_x)]

    if gen_cfg.d_type == 'continuous':
        d_values_dr = torch.from_numpy(ct_cfg.d_values_dr_np).to(device)
    else:
        d_values_dr = torch.tensor(gen_cfg.d_values,
                                   device=device,
                                   dtype=mcf_cuda.tdtype('int', precision),
                                   )
    output_yes = gen_cfg.with_output and gen_cfg.verbose
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, output_yes)

    idx_iter = range(n_x)
    if use_tqdm:
        idx_iter = tqdm(idx_iter, total=n_x, desc='IATE CUDA')
    for idx in idx_iter:
        if int_cfg.weight_as_sparse:
            weights_idx = [weights_tx[idx] for weights_tx in weights]
        else:
            weights_idx = weights[idx]

        if gen_tv and tv_scores_pred is not None:
            tv_scores_pred_idx = [None if score is None else score[idx, :].reshape(1, -1)
                                  for score in tv_scores_pred
                                  ]
        else:
            tv_scores_pred_idx = None

        ret_all_i = iate_func1_for_cuda(
            idx=idx,
            weights_i=weights_idx,
            cl_dat=cl_dat,
            no_of_cluster=no_of_cluster,
            w_dat=w_dat,
            w_ate=w_ate,
            y_dat=y_dat,
            no_of_out=no_of_out,
            n_y=n_y,
            ct_cfg=ct_cfg,
            int_cfg=int_cfg,
            gen_cfg=gen_cfg,
            p_cfg=p_cfg,
            gen_tv=gen_tv,
            d_tv_train=d_tv_train,
            x_tv_train=x_tv_train,
            x_tv_pred=(x_tv_pred[idx, :].reshape(1, -1)
                       if gen_tv and x_tv_pred is not None else None),
            tv_scores_train=tv_scores_train,
            tv_scores_pred=tv_scores_pred_idx,
            gen_tv_cfg=gen_tv_cfg,
            treat_main_tv=treat_main_tv,
            tv_estimator=tv_estimator,
            tv_sub_treat=tv_sub_treat,
            tv_d_all_values=tv_d_all_values,
            tv_min_subtreat=tv_min_subtreat,
            tv_cv_k=tv_cv_k,
            tv_penalize_version=gen_tv_cfg.penalize_version,
            iate_se_flag=iate_se_flag,
            se_boot_iate=se_boot_iate,
            iate_m_ate_flag=iate_m_ate_flag,
            d_values_dr=d_values_dr,
            precision=precision,
            zero_tol=int_cfg.zero_tol,
            sum_tol=int_cfg.sum_tol,
            iv=iv,
            )
        (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9, share_censored,
         ) = assign_ret_all_i_cuda(pot_y=pot_y,
                                   pot_y_var=pot_y_var,
                                   pot_y_m_ate=pot_y_m_ate,
                                   pot_y_m_ate_var=pot_y_m_ate_var,
                                   l1_to_9=l1_to_9,
                                   share_censored=share_censored,
                                   ret_all_i=ret_all_i,
                                   n_x=n_x,
                                   idx=idx,
                                   )
        mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx + 1, total=n_x)

    (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np, share_censored_np, l1_to_9_np,
     ) = gpu_to_numpy(pot_y=pot_y,
                      pot_y_var=pot_y_var,
                      pot_y_m_ate=pot_y_m_ate,
                      pot_y_m_ate_var=pot_y_m_ate_var,
                      share_censored=share_censored,
                      l1_to_9=l1_to_9,
                      )
    return (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np, l1_to_9_np,
            share_censored_np,
            )


def gpu_to_numpy(*, pot_y: torch.Tensor,
                 pot_y_var: TensorLike,
                 pot_y_m_ate: TensorLike,
                 pot_y_m_ate_var: TensorLike,
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

    def to_numpy(x: TensorLike):
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

    return (pot_y_np, pot_y_var_np, pot_y_m_ate_np, pot_y_m_ate_var_np, share_censored_np,
            l1_to_9_list,
            )


def assign_ret_all_i_cuda(*, pot_y: torch.Tensor,
                          pot_y_var: TensorLike,
                          pot_y_m_ate: TensorLike,
                          pot_y_m_ate_var: TensorLike,
                          l1_to_9: list[tuple[torch.Tensor | str, ...] | None],
                          share_censored: torch.Tensor,
                          ret_all_i: Any,
                          n_x: int,
                          idx: int | None = None,
                          ) -> tuple[torch.Tensor,
                                     TensorLike,
                                     TensorLike,
                                     TensorLike,
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

    return pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9, share_censored


def iate_func1_for_cuda(*, idx: int,
                        weights_i: list[Any] | torch.Tensor,
                        cl_dat: TensorLike,
                        no_of_cluster: int | None,
                        w_dat: TensorLike,
                        w_ate: TensorLike,
                        y_dat: torch.Tensor,
                        no_of_out: int,
                        n_y: int,
                        ct_cfg: 'CtGrid',
                        int_cfg: 'IntCfg',
                        gen_cfg: 'GenCfg',
                        p_cfg: 'PCfg',
                        gen_tv_cfg: 'GenTvCfg',
                        # p_ba_cfg: 'PBaCfg',
                        gen_tv: bool = False,
                        d_tv_train: TensorLike = None,
                        x_tv_train: TensorLike = None,
                        x_tv_pred: TensorLike = None,
                        tv_scores_train: list[TensorLike] | None = None,
                        tv_scores_pred: list[TensorLike] | None = None,
                        treat_main_tv: TensorLike = None,
                        tv_estimator: bool = True,
                        tv_sub_treat: TensorLike = None,
                        tv_d_all_values: TensorLike = None,
                        tv_min_subtreat: int | None = 10,
                        tv_cv_k: int | None = 5,
                        tv_penalize_version: bool | list[bool] = False,
                        iate_se_flag: bool = False,
                        se_boot_iate: int = 0,
                        iate_m_ate_flag: bool = False,
                        d_values_dr: torch.Tensor | None = None,
                        precision: int = 32,
                        iv: bool = False,
                        sum_tol: float = 1e-8,
                        zero_tol: float = 1e-10,
                        ) -> tuple[int,
                                   torch.Tensor,
                                   TensorLike,
                                   TensorLike,
                                   TensorLike,
                                   tuple[torch.Tensor | str, ...],
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
    int_dtype = mcf_cuda.tdtype('float', int(precision * 2))
    out_dtype = mcf_cuda.tdtype('float', precision)

    ret_ma = None

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

    pot_y_i = torch.empty((no_of_treat_dr, no_of_out), device=device, dtype=dtype)
    if gen_tv:
        pot_y_i = pot_y_i.repeat_interleave(treat_main_tv, dim=0)

    pot_y_m_ate_i = torch.empty_like(pot_y_i) if iate_m_ate_flag else None
    share_i = torch.zeros(pot_y_i.shape[0], device=device, dtype=dtype)

    pot_y_var_i = torch.empty_like(pot_y_i) if iate_se_flag else None
    pot_y_m_ate_var_i = torch.empty_like(pot_y_i) if iate_se_flag and iate_m_ate_flag else None
    cluster_std = p_cfg.cluster_std if iate_se_flag else False

    def weight_var_iate(w_i: torch.Tensor,
                        w_index: torch.Tensor,
                        y_col: torch.Tensor,
                        cl_i: TensorLike,
                        residual_dat: TensorLike, *,
                        y_local: TensorLike = None,
                        ) -> tuple[torch.Tensor, TensorLike, torch.Tensor]:
        """Keep the full variance sample and return weights in caller order."""
        expand = int_cfg.keep_w0 and iate_se_flag
        if expand:
            w_use = w_i.new_zeros((n_y,))
            w_use[w_index] = w_i
            y_use = y_col
            cl_use = cl_dat if cluster_std else None
            if residual_dat is None:
                r_use = None
            else:
                r_use = residual_dat.new_zeros((n_y,))
                r_use[w_index] = residual_dat.reshape(-1)
        else:
            w_use = w_i
            y_use = y_col[w_index] if y_local is None else y_local
            cl_use = cl_i
            r_use = residual_dat

        est, variance, w_ret = mcf_est_cuda.weight_var_cuda(
            w_use, y_use, cl_use, p_cfg,
            residual_dat=r_use, se_yes=iate_se_flag, bootstrap=se_boot_iate,
            keep_all=int_cfg.keep_w0, precision=precision, normalize=not iv,
            zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol, seed=123345, min_obs=5,
            )
        if expand and not (cluster_std and cl_use is not None and se_boot_iate < 1):
            w_ret = w_ret[w_index]

        return est, variance, w_ret

    if cluster_std:
        if no_of_cluster is None:
            raise ValueError('no_of_cluster must be provided when cluster_std is True.')
        w_add = torch.zeros((pot_y_i.shape[0], no_of_cluster), device=device, dtype=dtype)
    else:
        w_add = torch.zeros((pot_y_i.shape[0], n_y), device=device, dtype=dtype)

    w_add_unc = torch.zeros((pot_y_i.shape[0], n_y), device=device, dtype=dtype)

    no_of_m_treat = no_of_treat
    no_treat_per_main = [1] * no_of_m_treat if not gen_tv else treat_main_tv

    if gen_tv:
        if len(tv_penalize_version) == 1:
            tv_penalize_version = tv_penalize_version * no_of_m_treat
        tv_penalize_version_cuda = torch.tensor(tv_penalize_version, device=device, dtype=torch.bool
                                                )
    cl_i_both: torch.Tensor | None = None
    t_idx = 0

    for t_m_idx in range(no_of_m_treat):
        extra_weight_p1 = continuous and (t_m_idx < no_of_m_treat - 1)

        if int_cfg.weight_as_sparse:
            coalesced_tensor = weights_i[t_m_idx].coalesce()
            w_index = coalesced_tensor.indices().reshape(-1)
            w_i_t = coalesced_tensor.values()
        else:
            w_index = weights_i[t_m_idx][0]
            w_i_t = weights_i[t_m_idx][1].clone()

        if extra_weight_p1:
            if int_cfg.weight_as_sparse:
                coalesced_tensor_p1 = weights_i[t_m_idx + 1].coalesce()
                w_index_p1 = coalesced_tensor_p1.indices().reshape(-1)
            else:
                w_index_p1 = weights_i[t_m_idx + 1][0]

            w_index_both = torch.unique(torch.cat((w_index, w_index_p1)))

            w_i = torch.zeros(n_y, device=device, dtype=dtype)
            w_i[w_index] = w_i_t

            w_i_p1 = torch.zeros_like(w_i)
            if int_cfg.weight_as_sparse:
                w_i_p1[w_index_p1] = coalesced_tensor_p1.values()
            else:
                w_i_p1[w_index_p1] = weights_i[t_m_idx + 1][1].clone()

            w_i = w_i[w_index_both]
            w_i_p1 = w_i_p1[w_index_both]
        else:
            w_index_both = w_index
            w_i = w_i_t

        if gen_cfg.weighted and w_dat is not None:
            weight_index = w_index_both if extra_weight_p1 else w_index
            w_t = w_dat[weight_index].reshape(-1)
            w_i = w_i * w_t

            if extra_weight_p1:
                w_i_p1 = w_i_p1 * w_t

        w_i_sum = w_i.sum()
        w_i_sum_val = float(w_i_sum)
        if not (1.0 - sum_tol < w_i_sum_val < 1.0 + sum_tol) and not (continuous or iv):
            w_i = w_i / w_i_sum

        results_container_w, results_container_r = None, None

        for t_sub_idx in range(no_treat_per_main[t_m_idx]):
            if gen_tv:
                scores_train_t = tv_scores_train[t_m_idx] if tv_scores_train is not None else None
                scores_pred_t = tv_scores_pred[t_m_idx] if tv_scores_pred is not None else None

                (w_i, results_container_w, version_res_dat, results_container_r, _, _, _,
                 ) = version_wregr_cuda(w_i,
                                        y_train=y_dat,
                                        d_train=d_tv_train,
                                        x_train=x_tv_train,
                                        x_pred=x_tv_pred,
                                        d_sub_only=tv_sub_treat[t_m_idx],
                                        d_all_values=tv_d_all_values,
                                        tv_min_subtreat=tv_min_subtreat,
                                        cv_k=tv_cv_k,
                                        container_w=results_container_w,
                                        container_r=results_container_r,
                                        w_index=w_index,
                                        treat_idx=t_idx,
                                        maintreat_idx=t_m_idx,
                                        subtreat_idx=t_sub_idx,
                                        int_dtype=int_dtype,
                                        out_dtype=out_dtype,
                                        zero_tol=int_cfg.zero_tol,
                                        ridge=tv_estimator,
                                        penalize_version=tv_penalize_version_cuda[t_m_idx],
                                        return_residuals=True,
                                        standardize_x=True,
                                        interacted=gen_tv_cfg.specification == 'interacted',
                                        cfg=gen_tv_cfg,
                                        scores_train=scores_train_t,
                                        scores_pred=scores_pred_t,
                                        allow_local_version_scores=False,
                                        use_train_score_cache=True,
                                        )
            else:
                version_res_dat = None

            w_i_unc = w_i.clone()

            if p_cfg.max_weight_share < 1 and not continuous:
                denom_abs = w_i.abs().sum()
                if denom_abs > sum_tol:
                    max_share_now = w_i.abs().max() / denom_abs
                    if max_share_now > p_cfg.max_weight_share + zero_tol:
                        if iv:
                            w_i, _, share_i[t_idx] = (
                                mcf_gp.bound_norm_weights_not_one_cuda(w_i,
                                                                       p_cfg.max_weight_share,
                                                                       zero_tol=int_cfg.zero_tol,
                                                                       sum_tol=int_cfg.sum_tol,
                                                                       )
                                )
                        else:
                            w_i, _, share_i[t_idx] = mcf_gp.bound_norm_weights_cuda(
                                w_i,
                                max_weight_share=p_cfg.max_weight_share,
                                zero_tol=int_cfg.zero_tol,
                                sum_tol=int_cfg.sum_tol,
                                negative_weights_possible=gen_tv,
                                )
                    else:
                        share_i[t_idx] = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    share_i[t_idx] = torch.tensor(0.0, device=device, dtype=dtype)

            if extra_weight_p1:
                w_i_unc_p1 = w_i_p1.clone()

            if cluster_std:
                weight_index = w_index_both if extra_weight_p1 else w_index
                w_all_i, w_all_i_unc = get_walli(
                    weight_index, n_y, w_i, w_i_unc)
                cl_i = cl_dat[weight_index]

                if extra_weight_p1:
                    w_all_i_p1, w_all_i_unc_p1 = get_walli(
                        w_index_both, n_y, w_i_p1, w_i_unc_p1)
                    cl_i_both = cl_dat[w_index_both]
                else:
                    cl_i_both = cl_i
            else:
                cl_i, cl_i_both = None, None

            for o_idx in range(no_of_out):
                if version_res_dat is None or version_res_dat.ndim == 1:
                    vres = version_res_dat
                else:
                    vres = version_res_dat[:, o_idx]
                if continuous:
                    y_dat_cont = y_dat[w_index_both, o_idx]

                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        if extra_weight_p1:
                            w_i_cont = w10 * w_i + w01 * w_i_p1
                            w_i_unc_cont = w10 * w_i_unc + w01 * w_i_unc_p1
                            cl_i_cont = cl_i_both
                        else:
                            w_i_cont = w_i
                            w_i_unc_cont = w_i_unc
                            cl_i_cont = cl_i_both

                        w_i_cont = w_i_cont / w_i_cont.sum()
                        w_i_unc_cont = w_i_unc_cont / w_i_unc_cont.sum()

                        if p_cfg.max_weight_share < 1:
                            if iv:
                                w_i_cont, _, share_cont = (
                                    mcf_gp.bound_norm_weights_not_one_cuda(
                                        w_i_cont,
                                        p_cfg.max_weight_share,
                                        zero_tol=int_cfg.zero_tol,
                                        sum_tol=int_cfg.sum_tol,
                                        )
                                    )
                            else:
                                w_i_cont, _, share_cont = (
                                    mcf_gp.bound_norm_weights_cuda(
                                        w_i_cont,
                                        max_weight_share=p_cfg.max_weight_share,
                                        zero_tol=int_cfg.zero_tol,
                                        sum_tol=int_cfg.sum_tol,
                                        negative_weights_possible=gen_tv,
                                        )
                                    )
                            if i == 0:
                                share_i[t_idx] = share_cont

                        ret = weight_var_iate(w_i_cont, w_index_both,
                                              y_dat[:, o_idx], cl_i_cont, vres,
                                              y_local=y_dat_cont,
                                              )
                        ti_idx = index_full[t_idx, i]
                        pot_y_i[ti_idx, o_idx] = ret[0]
                        if pot_y_var_i is not None and ret[1] is not None:
                            # pylint: disable-next=unsupported-assignment-operation
                            pot_y_var_i[ti_idx, o_idx] = ret[1]

                        if cluster_std:
                            if extra_weight_p1:
                                w_cont = w10 * w_all_i + w01 * w_all_i_p1
                            else:
                                w_cont = w_all_i

                            ret2 = mcf_est_cuda.aggregate_cluster_nonzero_w_cuda(
                                cl_dat,
                                w_cont,
                                y_dat=y_dat[:, o_idx],
                                sweights=w_dat,
                                precision=precision,
                                zero_tol=int_cfg.zero_tol,
                                sum_tol=int_cfg.sum_tol,
                                )
                            if o_idx == 0:
                                w_add[ti_idx, :] = ret2[0].clone()
                                if w_ate is None:
                                    if extra_weight_p1:
                                        w_diff = w10 * w_all_i_unc + w01 * w_all_i_unc_p1
                                    else:
                                        w_diff = w_all_i_unc
                                else:
                                    if extra_weight_p1:
                                        w_ate_cont = (w10 * w_ate[t_idx, :]
                                                      + w01 * w_ate[t_idx + 1, :]
                                                      )
                                        w_ate_cont = w_ate_cont / w_ate_cont.sum()
                                        w_diff = (w10 * w_all_i_unc
                                                  + w01 * w_all_i_unc_p1
                                                  - w_ate_cont
                                                  )
                                    else:
                                        w_diff = w_all_i_unc - w_ate[t_idx, :]

                            if iate_m_ate_flag:
                                ret_ma = mcf_est_cuda.weight_var_cuda(
                                    w_diff,
                                    y_dat[:, o_idx],
                                    cl_dat,
                                    p_cfg,
                                    residual_dat=None,
                                    normalize=False,
                                    bootstrap=se_boot_iate,
                                    se_yes=iate_se_flag,
                                    keep_all=int_cfg.keep_w0,
                                    precision=precision,
                                    zero_tol=int_cfg.zero_tol,
                                    sum_tol=int_cfg.sum_tol,
                                    seed=123345,
                                    min_obs=5,
                                    )
                        else:
                            if o_idx == 0:
                                w_add[ti_idx, w_index_both] = ret[2]
                                w_i_unc_sum = w_i_unc_cont.sum()
                                w_i_unc_sum_val = float(w_i_unc_sum)

                                if (not (1.0 - sum_tol < w_i_unc_sum_val < 1.0 + sum_tol)
                                        and not iv):
                                    w_add_unc[ti_idx, w_index_both] = w_i_unc_cont / w_i_unc_sum
                                else:
                                    w_add_unc[ti_idx, w_index_both] = w_i_unc_cont

                                if w_ate is None:
                                    w_diff = w_add_unc[ti_idx, :]
                                else:
                                    if extra_weight_p1:
                                        w_ate_cont = (w10 * w_ate[t_idx, :]
                                                      + w01 * w_ate[t_idx + 1, :]
                                                      )
                                        w_ate_cont = w_ate_cont / w_ate_cont.sum()
                                        w_diff = w_add_unc[ti_idx, :] - w_ate_cont
                                    else:
                                        w_diff = w_add_unc[ti_idx, :] - w_ate[t_idx, :]

                            if iate_m_ate_flag:
                                ret_ma = mcf_est_cuda.weight_var_cuda(
                                    w_diff,
                                    y_dat[:, o_idx],
                                    None,
                                    p_cfg,
                                    residual_dat=None,
                                    normalize=False,
                                    bootstrap=se_boot_iate,
                                    se_yes=iate_se_flag,
                                    keep_all=int_cfg.keep_w0,
                                    precision=precision,
                                    zero_tol=int_cfg.zero_tol,
                                    sum_tol=int_cfg.sum_tol,
                                    seed=123345,
                                    min_obs=5,
                                )
                        if pot_y_m_ate_i is not None:
                            pot_y_m_ate_i[ti_idx, o_idx] = ret_ma[0]
                        if pot_y_m_ate_var_i is not None and ret_ma[1] is not None:
                            # pylint: disable-next=unsupported-assignment-operation
                            pot_y_m_ate_var_i[ti_idx, o_idx] = ret_ma[1]
                        if not extra_weight_p1:
                            break

                else:
                    ret = weight_var_iate(w_i, w_index, y_dat[:, o_idx], cl_i, vres)
                    pot_y_i[t_idx, o_idx] = ret[0]
                    if pot_y_var_i is not None and ret[1] is not None:
                        # pylint: disable-next=unsupported-assignment-operation
                        pot_y_var_i[t_idx, o_idx] = ret[1]

                    if cluster_std:
                        ret2 = mcf_est_cuda.aggregate_cluster_nonzero_w_cuda(
                            cl_dat,
                            w_all_i,
                            y_dat=y_dat[:, o_idx],
                            sweights=w_dat,
                            precision=precision,
                            zero_tol=int_cfg.zero_tol,
                            sum_tol=int_cfg.sum_tol,
                            )
                        if o_idx == 0:
                            w_add[t_idx, :] = ret2[0].clone()
                            if w_ate is None:
                                w_diff = w_all_i_unc
                            else:
                                w_diff = w_all_i_unc - w_ate[t_idx, :]

                        if iate_m_ate_flag:
                            ret_ma = mcf_est_cuda.weight_var_cuda(w_diff,
                                                                  y_dat[:, o_idx],
                                                                  cl_dat,
                                                                  p_cfg,
                                                                  residual_dat=None,
                                                                  normalize=False,
                                                                  bootstrap=se_boot_iate,
                                                                  se_yes=iate_se_flag,
                                                                  keep_all=int_cfg.keep_w0,
                                                                  precision=precision,
                                                                  zero_tol=int_cfg.zero_tol,
                                                                  sum_tol=int_cfg.sum_tol,
                                                                  seed=123345,
                                                                  min_obs=5,
                                                                  )
                    else:
                        if o_idx == 0:
                            w_add[t_idx, w_index] = ret[2]
                            w_i_unc_sum = w_i_unc.sum()
                            w_i_unc_sum_val = float(w_i_unc_sum)

                            if (not (1.0 - sum_tol < w_i_unc_sum_val < 1.0 + sum_tol)
                                    and not iv):
                                w_add_unc[t_idx, w_index] = w_i_unc / w_i_unc_sum
                            else:
                                w_add_unc[t_idx, w_index] = w_i_unc

                            if w_ate is None:
                                w_diff = w_add_unc[t_idx, :]
                            else:
                                w_diff = w_add_unc[t_idx, :] - w_ate[t_idx, :]

                        if iate_m_ate_flag:
                            ret_ma = mcf_est_cuda.weight_var_cuda(w_diff,
                                                                  y_dat[:, o_idx],
                                                                  None,
                                                                  p_cfg,
                                                                  residual_dat=None,
                                                                  normalize=False,
                                                                  bootstrap=se_boot_iate,
                                                                  se_yes=iate_se_flag,
                                                                  keep_all=int_cfg.keep_w0,
                                                                  precision=precision,
                                                                  zero_tol=int_cfg.zero_tol,
                                                                  sum_tol=int_cfg.sum_tol,
                                                                  seed=123345,
                                                                  min_obs=5,
                                                                  )
                    if pot_y_m_ate_i is not None:
                        pot_y_m_ate_i[t_idx, o_idx] = ret_ma[0]
                    if pot_y_m_ate_var_i is not None and ret_ma[1] is not None:
                        # pylint: disable-next=unsupported-assignment-operation
                        pot_y_m_ate_var_i[t_idx, o_idx] = ret_ma[1]

            t_idx += 1

    l1_to_9 = mcf_est_cuda.analyse_weights_cuda(
        w_add,
        None,
        gen_cfg,
        p_cfg,
        gen_tv_total_treat=t_idx if gen_tv else None,
        ate=False,
        continuous=continuous,
        no_of_treat_cont=no_of_treat_dr,
        d_values_cont=d_values_dr,
        precision=precision,
        zero_tol=zero_tol,
        sum_tol=sum_tol,
        )
    return idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i, l1_to_9, share_i


def tensor_to_gpu(weights_l_np: NDArray[Any] | Sequence[Any], *,
                  cl_dat_np: ArrayLike,
                  w_dat_np: ArrayLike,
                  w_ate_np: ArrayLike,
                  y_dat_np: NDArray[Any],
                  weight_as_sparse: bool,
                  iate_m_ate_flag: bool,
                  d_tv_train_np: ArrayLike = None,
                  x_tv_train_np: ArrayLike = None,
                  x_tv_pred_np: ArrayLike = None,
                  tv_sub_treat_np: list[int] | tuple[int, ...] | ArrayLike = None,
                  tv_d_all_values_np: list[int] | tuple[int, ...] | ArrayLike = None,
                  tv_scores_train_np: list[NDArray[Any] | None] | None = None,
                  tv_scores_pred_np: list[NDArray[Any] | None] | None = None,
                  precision: int = 32,
                  device: torch.device | str = 'cuda',
                  ) -> tuple[torch.Tensor | list[torch.Tensor],
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             TensorLike,
                             list[TensorLike] | None,
                             list[TensorLike] | None,
                             ]:
    """Convert numpy arrays to torch tensors and move them to GPU."""
    def _np_to_gpu_optional(arr_np: NDArray[Any] | None, dtype: Any) -> TensorLike:
        if arr_np is None:
            return None

        return torch.from_numpy(arr_np).to(device=device, dtype=dtype)

    def _np_list_to_gpu_optional(arr_list: list[NDArray[Any] | None] | None,
                                 dtype: Any,
                                 ) -> list[TensorLike] | None:
        if arr_list is None:
            return None

        return [None if arr is None else torch.from_numpy(arr).to(device=device, dtype=dtype,)
                for arr in arr_list
                ]
    dfloattype = mcf_cuda.tdtype('float', precision)
    dinttype = mcf_cuda.tdtype('int', precision)
    device = torch.device(device)

    if weight_as_sparse:
        weights = [scipy_sparse_to_torch_sparse(weight, precision, device=device).to(device)
                   for weight in weights_l_np
                   ]
    else:
        weights = torch.from_numpy(weights_l_np).to(device=device, dtype=dfloattype)

    # Keep the original label dtype; None prevents a dtype conversion.
    cl_dat = _np_to_gpu_optional(cl_dat_np, None)
    w_dat = _np_to_gpu_optional(w_dat_np, dfloattype)
    y_dat = torch.from_numpy(y_dat_np).to(device=device, dtype=dfloattype)

    if iate_m_ate_flag and w_ate_np is not None:
        w_ate = torch.from_numpy(w_ate_np).to(device=device, dtype=dfloattype)
    else:
        w_ate = None

    if d_tv_train_np is not None:
        d_tv_train = torch.from_numpy(d_tv_train_np).to(device=device, dtype=dinttype)
        x_tv_train = (torch.from_numpy(x_tv_train_np).to(device=device, dtype=dfloattype)
                      if x_tv_train_np is not None else None
                      )
        x_tv_pred = (torch.from_numpy(x_tv_pred_np).to(device=device, dtype=dfloattype)
                     if x_tv_pred_np is not None else None
                     )

        if isinstance(tv_sub_treat_np, (list, tuple)):
            tv_sub_treat = [
                torch.from_numpy(np.array(x)).to(device=device, dtype=dinttype)
                for x in tv_sub_treat_np
            ]
        else:
            tv_sub_treat = torch.from_numpy(tv_sub_treat_np).to(device=device, dtype=dinttype)

        if isinstance(tv_d_all_values_np, (list, tuple)):
            tv_d_all_values_np = np.array(tv_d_all_values_np)
        tv_d_all_values = torch.from_numpy(tv_d_all_values_np).to(device=device, dtype=dinttype)
    else:
        d_tv_train = x_tv_train = x_tv_pred = tv_sub_treat = None
        tv_d_all_values = None

    tv_scores_train = _np_list_to_gpu_optional(tv_scores_train_np, dfloattype)
    tv_scores_pred = _np_list_to_gpu_optional(tv_scores_pred_np, dfloattype)

    return (weights, cl_dat, w_dat, w_ate, y_dat, d_tv_train, x_tv_train,
            x_tv_pred, tv_sub_treat, tv_d_all_values, tv_scores_train, tv_scores_pred,
            )

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
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=coo.shape, dtype=dtype,
                                            device=device,
                                            )
    return sparse_tensor


def max_batch_size(n_x: int,
                   weights: Any,
                   max_share_weights: float,
                   weight_as_sparse: bool,
                   gen_cfg: 'GenCfg',
                   ) -> int:
    """Find maximum batch size for CUDA computations."""
    # in gigabytes
    in_gb = 1024 ** -3

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA batch-size calculation requested, but torch.cuda.is_available() '
                           'is False.'
                           )
    no_devices = torch.cuda.device_count()
    if no_devices < 1:
        raise RuntimeError('CUDA batch-size calculation requested, but PyTorch sees no CUDA '
                           'devices.'
                           )
    device_note = ''
    try:
        device = torch.cuda.current_device()
    except AssertionError:
        device = 0
        torch.cuda.set_device(device)
        device_note = ('\nInvalid current CUDA device detected. Resetting CUDA device to 0.'
                       )
    if not 0 <= device < no_devices:
        device_note = (f'\nInvalid current CUDA device id {device}. '
                       'Resetting CUDA device to 0.'
                       )
        device = 0
        torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory * in_gb

    allocated_memory = torch.cuda.memory_allocated(device=device) * in_gb
    reserved_memory = torch.cuda.memory_reserved(device=device) * in_gb
    reusable_memory = max(reserved_memory - allocated_memory, 0)

    try:
        free_driver_bytes, _ = torch.cuda.mem_get_info(device)
        free_driver_memory = free_driver_bytes * in_gb

        # Driver-free memory excludes PyTorch-reserved cached memory. Since cached
        # memory is reusable by PyTorch, add it back for this batch-size heuristic.
        free_memory = free_driver_memory + reusable_memory
    except RuntimeError:
        free_driver_memory = total_memory - reserved_memory

        # Fallback if cudaMemGetInfo is not available for some reason.
        free_memory = total_memory - allocated_memory

    free_memory = max(free_memory, 0)

    txt = '\n' + f'CUDA device:             {device} ({props.name})'
    txt += device_note
    txt += '\n' + f'Total GPU memory:        {total_memory:.3f} GB'
    txt += '\n' + f'Allocated GPU memory:    {allocated_memory:.3f} GB'
    txt += '\n' + f'Reserved GPU memory:     {reserved_memory:.3f} GB'
    txt += '\n' + f'Reusable cached memory:  {reusable_memory:.3f} GB'
    txt += '\n' + f'Driver-free GPU memory:  {free_driver_memory:.3f} GB'
    txt += '\n' + f'Usable free GPU memory:  {free_memory:.3f} GB'

    weight_bytes = mcf_sys.print_size_weight_matrix(weights,
                                                    weight_as_sparse,
                                                    gen_cfg.no_of_treat,
                                                    no_text=True,
                                                    )
    weight_memory = weight_bytes * in_gb

    txt += f'\nGPU memory needed for weight matrix: {weight_memory:.3f} GB'

    # Guard against degenerate cases.
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
