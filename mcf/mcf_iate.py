"""
Created on Mon Jun 19 15:51:30 2023.

@author: MLechner

Contains the functions needed for computing the IATE.

@author: MLechner
-*- coding: utf-8 -*-
"""
from collections import namedtuple
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd
try:
    import ray
except (ImportError, OSError):
    ray = None
try:
    from tqdm.auto import tqdm
except (ImportError, OSError):
    tqdm = None

from mcf import mcf_bias_adjustment as mcf_ba
from mcf import mcf_cuda
from mcf import mcf_estimation as mcf_est
from mcf import mcf_iate_cuda
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats as mcf_ps
from mcf.mcf_versions import get_tv_data, version_wregr
from mcf import mcfoptp_parallel_backend_ray_classical as mcf_ray
from mcf.mcfoptp_parallel_backend_forest_executor import (forest_executor_with_shared,
                                                          map_task_batches, forest_executor_context,
                                                          )
from mcf.mcfoptp_parallel_backends_base import TaskSpec, print_runtime_info


type ArrayLike = NDArray[Any] | None

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg, IntCfg, CtGrid, PBaCfg, GenTvCfg
    from mcf.mcf_init_predict import PCfg


def iate_est_mp(
        mcf_: 'ModifiedCausalForest',
        weights_dic: dict,
        w_ate: NDArray[Any], *,
        reg_round: bool = True,
        iv_scaling: bool = False,
        iv: bool = False,
        x_tv_df: pd.DataFrame | None = None,
        x_tv_pred_np: NDArray[Any] | None = None,
        print_progress: bool = True,
        ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], str]:
    """
    Estimate IATE and their standard errors, MP version.

    Parameters
    ----------
    mcf_ : mcf object.

    weights_dic : Dict.
              Contains weights and numpy data.
    balancing_test: Boolean.
              Default is False.
    reg_round : Boolean.
              First round of estimation

    Returns
    -------
    pot_y : Numpy array. Potential outcomes.
    pot_y_var : Numpy array. Variance of potential outcomes.

    """
    def warn_text_to_console() -> None:
        print('If prediction file is large, this step may take long. If '
              'nothing seems to happen, it may be worth to try do the '
              'estimation without using "sparse" weight matrix. This needs '
              'more memory, but could be substantially faster '
              '(int_weight_as_sparse = False).'
              )

    p_cfg, int_cfg, gen_cfg = mcf_.p_cfg, mcf_.int_cfg, mcf_.gen_cfg
    var_cfg, ct_cfg = mcf_.var_cfg, mcf_.ct_cfg
    p_ba_cfg, gen_tv_cfg = mcf_.p_ba_cfg, mcf_.gen_tv_cfg
    zero_tol = int_cfg.zero_tol
    progress = gen_cfg.with_output and gen_cfg.verbose and print_progress
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, progress)

    if reg_round and not iv_scaling:
        iate_se_flag, se_boot_iate = p_cfg.iate_se, p_cfg.se_boot_iate
        iate_m_ate_flag = p_cfg.iate_m_ate
    else:
        iate_se_flag = se_boot_iate = iate_m_ate_flag = False
    if gen_cfg.with_output and gen_cfg.verbose:
        print('\nComputing IATEs 1/2 (potential outcomes)')
    weights, y_dat = weights_dic['weights'], weights_dic['y_dat_np']
    w_dat = weights_dic['w_dat_np'] if gen_cfg.weighted else None
    if p_cfg.cluster_std and iate_se_flag:
        cl_dat = weights_dic['cl_dat_np']
        no_of_cluster = len(np.unique(cl_dat))
    else:
        no_of_cluster = cl_dat = None

    if p_ba_cfg.yes:
        # Extract information needed for bias adjustment
        ba_data = mcf_ba.get_ba_data_prediction(weights_dic, p_ba_cfg)
    else:
        ba_data = None

    if gen_tv_cfg.yes:
        d_train, x_tv_train, x_tv_pred = get_tv_data(x_tv_df, weights_dic, mcf_.var_cfg,
                                                     zero_tol=zero_tol, x_tv_pred_np=x_tv_pred_np,
                                                     )
        # Namedtuple is easy to pass through all function and access it
        Tv_data = namedtuple('TV_data', ['d_train', 'x_tv_train', 'x_tv_pred'])
        gen_tv_data = Tv_data(d_train, x_tv_train, x_tv_pred)
    else:
        gen_tv_data = None

    n_x = weights[0].shape[0] if int_cfg.weight_as_sparse else len(weights)
    n_y, no_of_out = len(y_dat), len(var_cfg.y_name)
    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        no_of_treat_dr, d_values_dr = no_of_treat, d_values

    pot_y = np.empty((n_x, no_of_treat_dr, no_of_out))
    nonzero = np.zeros(no_of_treat_dr)
    share_largest_q = np.zeros((no_of_treat_dr, 3))
    sum_larger = np.zeros((no_of_treat_dr, len(p_cfg.q_w)))

    if gen_tv_cfg.yes:
        # Add additional treatments
        treat_main = gen_tv_cfg.no_of_treat_per_main
        pot_y = np.repeat(pot_y, repeats=treat_main, axis=1)
        nonzero = np.repeat(nonzero, repeats=treat_main, axis=0)
        sum_larger = np.repeat(sum_larger, repeats=treat_main, axis=0)
        share_largest_q = np.repeat(share_largest_q, repeats=treat_main, axis=0)

    pot_y_m_ate = np.empty_like(pot_y) if iate_m_ate_flag else None
    pot_y_var = np.empty_like(pot_y) if iate_se_flag else None
    pot_y_m_ate_var = np.empty_like(pot_y) if (iate_se_flag and iate_m_ate_flag
                                               ) else None
    equal_0, mean_nonzero = np.zeros_like(nonzero), np.zeros_like(nonzero)
    std_nonzero, gini_all = np.zeros_like(nonzero), np.zeros_like(nonzero)
    gini_nonzero = np.zeros_like(nonzero)
    share_censored = np.zeros_like(nonzero)
    obs_larger = np.zeros_like(sum_larger)

    if w_ate is not None:
        w_ate = w_ate[0, :, :]
    if not p_cfg.iate_m_ate:
        w_ate = None
    l1_to_9 = [None for _ in range(n_x)]
    if gen_cfg.mp_automatic:
        maxworkers = mcf_sys.find_no_of_workers(gen_cfg.mp_parallel, gen_cfg.sys_share,
                                                zero_tol=int_cfg.zero_tol,
                                                )
    else:
        maxworkers = gen_cfg.mp_parallel

    cuda = int_cfg.cuda and maxworkers < 16 and not mcf_.low_mem_cfg.yes
    # else usually multiple CPUs are faster than single GPU

    if p_ba_cfg.yes and cuda:
        print('WARNING: Bias adjustment not yet fully implemented on Cuda. '
              'Non-cuda version of mcf for IATEs is used instead.'
              )
        cuda = False

    if gen_cfg.mp_parallel < 1.5 or cuda:
        #  Pytorch does not work well with ray. Need to set max_workers to 1.
        maxworkers = 1

    print_runtime_info(gen_cfg, int_cfg, maxworkers, txt_method='IATE 1/2')

    if gen_cfg.with_output and gen_cfg.verbose:
        print('Number of parallel processes (IATE): ', maxworkers)
    if int_cfg.weight_as_sparse:
        iterator = len(weights)   # Number of treatments
    kw_args = {'no_of_out': no_of_out, 'n_y': n_y, 'ct_cfg': ct_cfg, 'int_cfg': int_cfg,
               'gen_cfg': gen_cfg, 'p_cfg': p_cfg, 'iate_se_flag': iate_se_flag,
               'se_boot_iate': se_boot_iate, 'iate_m_ate_flag': iate_m_ate_flag,
               'gen_tv_cfg': gen_tv_cfg, 
               }
    if maxworkers == 1:
        if cuda:
            if gen_cfg.with_output and gen_cfg.verbose:
                print('Computing IATEs with Cuda')
            if gen_tv_cfg.yes:
                d_tv_train = gen_tv_data.d_train
                x_tv_train = gen_tv_data.x_tv_train
            else:
                d_tv_train = x_tv_train = None
            # The full weight matrix may not fit into GPU memory, batch it.
            # Up to share_weights of free GPU memory for weight matrix
            share_weights = 0.33
            batch_max_size = mcf_iate_cuda.max_batch_size(n_x, weights, share_weights,
                                                          int_cfg.weight_as_sparse, gen_cfg
                                                          )
            batch_idx_tuple = mcf_cuda.split_into_batches(n_x, batch_max_size)

            batch_iter = enumerate(batch_idx_tuple)
            if use_tqdm and len(batch_idx_tuple) > 1:
                batch_iter = enumerate(tqdm(batch_idx_tuple, total=len(batch_idx_tuple),
                                            desc='IATE 1/2', leave=False, dynamic_ncols=True,
                                            )
                                       )
            for batch_no, idx_ba in batch_iter:
                if ((not use_tqdm) and gen_cfg.with_output and gen_cfg.verbose
                        and (len(batch_idx_tuple) > 1)
                        ):
                    print('Batch: ', batch_no)
                if int_cfg.weight_as_sparse:
                    weights_ba = [weights[t_idx][idx_ba, :] for t_idx in range(iterator)]
                else:
                    weights_ba = weights[idx_ba[0]:idx_ba[-1]+1]

                x_tv_pred_ba = x_tv_pred[idx_ba, :] if gen_tv_cfg.yes else None

                (pot_y_ba, pot_y_var_ba, pot_y_m_ate_ba, pot_y_m_ate_var_ba, l1_to_9_ba,
                 share_censored_ba
                 ) = mcf_iate_cuda.iate_cuda(weights_ba,
                                             cl_dat_np=cl_dat, no_of_cluster=no_of_cluster,
                                             w_dat_np=w_dat, w_ate_np=w_ate, y_dat_np=y_dat,
                                             **kw_args,
                                             # ba_data, p_ba_cfg,bias adjustment not yet implemented
                                             d_tv_train_np=d_tv_train, x_tv_train_np=x_tv_train,
                                             x_tv_pred_np=x_tv_pred_ba, n_x=len(idx_ba),
                                             no_of_treat_dr=no_of_treat_dr,
                                             )
                pot_y[idx_ba, :] = pot_y_ba
                if pot_y_var is not None:
                    pot_y_var[idx_ba, :] = pot_y_var_ba
                if iate_m_ate_flag:
                    pot_y_m_ate[idx_ba] = pot_y_m_ate_ba
                    if pot_y_m_ate_var is not None:
                        pot_y_m_ate_var[idx_ba] = pot_y_m_ate_var_ba
                l1_to_9[idx_ba[0]:idx_ba[-1]+1] = l1_to_9_ba
                share_censored += share_censored_ba * len(idx_ba)
            share_censored /= n_x
        else:

            iterator_idx = range(n_x)
            if use_tqdm:
                iterator_idx = tqdm(iterator_idx, total=n_x, desc='IATE 1/2', leave=False,
                                    dynamic_ncols=True
                                    )
            for idx in iterator_idx:
                if int_cfg.weight_as_sparse:
                    weights_idx = [weights[t_idx][idx, :] for t_idx in range(iterator)]
                else:
                    weights_idx = weights[idx]
                ret_all_i = iate_func1_for_mp(idx, weights_idx,
                                              cl_dat=cl_dat, no_of_cluster=no_of_cluster,
                                              w_dat=w_dat, w_ate=w_ate, y_dat=y_dat,
                                              **kw_args,
                                              ba_data=ba_data, p_ba_cfg=p_ba_cfg,
                                              gen_tv_data=gen_tv_data, iv=iv,
                                              )
                (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9, share_censored
                 ) = assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                                      l1_to_9=l1_to_9, share_censored=share_censored,
                                      ret_all_i=ret_all_i, n_x=n_x, idx=idx,
                                      )
                mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx+1, total=n_x)
    else:
        rows_per_split = 1e9    # Just a large number; feature is not used
        no_of_splits = round(n_x / rows_per_split)
        no_of_splits = min(max(no_of_splits, maxworkers), n_x)
        if gen_cfg.with_output and gen_cfg.verbose:
            print('IATE-1: Avg. number of obs per split:',
                  f'{n_x / no_of_splits:5.2f}.',
                  ' Number of splits: ', no_of_splits)
        obs_idx_list = np.array_split(np.arange(n_x), no_of_splits)
        ray_err_txt='Ray does not startup in IATE estimation 1/2'
        if int_cfg.mp_use_old_ray:
            if ray is None or ray_iate_func1_for_mp_many_obs is None:
                raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                   'Install ray or use the new backend with mp_use_old_ray=False.'
                                   )
            if not ray.is_initialized():
                mcf_ray.init_ray_with_fallback(maxworkers, gen_cfg,
                                               mem_object_store=int_cfg.mem_object_store_3,
                                               mem_object_store_2=int_cfg.mem_object_store_2,
                                               ray_err_txt=ray_err_txt,
                                               )
            mcf_ray.print_object_store(gen_cfg, int_cfg.mem_object_store_3)
            y_dat_ref = mcf_ray.ray_put_all(y_dat)

            if int_cfg.weight_as_sparse:
                still_running = [ray_iate_func1_for_mp_many_obs.remote(
                    idx, [weights[t_idx][idx, :] for t_idx in range(iterator)],
                    cl_dat=cl_dat, no_of_cluster=no_of_cluster, w_dat=w_dat, w_ate=w_ate,
                    y_dat=y_dat_ref,
                    **kw_args,
                    ba_data=ba_data, p_ba_cfg=p_ba_cfg, gen_tv_data=gen_tv_data, iv=iv,
                    ) for idx in obs_idx_list
                    ]
                if gen_cfg.with_output and gen_cfg.verbose:
                    warn_text_to_console()
            else:
                still_running = [ray_iate_func1_for_mp_many_obs.remote(
                    idx, [weights[idxx] for idxx in idx],
                    cl_dat=cl_dat, no_of_cluster=no_of_cluster, w_dat=w_dat, w_ate=w_ate,
                    y_dat=y_dat_ref,
                    **kw_args,
                    ba_data=ba_data, p_ba_cfg=p_ba_cfg, gen_tv_data=gen_tv_data, iv=iv,
                    )
                    for idx in obs_idx_list
                    ]
            jdx = 0
            pbar = (tqdm(total=no_of_splits, desc='IATE 1/2', leave=False, dynamic_ncols=True)
                    if use_tqdm else None
                    )
            try:
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running, num_returns=1)
                    finished_res = ray.get(finished)
                    for ret_all_i_list in finished_res:
                        for ret_all_i in ret_all_i_list:
                            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9, share_censored
                             ) = assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var,
                                                  l1_to_9=l1_to_9, share_censored=share_censored,
                                                  ret_all_i=ret_all_i, n_x=n_x,
                                                  )
                        jdx += 1
                        if pbar is not None:
                            pbar.update(1)

                        mcf_gp.progress_clean_memory(output=output_clean, current_idx=jdx,
                                                     total=no_of_splits,
                                                     )
            finally:
                if pbar is not None:
                    pbar.close()

            *_, finished_res, finished = mcf_ray.ray_del_refs(f1=finished_res, f2=finished,
                                                              mp_ray_del=int_cfg.mp_ray_del,
                                                              )
        else:
            shared_iate_data = {'y_dat': y_dat}
            fail_txt='Failed to make executor in IATE estimation 1/2.'
            with forest_executor_with_shared(int_cfg=int_cfg, maxworkers=maxworkers,
                                             shared_obj=shared_iate_data,
                                             shared_name='iate_func1_data',
                                             ray_err_txt=ray_err_txt, fail_txt=fail_txt,
                                             ) as (executor, data_handle, maxworkers):
                if int_cfg.weight_as_sparse:
                    if gen_cfg.with_output and gen_cfg.verbose:
                        warn_text_to_console()

                    tasks = [TaskSpec(func=iate_func1_many_obs_backend,
                                      kwargs={'obs_idx': idx,
                                              'weights_chunk': [weights[t_idx][idx, :]
                                                                for t_idx in range(iterator)
                                                                ],
                                              'data': data_handle,
                                              'cl_dat': cl_dat,
                                              'no_of_cluster': no_of_cluster,
                                              'w_dat': w_dat,
                                              'w_ate': w_ate,
                                              'kw_args': kw_args,
                                              'ba_data': ba_data,
                                              'p_ba_cfg': p_ba_cfg,
                                              'gen_tv_data': gen_tv_data,
                                              'iv': iv,
                                              },
                                      name=f'iate_func1_{jdx}',
                                      )
                             for jdx, idx in enumerate(obs_idx_list)
                             ]
                else:
                    tasks = [TaskSpec(func=iate_func1_many_obs_backend,
                                      kwargs={'obs_idx': idx,
                                              'weights_chunk': [weights[idxx] for idxx in idx],
                                              'data': data_handle,
                                              'cl_dat': cl_dat,
                                              'no_of_cluster': no_of_cluster,
                                              'w_dat': w_dat,
                                              'w_ate': w_ate,
                                              'kw_args': kw_args,
                                              'ba_data': ba_data,
                                              'p_ba_cfg': p_ba_cfg,
                                              'gen_tv_data': gen_tv_data,
                                              'iv': iv,
                                              },
                                      name=f'iate_func1_{jdx}',
                                      )
                             for jdx, idx in enumerate(obs_idx_list)
                             ]
                jdx = 0
                pbar = (tqdm(total=no_of_splits, desc='IATE 1/2', leave=False, dynamic_ncols=True)
                        if use_tqdm else None
                        )
                try:
                    for ret_all_i_list in map_task_batches(executor=executor, tasks=tasks,
                                                           int_cfg=int_cfg, maxworkers=maxworkers,
                                                           min_worker_waves = (
                                                               4 if int_cfg.mp_backend == 'joblib'
                                                               else 1
                                                               ),
                                                           ):
                        for ret_all_i in ret_all_i_list:
                            (pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9,
                             share_censored,) = assign_ret_all_i(pot_y, pot_y_var, pot_y_m_ate,
                                                                 pot_y_m_ate_var, l1_to_9=l1_to_9,
                                                                 share_censored=share_censored,
                                                                 ret_all_i=ret_all_i, n_x=n_x,
                                                                 )
                        jdx += 1
                        if pbar is not None:
                            pbar.update(1)
                        mcf_gp.progress_clean_memory(output=output_clean, current_idx=jdx,
                                                     total=no_of_splits,
                                                     )
                finally:
                    if pbar is not None:
                        pbar.close()
    for idx in range(n_x):
        nonzero += l1_to_9[idx][0]
        equal_0 += l1_to_9[idx][1]
        mean_nonzero += l1_to_9[idx][2]
        std_nonzero += l1_to_9[idx][3]
        gini_all += l1_to_9[idx][4]
        gini_nonzero += l1_to_9[idx][5]
        share_largest_q += l1_to_9[idx][6]
        sum_larger += l1_to_9[idx][7]
        obs_larger += l1_to_9[idx][8]
    if gen_cfg.with_output:
        txt = '\n' + '=' * 100
        txt += ('\nAnalysis of weights (normalised to add to 1) of IATE'
                '(stats are averaged over all effects)')
        txt += mcf_ps.txt_weight_stat(nonzero=nonzero / n_x, equalzero=equal_0 / n_x,
                                      mean_nonzero=mean_nonzero / n_x,
                                      std_nonzero=std_nonzero / n_x, gini_all=gini_all / n_x,
                                      gini_nonzero=gini_nonzero / n_x,
                                      share_largest_q=share_largest_q / n_x,
                                      sum_larger=sum_larger / n_x, obs_larger=obs_larger / n_x,
                                      gen_cfg=gen_cfg, p_cfg=p_cfg, share_censored=share_censored,
                                      continuous=gen_cfg.d_type == 'continuous',
                                      d_values_cont=d_values_dr,
                                      )
    else:
        txt = ''

    return pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, txt


def iate_func1_many_obs_backend(*, obs_idx: Any,
                                weights_chunk: list[Any],
                                data: dict[str, Any],
                                cl_dat: Any,
                                no_of_cluster: int,
                                w_dat: Any,
                                w_ate: Any,
                                kw_args: dict[str, Any],
                                ba_data: Any,
                                p_ba_cfg: Any,
                                gen_tv_data: Any,
                                iv: bool,
                                ) -> Any:
    """Backend-agnostic wrapper for one IATE chunk."""
    return iate_func1_for_mp_many_obs(obs_idx, weights_chunk,
                                      cl_dat=cl_dat, no_of_cluster=no_of_cluster, w_dat=w_dat,
                                      w_ate=w_ate, y_dat=data['y_dat'], **kw_args, ba_data=ba_data,
                                      p_ba_cfg=p_ba_cfg, gen_tv_data=gen_tv_data, iv=iv,
                                      )


def iate_effects_print(mcf_: 'ModifiedCausalForest',
                       effect_dic: dict,
                       effect_m_ate_dic: dict,
                       y_pred_x_df: pd.DataFrame, *,
                       extra_title: str = '',
                       iv: bool = False,
                       print_progress: bool = True,
                       ) -> tuple[NDArray[Any], NDArray[Any], list[NDArray[Any], NDArray[Any]],
                                  pd.DataFrame, str
                                  ]:
    """Compute, print effects, add potential outcomes to prediction data."""
    p_cfg, int_cfg, gen_cfg = mcf_.p_cfg, mcf_.int_cfg, mcf_.gen_cfg
    var_cfg, ct_cfg, lc_cfg = mcf_.var_cfg, mcf_.ct_cfg, mcf_.lc_cfg
    gen_tv_cfg = mcf_.gen_tv_cfg
    # low_mem_cfg = mcf_.low_mem_cfg
    zero_tol = int_cfg.zero_tol
    progress = gen_cfg.with_output and gen_cfg.verbose and print_progress
    use_tqdm, output_clean = mcf_gp.tqdm_setup(tqdm, progress)

    if gen_cfg.d_type == 'continuous':
        no_of_treat, d_values = ct_cfg.grid_w, ct_cfg.grid_w_val
        d_values_dr = ct_cfg.d_values_dr_np
        no_of_treat_dr = len(d_values_dr)
        d_sub_values = None
    else:
        no_of_treat, d_values = gen_cfg.no_of_treat, gen_cfg.d_values
        no_of_treat_dr, d_values_dr = no_of_treat, d_values
        d_sub_values = gen_tv_cfg.d_dict['sub_treat'] if gen_tv_cfg.yes else None

    if gen_cfg.with_output and gen_cfg.verbose:
        if iv:
            print('\nComputing LIATEs 2/2 (effects)' + extra_title)
        else:
            print('\nComputing IATEs 2/2 (effects)' + extra_title)
    if gen_cfg.d_type == 'continuous':
        dim_3 = round(no_of_treat_dr - 1)
    else:
        no_of_treat_all = gen_tv_cfg.no_of_treat_all if gen_tv_cfg.yes else no_of_treat
        dim_3 = round(no_of_treat_all * (no_of_treat_all - 1) / 2)

    y_pot, y_pot_var = effect_dic['y_pot'], effect_dic['y_pot_var']
    if effect_m_ate_dic is None:
        y_pot_m_ate_var = y_pot_m_ate = None
    else:
        y_pot_m_ate = effect_m_ate_dic['y_pot']
        y_pot_m_ate_var = effect_m_ate_dic['y_pot_var']
    n_x, no_of_out = y_pot.shape[0], y_pot.shape[2]
    iate = np.empty((n_x, no_of_out, dim_3, 2))    # iate, iate_m_ate

    if p_cfg.iate_se:
        iate_se, iate_p = np.empty_like(iate), np.empty_like(iate)
    else:
        iate_se = iate_p = None

    if gen_cfg.mp_parallel < 1.5:
        maxworkers = 1
    else:
        if gen_cfg.mp_automatic:
            maxworkers = mcf_sys.find_no_of_workers(gen_cfg.mp_parallel,
                                                    gen_cfg.sys_share,
                                                    zero_tol=zero_tol,
                                                    )
        else:
            maxworkers = gen_cfg.mp_parallel

    print_runtime_info(gen_cfg, int_cfg, maxworkers, txt_method='IATE 2/2')

    effect_list = None

    no_of_treat_dr_all = gen_tv_cfg.no_of_treat_all if gen_tv_cfg.yes else no_of_treat_dr
    no_of_treat_all = gen_tv_cfg.no_of_treat_all if gen_tv_cfg.yes else no_of_treat
    kw_args = {'no_of_out':no_of_out, 'd_type': gen_cfg.d_type, 'd_values': d_values_dr,
               'no_of_treat': no_of_treat_dr_all,
               }
    if maxworkers == 1:
        se = getattr(p_cfg, 'iate_se', False)
        m_ate = getattr(p_cfg, 'iate_m_ate', False)
        iterator_idx = range(n_x)
        if use_tqdm:
            iterator_idx = tqdm(iterator_idx, total=n_x, desc='IATE 2/2',
                                leave=False, dynamic_ncols=True
                                )
        for idx in iterator_idx:
            match (se, m_ate):
                case (True, True):
                    y_pot_v = y_pot_var[idx]
                    y_pot_m = y_pot_m_ate[idx]
                    y_pot_m_v = y_pot_m_ate_var[idx]
                case (True, False):
                    y_pot_v = y_pot_var[idx]
                    y_pot_m = None
                    y_pot_m_v = None
                case (False, True):
                    y_pot_v = None
                    y_pot_m = y_pot_m_ate[idx]
                    y_pot_m_v = None
                case _:
                    y_pot_v = y_pot_m = y_pot_m_v = None

            ret_all_idx = iate_func2_for_mp(idx,
                                            **kw_args,
                                            pot_y_i=y_pot[idx], pot_y_var_i=y_pot_v,
                                            pot_y_m_ate_i=y_pot_m, pot_y_m_ate_var_i=y_pot_m_v,
                                            iate_se_flag=p_cfg.iate_se,
                                            iate_m_ate_flag=p_cfg.iate_m_ate,
                                            d_sub_values=d_sub_values,
                                            )
            mcf_gp.progress_clean_memory(output=output_clean, current_idx=idx+1, total=n_x)
            iate[idx, :, :, :] = ret_all_idx[1]
            if p_cfg.iate_se:
                iate_se[idx, :, :, :] = ret_all_idx[2]
                iate_p[idx, :, :, :] = ret_all_idx[3]
            if idx == n_x - 1:
                effect_list = ret_all_idx[4]
    else:
        if int_cfg.iate_chunk_size is None:
            # This function is executed after the low_mem branch
            # if low_mem_cfg.yes:
            #     number_indices_chunk = low_mem_cfg.max_chunksize
            # else:
            number_indices_chunk = np.ceil(n_x / maxworkers)
        else:
            number_indices_chunk = int_cfg.iate_chunk_size

        if number_indices_chunk > 1:
            indices_chuncks = get_chunck_of_indices(n_x, number_indices_chunk)
            no_of_chunks = len(indices_chuncks)
        else:
            indices_chuncks = None
            no_of_chunks = n_x

        if gen_cfg.with_output and gen_cfg.verbose:
            print(f'Number of chunks: {no_of_chunks}')
            print('Maximum number of observations in each chunk: '
                  f'{number_indices_chunk}'
                  )
        ray_err_txt = 'Ray not starting up in IATE estimation 2/2'
        if int_cfg.mp_use_old_ray:
            if (ray is None
                or ray_iate_func2_for_mp_single is None
                    or ray_iate_func2_for_mp_mult is None
                    ):
                raise ImportError('int_cfg.mp_use_old_ray=True, but ray is not installed. '
                                   'Install ray or use the new backend with mp_use_old_ray=False.'
                                   )
            if not ray.is_initialized():
                mcf_ray.init_ray_with_fallback(maxworkers, gen_cfg,
                                               mem_object_store=int_cfg.mem_object_store_3,
                                               mem_object_store_2=int_cfg.mem_object_store_2,
                                               ray_err_txt=ray_err_txt,
                                               )
            mcf_ray.print_object_store(gen_cfg, int_cfg.mem_object_store_3)

            if p_cfg.iate_se and p_cfg.iate_m_ate:
                if number_indices_chunk == 1:
                    still_running = [
                        ray_iate_func2_for_mp_single.remote(idx,
                                                            **kw_args,
                                                            y_pot_i=y_pot[idx],
                                                            y_pot_var_i=y_pot_var[idx],
                                                            y_pot_m_ate_i=y_pot_m_ate[idx],
                                                            y_pot_m_ate_var_i=y_pot_m_ate_var[idx],
                                                            iate_se_flag=True, iate_m_ate_flag=True,
                                                            d_sub_values=d_sub_values,
                                                            )
                        for idx in range(n_x)
                        ]
                else:
                    still_running = [
                        ray_iate_func2_for_mp_mult.remote(idx_ch,
                                                          **kw_args,
                                                          pot_y_chunk=y_pot[idx_ch],
                                                          pot_y_var_chunk=y_pot_var[idx_ch],
                                                          pot_y_m_ate_chunk=y_pot_m_ate[idx_ch],
                                                          pot_y_m_ate_var_chunk
                                                              =y_pot_m_ate_var[idx_ch],
                                                          iate_se_flag=True, iate_m_ate_flag=True,
                                                          d_sub_values=d_sub_values,
                                                          )
                        for idx_ch in indices_chuncks
                        ]
            elif p_cfg.iate_se and not p_cfg.iate_m_ate:
                if number_indices_chunk == 1:
                    still_running = [
                        ray_iate_func2_for_mp_single.remote(idx,
                                                            **kw_args,
                                                            y_pot_i=y_pot[idx],
                                                            y_pot_var_i=y_pot_var[idx],
                                                            y_pot_m_ate_i=None,
                                                            y_pot_m_ate_var_i=None,
                                                            iate_se_flag=True,
                                                            iate_m_ate_flag=False,
                                                            d_sub_values=d_sub_values,
                                                            )
                        for idx in range(n_x)
                        ]
                else:
                    still_running = [
                        ray_iate_func2_for_mp_mult.remote(idx_ch,
                                                          **kw_args,
                                                          pot_y_chunk=y_pot[idx_ch],
                                                          pot_y_var_chunk=y_pot_var[idx_ch],
                                                          pot_y_m_ate_chunk=None,
                                                          pot_y_m_ate_var_chunk=None,
                                                          iate_se_flag=True, iate_m_ate_flag=False,
                                                          d_sub_values=d_sub_values,
                                                          )
                        for idx_ch in indices_chuncks
                        ]
            elif not p_cfg.iate_se and p_cfg.iate_m_ate:
                if number_indices_chunk == 1:
                    still_running = [
                        ray_iate_func2_for_mp_single.remote(idx,
                                                            **kw_args,
                                                            y_pot_i=y_pot[idx], y_pot_var_i=None,
                                                            y_pot_m_ate_i=y_pot_m_ate[idx],
                                                            y_pot_m_ate_var_i=None,
                                                            iate_se_flag=False,
                                                            iate_m_ate_flag=True,
                                                            d_sub_values=d_sub_values,
                                                            )
                        for idx in range(n_x)
                        ]
                else:
                    still_running = [
                        ray_iate_func2_for_mp_mult.remote(idx_ch,
                                                          **kw_args,
                                                          pot_y_chunk=y_pot[idx_ch],
                                                          pot_y_var_chunk=None,
                                                          pot_y_m_ate_chunk=y_pot_m_ate[idx_ch],
                                                          pot_y_m_ate_var_chunk=None,
                                                          iate_se_flag=False, iate_m_ate_flag=True,
                                                          d_sub_values=d_sub_values,
                                                          )
                        for idx_ch in indices_chuncks
                        ]
            else:
                if number_indices_chunk == 1:
                    still_running = [
                        ray_iate_func2_for_mp_single.remote(idx,
                                                            **kw_args,
                                                            y_pot_i=y_pot[idx], y_pot_var_i=None,
                                                            y_pot_m_ate_i=None,
                                                            y_pot_m_ate_var_i=None,
                                                            iate_se_flag=False,
                                                            iate_m_ate_flag=False,
                                                            d_sub_values=d_sub_values,
                                                            )
                        for idx in range(n_x)
                        ]
                else:
                    still_running = [
                        ray_iate_func2_for_mp_mult.remote(idx_ch,
                                                          **kw_args,
                                                          pot_y_chunk=y_pot[idx_ch],
                                                          pot_y_var_chunk=None,
                                                          pot_y_m_ate_chunk=None,
                                                          pot_y_m_ate_var_chunk=None,
                                                          iate_se_flag=False, iate_m_ate_flag=False,
                                                          d_sub_values=d_sub_values,
                                                          )
                        for idx_ch in indices_chuncks
                        ]
            if number_indices_chunk == 1:
                jdx = 0
                pbar = (tqdm(total=n_x, desc='IATE 2/2', leave=False, dynamic_ncols=True)
                        if use_tqdm else None
                        )
                try:
                    while len(still_running) > 0:
                        finished, still_running = ray.wait(still_running, num_returns=1)
                        finished_res = ray.get(finished)
                        # IATE 2/2, old-Ray, number_indices_chunk == 1
                        for ret_all_i2 in finished_res:
                            iix = ret_all_i2[0]
                            iate[iix, :, :, :] = ret_all_i2[1]
                            if p_cfg.iate_se:
                                iate_se[iix, :, :, :] = ret_all_i2[2]
                                iate_p[iix, :, :, :] = ret_all_i2[3]
                            if jdx == n_x - 1:
                                effect_list = ret_all_i2[4]
                            jdx += 1
                            if pbar is not None:
                                pbar.update(1)
                            mcf_gp.progress_clean_memory(output=output_clean, current_idx=jdx,
                                                         total=n_x,
                                                         )
                finally:
                    if pbar is not None:
                        pbar.close()
            else:
                chunk_jdx = 0
                obs_jdx = 0
                pbar = (tqdm(total=len(indices_chuncks), desc='IATE 2/2', leave=False,
                             dynamic_ncols=True
                             )
                        if use_tqdm else None
                        )
                try:
                    while len(still_running) > 0:
                        finished, still_running = ray.wait(still_running, num_returns=1)
                        finished_res = ray.get(finished)
                        for ret_all_i2_all in finished_res:
                            for ret_all_i2 in ret_all_i2_all:
                                iix = ret_all_i2[0]
                                iate[iix, :, :, :] = ret_all_i2[1]
                                if p_cfg.iate_se:
                                    iate_se[iix, :, :, :] = ret_all_i2[2]
                                    iate_p[iix, :, :, :] = ret_all_i2[3]
                                if obs_jdx == n_x - 1:
                                    effect_list = ret_all_i2[4]
                                obs_jdx += 1

                            chunk_jdx += 1
                            if pbar is not None:
                                pbar.update(1)

                            mcf_gp.progress_clean_memory(output=output_clean,
                                                         current_idx=chunk_jdx,
                                                         total=len(indices_chuncks))
                finally:
                    if pbar is not None:
                        pbar.close()
            # chunk_jdx = 0
            # obs_jdx = 0
            # pbar = (tqdm(total=len(indices_chuncks), desc='IATE 2/2', leave=False,
            #              dynamic_ncols=True
            #              )
            #         if use_tqdm else None
            #         )
            # try:
            #     while len(still_running) > 0:
            #         finished, still_running = ray.wait(still_running, num_returns=1)
            #         finished_res = ray.get(finished)
            #         for ret_all_i2_all in finished_res:
            #             for ret_all_i2 in ret_all_i2_all:
            #                 iix = ret_all_i2[0]
            #                 iate[iix, :, :, :] = ret_all_i2[1]
            #                 if p_cfg.iate_se:
            #                     iate_se[iix, :, :, :] = ret_all_i2[2]
            #                     iate_p[iix, :, :, :] = ret_all_i2[3]
            #                 if obs_jdx == n_x-1:
            #                     effect_list = ret_all_i2[4]
            #                 obs_jdx += 1
            #             chunk_jdx += 1
            #             mcf_gp.progress_clean_memory(output=output_clean, current_idx=chunk_jdx,
            #                                          total=len(indices_chuncks),
            #                                          )
            #         # if number_indices_chunk == 1:
            #         #     for ret_all_i2 in finished_res:
            #         #         iix = ret_all_i2[0]
            #         #         iate[iix, :, :, :] = ret_all_i2[1]
            #         #         if p_cfg.iate_se:
            #         #             iate_se[iix, :, :, :] = ret_all_i2[2]
            #         #             iate_p[iix, :, :, :] = ret_all_i2[3]
            #         #         if jdx == n_x-1:
            #         #             effect_list = ret_all_i2[4]
            #         #         mcf_gp.progress_clean_memory(
            #         #             output=gen_cfg.with_output and gen_cfg.verbose
            #                         and print_progress,
            #         #             current_idx=jdx+1, total=n_x,
            #         #             )
            #         #         jdx += 1
            #         # else:
            #         #     for ret_all_i2_all in finished_res:
            #         #         for ret_all_i2 in ret_all_i2_all:
            #         #             iix = ret_all_i2[0]
            #         #             iate[iix, :, :, :] = ret_all_i2[1]
            #         #             if p_cfg.iate_se:
            #         #                 iate_se[iix, :, :, :] = ret_all_i2[2]
            #         #                 iate_p[iix, :, :, :] = ret_all_i2[3]
            #         #             if jdx == n_x-1:
            #         #                 effect_list = ret_all_i2[4]
            #         #             mcf_gp.progress_clean_memory(
            #         #                 output=gen_cfg.with_output and gen_cfg.verbose and
            #  print_progress,
            #         #                 current_idx=jdx+1, total=len(indices_chuncks),
            #         #                 )
            #         #             jdx += 1
            # finally:
            #     if pbar is not None:
            #         pbar.close()

            *_, finished_res, finished = mcf_ray.ray_del_refs(f1=finished_res, f2=finished,
                                                              mp_ray_del=int_cfg.mp_ray_del,
                                                              )
        else:
            with forest_executor_context(int_cfg=int_cfg, maxworkers=maxworkers,
                                         ray_err_txt=ray_err_txt,
                                         fail_txt='Failed to make executor in IATE estimation 2/2.',
                                         ) as (executor, maxworkers):
                iate_se_flag = p_cfg.iate_se is True
                iate_m_ate_flag = p_cfg.iate_m_ate is True

                if number_indices_chunk == 1:
                    tasks = [TaskSpec(func=iate_func2_single_backend,
                                      kwargs={'idx': idx,
                                              'kw_args': kw_args,
                                              'y_pot_i': y_pot[idx],
                                              'y_pot_var_i': (y_pot_var[idx] if iate_se_flag
                                                             else None),
                                              'y_pot_m_ate_i': (y_pot_m_ate[idx] if iate_m_ate_flag
                                                                else None),
                                              'y_pot_m_ate_var_i': (y_pot_m_ate_var[idx]
                                                  if iate_se_flag and iate_m_ate_flag else None),
                                              'iate_se_flag': iate_se_flag,
                                              'iate_m_ate_flag': iate_m_ate_flag,
                                              'd_sub_values': d_sub_values,
                                              },
                                      name=f'iate_func2_single_{idx}',
                                      )
                             for idx in range(n_x)
                             ]
                    chunk_jdx = 0
                    pbar = (tqdm(total=n_x, desc='IATE 2/2', leave=False, dynamic_ncols=True)
                            if use_tqdm else None
                            )
                    try:
                        for ret_all_i2 in map_task_batches(executor=executor, tasks=tasks,
                                                           int_cfg=int_cfg, maxworkers=maxworkers,
                                                           min_worker_waves = (
                                                               4 if int_cfg.mp_backend == 'joblib'
                                                               else 1
                                                               ),
                                                           ):
                            iix = ret_all_i2[0]
                            iate[iix, :, :, :] = ret_all_i2[1]

                            if p_cfg.iate_se:
                                iate_se[iix, :, :, :] = ret_all_i2[2]
                                iate_p[iix, :, :, :] = ret_all_i2[3]

                            if chunk_jdx == n_x - 1:
                                effect_list = ret_all_i2[4]
                            chunk_jdx += 1
                            if pbar is not None:
                                pbar.update(1)
                            mcf_gp.progress_clean_memory(output=output_clean, current_idx=chunk_jdx,
                                                         total=n_x,
                                                         )
                    finally:
                        if pbar is not None:
                            pbar.close()
                else:
                    tasks = [TaskSpec(func=iate_func2_mult_backend,
                                      kwargs={'idx_chunk': idx_ch,
                                              'kw_args': kw_args,
                                              'pot_y_chunk': y_pot[idx_ch],
                                              'pot_y_var_chunk': (y_pot_var[idx_ch] if iate_se_flag
                                                                  else None),
                                              'pot_y_m_ate_chunk': (y_pot_m_ate[idx_ch]
                                                                    if iate_m_ate_flag else None),
                                              'pot_y_m_ate_var_chunk': (y_pot_m_ate_var[idx_ch]
                                                                        if iate_se_flag
                                                                        and iate_m_ate_flag
                                                                        else None),
                                              'iate_se_flag': iate_se_flag,
                                              'iate_m_ate_flag': iate_m_ate_flag,
                                              'd_sub_values': d_sub_values,
                                              },
                                      name=f'iate_func2_mult_{jdx}',
                                      )
                             for jdx, idx_ch in enumerate(indices_chuncks)
                             ]
                    chunk_jdx = 0
                    obs_jdx = 0
                    pbar = (tqdm(total=len(indices_chuncks), desc='IATE 2/2', leave=False,
                                 dynamic_ncols=True
                                 )
                            if use_tqdm else None
                            )
                    try:
                        for ret_all_i2_all in map_task_batches(executor=executor, tasks=tasks,
                                                               int_cfg=int_cfg,
                                                               maxworkers=maxworkers,
                                                               min_worker_waves = (
                                                                   4 if int_cfg.mp_backend ==
                                                                   'joblib' else 1
                                                                   ),
                                                               ):
                            for ret_all_i2 in ret_all_i2_all:
                                iix = ret_all_i2[0]
                                iate[iix, :, :, :] = ret_all_i2[1]

                                if p_cfg.iate_se:
                                    iate_se[iix, :, :, :] = ret_all_i2[2]
                                    iate_p[iix, :, :, :] = ret_all_i2[3]

                                if obs_jdx == n_x - 1:
                                    effect_list = ret_all_i2[4]

                                obs_jdx += 1

                            chunk_jdx += 1
                            if pbar is not None:
                                pbar.update(1)

                            mcf_gp.progress_clean_memory(output=output_clean, current_idx=chunk_jdx,
                                                         total=len(indices_chuncks),
                                                         )
                    finally:
                        if pbar is not None:
                            pbar.close()
    if gen_cfg.with_output:
        txt_iate_long, txt_iate = mcf_ps.print_iate(iate, iate_se,
                                                    iate_p=iate_p, effect_list=effect_list,
                                                    gen_cfg=gen_cfg, p_cfg=p_cfg,
                                                    var_cfg=var_cfg, extra_title=extra_title,
                                                    gen_tv_cfg_yes=gen_tv_cfg.yes,
                                                    zero_tol=zero_tol,
                                                    )
        mcf_ps.print_mcf(gen_cfg, txt_iate_long, summary=True, non_summary=False)
        if p_cfg.iate_m_ate:
            mcf_ps.print_mcf(gen_cfg, effect_dic['txt_weights'], summary=False)
    else:
        txt_iate = ''

    # Add results to data file
    y_pot_np = np.empty((n_x, no_of_out * no_of_treat_dr))
    if p_cfg.iate_se:
        y_pot_se_np = np.empty_like(y_pot_np)
    if gen_cfg.d_type == 'continuous':
        dim = round(no_of_out * (no_of_treat_dr - 1))
    else:
        dim = round(no_of_out * no_of_treat_all * (no_of_treat_all - 1) / 2)
    iate_np = np.empty((n_x, dim))
    if p_cfg.iate_m_ate:
        iate_mate_np = np.empty_like(iate_np)
    if p_cfg.iate_se:
        iate_se_np = np.empty_like(iate_np)
        if p_cfg.iate_m_ate:
            iate_mate_se_np = np.empty_like(iate_np)

    jdx = j2dx = jdx_unlc = 0
    name_pot, name_eff, name_eff0 = [], [], []
    y_pot_unlc_np = None
    if lc_cfg.uncenter_po and isinstance(y_pred_x_df, (pd.Series, pd.DataFrame)):
        name_pot_unlc, y_pot_unlc_np = [], np.empty((n_x, no_of_treat_dr))

        if isinstance(var_cfg.y_tree_name, list):
            y_tree_name = var_cfg.y_tree_name[0]
        else:
            y_tree_name = var_cfg.y_tree_name
        y_pred_x_np = mcf_gp.to_numpy_big_data(y_pred_x_df, int_cfg.obs_bigdata)
    else:
        name_pot_unlc = y_tree_name = name_y_pot_unlc = y_pred_x_np = None
    for o_idx, o_name in enumerate(var_cfg.y_name):
        for t_idx, t_name in enumerate(d_values_dr):
            name_pot += [o_name + str(t_name)]
            y_pot_np[:, jdx] = y_pot[:, t_idx, o_idx]

            if o_name == y_tree_name and lc_cfg.uncenter_po:
                name_pot_unlc += [o_name + str(t_name) + '_un_lc']
                y_pot_unlc_np[:, jdx_unlc] = y_pot_np[:, jdx] + y_pred_x_np[:, o_idx]

                jdx_unlc += 1
            if p_cfg.iate_se:
                y_pot_se_np[:, jdx] = np.sqrt(y_pot_var[:, t_idx, o_idx])
            jdx += 1
        for t2_idx, t2_name in enumerate(effect_list):
            name_eff += [o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            if np.isclose(t2_name[1], d_values_dr[0], atol=zero_tol, rtol=zero_tol):
                # Usually, control
                name_eff0 += [o_name + str(t2_name[0]) + 'vs' + str(t2_name[1])]
            iate_np[:, j2dx] = iate[:, o_idx, t2_idx, 0]
            if p_cfg.iate_m_ate:
                iate_mate_np[:, j2dx] = iate[:, o_idx, t2_idx, 1]

            if p_cfg.iate_se:
                iate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 0]
                if p_cfg.iate_m_ate:
                    iate_mate_se_np[:, j2dx] = iate_se[:, o_idx, t2_idx, 1]
            j2dx += 1
    name_y_pot = [s + '_pot' for s in name_pot]
    uncenter = lc_cfg.uncenter_po and isinstance(y_pred_x_df, (pd.Series, pd.DataFrame,))
    if uncenter:
        name_y_pot_unlc = [s + '_pot' for s in name_pot_unlc]
    txt = '_liate' if iv else '_iate'
    name_iate = [s + txt for s in name_eff]
    name_iate0 = [s + txt for s in name_eff0]
    name_y_pot_se = name_iate_se = None
    name_iate_mate = name_iate_mate0 = name_iate_mate_se = None
    name_iate_se0 = name_iate_mate_se0 = None
    if p_cfg.iate_m_ate:
        txt = '_liatemlate' if iv else '_iatemate'
        name_iate_mate = [s + txt for s in name_eff]
        name_iate_mate0 = [s + txt for s in name_eff0]

    if p_cfg.iate_se:
        name_y_pot_se = [s + '_pot_se' for s in name_pot]
        txt = '_liate_se' if iv else '_iate_se'
        name_iate_se = [s + txt for s in name_eff]
        name_iate_se0 = [s + txt for s in name_eff0]
        if p_cfg.iate_m_ate:
            txt = '_liatemlate_se' if iv else '_iatemate_se'
            name_iate_mate_se = [s + txt for s in name_eff]
            name_iate_mate_se0 = [s + txt for s in name_eff0]
    if gen_cfg.with_output or gen_cfg.return_iate_sp:
        y_pot_df = pd.DataFrame(data=y_pot_np, columns=name_y_pot)
        iate_df = pd.DataFrame(data=iate_np, columns=name_iate)
        if p_cfg.iate_m_ate:
            iate_mate_df = pd.DataFrame(data=iate_mate_np, columns=name_iate_mate)
        else:
            iate_mate_df = None
        if p_cfg.iate_se:
            y_pot_se_df = pd.DataFrame(data=y_pot_se_np, columns=name_y_pot_se)
            iate_se_df = pd.DataFrame(data=iate_se_np, columns=name_iate_se)
            if p_cfg.iate_m_ate:
                iate_mate_se_df = pd.DataFrame(data=iate_mate_se_np, columns=name_iate_mate_se)
            else:
                iate_mate_se_df = None
        else:
            y_pot_se_df = iate_se_df = iate_mate_se_df = None

        se = bool(getattr(p_cfg, 'iate_se'))
        m_ate = bool(getattr(p_cfg, 'iate_m_ate'))
        match (se, m_ate):
            case (True, True):
                df_list = [y_pot_df, y_pot_se_df, iate_df, iate_se_df, iate_mate_df, iate_mate_se_df
                           ]
            case (True, False):
                df_list = [y_pot_df, y_pot_se_df, iate_df, iate_se_df]
            case (False, True):
                df_list = [y_pot_df, iate_df, iate_mate_df]
            case _:
                df_list = [y_pot_df, iate_df]

        if uncenter:
            pot_y_unlc_df = pd.DataFrame(data=y_pot_unlc_np, columns=name_y_pot_unlc)
            df_list.append(pot_y_unlc_df)

        results_df = pd.concat(df_list, axis=1)
        if gen_cfg.with_output:
            mcf_ps.print_mcf(gen_cfg, '\nIndividualized ATE ' + extra_title, summary=True)
            mcf_ps.print_descriptive_df(gen_cfg, results_df, varnames='all', summary=True)
    names_pot_iate = {'names_y_pot': name_y_pot, 'names_y_pot_se': name_y_pot_se,
                      'names_iate': name_iate, 'names_iate_se': name_iate_se,
                      'names_iate_mate': name_iate_mate, 'names_iate_mate_se': name_iate_mate_se,
                      }
    names_pot_iate0 = {'names_y_pot': name_y_pot, 'names_y_pot_se': name_y_pot_se,
                       'names_iate': name_iate0, 'names_iate_se': name_iate_se0,
                       'names_iate_mate': name_iate_mate0, 'names_iate_mate_se': name_iate_mate_se0,
                       }
    if uncenter:
        names_pot_iate['names_y_pot_uncenter'] = name_y_pot_unlc
        names_pot_iate0['names_y_pot_uncenter'] = name_y_pot_unlc

    if not gen_cfg.return_iate_sp:
        results_df = None

    return iate, iate_se, (names_pot_iate, names_pot_iate0), results_df, txt_iate


def iate_func2_single_backend(*, idx: int,
                              kw_args: dict[str, Any],
                              y_pot_i: Any,
                              y_pot_var_i: Any,
                              y_pot_m_ate_i: Any,
                              y_pot_m_ate_var_i: Any,
                              iate_se_flag: bool,
                              iate_m_ate_flag: bool,
                              d_sub_values: Any,
                              ) -> Any:
    """Backend-agnostic wrapper for one-observation IATE post-processing."""
    return iate_func2_for_mp_single(idx,
                                    **kw_args,
                                    y_pot_i=y_pot_i, y_pot_var_i=y_pot_var_i,
                                    y_pot_m_ate_i=y_pot_m_ate_i,
                                    y_pot_m_ate_var_i=y_pot_m_ate_var_i,
                                    iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                                    d_sub_values=d_sub_values,
                                    )


def iate_func2_mult_backend(*,
                            idx_chunk: Any,
                            kw_args: dict[str, Any],
                            pot_y_chunk: Any,
                            pot_y_var_chunk: Any,
                            pot_y_m_ate_chunk: Any,
                            pot_y_m_ate_var_chunk: Any,
                            iate_se_flag: bool,
                            iate_m_ate_flag: bool,
                            d_sub_values: Any,
                            ) -> Any:
    """Backend-agnostic wrapper for chunked IATE post-processing."""
    return iate_func2_for_mp_mult(idx_chunk,
                                  **kw_args,
                                  pot_y_chunk=pot_y_chunk, pot_y_var_chunk=pot_y_var_chunk,
                                  pot_y_m_ate_chunk=pot_y_m_ate_chunk,
                                  pot_y_m_ate_var_chunk=pot_y_m_ate_var_chunk,
                                  iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                                  d_sub_values=d_sub_values,
                                  )


def iate_func1_for_mp(idx: int,
                      weights_i: list[int], *,
                      cl_dat: ArrayLike,
                      no_of_cluster: int,
                      w_dat: ArrayLike,
                      w_ate: ArrayLike,
                      y_dat: ArrayLike,
                      no_of_out: int,
                      n_y: int,
                      ct_cfg: 'CtGrid', int_cfg: 'IntCfg', gen_cfg: 'GenCfg', p_cfg: 'PCfg',
                      ba_data: Any,
                      p_ba_cfg: 'PBaCfg',
                      gen_tv_data: Any,
                      gen_tv_cfg: 'GenTvCfg',
                      iate_se_flag: bool,
                      se_boot_iate: bool,
                      iate_m_ate_flag: bool,
                      iv: bool = False,
                      ) -> tuple[int,
                                 NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any],
                                 tuple[NDArray[Any], ..., str],
                                 NDArray[Any],
                                 ]:
    """
    Compute function to be looped over observations for Multiprocessing.

    Parameters
    ----------
    idx : Int. Counter. Element of prediction data.
    weights_i : List of int. Indices of non-zero weights.
                Alternative: Sparse csr matrix
    cl_dat : Numpy vector. Cluster variable.
    no_of_cluster : Int. Number of clusters.
    w_dat : Numpy vector. Sampling weights.
    w_ate : Numpy array. Weights for ATE.
    y_dat : Numpy array. Outcome variable.
    no_of_out : Int. Number of outcomes.
    n_y : Int. Length of outcome data.
    ct_cfg, int_cfg, gen_cfg, p_cfg : Dict, DC. Parameters.
    iate_se_flag : Boolean. Compute standard errors.
    se_boot_iate : Boolean. Compute bootstrap standard errors.
    iate_m_ate_flag : Boolean. Compute difference to average potential outcome.
    ...

    Returns
    -------
    idx: Int. Counter.
    pot_y_i: Numpy array.
    pot_y_var_i: Numpy array.
    pot_y_m_ate_i: Numpy array.
    pot_y_m_ate_var_i: Numpy array.
    l1_to_9: Tuple of lists.
    """
    def get_walli(w_index, n_y, w_i, w_i_unc):
        w_all_i = np.zeros(n_y)
        w_all_i[w_index] = w_i
        w_all_i_unc = np.zeros_like(w_all_i)
        w_all_i_unc[w_index] = w_i_unc
        return w_all_i, w_all_i_unc

    def get_weights_idx(weights_i, t_idx, weight_as_sparse):
        """Extract weights and their position in training data."""
        if weight_as_sparse:
            w_index = weights_i[t_idx].col    # scr with indices
            w_i = weights_i[t_idx].data
        else:
            w_index = weights_i[t_idx][0]    # Indices of non-zero weights
            w_i = weights_i[t_idx][1].copy()
        return w_i, w_index

    def adjust_for_weighting(w_i, w_dat, w_index, weight_as_sparse):
        """Adjust for sample weights."""
        w_t = w_dat[w_index].ravel()
        if weight_as_sparse:
            w_i = w_i * w_t
        else:
            w_i *= w_t
        return w_i, w_t

    def normalise_w(w_i, eps):
        """Normalize weights."""
        w_i_sum = np.sum(w_i)
        if abs(w_i_sum) > zero_tol and not (1 - eps) < w_i_sum < (1 + eps):
            w_i = w_i / w_i_sum
        return w_i

    # Define or extract some constants and flags
    sum_tol = int_cfg.sum_tol
    zero_tol = int_cfg.zero_tol
    weighted = gen_cfg.weighted
    weight_as_sparse = int_cfg.weight_as_sparse
    keep_w0 = int_cfg.keep_w0
    max_weight_share = p_cfg.max_weight_share
    gen_tv = gen_tv_cfg.yes
    p_ba = p_ba_cfg.yes

    weight_var = mcf_est.weight_var   # This is shortcut name for a function
    aggregate_cluster_nonzero_w = mcf_est.aggregate_cluster_nonzero_w

    cluster_std = p_cfg.cluster_std if iate_se_flag else False

    bound_norm_weights_not_one = mcf_gp.bound_norm_weights_not_one
    bound_norm_weights = mcf_gp.bound_norm_weights

    if gen_tv:
        d_tv_train = gen_tv_data.d_train
        x_tv_train = gen_tv_data.x_tv_train
        x_tv_pred = None if x_tv_train is None else gen_tv_data.x_tv_pred[idx, :].reshape(1, -1)

    if gen_cfg.d_type == 'continuous':
        continuous, d_values_dr = True, ct_cfg.d_values_dr_np
        no_of_treat = ct_cfg.grid_w
        i_w01, i_w10 = ct_cfg.w_to_dr_int_w01, ct_cfg.w_to_dr_int_w10
        index_full = ct_cfg.w_to_dr_index_full
        no_of_treat_dr = len(d_values_dr)
    else:
        continuous, d_values_dr = False, gen_cfg.d_values
        no_of_treat = no_of_treat_dr = gen_cfg.no_of_treat
        i_w01 = i_w10 = index_full = None
    pot_y_i = np.empty((no_of_treat_dr, no_of_out))

    if gen_tv:
        treat_main = gen_tv_cfg.no_of_treat_per_main
        pot_y_i = np.repeat(pot_y_i, repeats=treat_main, axis=0)
    else:
        treat_main = None

    pot_y_m_ate_i = np.empty_like(pot_y_i) if iate_m_ate_flag else None

    if iate_se_flag:
        pot_y_var_i = np.empty_like(pot_y_i)
        pot_y_m_ate_var_i = np.empty_like(pot_y_i) if iate_m_ate_flag else None
    else:
        pot_y_var_i = pot_y_m_ate_var_i = None

    tmp_diff = np.zeros(n_y)  # new: reusable scratch for w_diff (non-cluster)

    if p_ba and p_ba_cfg.adj_method == 'w_obs' and not continuous:
        # Computing evaluation weights requires access to all weights
        w_iate = np.zeros((no_of_treat, n_y), dtype=np.float64)
        for t_idx in range(no_of_treat):
            w_i, w_index = get_weights_idx(weights_i, t_idx, weight_as_sparse)
            if weighted:
                w_i, w_t = adjust_for_weighting(w_i, w_dat, w_index, weight_as_sparse)
            else:
                w_t = None
            if not (continuous or iv):
                w_i = normalise_w(w_i, sum_tol)
            w_iate[t_idx, w_index] = w_i
        ba_data.weights_eval = mcf_ba.get_weights_eval_ba(w_iate, no_of_treat, zero_tol=zero_tol)
    share_i = np.zeros(pot_y_i.shape[0])
    w_add = (np.zeros((pot_y_i.shape[0], no_of_cluster)) if cluster_std
             else np.zeros((pot_y_i.shape[0], n_y))
             )
    # From here onwards, distinguish between main and subtreated
    no_of_m_treat = no_of_treat
    if gen_tv:
        no_treat_per_main = gen_tv_cfg.no_of_treat_per_main
    else:
        no_treat_per_main = [1] * no_of_m_treat   # No treatment versions

    t_idx = 0    # All treatments
    for t_m_idx in range(no_of_m_treat):

        extra_weight_p1 = continuous and t_m_idx < no_of_m_treat-1
        w_i_t, w_index = get_weights_idx(weights_i, t_m_idx, weight_as_sparse)

        if extra_weight_p1:  # Only continuous case
            w_index_p1 = (weights_i[t_m_idx+1].col if weight_as_sparse
                          else weights_i[t_m_idx+1][0]
                          )
            w_index_both = np.unique(np.concatenate((w_index, w_index_p1)))
            w_i = np.zeros(n_y)
            w_i[w_index] = w_i_t
            w_i_p1 = np.zeros_like(w_i)
            if weight_as_sparse:
                w_i_p1[w_index_p1] = weights_i[t_m_idx+1].data
            else:
                w_i_p1[w_index_p1] = weights_i[t_m_idx+1][1].copy()
            w_i = w_i[w_index_both]
            w_i_p1 = w_i_p1[w_index_both]
        else:
            w_index_both = w_index
            w_i = w_i_t

        w_t_p1 = w_dat[w_index_p1].ravel() if weighted and extra_weight_p1 else None
        if weighted:
            w_i, w_t = adjust_for_weighting(w_i, w_dat, w_index, weight_as_sparse)
        else:
            w_t = None

        if not (continuous or iv):
            w_i = normalise_w(w_i, sum_tol)

        if p_ba:
            # Bias correction
            w_i = mcf_ba.bias_correction_wregr(w_i,
                                               y_dat[w_index, 0],
                                               ba_data,
                                               int_dtype=np.float64, out_dtype=np.float32,
                                               pos_weights_only=p_ba_cfg.pos_weights_only,
                                               w_index=w_index,
                                               zero_tol=zero_tol,
                                               ridge=p_ba_cfg.ridge,
                                               cv_k=p_ba_cfg.cv_k,
                                               )
        # Expand weights
        results_container_w, results_container_r = None, None
        for t_sub_idx in range(no_treat_per_main[t_m_idx]):
            if gen_tv:
                (w_i, results_container_w, version_res_dat, results_container_r, _, _, _
                 ) = version_wregr(w_i,
                                   y_train=y_dat[:, 0],
                                   d_train=d_tv_train,
                                   x_train=x_tv_train,
                                   x_pred=x_tv_pred,
                                   cfg=gen_tv_cfg,
                                   container_w=results_container_w,
                                   container_r=results_container_r,
                                   w_index=w_index,
                                   treat_idx=t_idx,
                                   maintreat_idx=t_m_idx,
                                   subtreat_idx=t_sub_idx,
                                   int_dtype=np.float64, out_dtype=np.float32,
                                   zero_tol=int_cfg.zero_tol,
                                   ridge=gen_tv_cfg.estimator == 'ridge',
                                   penalize_version=gen_tv_cfg.penalize_version[t_m_idx],
                                   return_residuals=True,
                                   standardize_x=True,
                                   )
            else:
                version_res_dat = None
            w_i_unc = w_i.copy()

            if max_weight_share < 1 and not continuous and not p_ba:
                denom = w_i.sum()
                # handle degenerate all-zero case gracefully
                if np.abs(denom) > sum_tol:
                    max_share_now = w_i.max() / denom
                    if max_share_now > (max_weight_share + zero_tol):
                        # only call the heavier routine if it’s actually needed
                        if iv:
                            w_i, _, share_i[t_idx] = bound_norm_weights_not_one(
                                w_i, max_weight_share,
                                zero_tol=zero_tol, sum_tol=sum_tol,
                                )
                        else:
                            w_i, _, share_i[t_idx] = bound_norm_weights(
                                w_i,
                                max_weight_share=max_weight_share, zero_tol=zero_tol,
                                sum_tol=sum_tol, negative_weights_possible=p_ba or gen_tv,
                                )
                    else:
                        # keep weights as-is and record the current max share
                        # (for diagnostics)
                        share_i[t_idx] = max_share_now
                else:
                    share_i[t_idx] = 0.0

            if extra_weight_p1:
                w_i_unc_p1 = np.copy(w_i_p1)
            if cluster_std:
                w_all_i, w_all_i_unc = get_walli(w_index, n_y, w_i, w_i_unc)
                cl_i = cl_dat[w_index]
                if extra_weight_p1:
                    w_all_i_p1, w_all_i_unc_p1 = get_walli(w_index_p1, n_y, w_i_p1, w_i_unc_p1)
                    cl_i_both = cl_dat[w_index_both]
                else:
                    cl_i_both = None
            else:
                cl_i = cl_i_both = None

            for o_idx in range(no_of_out):
                y_col = y_dat[:, o_idx]
                if continuous:
                    y_dat_cont = y_col[w_index_both]
                    for i, (w01, w10) in enumerate(zip(i_w01, i_w10)):
                        if extra_weight_p1:
                            w_i_cont = w10 * w_i + w01 * w_i_p1
                            w_i_unc_cont = w10 * w_i_unc + w01 * w_i_unc_p1
                            w_t_cont = None if w_t is None else w10 * w_t + w01 * w_t_p1
                            cl_i_cont = cl_i_both
                        else:
                            w_i_cont, w_t_cont, cl_i_cont = w_i, w_t, cl_i_both
                            w_i_unc_cont = w_i_unc
                        w_i_cont = w_i_cont / np.sum(w_i_cont)
                        if w_t_cont is not None:
                            w_t_cont = w_t_cont / np.sum(w_t_cont)
                        w_i_unc_cont = w_i_unc_cont / np.sum(w_i_unc_cont)

                        if max_weight_share < 1:
                            if iv:
                                (w_i_cont, _, share_cont
                                 ) = bound_norm_weights_not_one(w_i_cont, max_weight_share,
                                                                zero_tol=zero_tol, sum_tol=sum_tol,
                                                                )
                            else:
                                w_i_cont, _, share_cont = bound_norm_weights(
                                    w_i_cont,
                                    max_weight_share=max_weight_share, zero_tol=zero_tol,
                                    sum_tol=sum_tol, negative_weights_possible=p_ba or gen_tv,
                                    )
                            if i == 0:
                                share_i[t_idx] = share_cont
                        ret = weight_var(
                            w_i_cont, y_dat_cont, cl_i_cont, p_cfg,
                            residual_dat=version_res_dat,
                            weighted=gen_cfg.weighted, weights=w_t_cont, se_yes=iate_se_flag,
                            bootstrap=se_boot_iate, keep_all=keep_w0,
                            normalize=not iv,
                            zero_tol=zero_tol, sum_tol=sum_tol,
                            seed=123345, min_obs=5,
                            )
                        ti_idx = index_full[t_idx, i]  # pylint: disable=E1136
                        pot_y_i[ti_idx, o_idx] = ret[0]
                        if iate_se_flag:
                            pot_y_var_i[ti_idx, o_idx] = ret[1]
                        if cluster_std:
                            w_cont = ((w10 * w_all_i + w01 * w_all_i_p1) if extra_weight_p1
                                      else w_all_i
                                      )
                            if o_idx == 0:
                                ret2 = aggregate_cluster_nonzero_w(cl_dat, w_cont,
                                                                   y_dat=y_col, sweights=w_dat,
                                                                   )
                                w_add[ti_idx, :] = ret2[0].copy()

                                if iate_m_ate_flag:
                                    if w_ate is None:
                                        w_diff = ((w10 * w_all_i_unc + w01 * w_all_i_unc_p1)
                                                  if extra_weight_p1 else w_all_i_unc
                                                  )
                                    else:
                                        if extra_weight_p1:
                                            w_ate_cont = (
                                                w10 * w_ate[t_idx, :] + w01 * w_ate[t_idx+1, :]
                                                )
                                            w_ate_cont /= np.sum(w_ate_cont)
                                            w_diff = (w10 * w_all_i_unc + w01 * w_all_i_unc_p1
                                                      - w_ate_cont
                                                      )
                                        else:
                                            w_diff = w_all_i_unc - w_ate[t_idx, :]
                            if iate_m_ate_flag:
                                ret = weight_var(
                                    w_diff, y_col, cl_dat, p_cfg,
                                    residual_dat=None,
                                    weighted=gen_cfg.weighted, normalize=False, weights=w_dat,
                                    bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                    keep_all=keep_w0,
                                    zero_tol=zero_tol, sum_tol = sum_tol,
                                    seed=123345, min_obs=5,
                                )
                        else:
                            if o_idx == 0:
                                w_add[ti_idx, w_index_both] = ret[2]
                                if iate_m_ate_flag:
                                    # build normalized uncentered weights directly
                                    # into scratch
                                    tmp_diff.fill(0.0)
                                    tmp_diff[w_index_both] = w_i_unc_cont
                                    s_unc = tmp_diff[w_index_both].sum()
                                    if not (1 - sum_tol) < s_unc < (1 + sum_tol):
                                        tmp_diff[w_index_both] /= s_unc

                                    if w_ate is None:
                                        w_diff = tmp_diff
                                    else:
                                        if extra_weight_p1:
                                            w_ate_cont = (w10 * w_ate[t_idx, :]
                                                          + w01 * w_ate[t_idx+1, :]
                                                          )
                                            w_ate_cont /= np.sum(w_ate_cont)
                                            w_diff = tmp_diff - w_ate_cont
                                        else:
                                            w_diff = tmp_diff - w_ate[t_idx, :]

                            if iate_m_ate_flag:
                                ret = weight_var(w_diff, y_col, None, p_cfg,
                                                 residual_dat=None, weighted=gen_cfg.weighted,
                                                 normalize=False, weights=w_dat,
                                                 bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                                 keep_all=keep_w0, zero_tol=zero_tol,
                                                 sum_tol=sum_tol, seed=123345, min_obs=5,
                                                 )
                        if iate_m_ate_flag:
                            pot_y_m_ate_i[ti_idx, o_idx] = ret[0]
                        if iate_se_flag and iate_m_ate_flag:
                            pot_y_m_ate_var_i[ti_idx, o_idx] = ret[1]
                        if not extra_weight_p1:
                            break
                else:  # discrete treatment
                    ret = weight_var(w_i, y_col[w_index], cl_i, p_cfg,
                                     residual_dat=version_res_dat,
                                     weighted=gen_cfg.weighted, weights=w_t, se_yes=iate_se_flag,
                                     bootstrap=se_boot_iate, keep_all=keep_w0,
                                     normalize=not iv, zero_tol=zero_tol, sum_tol=sum_tol,
                                     seed=123345, min_obs=5,
                                     )
                    pot_y_i[t_idx, o_idx] = ret[0]
                    if iate_se_flag:
                        pot_y_var_i[t_idx, o_idx] = ret[1]
                    if cluster_std:
                        if o_idx == 0:
                            ret2 = aggregate_cluster_nonzero_w(cl_dat, w_all_i,
                                                               y_dat=y_col, sweights=w_dat,
                                                               )
                            w_add[t_idx, :] = ret2[0].copy()
                            if w_ate is None:
                                w_diff = w_all_i_unc  # Dummy if no w_ate
                            else:
                                w_diff = w_all_i_unc - w_ate[t_idx, :]
                        if iate_m_ate_flag:
                            ret = weight_var(
                                w_diff, y_col, cl_dat, p_cfg,
                                residual_dat=None,
                                weighted=gen_cfg.weighted, normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate,
                                se_yes=iate_se_flag, keep_all=keep_w0,
                                zero_tol=zero_tol, sum_tol=sum_tol,
                                seed=123345, min_obs=5,
                                )
                    else:
                        if o_idx == 0:
                            w_add[t_idx, w_index] = ret[2]
                            if iate_m_ate_flag:
                                # build normalized uncentered weights directly
                                # into scratch
                                tmp_diff.fill(0.0)
                                tmp_diff[w_index] = w_i_unc
                                s_unc = tmp_diff[w_index].sum()
                                if not (1 - sum_tol) < s_unc < (1 + sum_tol):
                                    tmp_diff[w_index] /= s_unc

                                w_diff = tmp_diff if w_ate is None else tmp_diff - w_ate[t_idx, :]
                        if iate_m_ate_flag:
                            ret = weight_var(
                                w_diff, y_col, None, p_cfg,
                                residual_dat=None,
                                weighted=gen_cfg.weighted, normalize=False, weights=w_dat,
                                bootstrap=se_boot_iate, se_yes=iate_se_flag,
                                keep_all=keep_w0,
                                zero_tol=zero_tol, sum_tol=sum_tol,
                                seed=123345, min_obs=5,
                                )
                    if iate_m_ate_flag:
                        pot_y_m_ate_i[t_idx, o_idx] = ret[0]
                    if iate_m_ate_flag and iate_se_flag:
                        pot_y_m_ate_var_i[t_idx, o_idx] = ret[1]
            t_idx += 1

    l1_to_9 = mcf_est.analyse_weights(w_add, None, gen_cfg, p_cfg,
                                      gen_tv_total_treat=
                                          gen_tv_cfg.no_of_treat_all if gen_tv else None,
                                      ate=False, continuous=continuous,
                                      no_of_treat_cont=no_of_treat_dr, d_values_cont=d_values_dr,
                                      zero_tol=zero_tol,
                                      )
    return idx, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i, l1_to_9, share_i


def assign_ret_all_i(pot_y: NDArray[Any],
                     pot_y_var: ArrayLike,
                     pot_y_m_ate: ArrayLike,
                     pot_y_m_ate_var: ArrayLike, *,
                     l1_to_9: tuple[NDArray[Any], ..., str],
                     share_censored: float,
                     ret_all_i: Any,
                     n_x: int,
                     idx: int | None = None
                     ) -> tuple[NDArray[Any], ArrayLike, ArrayLike, ArrayLike, Any, Any]:
    """Use to avoid duplicate code."""
    if idx is None:
        idx = ret_all_i[0]
    pot_y[idx, :, :] = ret_all_i[1]
    if pot_y_m_ate is not None:
        pot_y_m_ate[idx, :, :] = ret_all_i[3]
    if pot_y_var is not None:
        pot_y_var[idx, :, :] = ret_all_i[2]
    if pot_y_m_ate_var is not None:
        pot_y_m_ate_var[idx, :, :] = ret_all_i[4]
    l1_to_9[idx] = ret_all_i[5]
    share_censored += ret_all_i[6] / n_x

    return pot_y, pot_y_var, pot_y_m_ate, pot_y_m_ate_var, l1_to_9, share_censored


# @ray.remote
def _ray_iate_func1_for_mp_many_obs_impl(idx_list, weights_list, *,
                                         cl_dat, no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y,
                                         ct_cfg, int_cfg, gen_cfg, p_cfg, ba_data, p_ba_cfg,
                                         gen_tv_data, gen_tv_cfg, iate_se_flag, se_boot_iate,
                                         iate_m_ate_flag,
                                         iv=False,
                                         ):
    """Compute IATE for several obs in one loop (MP)."""
    return iate_func1_for_mp_many_obs(idx_list, weights_list,
                                      cl_dat=cl_dat, no_of_cluster=no_of_cluster, w_dat=w_dat,
                                      w_ate=w_ate, y_dat=y_dat, no_of_out=no_of_out, n_y=n_y,
                                      ct_cfg=ct_cfg, int_cfg=int_cfg, gen_cfg=gen_cfg, p_cfg=p_cfg,
                                      ba_data=ba_data, p_ba_cfg=p_ba_cfg, gen_tv_data=gen_tv_data,
                                      gen_tv_cfg=gen_tv_cfg, iate_se_flag=iate_se_flag,
                                      se_boot_iate=se_boot_iate, iate_m_ate_flag=iate_m_ate_flag,
                                      iv=iv
                                      )


ray_iate_func1_for_mp_many_obs = (ray.remote(_ray_iate_func1_for_mp_many_obs_impl)
                                  if ray is not None else None
                                  )


def iate_func1_for_mp_many_obs(idx_list, weights_list, *,
                               cl_dat, no_of_cluster, w_dat, w_ate, y_dat, no_of_out, n_y, ct_cfg,
                               int_cfg, gen_cfg, p_cfg, ba_data, p_ba_cfg, gen_tv_data, gen_tv_cfg,
                               iate_se_flag, se_boot_iate, iate_m_ate_flag,
                               iv=False
                               ):
    """Compute IATE for several obs in one loop (MP)."""
    ret_all = []
    if int_cfg.weight_as_sparse:
        iterator = len(weights_list)
    for i, idx_org in enumerate(idx_list):
        if int_cfg.weight_as_sparse:
            weights_i = [weights_list[t_idx][i, :] for t_idx in range(iterator)]
        else:
            weights_i = weights_list[i]
        ret = iate_func1_for_mp(idx_org, weights_i,
                                cl_dat=cl_dat, no_of_cluster=no_of_cluster, w_dat=w_dat,
                                w_ate=w_ate, y_dat=y_dat, no_of_out=no_of_out, n_y=n_y,
                                ct_cfg=ct_cfg, int_cfg=int_cfg, gen_cfg=gen_cfg, p_cfg=p_cfg,
                                ba_data=ba_data, p_ba_cfg=p_ba_cfg, gen_tv_data=gen_tv_data,
                                gen_tv_cfg=gen_tv_cfg, iate_se_flag=iate_se_flag,
                                se_boot_iate=se_boot_iate, iate_m_ate_flag=iate_m_ate_flag, iv=iv,
                                )
        ret_all.append(ret)

    return ret_all


# @ray.remote
def _ray_iate_func2_for_mp_single_impl(idx, *,
                                       no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i,
                                       pot_y_m_ate_var_i, d_type, d_values, no_of_treat,
                                       iate_se_flag, iate_m_ate_flag,
                                       d_sub_values=None,
                                       ):
    """Make function compatible with Ray."""
    return iate_func2_for_mp(idx,
                             no_of_out=no_of_out, pot_y_i=pot_y_i, pot_y_var_i=pot_y_var_i,
                             pot_y_m_ate_i=pot_y_m_ate_i, pot_y_m_ate_var_i=pot_y_m_ate_var_i,
                             d_type=d_type, d_values=d_values, no_of_treat=no_of_treat,
                             iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                             d_sub_values=d_sub_values
                             )


ray_iate_func2_for_mp_single = (ray.remote(_ray_iate_func2_for_mp_single_impl)
                                  if ray is not None else None
                                  )


def iate_func2_for_mp_single(idx, *,
                             no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
                             d_type, d_values, no_of_treat, iate_se_flag, iate_m_ate_flag,
                             d_sub_values=None,
                             ):
    """Make function compatible with Ray."""
    return iate_func2_for_mp(idx,
                             no_of_out=no_of_out, pot_y_i=pot_y_i, pot_y_var_i=pot_y_var_i,
                             pot_y_m_ate_i=pot_y_m_ate_i, pot_y_m_ate_var_i=pot_y_m_ate_var_i,
                             d_type=d_type, d_values=d_values, no_of_treat=no_of_treat,
                             iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                             d_sub_values=d_sub_values
                             )


# @ray.remote
def _ray_iate_func2_for_mp_mult_impl(idx_list, *,
                                    no_of_out, pot_y_chunk, pot_y_var_chunk, pot_y_m_ate_chunk,
                                    pot_y_m_ate_var_chunk, d_type, d_values, no_of_treat,
                                    iate_se_flag,
                                    iate_m_ate_flag, d_sub_values=None
                                    ):
    """Make function compatible with Ray for multiple indices."""
    if pot_y_var_chunk is None and pot_y_m_ate_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=None, pot_y_m_ate_i=None, pot_y_m_ate_var_i=None,
                                       d_type=d_type, d_values=d_values, no_of_treat=no_of_treat,
                                       iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    elif pot_y_var_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=None, pot_y_m_ate_i=pot_y_m_ate_chunk[pos],
                                       pot_y_m_ate_var_i=None, d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag, d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    elif pot_y_m_ate_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=pot_y_var_chunk[pos], pot_y_m_ate_i=None,
                                       pot_y_m_ate_var_i=None, d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    else:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=pot_y_var_chunk[pos],
                                       pot_y_m_ate_i=pot_y_m_ate_chunk[pos],
                                       pot_y_m_ate_var_i=pot_y_m_ate_var_chunk[pos],
                                       d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]

    return iate_list


ray_iate_func2_for_mp_mult = (ray.remote(_ray_iate_func2_for_mp_mult_impl) if ray is not None
                              else None
                              )


def iate_func2_for_mp_mult(idx_list, *,
                           no_of_out, pot_y_chunk, pot_y_var_chunk, pot_y_m_ate_chunk,
                           pot_y_m_ate_var_chunk, d_type, d_values, no_of_treat, iate_se_flag,
                           iate_m_ate_flag, d_sub_values=None
                           ):
    """Make function compatible with Ray for multiple indices."""
    if pot_y_var_chunk is None and pot_y_m_ate_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=None, pot_y_m_ate_i=None, pot_y_m_ate_var_i=None,
                                       d_type=d_type, d_values=d_values, no_of_treat=no_of_treat,
                                       iate_se_flag=iate_se_flag, iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    elif pot_y_var_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=None, pot_y_m_ate_i=pot_y_m_ate_chunk[pos],
                                       pot_y_m_ate_var_i=None, d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag, d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    elif pot_y_m_ate_chunk is None and pot_y_m_ate_var_chunk is None:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=pot_y_var_chunk[pos], pot_y_m_ate_i=None,
                                       pot_y_m_ate_var_i=None, d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    else:
        iate_list = [iate_func2_for_mp(idx, no_of_out=no_of_out, pot_y_i=pot_y_chunk[pos],
                                       pot_y_var_i=pot_y_var_chunk[pos],
                                       pot_y_m_ate_i=pot_y_m_ate_chunk[pos],
                                       pot_y_m_ate_var_i=pot_y_m_ate_var_chunk[pos],
                                       d_type=d_type, d_values=d_values,
                                       no_of_treat=no_of_treat, iate_se_flag=iate_se_flag,
                                       iate_m_ate_flag=iate_m_ate_flag,
                                       d_sub_values=d_sub_values,
                                       )
                     for pos, idx in enumerate(idx_list)
                     ]
    return iate_list



def iate_func2_for_mp(idx, *,
                      no_of_out, pot_y_i, pot_y_var_i, pot_y_m_ate_i, pot_y_m_ate_var_i,
                      d_type, d_values, no_of_treat, iate_se_flag, iate_m_ate_flag,
                      d_sub_values=None,
                      ):
    """
    Do computations for IATE with MP. Second chunck.

    Parameters
    ----------
    idx : Int. Counter.
    no_of_out : Int. Number of outcomes.
    pot_y_i : Numpy array.
    pot_y_var_i : Numpy array.
    pot_y_m_ate_i : Numpy array.
    pot_y_m_ate_var_i : Numpy array.
    c_dict : Dict. Parameters.

    Returns
    -------
    iate_i : Numpy array.
    iate_se_i : Numpy array.
    iate_p_i : Numpy array.
    effect_list : List.
    """
    # obs x outcome x effects x type_of_effect
    if d_type == 'continuous':
        dim = (no_of_out, no_of_treat - 1, 2)
    else:
        dim = (no_of_out, round(no_of_treat * (no_of_treat - 1) / 2), 2)
    iate_i = np.empty(dim)
    if iate_se_flag:
        iate_se_i = np.empty(dim)  # obs x outcome x effects x type_of_effect
        iate_p_i = np.empty_like(iate_se_i)
    else:
        iate_se_i = iate_p_i = None
    iterator = 2 if iate_m_ate_flag else 1
    compute_comparison_label = True

    old_filters = warnings.filters.copy()
    warnings.filterwarnings('error', category=RuntimeWarning)
    try:
        effect_list = None
        for o_i in range(no_of_out):
            for jdx in range(iterator):
                if jdx == 0:
                    pot_y_ao = pot_y_i[:, o_i]
                    pot_y_var_ao = pot_y_var_i[:, o_i] if iate_se_flag else None
                else:
                    pot_y_ao = pot_y_m_ate_i[:, o_i]
                    pot_y_var_ao = pot_y_m_ate_var_i[:, o_i] if iate_se_flag else None
                ret = mcf_est.effect_from_potential(pot_y_ao, pot_y_var_ao, d_values,
                                                    se_yes=iate_se_flag,
                                                    continuous=d_type == 'continuous',
                                                    return_comparison=compute_comparison_label,
                                                    d_sub_values=d_sub_values,
                                                    )
                if compute_comparison_label:
                    effect_list = ret[4]
                if iate_se_flag:
                    iate_i[o_i, :, jdx], iate_se_i[o_i, :, jdx], _, iate_p_i[o_i, :, jdx] = ret[:4]
                else:
                    iate_i[o_i, :, jdx], _, _, _ = ret[:4]
                compute_comparison_label = False
    finally:
        warnings.filters = old_filters

    return idx, iate_i, iate_se_i, iate_p_i, effect_list


def get_chunck_of_indices(n_x, size_per_chunk):
    """Create chuncks of indices as list to be used for array splitting."""
    indices = np.arange(n_x)
    num_chunks = np.ceil(size_per_chunk)

    return np.array_split(indices, num_chunks)
