"""
Created on Tue Mar 31 14:05:57 2026.

@author: MLechner

# -*- coding: utf-8 -*-

Contains main level files for the two different ways of computing effects (with full and sequential
weight matrix).
"""
from copy import deepcopy
from time import time
from typing import Any, TYPE_CHECKING
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
try:
    import ray
except ImportError:
    ray = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import mcf.mcf_effect_helpers as he
from mcf.mcf_ate import ate_est
from mcf.mcf_ate_low_memory import ate_weights, ate_estimate_pot
from mcf.mcf_estimation import aggregate_pots
from mcf.mcf_gate import bgate_est, gate_est
from mcf.mcf_gate_low_memory import (bgate_est_low_memory, gate_weights, gate_estimate_pot,
                                     prepare_gate_context,
                                     )
from mcf.mcf_general import progress_clean_memory, split_dataframe, tqdm_setup
from mcf.mcf_general_sys import find_no_of_workers, print_mememory_statistics
from mcf.mcf_iate import iate_est_mp
from mcf.mcf_qiate import qiate_est
from mcf.mcf_versions import fit_tv_feature_spec, transform_x_tv
from mcf.mcf_weight import get_weights_mp
from mcf.mcfoptp_parallel_backend_ray_classical import (init_ray_with_fallback, print_object_store,
                                                        ray_del_refs,
                                                        )
from mcf.mcfoptp_parallel_backend_forest_executor import make_forest_executor, map_task_batches
from mcf.mcfoptp_parallel_backends_base import print_runtime_info, SequentialExecutor, TaskSpec

type ArrayLike = NDArray[Any] | None
if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg
    from mcf.mcf_effect_helpers import EffParaFlags, EffectDicts


def effect_with_sequential_weights(mcf_: 'ModifiedCausalForest', *,
                                   data_df: DataFrame,
                                   forest_dic: dict,
                                   eff_flags: 'EffParaFlags',
                                   effect: 'EffectDicts',
                                   time_: he.TimeState,
                                   round_: str,
                                   idx_round: int,
                                   fold: int,
                                   ) -> None:   # Dataclasses (effect, time) will be updated
    """Compute the effects by first computing weights and then effects with full weight matrix."""
    gen_cfg, int_cfg, p_cfg = mcf_.gen_cfg, mcf_.int_cfg, mcf_.p_cfg
    use_tqdm, output_clean = tqdm_setup(tqdm, gen_cfg.with_output and gen_cfg.verbose)
    # mcf_seq = deepcopy(mcf_)    Needs far too much memory
    # mcf_seq.gen_cfg.mp_parallel = 1
    # mcf_seq.gen_cfg.with_output = False
    mcf_seq = make_mcf_seq_for_low_memory(mcf_)
    start_time_w = time()
    # Split dataframe in pieces of approximate equal size
    data_df_split, indices_split = split_dataframe(data_df,
                                                   max_chunk_size=mcf_.low_mem_cfg.max_chunksize,
                                                   reset_index=True,
                                                   )
    no_of_chunks = len(data_df_split)
    n_rows = len(data_df)

    gate_context = None
    if p_cfg.gate and (round_ == 'regular' or eff_flags.gate_eff):
        # Precompute bandwidth for continuous Z with the full sample
        gate_context = prepare_gate_context(mcf_seq, data_df=data_df)

    if mcf_.gen_tv_cfg.yes and p_cfg.iate and (round_ == 'regular' or eff_flags.iate_eff):
        # Precompute normalisation and dummy creation on full sample
        x_tv_pred_full_df = data_df[mcf_.var_cfg.x_name_tv].copy()
    else:
        x_tv_pred_full_df = None

    maxworkers = get_maxworkers(gen_cfg=gen_cfg, zero_tol=int_cfg.zero_tol)
    if gen_cfg.with_output and gen_cfg.verbose and int_cfg.memory_print:
        print_mememory_statistics(gen_cfg, 'Sequential weighting and IATE estimation: Start')

    if gen_cfg.with_output and gen_cfg.verbose:
        print_runtime_info(gen_cfg, int_cfg, maxworkers, txt_method='sequential weights and IATEs')

    w_ate = w_bala = w_gate = None
    pot_iate = pot_iate_var = weights_dic_data = txt_iate = None
    kw_get_weights = {'forest_dic': forest_dic, 'round_': round_, 'mcf_seq': mcf_seq,
                      'eff_flags':eff_flags, 'gate_context': gate_context,
                      'indices_split': indices_split, 'x_tv_pred_full_df': x_tv_pred_full_df,
                      }
    time_.weight += time() - start_time_w
    if maxworkers == 1:
        if use_tqdm:
            iterator = enumerate(tqdm(data_df_split, total=no_of_chunks, desc='Sequential weights',
                                      leave=False, dynamic_ncols=True,
                                      )
                                 )
        else:
            iterator = enumerate(data_df_split)
        for chunk_idx, data_df_chunk in iterator:
            (_, time_chunk, weights_dic_data_chunk, pot_iate_chunk, pot_iate_var_chunk,
             txt_iate_chunk, w_ate_chunk, w_bala_chunk, w_gate_chunk,
             ) = get_weights_aggregateweights_iate_chunk(data_df_chunk,
                                                         chunk_idx=chunk_idx,
                                                         **kw_get_weights,
                                                         )
            (time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var
             ) = update_weights_iates(time_=time_,
                                      w_ate=w_ate, w_bala=w_bala, w_gate=w_gate,
                                      pot_iate=pot_iate, pot_iate_var=pot_iate_var,
                                      time_chunk=time_chunk,
                                      w_ate_chunk=w_ate_chunk, w_bala_chunk=w_bala_chunk,
                                      w_gate_chunk=w_gate_chunk,
                                      pot_iate_chunk=pot_iate_chunk,
                                      pot_iate_var_chunk=pot_iate_var_chunk,
                                      indices=indices_split[chunk_idx], n_rows=n_rows
                                      )
            progress_clean_memory(output=output_clean, current_idx=chunk_idx + 1,
                                  total=no_of_chunks, clean_mem=False
                                  )
            if chunk_idx == 0:
                weights_dic_data = weights_dic_data_chunk
                txt_iate = txt_iate_chunk

    elif (int_cfg.mp_use_old_ray and ray is not None
              and ray_get_weights_aggregateweights_iate_chunk is not None
              ):
        if not ray.is_initialized():
            ray_err_txt='Ray does not start up in weight estimation 1'
            init_ray_with_fallback(maxworkers, gen_cfg,
                                   mem_object_store=int_cfg.mem_object_store_3,
                                   mem_object_store_2=int_cfg.mem_object_store_2,
                                   ray_err_txt=ray_err_txt,
                                   )
        print_object_store(gen_cfg, int_cfg.mem_object_store_3)
        kw_get_weights_ref = ray.put(kw_get_weights)

        still_running = [ray_get_weights_aggregateweights_iate_chunk.remote(data_df_chunk,
                                                                            chunk_idx=chunk_idx,
                                                                            kw=kw_get_weights_ref,
                                                                            )
                         for chunk_idx, data_df_chunk in enumerate(data_df_split)
                         ]
        jdx = 0
        pbar = (tqdm(total=no_of_chunks, desc='Sequential weights', leave=False, dynamic_ncols=True,
                     )
                if use_tqdm else None
                )
        try:
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running, num_returns=1)
                finished_res = ray.get(finished)
                for result in finished_res:
                    (chunk_idx, time_chunk, weights_dic_data_chunk,
                     pot_iate_chunk, pot_iate_var_chunk, txt_iate_chunk,
                     w_ate_chunk, w_bala_chunk, w_gate_chunk,
                     ) = result

                    (time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var
                     ) = update_weights_iates(time_=time_,
                                              w_ate=w_ate, w_bala=w_bala, w_gate=w_gate,
                                              pot_iate=pot_iate, pot_iate_var=pot_iate_var,
                                              time_chunk=time_chunk,
                                              w_ate_chunk=w_ate_chunk,
                                              w_bala_chunk=w_bala_chunk,
                                              w_gate_chunk=w_gate_chunk,
                                              pot_iate_chunk=pot_iate_chunk,
                                              pot_iate_var_chunk=pot_iate_var_chunk,
                                              indices=indices_split[chunk_idx],
                                              n_rows=n_rows,
                                              add_chunk_time=False,
                                              )

                    if chunk_idx == 0:
                        weights_dic_data = weights_dic_data_chunk
                        txt_iate = txt_iate_chunk

                    jdx += 1
                    if pbar is not None:
                        pbar.update(1)

                    progress_clean_memory(output=output_clean, current_idx=jdx, total=no_of_chunks,
                                          clean_mem=False
                                          )
        finally:
            if pbar is not None:
                pbar.close()

        (kw_get_weights_ref, finished_res, finished,    # pylint: disable=unbalanced-tuple-unpacking
         ) = ray_del_refs(kw_get_weights_ref,
                          f1=finished_res, f2=finished, mp_ray_del=int_cfg.mp_ray_del,
                          )
    else:
        start_time_w = time()
        try:
            ray_err_txt = 'Ray does not start up in sequential weight estimation.'
            executor = make_forest_executor(int_cfg=int_cfg, maxworkers=maxworkers,
                                            ray_err_txt=ray_err_txt,
                                            )
        except (RuntimeError, ImportError) as exc:
            if int_cfg.mp_backend in ('ray', 'joblib'):
                maxworkers = 1
                executor = SequentialExecutor()
            else:
                failtext = 'Failed to make executor in sequential weight estimation.'
                raise RuntimeError(failtext) from exc
        time_.weight += time() - start_time_w
        try:
            start_time_w = time()
            data_handle = executor.put_shared(kw_get_weights, name='sequential_weight_data')
            tasks = [TaskSpec(func=backend_get_weights_aggregateweights_iate_chunk,
                              args=(data_df_chunk,),
                              kwargs={'chunk_idx': chunk_idx, 'data': data_handle},
                              name=f'sequential_weight_chunk_{chunk_idx}',
                              )
                     for chunk_idx, data_df_chunk in enumerate(data_df_split)
                     ]
            (time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var,
             weights_dic_data, txt_iate,
             ) = sequential_weights_parallel(executor=executor, tasks=tasks,
                                             int_cfg=int_cfg, gen_cfg=gen_cfg,
                                             maxworkers=maxworkers, no_of_chunks=no_of_chunks,
                                             time_=time_,
                                             w_ate=w_ate, w_bala=w_bala, w_gate=w_gate,
                                             pot_iate=pot_iate, pot_iate_var=pot_iate_var,
                                             indices_split=indices_split, n_rows=n_rows,
                                             weights_dic_data=weights_dic_data,
                                             txt_iate=txt_iate,
                                             )
            if maxworkers > 1:
                time_.weight += time() - start_time_w   # time_ not updated inside function
        finally:
            executor.shutdown()
    if gen_cfg.with_output and gen_cfg.verbose and int_cfg.memory_print:
        print_mememory_statistics(gen_cfg, 'Sequential weighting and IATE estimation: End')

    # 3) Compute IATEs (effects)
    if p_cfg.iate and (round_ == 'regular' or eff_flags.iate_eff):
        effect.iate_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                           pot_iate, pot_iate_var,
                                                           effect_dic=effect.iate_dic[idx_round],
                                                           txt=txt_iate, fold=fold, title='IATE',
                                                           )
        del pot_iate, pot_iate_var   # Remove fromn memory
        time_.iate += dtime

    # 4) Compute aggregate effects
    if he.need_ate_weights(round_, eff_flags, p_cfg.gate or p_cfg.bgate or p_cfg.cbgate):
        # Estimate ATE n fold
        (w_ate, pot_ate, pot_ate_var, txt_ate, dtime,
         ) = ate_estimate_pot(mcf_,
                              data_df=data_df,
                              w_ate=w_ate, weights_dic=weights_dic_data,
                              balancing_test=False, w_ate_only=False, with_output=True,
                              iv=gen_cfg.iv, pred_alloc=False,
                              )
        time_.ate += dtime
        # Aggregate ATEs over folds
        effect.ate_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                          pot_ate, pot_ate_var,
                                                          effect_dic=effect.ate_dic[idx_round],
                                                          txt=txt_ate, fold=fold, title='ATE',
                                                          )
        time_.ate += dtime

        # Compute balancing tests
        if p_cfg.bt_yes:
            (_, pot_bala, pot_bala_var, txt_bala, dtime,
             ) = ate_estimate_pot(mcf_,
                                  data_df=data_df,
                                  w_ate=w_bala, weights_dic=weights_dic_data,
                                  balancing_test=True, w_ate_only=False, with_output=True,
                                  iv=gen_cfg.iv, pred_alloc=False,
                                  )
            time_.bala += dtime
            # Aggregate Balancing results over folds
            effect.bala_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                               pot_bala, pot_bala_var,
                                                               effect_dic
                                                                   =effect.bala_dic[idx_round],
                                                               txt=txt_bala, fold=fold,
                                                               title='Balancing check',
                                                               )
            time_.bala += dtime

    # Compute GATEs
    if p_cfg.gate and (round_ == 'regular' or eff_flags.gate_eff):
        (pot_gate, pot_gate_var, pot_gate_mate, pot_gate_mate_var, effect.gate_est_dic, txt_gate,
         dtime) = gate_estimate_pot(mcf_,
                                    data_df=data_df,
                                    w_gate_all=w_gate, w_atemain=w_ate,
                                    weights_dic_data=weights_dic_data,
                                    gate_type='GATE', paras_cbgate=None, z_name_cbgate=None,
                                    gate_context=gate_context,
                                    with_output=True, iv=gen_cfg.iv,
                                    )
        time_.gate += dtime
        effect.gate_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                           pot_gate, pot_gate_var,
                                                           effect_dic=effect.gate_dic[idx_round],
                                                           txt=txt_gate, fold=fold,
                                                           pot_is_list=True, title='GATE',
                                                           )
        time_.gate += dtime
        if pot_gate_mate is not None:
            (effect.gate_m_ate_dic[idx_round], dtime,
             ) = aggregate_pots(mcf_,
                                pot_gate_mate, pot_gate_mate_var,
                                effect_dic=effect.gate_m_ate_dic[idx_round], txt=txt_gate,
                                fold=fold,
                                pot_is_list=True, title='GATE minus ATE',
                                )
            time_.gate += dtime

    time_bgate_start = time()
    if p_cfg.bgate and (round_ == 'regular' or eff_flags.gate_eff):
        (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
         effect.bgate_est_dic, txt_w_f, effect.txt_b
         ) = bgate_est_low_memory(mcf_,
                                  data_df=data_df, forest_dic=forest_dic, gate_type='BGATE',
                                  )
        effect.bgate_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                            y_pot_bgate_f, y_pot_var_bgate_f,
                                                            effect_dic=effect.bgate_dic[idx_round],
                                                            txt=txt_w_f, fold=fold,
                                                            pot_is_list=True, title='BGATE',
                                                            )
        if y_pot_mate_bgate_f is not None:
            (effect.bgate_m_ate_dic[idx_round], dtime,
             ) = aggregate_pots(mcf_,
                                y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
                                effect_dic=effect.bgate_m_ate_dic[idx_round],
                                txt=txt_w_f, fold=fold, pot_is_list=True, title='BGATE minus ATE',
                                )
    time_.bgate += time() - time_bgate_start

    # CBGATE
    time_cbg_start = time()
    if p_cfg.cbgate and (round_ == 'regular' or eff_flags.gate_eff):
        (y_pot_cbgate_f, y_pot_var_cbgate_f, y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
         effect.cbgate_est_dic, txt_w_f, effect.txt_am,
         ) = bgate_est_low_memory(mcf_,
                                  data_df=data_df, forest_dic=forest_dic, gate_type='CBGATE',
                                  )
        effect.cbgate_dic[idx_round], dtime = aggregate_pots(mcf_,
                                                             y_pot_cbgate_f, y_pot_var_cbgate_f,
                                                             effect_dic
                                                                 =effect.cbgate_dic[idx_round],
                                                             txt=txt_w_f, fold=fold,
                                                             pot_is_list=True, title='CBGATE',
                                                             )
        if y_pot_mate_cbgate_f is not None:
            (effect.cbgate_m_ate_dic[idx_round], dtime
             ) = aggregate_pots(mcf_,
                                y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
                                effect_dic=effect.cbgate_m_ate_dic[idx_round], txt=txt_w_f,
                                fold=fold, pot_is_list=True, title='CBGATE minus ATE',
                                )
    time_.cbgate += time() - time_cbg_start
    if int_cfg.del_forest:
        del forest_dic['forest']


def effect_with_full_weights(mcf_: 'ModifiedCausalForest', *,
                             data_df: DataFrame,
                             forest_dic: dict,
                             eff_flags: 'EffParaFlags',
                             effect: 'EffectDicts',
                             time_: he.TimeState,
                             round_: str,
                             idx_round: int,
                             fold: int,
                             ) -> None:    # Dataclasses (effect, time) will be updated
    """Compute the effects by first computing weights and then effects with full weight matrix."""
    weights_dic, delta_time = get_weights_mp(data_df, forest_dic,
                                             reg_round=round_ == 'regular',
                                             cf_cfg=mcf_.cf_cfg, ct_cfg=mcf_.ct_cfg,
                                             gen_cfg=mcf_.gen_cfg, int_cfg=mcf_.int_cfg,
                                             p_cfg=mcf_.p_cfg, var_cfg=mcf_.var_cfg,
                                             gen_tv_cfg_yes=mcf_.gen_tv_cfg.yes,
                                             )
    time_.weight += delta_time

    time_a_start = time()
    if round_ == 'regular' or eff_flags.ate_eff:
        # Estimate ATE n fold
        w_ate, y_pot_f, y_pot_var_f, txt_w_f = ate_est(mcf_, data_df, weights_dic)

        # Aggregate ATEs over folds
        effect.ate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                      y_pot_f, y_pot_var_f,
                                                      effect_dic=effect.ate_dic[idx_round],
                                                      txt=txt_w_f, fold=fold, title='ATE',
                                                      )
    else:
        w_ate = None
    time_.ate += time() - time_a_start

    # Compute balancing tests
    time_b_start = time()
    if round_ == 'regular' or eff_flags.ate_eff:
        if mcf_.p_cfg.bt_yes:
            _, y_pot_f, y_pot_var_f, txt_w_f = ate_est(mcf_,
                                                       data_df, weights_dic,
                                                       balancing_test=True,
                                                       )
            # Aggregate Balancing results over folds
            effect.bala_dic[idx_round], _ = aggregate_pots(mcf_,
                                                           y_pot_f, y_pot_var_f,
                                                           effect_dic=effect.bala_dic[idx_round],
                                                           txt=txt_w_f, fold=fold,
                                                           title='Balancing check: ',
                                                           )
    time_.bala += time() - time_b_start

    # BGATE
    time_bgate_start = time()
    if mcf_.p_cfg.bgate and (round_ == 'regular' or eff_flags.gate_eff):
        (y_pot_bgate_f, y_pot_var_bgate_f, y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
         effect.bgate_est_dic, txt_w_f, effect.txt_b) = bgate_est(mcf_,
                                                                  data_df, weights_dic, w_ate,
                                                                  forest_dic,
                                                                  gate_type='BGATE',
                                                                  )
        effect.bgate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                        y_pot_bgate_f, y_pot_var_bgate_f,
                                                        effect_dic=effect.bgate_dic[idx_round],
                                                        txt=txt_w_f, fold=fold, pot_is_list=True,
                                                        title='BGATE',
                                                        )
        if y_pot_mate_bgate_f is not None:
            (effect.bgate_m_ate_dic[idx_round], _
             ) = aggregate_pots(mcf_,
                                y_pot_mate_bgate_f, y_pot_mate_var_bgate_f,
                                effect_dic=effect.bgate_m_ate_dic[idx_round], txt=txt_w_f,fold=fold,
                                pot_is_list=True, title='BGATE minus ATE',
                                )
    time_.bgate += time() - time_bgate_start

    # CBGATE
    time_cbg_start = time()
    if mcf_.p_cfg.cbgate and (round_ == 'regular' or eff_flags.gate_eff):
        (y_pot_cbgate_f, y_pot_var_cbgate_f, y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
         effect.cbgate_est_dic, txt_w_f, effect.txt_am,) = bgate_est(mcf_,
                                                                     data_df, weights_dic, w_ate,
                                                                     forest_dic,
                                                                     gate_type='CBGATE',
                                                                     )
        effect.cbgate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                         y_pot_cbgate_f, y_pot_var_cbgate_f,
                                                         effect_dic=effect.cbgate_dic[idx_round],
                                                         txt=txt_w_f, fold=fold, pot_is_list=True,
                                                         title='CBGATE',
                                                         )
        if y_pot_mate_cbgate_f is not None:
            (effect.cbgate_m_ate_dic[idx_round], _
             ) = aggregate_pots(mcf_,
                                y_pot_mate_cbgate_f, y_pot_mate_var_cbgate_f,
                                effect_dic=effect.cbgate_m_ate_dic[idx_round],
                                txt=txt_w_f, fold=fold, pot_is_list=True, title='CBGATE minus ATE',
                                )
    time_.cbgate += time() - time_cbg_start
    if mcf_.int_cfg.del_forest:
        del forest_dic['forest']
    # IATE
    time_i_start = time()
    if mcf_.p_cfg.iate and (round_ == 'regular' or eff_flags.iate_eff):
        (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f, txt_w_f
         ) = iate_est_mp(mcf_,
                         weights_dic,
                         w_ate=w_ate, reg_round=round_ == 'regular',
                         x_tv_df=(data_df[mcf_.var_cfg.x_name_tv] if mcf_.gen_tv_cfg.yes
                                  else None),
                         )
        effect.iate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                       y_pot_f, y_pot_var_f,
                                                       effect_dic=effect.iate_dic[idx_round],
                                                       txt=txt_w_f, fold=fold, title='IATE',
                                                       )
        if y_pot_m_ate_f is not None:
            effect.iate_m_ate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                                 y_pot_m_ate_f, y_pot_m_ate_var_f,
                                                                 effect_dic=effect.iate_m_ate_dic[
                                                                     idx_round],
                                                                 txt=txt_w_f, fold=fold,
                                                                 title='IATE minus ATE',
                                                                 )
    time_.iate += time() - time_i_start

    # QIATE
    time_q_start = time()
    if mcf_.p_cfg.qiate and mcf_.p_cfg.iate and (round_ == 'regular' or eff_flags.qiate_eff):
        (y_pot_qiate_f, y_pot_var_qiate_f,
         y_pot_mmed_qiate_f, y_pot_mmed_var_qiate_f, y_pot_mopp_qiate_f,
         y_pot_mopp_var_qiate_f, effect.qiate_est_dic, txt_w_f) = qiate_est(mcf_,
                                                                            data_df, weights_dic,
                                                                            y_pot_f,
                                                                            y_pot_var=y_pot_var_f,
                                                                            )
        effect.qiate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                        y_pot_qiate_f, y_pot_var_qiate_f,
                                                        effect_dic=effect.qiate_dic[idx_round],
                                                        txt=txt_w_f, fold=fold, pot_is_list=True,
                                                        title='QIATE',
                                                        )
        if y_pot_mmed_qiate_f is not None:
            effect.qiate_m_med_dic[idx_round], _ = aggregate_pots(mcf_,
                                                                  y_pot_mmed_qiate_f,
                                                                  y_pot_mmed_var_qiate_f,
                                                                  effect_dic=effect.qiate_m_med_dic[
                                                                     idx_round],
                                                                  txt=txt_w_f, fold=fold,
                                                                  pot_is_list=True,
                                                                  title='QIATE minus QIATE(median))'
                                                                  )
        if y_pot_mopp_qiate_f is not None:
            effect.qiate_m_opp_dic[idx_round], _ = aggregate_pots(mcf_,
                                                                  y_pot_mopp_qiate_f,
                                                                  y_pot_mopp_var_qiate_f,
                                                                  effect_dic=effect.qiate_m_opp_dic[
                                                                   idx_round],
                                                                  txt=txt_w_f,  fold=fold,
                                                                  pot_is_list=True,
                                                                  title='QIATE(q) minus QIATE(1-q))'
                                                                  )
    time_.qiate += time() - time_q_start  # QIATE

    # GATE
    time_g_start = time()
    if mcf_.p_cfg.gate and (round_ == 'regular' or eff_flags.gate_eff):
        (y_pot_gate_f, y_pot_var_gate_f, y_pot_mate_gate_f, y_pot_mate_var_gate_f,
         effect.gate_est_dic, txt_w_f) = gate_est(mcf_, data_df, weights_dic, w_ate)

        effect.gate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                       y_pot_gate_f, y_pot_var_gate_f,
                                                       effect_dic=effect.gate_dic[idx_round],
                                                       txt=txt_w_f, fold=fold, pot_is_list=True,
                                                       title='GATE',
                                                       )
        if y_pot_mate_gate_f is not None:
            effect.gate_m_ate_dic[idx_round], _ = aggregate_pots(mcf_,
                                                                 y_pot_mate_gate_f,
                                                                 y_pot_mate_var_gate_f,
                                                                 effect_dic=effect.gate_m_ate_dic[
                                                                     idx_round],
                                                                 txt=txt_w_f, fold=fold,
                                                                 pot_is_list=True,
                                                                 title='GATE minus ATE',
                                                                 )
    time_.gate += time() - time_g_start


def get_weights_aggregateweights_iate_chunk(data_df_chunk: DataFrame, *,
                                            forest_dic: dict,
                                            round_: str,
                                            mcf_seq: 'ModifiedCausalForest',
                                            eff_flags: 'EffParaFlags',
                                            chunk_idx: int,
                                            gate_context: he.GateContext | None = None,
                                            indices_split: list[NDArray[Any]] | None = None,
                                            x_tv_pred_full_df: DataFrame | None = None,
                                            ) -> tuple[int,
                                                       he.TimeState,
                                                       dict,
                                                       NDArray[Any],
                                                       ArrayLike,
                                                       str,
                                                       ArrayLike,
                                                       ArrayLike,
                                                       list[ArrayLike] | None,
                                                       ]:
    """Compute IATEs for this chunk of data and accumulate weights for aggregate effects."""
    time_chunk = he.TimeState()     # Initialise timing
    # Get IATE weights for the chunk of the prediction data
    weights_dic, dtime = get_weights_mp(data_df_chunk, forest_dic,
                                        reg_round=round_ == 'regular',
                                        cf_cfg=mcf_seq.cf_cfg, ct_cfg=mcf_seq.ct_cfg,
                                        gen_cfg=mcf_seq.gen_cfg, int_cfg=mcf_seq.int_cfg,
                                        p_cfg=mcf_seq.p_cfg, var_cfg=mcf_seq.var_cfg,
                                        gen_tv_cfg_yes=mcf_seq.gen_tv_cfg.yes,
                                        print_progress=False,
                                        )
    time_chunk.weight += dtime

    # The following parameters will not be computed because they need access to the full
    # weight matrix:
    # IATE - ATE, QIATEs (needs individual weights at least for inference, but most likely in the
    #                     implementation also for the point estimator)

    # 1) Compute IATE (potential outcomes)
    # IATE
    if mcf_seq.p_cfg.iate and (round_ == 'regular' or eff_flags.iate_eff):
        if x_tv_pred_full_df is not None and indices_split is not None:
            tv_spec = fit_tv_feature_spec(weights_dic['x_tv_dat'],
                                          mcf_seq.var_cfg,
                                          zero_tol=mcf_seq.int_cfg.zero_tol,
                                          )
            x_tv_pred_chunk = transform_x_tv(x_tv_pred_full_df.iloc[indices_split[chunk_idx]],
                                             tv_spec,
                                             )
        else:
            x_tv_pred_chunk = None

        pot_iate, pot_iate_var, txt_iate, dtime = iate_pot_est(mcf_seq,
                                                               data_df=data_df_chunk,
                                                               weights_dic=weights_dic,
                                                               round_=round_,
                                                               x_tv_pred_np=x_tv_pred_chunk,
                                                               print_progress=False,
                                                               )
        time_chunk.iate += dtime
    else:
        pot_iate, pot_iate_var, txt_iate = None, None, ''

    # Accumulate weights for ATE and balancing test
    gate_like_yes = mcf_seq.p_cfg.gate or mcf_seq.p_cfg.bgate or mcf_seq.p_cfg.cbgate
    if he.need_ate_weights(round_, eff_flags, gate_like_yes):
        w_ate, dtime = ate_weights(mcf_seq,
                                   data_df=data_df_chunk, weights_dic=weights_dic,
                                   balancing_test=False, w_ate_only=False,
                                   with_output=chunk_idx==0, iv=mcf_seq.gen_cfg.iv,
                                   pred_alloc=False,
                                   )
        time_chunk.ate += dtime

        if mcf_seq.p_cfg.bt_yes:
            w_bala, dtime = ate_weights(mcf_seq,
                                        data_df=data_df_chunk, weights_dic=weights_dic,
                                        balancing_test=True, w_ate_only=False,
                                        with_output=chunk_idx==0, iv=mcf_seq.gen_cfg.iv,
                                        pred_alloc=False,
                                        )
            time_chunk.bala += dtime
        else:
            w_bala = None
    else:
        w_ate = w_bala = None

    # Accumulate weights for GATE estimation
    if mcf_seq.p_cfg.gate and (round_ == 'regular' or eff_flags.gate_eff):
        w_gate, dtime = gate_weights(mcf_seq,
                                     data_df=data_df_chunk, weights_dic=weights_dic,
                                     gate_context=gate_context,
                                     with_output=chunk_idx==0, iv=mcf_seq.gen_cfg.iv,
                                    )
        time_chunk.gate += dtime
    else:
        w_gate = None

    if chunk_idx == 0:
        del weights_dic['weights']   # Other information in weights_dic is independent of chuncks
        weights_dic_data = weights_dic
    else:
        weights_dic_data = weights_dic = None

    return (chunk_idx, time_chunk, weights_dic_data, pot_iate, pot_iate_var, txt_iate,
            w_ate, w_bala, w_gate,
            )


def backend_get_weights_aggregateweights_iate_chunk(data_df_chunk: DataFrame, *,
                                                    chunk_idx: int,
                                                    data: dict[str, Any],
                                                    ) -> tuple[int, he.TimeState, dict,
                                                               NDArray[Any], ArrayLike, str,
                                                               ArrayLike, ArrayLike,
                                                               list[ArrayLike] | None,
                                                               ]:
    """Backend wrapper using a direct shared-data argument."""
    return get_weights_aggregateweights_iate_chunk(data_df_chunk, **data, chunk_idx=chunk_idx)


# The following is used for the case when an ray import fails
def ray_get_weights_aggregateweights_iate_chunk_plain(data_df_chunk: DataFrame, *,
                                                      chunk_idx: int,
                                                      kw: dict[str, Any],
                                                      ) -> tuple[int, he.TimeState, dict,
                                                                 NDArray[Any], ArrayLike, str,
                                                                 ArrayLike, ArrayLike,
                                                                 list[ArrayLike] | None,
                                                                 ]:
    """Ray wrapper unpacking shared keyword dictionary."""
    return get_weights_aggregateweights_iate_chunk(data_df_chunk,
                                                   **kw,
                                                   chunk_idx=chunk_idx,
                                                   )

if ray is not None:
    ray_get_weights_aggregateweights_iate_chunk = ray.remote(
        ray_get_weights_aggregateweights_iate_chunk_plain
        )
else:
    ray_get_weights_aggregateweights_iate_chunk = None   # # pylint: disable=C0103


def sequential_weights_parallel(*, executor: Any,
                                tasks: list[TaskSpec],
                                maxworkers: int,
                                int_cfg: Any,
                                gen_cfg: Any,
                                no_of_chunks: int,
                                time_: he.TimeState,
                                w_ate: ArrayLike,
                                w_bala: ArrayLike,
                                w_gate: ArrayLike,
                                pot_iate: ArrayLike,
                                pot_iate_var: ArrayLike,
                                indices_split: list[NDArray[Any]],
                                n_rows: int,
                                weights_dic_data: dict | None,
                                txt_iate: str | None,
                                ) -> tuple[he.TimeState,
                                           ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,
                                           dict | None, str | None
                                           ]:
    """Run chunk tasks with backend-agnostic progress-bar support."""
    use_tqdm, output_clean = tqdm_setup(tqdm, gen_cfg.with_output and gen_cfg.verbose)

    pbar = (tqdm(total=no_of_chunks, desc='Sequential weights', leave=False, dynamic_ncols=True,)
            if use_tqdm else None
            )
    jdx = 0
    try:
        for result in map_task_batches(executor=executor, tasks=tasks,
                                       int_cfg=int_cfg, maxworkers=maxworkers,
                                       min_worker_waves=1,
                                       stream_results=True,
                                       ):
            (chunk_idx, time_chunk, weights_dic_data_chunk, pot_iate_chunk, pot_iate_var_chunk,
             txt_iate_chunk, w_ate_chunk, w_bala_chunk, w_gate_chunk,
             ) = result

            (time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var
             ) = update_weights_iates(time_=time_,
                                      w_ate=w_ate, w_bala=w_bala, w_gate=w_gate,
                                      pot_iate=pot_iate, pot_iate_var=pot_iate_var,
                                      time_chunk=time_chunk,
                                      w_ate_chunk=w_ate_chunk, w_bala_chunk=w_bala_chunk,
                                      w_gate_chunk=w_gate_chunk,
                                      pot_iate_chunk=pot_iate_chunk,
                                      pot_iate_var_chunk=pot_iate_var_chunk,
                                      indices=indices_split[chunk_idx], n_rows=n_rows,
                                      add_chunk_time=False,
                                      )
            if chunk_idx == 0:
                weights_dic_data = weights_dic_data_chunk
                txt_iate = txt_iate_chunk

            jdx += 1
            if pbar is not None:
                pbar.update(1)

            progress_clean_memory(output=output_clean, current_idx=jdx, total=no_of_chunks,
                                  clean_mem=False,
                                  )
    finally:
        if pbar is not None:
            pbar.close()

    return time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var, weights_dic_data, txt_iate


def iate_pot_est(mcf_: 'ModifiedCausalForest', *,
                 data_df: DataFrame,
                 weights_dic: dict[str, Any],
                 round_: str,
                 x_tv_pred_np: NDArray[Any] | None = None,
                 print_progress: bool = True,
                 ) -> tuple[NDArray, ArrayLike, str, float]:
    """Compute potential outcomes for IATE."""
    time_start = time()
    if mcf_.gen_tv_cfg.yes and x_tv_pred_np is None:
        x_tv_df = data_df[mcf_.var_cfg.x_name_tv]
    else:
        x_tv_df = None

    pot_chunk, pot_var_chunk, _, _, txt_w = iate_est_mp(mcf_,
                                                        weights_dic,
                                                        w_ate=None, reg_round=round_ == 'regular',
                                                        x_tv_df=x_tv_df, x_tv_pred_np=x_tv_pred_np,
                                                        print_progress=print_progress,
                                                        )
    return pot_chunk, pot_var_chunk, txt_w, time() - time_start


def combine_iate(pot_iate: ArrayLike,
                 pot_iate_var: ArrayLike, *,
                 pot_chunk: ArrayLike,
                 pot_var_chunk: ArrayLike,
                 indices: NDArray,
                 rows_total: int,
                 ) -> tuple[ArrayLike, ArrayLike, float]:
    """Stack IATEs."""
    if pot_chunk is None:
        return None, None, 0

    time_start = time()
    if pot_iate is None:
        pot_iate = np.empty((rows_total, pot_chunk.shape[1], pot_chunk.shape[2]),
                            dtype=pot_chunk.dtype,
                            )
        if pot_var_chunk is not None:
            pot_iate_var = np.empty_like(pot_iate)

    pot_iate[indices, :] = pot_chunk
    if pot_var_chunk is not None:
        pot_iate_var[indices, :] = pot_var_chunk

    return pot_iate, pot_iate_var, time() - time_start


def get_maxworkers(*, gen_cfg: 'GenCfg', zero_tol: float) -> int:
    """Get number of maxworkers."""
    if gen_cfg.mp_parallel < 1.5:
        return 1

    return (find_no_of_workers(gen_cfg.mp_parallel, gen_cfg.sys_share, zero_tol=zero_tol)
            if gen_cfg.mp_automatic else gen_cfg.mp_parallel
            )


def update_weights_iates(*,
                         time_: he.TimeState,
                         w_ate: ArrayLike,
                         w_bala: ArrayLike,
                         w_gate: ArrayLike,
                         pot_iate: ArrayLike,
                         pot_iate_var: ArrayLike,
                         time_chunk: he.TimeState,
                         w_ate_chunk: ArrayLike,
                         w_bala_chunk: ArrayLike,
                         w_gate_chunk: list[ArrayLike] | None,
                         pot_iate_chunk: ArrayLike,
                         pot_iate_var_chunk:  ArrayLike,
                         indices: NDArray,
                         n_rows: int,
                         add_chunk_time: bool = True,
                         ) -> tuple[he.TimeState,
                                    ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,
                                    ]:
    """Update weights and IATEs with results from next chunk."""
    if add_chunk_time:
        he.add_dataclass_attributes_inplace(time_, time_chunk)  # Update time

    if w_ate_chunk is not None:
        w_ate, dtime = he.accumulate_weights(w_ate, w_ate_chunk)
        time_.ate += dtime

    if w_bala_chunk is not None:
        w_bala, dtime = he.accumulate_weights(w_bala, w_bala_chunk)
        time_.bala += dtime

    if w_gate_chunk is not None:
        if w_gate is None:
            w_gate = [None] * len(w_gate_chunk)
        for jdx, gate_j in enumerate(w_gate_chunk):
            w_gate[jdx], dtime = he.accumulate_weights(w_gate[jdx], gate_j)
            time_.gate += dtime

    if pot_iate_chunk is not None:
        pot_iate, pot_iate_var, dtime = combine_iate(pot_iate, pot_iate_var,
                                                     pot_chunk=pot_iate_chunk,
                                                     pot_var_chunk=pot_iate_var_chunk,
                                                     indices=indices,
                                                     rows_total=n_rows,
                                                     )
        time_.iate += dtime

    return time_, w_ate, w_bala, w_gate, pot_iate, pot_iate_var


def make_mcf_seq_for_low_memory(mcf_: 'ModifiedCausalForest') -> Any:
    """Create a small MCF-like object for low-memory chunk workers."""
    mcf_seq = SimpleNamespace(gen_cfg=deepcopy(mcf_.gen_cfg),
                              int_cfg=deepcopy(mcf_.int_cfg),
                              p_cfg=deepcopy(mcf_.p_cfg),
                              var_cfg=deepcopy(mcf_.var_cfg),
                              ct_cfg=deepcopy(mcf_.ct_cfg),
                              cf_cfg=deepcopy(mcf_.cf_cfg),
                              p_ba_cfg=deepcopy(mcf_.p_ba_cfg),
                              gen_tv_cfg=deepcopy(mcf_.gen_tv_cfg),
                              var_x_values=deepcopy(mcf_.var_x_values),
                              low_mem_cfg=deepcopy(mcf_.low_mem_cfg),
                              )
    mcf_seq.gen_cfg.mp_parallel = 1
    mcf_seq.gen_cfg.with_output = False

    return mcf_seq
