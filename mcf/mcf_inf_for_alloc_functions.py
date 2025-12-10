"""Created on Fri Mar 21 07:37:16 2025.

Contains functions for the estimation and inference of values of allocations,
when policy scores evaluated here are based on mcf training.

# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING
from time import time

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import pandas as pd

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_common_support_functions as mcf_cs
from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_init_update_helper_functions as mcf_init_update
from mcf import mcf_general as mcf_gp
from mcf import mcf_local_centering_functions as mcf_lc
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_weight_functions as mcf_w

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def predict_different_allocations_main(self,
                                       data_df: DataFrame,
                                       allocations_df: DataFrame | None = None
                                       ) -> tuple[dict, Path]:
    """
    Predict average potential outcomes + IATEs for different allocations.

    meth:`~ModifiedCausalForest.train` method must be run beforehand.

    Parameters
    ----------
    data_df : DataFrame
        Data used to compute the predictions. It must contain information
        about features (and treatment if effects for treatment specific
        subpopulations are desired as well).
    allocations_df : Dataframe or None, optional
        Different allocations which are to be evaluated. The length of this
        dataframe must be the same as the length of data_df.
        Default is None.

    Returns
    -------
    results : Dictionary.
        Results. This dictionary has the following structure:
        'ate': Average treatment effects
        'ate_se': Standard error of average treatment effects
        'ate_effect_list': List with name with estiamted effects
        'alloc_df': Dataframe with value and variance of value for all
                    allocations investigated.

    outpath : Pathlib object
        Location of directory in which output is saved.
    """
    time_start = time()
    report = {}
    txt = ''

    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    if allocations_df is not None:
        allocations_df, alloc_names = mcf_data.data_frame_vars_lower(
            allocations_df)
    else:
        alloc_names = None
    # Initialise again with data information
    data_df = mcf_init_update.p_update_pred(self, data_df)
    # Check treatment data
    data_df = basic_data_compability_diff_allocations(self,
                                                      data_df,
                                                      allocations_df
                                                      )
    mcf_init_update.int_update_pred(self, len(data_df))
    mcf_init_update.post_update_pred(self, data_df)
    if self.gen_cfg.with_output:
        mcf_ps.print_dic_values_all(self, summary_top=True, summary_dic=False,
                                    train=False
                                    )
    # Prepare data: Add and recode variables for GATES (Z). Recode categorical
    #    variables to prime numbers, cont. vars. This is done here as we need
    #    the same variables as were used in training.
    data_df = mcf_data.create_xz_variables(self, data_df, train=False)
    if self.gen_cfg.with_output and self.gen_cfg.verbose:
        mcf_data.print_prime_value_corr(self.data_train_dict,
                                        self.gen_cfg, summary=False
                                        )
    # Clean data and remove missings and unncessary variables
    if self.dc_cfg.clean_data:
        data_df, report['alloc_prediction_obs'] = mcf_data.clean_data(
            self, data_df, train=False, d_alloc_names=alloc_names
            )
    else:
        report['alloc_prediction_obs'] = len(data_df)
    time_1 = time()

    # Enforce common support as determined in training
    if self.cs_cfg.type_:
        (data_df, _, report['alloc_cs_p_share_deleted'],
         report['alloc_cs_p_obs_remain'], _, _, _, _
         ) = mcf_cs.common_support_p_score(self, data_df, None, train=False,
                                           p_score_only=False
                                           )
    else:
        report['alloc_cs_p_share_deleted'] = 0
        report['alloc_cs_p_obs_remain'] = len(data_df)

    data_df = data_df.copy().reset_index(drop=True)

    time_2 = time()

    # Enforce local centering for IATE as (and if) enforced in training
    if self.lc_cfg.yes and self.lc_cfg.uncenter_po:
        (_, _, y_pred_x_df, _) = mcf_lc.local_centering(self, data_df,
                                                        None, train=False)
    else:
        y_pred_x_df = 0

    time_3 = time()
    time_delta_weight = time_delta_ate = time_delta_alloc = 0

    ate_dic = w_alloc_np = w_alloc_var_np = None

    only_one_fold_one_round = (self.cf_cfg.folds == 1
                               and len(self.cf_cfg.est_rounds) == 1)
    for fold in range(self.cf_cfg.folds):
        time_w_start = time()
        if only_one_fold_one_round:
            forest_dic = self.forest[fold][0]
        else:
            forest_dic = deepcopy(self.forest[fold][0])
        if self.gen_cfg.with_output and self.gen_cfg.verbose:
            print(f'\n\nWeight maxtrix {fold+1} / {self.cf_cfg.folds} forests')
        weights_dic = mcf_w.get_weights_mp(
            self, data_df, forest_dic, reg_round='regular', with_output=True)
        time_delta_weight += time() - time_w_start

        time_a_start = time()
        # Estimate ATE n fold
        (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
            self, data_df, weights_dic, pred_alloc=True)
        # Aggregate ATEs over folds
        ate_dic = mcf_est.aggregate_pots(
            self, y_pot_f, y_pot_var_f, txt_w_f, ate_dic, fold,
            title='All in one treatment')
        if self.int_cfg.del_forest:
            del forest_dic['forest']
        time_delta_ate += time() - time_a_start

        time_i_start = time()
        if allocations_df is not None:
            w_alloc_f_np, w_alloc_var_f_np, names, txt_w_f = alloc_est(
                self, data_df, weights_dic, alloc_names=alloc_names
                )
            if fold == 0:
                w_alloc_np = np.zeros_like(w_alloc_f_np)
                w_alloc_var_np = np.zeros_like(w_alloc_f_np)
                alloc_all_names = names

            w_alloc_np += w_alloc_f_np
            w_alloc_var_np += w_alloc_var_f_np
            txt += txt_w_f
        time_delta_alloc += time() - time_i_start

    if allocations_df is not None:
        time_i_start = time()
        w_alloc_np /= self.cf_cfg.folds
        w_alloc_var_np /= (self.cf_cfg.folds ** 2)
        time_delta_alloc += time() - time_i_start

    if not only_one_fold_one_round and self.int_cfg.del_forest:
        self.forest[fold] = None
        # Without this and the following delete, it becomes impossible
        # to reuse the same forest for several data sets, which is bad.

    if self.int_cfg.del_forest:
        self.forest = None

    del weights_dic

    # ATE (corresponding to all are allocated into the same treatment)
    time_a_start = time()
    ate, ate_se, ate_effect_list = mcf_ate.ate_effects_print(
        self, ate_dic, y_pred_x_df, balancing_test=False, pred_alloc=True,
        )
    time_delta_ate += time() - time_a_start

    if allocations_df is not None:
        time_i_start = time()
        alloc_df, txt_alloc = alloc_effects_print(self,
                                                  w_alloc_np, w_alloc_var_np,
                                                  y_pred_x_df, len(data_df),
                                                  alloc_names=alloc_all_names,
                                                  alloc_single_names=alloc_names
                                                  )
        time_delta_alloc += time() - time_i_start
    else:
        alloc_df, txt_alloc = None, ''
    # Collect results
    results = {
        # Keep well tested & well formated structure + names of predict method
        'ate': ate, 'ate_se': ate_se, 'ate_effect_list': ate_effect_list,
        'alloc_df': alloc_df,
        }
    if self.gen_cfg.with_output:
        report['alloc_mcf_pred_results'] = results
        self.report['predict_list'].append(report.copy())
        self.report['alloc_welfare_allocations'] = txt_alloc

    time_end = time()
    if self.gen_cfg.with_output:
        time_string = [
            'Data preparation and stats II:                  ',
            'Common support:                                 ',
            'Local centering (recoding of Y):                ',
            'Weights:                                        ',
            'All-in-1-treatment rule:                        ',
            'Other allocation rules:                         ',
            '\nTotal time prediction for allocations:          ']
        time_difference = [
            time_1 - time_start, time_2 - time_1, time_3 - time_2,
            time_delta_weight, time_delta_ate, time_delta_alloc,
            time_end - time_start]
        mcf_ps.print_mcf(self.gen_cfg, self.time_strings['time_train'])
        time_pred = mcf_ps.print_timing(
            self.gen_cfg, 'Predictions for allocations', time_string,
            time_difference, summary=True)
        self.time_strings['time_pred'] = time_pred

    return results, self.gen_cfg.outpath


def alloc_est(mcf_: 'ModifiedCausalForest',
              data_df: DataFrame,
              weights_dic: dict,
              alloc_names: list[str] | None = None
              ) -> tuple[NDArray[Any], NDArray[Any], list[str], str]:
    """Estimate values of allocations & their SE using scores & IATEs.

    Parameters
    ----------
    mcf_ : mcf object.
    data_df : DataFrame. Prediction data.
    weights_dic : Dict.
              Contains weights and numpy data.
    alloc_names : List of strings or None, optional
              Names of treatment allocations to be investigated (and contained
                                                                 in data_df)

    Returns
    -------
    allocations_np : Numpy array. Value of evaluations.
    allocations_var_np : Numpy array. Variance value of evaluations.
    effect_list : List of strings with name of effects (same order as ATE)

    """
    txt = ''
    int_cfg = mcf_.int_cfg
    p_ba_yes = mcf_.p_ba_cfg
    # Training data
    y_dat = weights_dic['y_dat_np']
    w_dat = weights_dic['w_dat_np'] if mcf_.gen_cfg.weighted else None

    if mcf_.gen_cfg.d_type == 'continuous':
        raise NotImplementedError('Evaluation of allocations is only available'
                                  'for discrete treatments.')
    if mcf_.p_cfg.choice_based_sampling:
        raise NotImplementedError('Evaluation of allocations is NOT available'
                                  'for choice based sampling.')
    if mcf_.gen_cfg.weighted:
        raise NotImplementedError('Evaluation of allocations is NOT available'
                                  'for weighted sampling.')

    d_alloc_np = np.int32(np.round(data_df[alloc_names].to_numpy()))
    no_of_alloc = len(alloc_names)

    n_p, n_y, no_of_out = len(data_df), len(y_dat), len(mcf_.var_cfg.y_name)

    # Step 1: Aggregate weights
    if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
        print('\n\nComputing values of allocations')

    # Compute weights for each alloc
    w_alloc = np.zeros((no_of_alloc, n_y))
    weights = weights_dic['weights']
    if int_cfg.weight_as_sparse:
        for j in range(n_p):
            w_add = np.zeros((no_of_alloc, n_y))
            d_values_j = np.unique(d_alloc_np[j, :])  # Should be fast
            for t_ind, _ in enumerate(mcf_.gen_cfg.d_values):
                if t_ind in d_values_j:
                    w_i_csr = weights[t_ind][j, :]   # copy, but still sparse
                    sum_wi = w_i_csr.sum()
                    if sum_wi <= int_cfg.zero_tol:
                        txt = f'\nEmpty leaf. Observation: {j}'
                        mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
                        raise RuntimeError(txt)
                    if not (1-int_cfg.sum_tol) < sum_wi < (1+int_cfg.sum_tol):
                        w_i_csr = w_i_csr.multiply(1 / sum_wi)
                    w_i_dense = w_i_csr.todense()
                    for jdx, d_j in enumerate(d_alloc_np[j, :]):
                        if d_j == t_ind:
                            w_add[jdx, :] = w_i_dense

            w_alloc += w_add

    else:
        for j, weight_i in enumerate(weights):
            w_add = np.zeros((no_of_alloc, n_y))
            d_values_j = np.unique(d_alloc_np[j, :])  # Should be fast
            for t_ind, _ in enumerate(mcf_.gen_cfg.d_values):
                w_i = weight_i[t_ind][1].copy()
                if t_ind in d_values_j:
                    sum_wi = np.sum(w_i)
                    if sum_wi <= int_cfg.zero_tol:
                        txt = (f'\nZero weight. Index: {weight_i[t_ind][0]}'
                               f'd_value: {t_ind}\nWeights: {w_i}')
                        mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
                        raise RuntimeError(txt)
                    if not (1-int_cfg.sum_tol) < sum_wi < (1+int_cfg.sum_tol):
                        w_i = w_i / sum_wi
                    for jdx, d_j in enumerate(d_alloc_np[j, :]):
                        if d_j == t_ind:
                            w_add[jdx, weight_i[t_ind][0]] = w_i
            w_alloc += w_add

    # Step 2: Get allocations
    # Step 2.1: Check and normalize weights
    sumw = np.sum(w_alloc, axis=1)
    w_alloc /= n_p
    for alloc_idx in range(no_of_alloc):
        if -int_cfg.sum_tol < sumw[alloc_idx] < int_cfg.sum_tol:
            if mcf_.gen_cfg.with_output:
                txt += (f'\nAloc name: {alloc_names[alloc_idx]}) '
                        '\nAllocation weights: '
                        f'{w_alloc[alloc_idx, :]}'
                        'Weights are all zero. Not good.'
                        ' Redo statistic without this variable.'
                        ' \nOr try to use more bootstraps.'
                        ' \nOr Sample may be too small.'
                        )
                mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
                raise RuntimeError(txt)

        if mcf_.p_cfg.max_weight_share < 1:
            w_alloc[alloc_idx, :], _, share = mcf_gp.bound_norm_weights(
                 w_alloc[alloc_idx, :],
                 mcf_.p_cfg.max_weight_share,
                 zero_tol=int_cfg.zero_tol, sum_tol=int_cfg.sum_tol,
                 negative_weights_possible=p_ba_yes,
                 )
            if mcf_.gen_cfg.with_output:
                txt += ('\nShare of weights censored at'
                        f'{mcf_.p_cfg.max_weight_share*100:8.3f}%: '
                        f'{share*100:8.3f}%  Alloc type: {alloc_idx:2} '
                        )

    cl_dat = weights_dic['cl_dat_np'] if mcf_.p_cfg.cluster_std else None

    # Step 2.2 Compute welfares and their variance for the allocations
    welfare_level = np.zeros((no_of_alloc, no_of_out))
    welfare_level_var = np.empty_like(welfare_level)
    for alloc_idx in range(no_of_alloc):
        for out_idx in range(no_of_out):
            # TODO: hier bias adjustment, oder vorher?
            ret = mcf_est.weight_var(
                w_alloc[alloc_idx, :], y_dat[:, out_idx],
                cl_dat, mcf_.gen_cfg, mcf_.p_cfg,
                weights=w_dat,
                bootstrap=mcf_.p_cfg.se_boot_ate,
                keep_all=int_cfg.keep_w0,
                se_yes=True,
                normalize=True,
                zero_tol=int_cfg.zero_tol,
                sum_tol = int_cfg.sum_tol,
                )
            welfare_level[alloc_idx, out_idx] = ret[0]
            welfare_level_var[alloc_idx, out_idx] = ret[1]

    # Step 2.3 Compute difference of welfare & their variance
    no_of_alloc_diff = np.int32(np.round(no_of_alloc * (no_of_alloc - 1) / 2))
    welfare_diff = np.zeros((no_of_alloc_diff, no_of_out))
    welfare_diff_var = np.empty_like(welfare_diff)
    alloc_diff_idx = 0
    alloc_diff_names = []
    for alloc1_idx, alloc1_name in enumerate(alloc_names):
        if alloc1_idx == no_of_alloc - 1:
            break
        alloc2_names = alloc_names[alloc1_idx+1:]
        for alloc2_idx, alloc2_name in enumerate(alloc2_names,
                                                 start=alloc1_idx + 1):
            alloc_diff_names.append(alloc1_name + '_m_' + alloc2_name)
            for out_idx in range(no_of_out):
                ret = mcf_est.weight_var(
                    w_alloc[alloc1_idx, :] - w_alloc[alloc2_idx, :],
                    y_dat[:, out_idx],
                    cl_dat, mcf_.gen_cfg, mcf_.p_cfg,
                    weights=w_dat,
                    bootstrap=mcf_.p_cfg.se_boot_ate,
                    keep_all=int_cfg.keep_w0,
                    se_yes=True,
                    normalize=False,
                    zero_tol=int_cfg.zero_tol,
                    sum_tol = int_cfg.sum_tol,
                    )
                welfare_diff[alloc_diff_idx, out_idx] = ret[0]
                welfare_diff_var[alloc_diff_idx, out_idx] = ret[1]
                alloc_diff_idx += 1

    all_names = [*alloc_names, *alloc_diff_names]
    welfare = np.concatenate((welfare_level, welfare_diff))
    welfare_var = np.concatenate((welfare_level_var, welfare_diff_var))

    return welfare, welfare_var, all_names, txt


def alloc_effects_print(mcf_: 'ModifiedCausalForest',
                        w_alloc_np: NDArray[Any],
                        w_alloc_var_np: NDArray[Any],
                        y_pred_lc: NDArray[Any],
                        obs: int,
                        alloc_names: list[str] | None = None,
                        alloc_single_names: list[str] | None = None
                        ) -> tuple[DataFrame, str]:
    """Compute ate's from potential outcomes and print them."""
    txt = ''
    if isinstance(y_pred_lc, (pd.Series, pd.DataFrame)):
        lc_yes, y_pred_lc_avg = True, np.mean(y_pred_lc, axis=0)
    else:
        lc_yes, y_pred_lc_avg = False, 0

    if mcf_.gen_cfg.with_output:
        txt += '\n' * 2 + '=' * 100
        txt += '\nEffects of Allocations Estimation \n' + '-' * 100 + '\n'
        txt += 'Average Outcomes'
        txt += '\n' + '-' * 100
        if mcf_.p_cfg.se_boot_ate > 1:
            txt += ('\nBootstrap standard errors with '
                    '{mcf_.p_cfg.se_boot_ate:<6} replications')
        for out_idx, out_name in enumerate(mcf_.var_cfg.y_name):
            out_name_txt = out_name[:-3] if lc_yes else out_name
            txt += '\nOutcome variable: ' + out_name_txt

            if lc_yes:
                (w_alloc_np[:len(alloc_single_names), out_idx]
                 ) += y_pred_lc_avg.iloc[out_idx]
            txt += '\n' + '- ' * 50
            txt += '\n    Outcomes of different allocations'

            stderr, t_val, p_val = mcf_est.compute_inference(
                w_alloc_np[:, out_idx],
                w_alloc_var_np[:, out_idx])
            txt += mcf_ps.print_effect(w_alloc_np[:, out_idx],  # Effect
                                       stderr,
                                       t_val,
                                       p_val,
                                       alloc_names,
                                       no_comparison=True
                                       )

            add_txt = ('\nNumber of observations used for these calculations: '
                       f'{obs}.'
                       '\nThere could be small differences in standard '
                       'errors compared to all-in-1-allocations, \ndue to '
                       'differences in computation.'
                       '\nNote that cross-fitting and efficient estimation of '
                       'IATEs (if used) is not accounted for.'
                       )
            txt += mcf_ps.print_se_info(mcf_.p_cfg.cluster_std,
                                        mcf_.p_cfg.se_boot_ate,
                                        additional_info=add_txt
                                        )
        txt += '\n' + '-' * 100
        mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True, non_summary=True)

    # Fill dataframe to return information about estimation
    row_names = ('est', 'var',)
    cols = len(mcf_.var_cfg.y_name) * len(alloc_names)
    rows = len(row_names)
    alloc_np = np.zeros((rows, cols))
    idx = 0
    col_names = []
    for out_idx, out_name in enumerate(mcf_.var_cfg.y_name):
        for alloc_idx, alloc in enumerate(alloc_names):
            col_names.append(out_name + '_' + alloc)
            alloc_np[0, idx] = w_alloc_np[alloc_idx, out_idx]
            alloc_np[1, idx] = w_alloc_var_np[alloc_idx, out_idx]
            idx += 1
    alloc_df = pd.DataFrame(data=alloc_np, index=row_names, columns=col_names)

    return alloc_df, txt


def basic_data_compability_diff_allocations(self,
                                            data_df: DataFrame,
                                            allocations_df: DataFrame
                                            ) -> DataFrame:
    """Perform basic data consistency checks and merge dataframes."""
    if allocations_df is None:
        return data_df

    if len(data_df) != len(allocations_df):
        raise ValueError(
            f'Length of DataFrame containing features ({len(data_df)}) and '
            f'DataFrame containing allocations {len(allocations_df)} must be '
            'the same.'
            )
    unique_values_all = set(np.unique(allocations_df.values.ravel()))
    _, d_values, _ = mcf_data.get_treat_info(self)
    if not unique_values_all.issubset(set(d_values)):
        diff = unique_values_all.difference(set(d_values))
        raise ValueError(
            f'Treatment values in specified allocations {unique_values_all} '
            'must be contained in treatment values used in training '
            f'{set(d_values)}. The following values were not used in training: '
            f'{diff}'
            )

    alloc_names = allocations_df.columns
    if min(d_values) != 0:
        for name in alloc_names:
            allocations_df[name] -= min(d_values)

    data_both_df = pd.concat((data_df, allocations_df), axis=1)

    return data_both_df
