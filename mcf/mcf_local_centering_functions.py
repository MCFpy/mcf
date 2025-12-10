"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for local centering.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from mcf import mcf_data_functions as mcf_data
from mcf import mcf_estimation_generic_functions as mcf_gf
from mcf.mcf_general import get_key_values_in_list, to_numpy_big_data
from mcf.mcf_general_sys import print_mememory_statistics
from mcf.mcf_print_stats_functions import print_mcf, print_descriptive_df
from mcf.mcf_variable_importance_functions import print_variable_importance

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def local_centering(mcf_: 'ModifiedCausalForest',
                    tree_df: pd.DataFrame,
                    fill_y_df: pd.DataFrame | None = None,
                    train: bool = True,
                    seed: int = 9324561,
                    title: str = ''
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Locally center all outcome variables."""
    lc_cfg, gen_cfg, var_cfg = mcf_.lc_cfg, mcf_.gen_cfg, mcf_.var_cfg
    var_x_type, data_train_dict = mcf_.var_x_type, mcf_.data_train_dict
    cf_cfg, int_cfg = mcf_.cf_cfg, mcf_.int_cfg
    if gen_cfg.with_output:
        print_lc_header(gen_cfg, lc_cfg, title=title)

    x_lc_np = None

    y_0_dtype = (np.float64 if len(tree_df) < int_cfg.obs_bigdata
                 else np.float32)

    x_name, x_type = get_key_values_in_list(var_x_type)
    # Get correct dataframes depending on task and cross-validation
    if train:
        if lc_cfg.cs_cv:   # Crossvalidate ... only tree data is used
            tree_mcf_df, fill_y_mcf_df = tree_df.copy(), fill_y_df
        else:  # Use lc.cs_share of data for lc (and cs) estim. only
            # Take the same share of obs. from both input samples
            if fill_y_df is None:
                fill_y_lc_df = fill_y_mcf_df = None
                tree_mcf_df, tree_lc_df = train_test_split(
                    tree_df, test_size=lc_cfg.cs_share, random_state=seed)
            else:
                tree_mcf_df, tree_lc_df = train_test_split(
                    tree_df,
                    test_size=lc_cfg.cs_share,
                    random_state=seed
                    )
                fill_y_mcf_df, fill_y_lc_df = train_test_split(
                    fill_y_df,
                    test_size=lc_cfg.cs_share,
                    random_state=seed
                    )
            data_lc_df = pd.concat((tree_lc_df, fill_y_lc_df), axis=0)
            x_lc_df, _ = mcf_data.get_x_data(data_lc_df, x_name)
        if fill_y_df is not None:
            x_fy_df, _ = mcf_data.get_x_data(fill_y_mcf_df, x_name)
    else:
        tree_mcf_df, fill_y_mcf_df = tree_df, None

    x_tree_df, obs_mcf = mcf_data.get_x_data(tree_mcf_df, x_name)

    # Create dummies for categorical variables
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    # Get covariates (as dummies if categorical)
    if names_unordered:  # List is not empty; predict and train
        x_tree_df, dummy_names = mcf_data.dummies_for_unord(
            x_tree_df, names_unordered, data_train_dict=data_train_dict)
    else:
        dummy_names = None

    x_tree_np = to_numpy_big_data(x_tree_df, int_cfg.obs_bigdata)
    if gen_cfg.with_output and gen_cfg.verbose:
        print_mememory_statistics(gen_cfg,
                                  'Local centering: Prior to estimation'
                                  )
    lc_r2_txt = None
    if train:
        max_workers = 1 if int_cfg.replication else gen_cfg.mp_parallel
        if names_unordered:
            if fill_y_df is not None:
                x_fy_df, _ = mcf_data.dummies_for_unord(
                    x_fy_df, names_unordered, data_train_dict=data_train_dict)
            if not lc_cfg.cs_cv:
                x_lc_df, _ = mcf_data.dummies_for_unord(
                    x_lc_df, names_unordered, data_train_dict=data_train_dict)
        if not lc_cfg.cs_cv:
            x_lc_np = to_numpy_big_data(x_lc_df, int_cfg.obs_bigdata)
        if fill_y_df is not None:
            x_fy_np = to_numpy_big_data(x_fy_df, int_cfg.obs_bigdata)

        y_tree_np = to_numpy_big_data(tree_mcf_df[var_cfg.y_name],
                                      int_cfg.obs_bigdata)

        # Find best estimator for first outcome
        (estimator, params, txt_sel, _, transform_x, txt_mse
         ) = mcf_gf.best_regression(
            x_tree_np,  y_tree_np[:, 0].ravel(), estimator=lc_cfg.estimator,
            boot=cf_cfg.boot, seed=seed, max_workers=max_workers,
            test_share=lc_cfg.cs_share,
            cross_validation_k=lc_cfg.cs_cv_k if lc_cfg.cs_cv else 0,
            obs_bigdata=int_cfg.obs_bigdata
            )
        if gen_cfg.with_output and gen_cfg.verbose:
            print_mememory_statistics(gen_cfg,
                                      'Local centering training: '
                                      'After best_regression'
                                      )
        forests_all = []
        y_x_tree = np.zeros((len(x_tree_np), len(var_cfg.y_name)),
                            dtype=y_0_dtype)
        if fill_y_df is not None:
            y_x_fy = np.zeros((len(x_fy_np), len(var_cfg.y_name)),
                              dtype=y_0_dtype)
        if lc_cfg.cs_cv:
            index = np.arange(obs_mcf)       # indices
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(index)
            index_folds = np.array_split(index, lc_cfg.cs_cv_k)
            for fold_pred in range(lc_cfg.cs_cv_k):
                fold_train = [x for idx, x in enumerate(index_folds)
                              if idx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                if transform_x:
                    scaler, x_train, x_pred = mcf_gf.scale(
                        x_tree_np[index_train], x_tree_np[index_pred])
                    if fill_y_df is not None:
                        x_fy_np_ = scaler.transform(x_fy_np)
                    else:
                        x_fy_np_ = None
                else:
                    x_train = x_tree_np[index_train]
                    x_pred = x_tree_np[index_pred]
                    if fill_y_df is not None:
                        x_fy_np_ = x_fy_np.copy()
                    else:
                        x_fy_np_ = None
                    scaler = None
                forests_y = []
                for idx, _ in enumerate(var_cfg.y_name):
                    y_train = y_tree_np[index_train, idx].ravel()
                    y_obj = mcf_gf.regress_instance(estimator, params)
                    if y_obj is None:
                        mean = np.average(y_train)
                        y_x_tree[index_pred, idx] = mean
                        if fill_y_df is not None:
                            y_x_fy[:, idx] += mean
                        y_obj = mean
                    else:
                        y_obj.fit(x_train, y_train)
                        y_x_tree[index_pred, idx] = y_obj.predict(x_pred)
                        if fill_y_df is not None:
                            y_x_fy[:, idx] += y_obj.predict(x_fy_np_)
                    forests_y.append((deepcopy(y_obj), deepcopy(scaler),))
                    if gen_cfg.with_output and gen_cfg.verbose:
                        print_mememory_statistics(gen_cfg,
                                                  'Local centering training: '
                                                  'During cross-fitting'
                                                  )
                forests_all.append(deepcopy(forests_y))
            if gen_cfg.with_output:
                for idx, y_name in enumerate(var_cfg.y_name):
                    txt_return = print_lc_info(
                        y_name, False, gen_cfg, txt_sel,
                        y_x_tree[:, idx].ravel(), y_tree_np[:, idx].ravel())
                    if idx == 0:
                        lc_r2_txt = txt_return
            if fill_y_df is not None:
                y_x_fy /= lc_cfg.cs_cv_k
        else:
            forests_y = []
            if transform_x:
                scaler, x_lc_np_, x_tree_np_ = mcf_gf.scale(x_lc_np,
                                                            x_test=x_tree_np
                                                            )
                if fill_y_df is not None:
                    x_fy_np_ = scaler.transform(x_fy_np)
            else:
                x_lc_np_, x_tree_np_ = x_lc_np, x_tree_np
                if fill_y_df is not None:
                    x_fy_np_ = x_fy_np
                scaler = None
            for idx, y_name in enumerate(var_cfg.y_name):
                y_lc_np = to_numpy_big_data(
                    data_lc_df[y_name], int_cfg.obs_bigdata).ravel()
                y_obj = mcf_gf.regress_instance(estimator, params)
                if y_obj is None:
                    mean = np.average(y_lc_np)
                    y_x_tree[:, idx] = mean
                    if fill_y_df is not None:
                        y_x_fy[:, idx] = mean
                    y_obj = mean
                else:
                    y_obj.fit(x_lc_np_, y_lc_np)
                    y_x_tree[:, idx] = y_obj.predict(x_tree_np_)
                    if fill_y_df is not None:
                        y_x_fy[:, idx] = y_obj.predict(x_fy_np_)
                forests_y.append((deepcopy(y_obj), deepcopy(scaler)))
                if gen_cfg.with_output:
                    txt_return = print_lc_info(
                        y_name, False, gen_cfg, txt_sel,
                        y_x_tree[:, idx].ravel(),
                        to_numpy_big_data(tree_mcf_df[y_name],
                                          int_cfg.obs_bigdata).ravel())
                    if idx == 0:
                        lc_r2_txt = txt_return
            if gen_cfg.with_output and gen_cfg.verbose:
                print_mememory_statistics(gen_cfg,
                                          'Local centering training: '
                                          'During prediction')
            forests_all.append(deepcopy(forests_y))
        if isinstance(var_cfg.y_name, str):
            var_cfg.y_name = [var_cfg.y_name]
        y_cent_name = [name + '_lc' for name in var_cfg.y_name]
        y_x_name = [name + '_Ey_x' for name in var_cfg.y_name]
        var_cfg.y_name_lc, var_cfg.y_name_ey_x = y_cent_name, y_x_name

        # Use predictions to center outcomes
        y_m_yx_tree = to_numpy_big_data(tree_mcf_df[var_cfg.y_name],
                                        int_cfg.obs_bigdata) - y_x_tree
        # Transfer back to pandas DataFrames
        y_m_yx_tree_df = pd.DataFrame(data=y_m_yx_tree, columns=y_cent_name)
        y_x_tree_df = pd.DataFrame(data=y_x_tree, columns=y_x_name)
        # Reset indices for existing dataframes
        tree_mcf_df = tree_mcf_df.reset_index(drop=True)
        tree_add_y_lc_df = pd.concat((tree_mcf_df, y_m_yx_tree_df, y_x_tree_df
                                      ), axis=1)
        if fill_y_df is not None:
            y_m_yx_fy = to_numpy_big_data(fill_y_mcf_df[var_cfg.y_name],
                                          int_cfg.obs_bigdata) - y_x_fy
            y_m_yx_fy_df = pd.DataFrame(data=y_m_yx_fy, columns=y_cent_name)
            y_x_fy_df = pd.DataFrame(data=y_x_fy, columns=y_x_name)
            fill_y_mcf_df = fill_y_mcf_df.reset_index(drop=True)
            fill_add_y_lc_df = pd.concat(
                (fill_y_mcf_df, y_m_yx_fy_df, y_x_fy_df), axis=1)
        else:
            fill_add_y_lc_df = None

        if gen_cfg.with_output:
            print_mcf(gen_cfg, txt_mse, summary=False)

            if names_unordered:
                names_unordered_print = (
                    data_train_dict['prime_old_name_dict']
                    )
            else:
                names_unordered_print = None
            for y_idx, objects in enumerate(forests_y):
                obj, scaler = objects
                if obj is not None and not isinstance(obj, float):
                    # Only if not mean (makes no sense)
                    print_variable_importance(
                        deepcopy(obj),
                        x_tree_df, tree_mcf_df[[var_cfg.y_name[y_idx]]],
                        x_name, names_unordered, dummy_names, gen_cfg,
                        summary=True, name_label_dict=names_unordered_print,
                        obs_bigdata=int_cfg.obs_bigdata,
                        classification=False,
                        scaler=scaler
                        )
            all_y_name = [*var_cfg.y_name, *y_cent_name, *y_x_name]
            if fill_y_df is None:
                print_descriptive_df(
                    gen_cfg, tree_add_y_lc_df, varnames=all_y_name,
                    summary=False
                    )
            else:
                print_descriptive_df(
                    gen_cfg,
                    pd.concat((tree_add_y_lc_df, fill_add_y_lc_df), axis=0),
                    varnames=all_y_name, summary=False
                    )
        lc_cfg.forests = forests_all
        var_cfg = adjust_y_names(var_cfg, gen_cfg, var_cfg.y_name,
                                 y_cent_name, summary=True
                                 )
        y_x_df = None

        mcf_.lc_cfg = lc_cfg
        mcf_.var_cfg = var_cfg
        # ---------
    else:  # Predict Ey|X
        y_x_np = np.zeros((len(x_tree_np), len(var_cfg.y_name)),
                          dtype=y_0_dtype)
        if lc_cfg.forests == []:
            raise ValueError('Estimator used for centering was not saved for '
                             'prediction.'
                             )
        no_of_folds = len(lc_cfg.forests)
        for forest in lc_cfg.forests:   # Loops over training folds
            for idx, forest_y in enumerate(forest):  # Loops over outcomes
                scaler = forest_y[1]
                if scaler is not None:
                    x_tree_np_ = scaler.transform(x_tree_np)
                else:
                    x_tree_np_ = x_tree_np
                if isinstance(forest_y[0], float):
                    y_x_np[:, idx] += forest_y[0]
                else:
                    y_x_np[:, idx] += forest_y[0].predict(x_tree_np_)

        y_x_np /= no_of_folds
        tree_add_y_lc_df = fill_add_y_lc_df = None
        y_x_df = pd.DataFrame(data=y_x_np, columns=var_cfg.y_name_ey_x)

    if fill_y_df is None:
        fill_add_y_lc_df = None
    if gen_cfg.with_output and gen_cfg.verbose:
        print_mememory_statistics(gen_cfg,
                                  'Local centering: End of function.'
                                  )
    return tree_add_y_lc_df, fill_add_y_lc_df, y_x_df, lc_r2_txt


def print_lc_header(gen_cfg: Any, lc_cfg: Any, title: str = '') -> None:
    """Print the header."""
    print_mcf(gen_cfg, '=' * 100 + f'\nLocal Centering  {title}', summary=True)
    if gen_cfg.verbose:
        txt = '\nLocal centering is done '
        if lc_cfg.cs_cv:
            txt += 'by cross-validation'
        else:
            txt += 'in an independent random sample'
        print_mcf(gen_cfg, txt, summary=False)


def print_lc_info(y_name: str,
                  y_as_classifier: bool,
                  gen_cfg: Any,
                  txt_sel: str,
                  y_pred: NDArray[Any],
                  y_true: NDArray[Any],
                  summary=True
                  ) -> str:
    """Compute some basic info on outcome regression in local centering."""
    if y_as_classifier:
        fit = accuracy_score(y_true, y_pred, normalize=True)
        method = 'Accuracy score'
    else:
        fit = r2_score(y_true, y_pred)
        method = 'R2'
    txt = '\n' + 100 * '-' + '\nFit of Ey|x for local centering (Best method: '
    txt += (txt_sel + '). ' + method + f' for {y_name}: {fit:5.2%} \n'
            + '- ' * 50)
    print_mcf(gen_cfg, txt, summary=summary)
    txt_return = txt_sel + ' of Ey|x (' + method + f') for {y_name}: {fit:5.2%}'

    return txt_return


def adjust_y_names(var_cfg: Any,
                   gen_cfg: Any,
                   y_name_old: list[str],
                   y_name_new: list[str],
                   summary: bool = True
                   ) -> dict:
    """
    Switch variables names of y in dictionary.

    Parameters
    ----------
    var_cfg : VarCfg Dataclass. Variables.
    y_name_old : List of strings. Old variable names.
    y_name_new : List of strings. New variable names.
    with_output : Boolean.

    Returns
    -------
    var_cfg : VarCfg Dataclass. Modified variable names.

    """
    var_cfg.y_tree_name_unc = var_cfg.y_tree_name.copy()
    for indx, y_name in enumerate(y_name_old):
        if (var_cfg.y_tree_name is None
            or var_cfg.y_tree_name == []
                or y_name == var_cfg.y_tree_name[0]):
            var_cfg.y_tree_name = [y_name_new[indx]]
            break
    var_cfg.y_name = y_name_new
    if gen_cfg.with_output:
        txt = '\n' + 'New variable to build trees in RF: '
        txt += f' {var_cfg.y_tree_name[0]}'
        print_mcf(gen_cfg, txt, summary=summary)

    return var_cfg
