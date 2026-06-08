"""
Contains functions for implementing weight based bias correction.

Created on 10.11.2025

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mcf import mcf_estimation_weighted_regression as mcf_ewr
from mcf.mcf_estimation_weighted_regression import standardize_col
from mcf.mcf_data import get_treat_info
from mcf.mcf_print_stats import print_mcf
from mcf.mcf_estimation_generic import best_regression, scale, regress_instance
from mcf.mcf_estimation import predict_proba_aligned
from mcf.mcf_init_update_helper import pa_ba_update_train

type ArrayLike = NDArray[Any] | None

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import PBaCfg


def get_weights_eval_ba(weights: NDArray[np.floating],
                        no_of_treat: int,
                        zero_tol: float = 1e-10,
                        ) -> NDArray[np.floating]:
    """Get weights for weighted evaluation distribution."""
    w_dim, shape = weights.ndim, weights.shape
    # Initialise weight matrix
    size = (shape[0], shape[2]) if w_dim == 3 else shape[1]
    weights_eval = np.zeros(size, dtype=weights.dtype)
    # Share of Number of treated to be used for reweighthing weights
    nonzero = (np.abs(weights) >= zero_tol).mean(axis=-1, dtype=weights.dtype)
    if np.any(nonzero <= zero_tol):
        raise ValueError('Cannot compute bias-adjustment evaluation weights because at least one '
                         'treatment cell has no nonzero weights.'
                         )
    for t_idx in range(no_of_treat):
        # Add the weights, reweighted by relative treatment share
        if w_dim == 3:
            weights_eval += weights[:, t_idx, :] / nonzero[:, t_idx].reshape(-1, 1)
        else:
            weights_eval += weights[t_idx, :] / nonzero[t_idx]

    return weights_eval


@dataclass(slots=True, kw_only=True)
class BaData:
    """Holds the data used in ba_adjustments within each fold."""

    x_ba_train: ArrayLike
    prog_sc_train: ArrayLike
    prop_sc_train: ArrayLike

    x_ba_eval: ArrayLike
    prog_sc_eval: ArrayLike
    prop_sc_eval: ArrayLike

    weights_eval: ArrayLike


# TODO this needs to be adjusted for the prediction data, if ever implemented
def get_ba_data_prediction(weights_dic: dict, p_ba_cfg: 'PBaCfg') -> Any:
    """Get data for bias adjustment as instance of named tuple."""
    ba_data = BaData(x_ba_train=weights_dic['x_ba_dat'],
                     prog_sc_train=weights_dic['prog_dat_np'],
                     prop_sc_train=weights_dic['prop_dat_np'],
                     x_ba_eval=p_ba_cfg.x_ba_eval,
                     prog_sc_eval=p_ba_cfg.prog_score_eval,
                     prop_sc_eval=p_ba_cfg.prop_score_eval,
                     weights_eval = None  # If needed, it will be changed outside
                     )
    return ba_data


# TODO: In the future: Create CUDA version
def bias_correction_wregr(weights: NDArray[np.floating],
                          y_dat,
                          ba_data: Any, *,
                          int_dtype: DTypeLike = np.float64,
                          out_dtype: DTypeLike = np.float64,
                          pos_weights_only: bool = False,
                          w_index: ArrayLike = None,
                          zero_tol: float = 1e-10,
                          ridge: bool = True,
                          cv_k: int = 5,
                          ) -> NDArray[np.floating]:
    """Adjust the weights to account for the bias correction."""
    def regressors_add(regr_train: list[NDArray[Any]],
                       regr_eval: list[NDArray[Any]],
                       x_train: NDArray[Any],
                       x_eval: NDArray[Any],
                       ) -> tuple[list[NDArray[Any]], list[NDArray[Any]],]:
        """Build the regressor matrices."""
        regr_train = np.concatenate((regr_train, x_train), axis=1)
        if x_eval is None or np.all(np.isclose(x_eval, 0, atol=zero_tol, rtol=zero_tol)):
            regr_eval = np.concatenate((regr_eval, np.zeros_like(x_train)), axis=1,)
        else:
            regr_eval = np.concatenate((regr_eval, x_eval), axis=1)

        return regr_train, regr_eval

    # Build regressor matrix  (all regressors are 2D arrays, if they exist)
    w_nonzero = np.abs(weights) > zero_tol  # Use only rows with nonzero weights
    #                             otherwise they may come from a different treatment
    obs = int(np.sum(w_nonzero))

    regr_train = np.ones((obs, 1), dtype=out_dtype)  # Constant term
    eva = (ba_data.x_ba_eval, ba_data.prog_sc_eval, ba_data.prop_sc_eval)
    if all(a is None for a in eva):  # No evaluation data provided
        regr_eval = np.ones_like(regr_train)
    else:
        obs_eva = max(len(a) for a in eva if a is not None)
        regr_eval = np.ones((obs_eva, 1))

    if ba_data.x_ba_train is not None:
        if w_index is None:
            x_ba_train = ba_data.x_ba_train[w_nonzero]
        else:
            x_ba_train = ba_data.x_ba_train[w_index][w_nonzero]

        regr_train, regr_eval = regressors_add(regr_train, regr_eval, x_ba_train, ba_data.x_ba_eval)
    if ba_data.prog_sc_train is not None:
        if w_index is None:
            prog_sc_train = ba_data.prog_sc_train[w_nonzero]
        else:
            prog_sc_train = ba_data.prog_sc_train[w_index][w_nonzero]

        regr_train, regr_eval = regressors_add(regr_train, regr_eval,
                                               prog_sc_train, ba_data.prog_sc_eval,
                                               )
    if ba_data.prop_sc_train is not None:
        if w_index is None:
            prop_sc_train = ba_data.prop_sc_train[w_nonzero]
        else:
            prop_sc_train = ba_data.prop_sc_train[w_index][w_nonzero]

        regr_train, regr_eval = regressors_add(regr_train, regr_eval, prop_sc_train,
                                               ba_data.prop_sc_eval,
                                               )
    # run WOLS & obtain weights from this regression
    if ridge:
        # choose penalty
        w_non_zero = weights[w_nonzero].reshape(-1, 1)
        grid=np.zeros(31)
        grid[1:] = np.logspace(-5, 2, 30)
        best_lam, _ = mcf_ewr.ridge_cv_penalty(regr_train, y_dat[w_nonzero],
                                               w_non_zero,
                                               lambdas=grid,
                                               k=cv_k,
                                               shuffle=True,
                                               seed=1234566,
                                               not_penalize_first_k=1,
                                               dtype=int_dtype,
                                               )
        weights_ba = mcf_ewr.ridge_equivalent_weights_for_mean(regr_train,
                                                               w_non_zero,
                                                               y=None,
                                                               lam=best_lam,
                                                               x_eval=regr_eval,
                                                               int_dtype=int_dtype,
                                                               out_dtype=out_dtype,
                                                               weights_eval=ba_data.weights_eval,
                                                               not_penalize_first_k=1,
                                                               return_residuals=False,
                                                               )
    else:
        weights_ba = mcf_ewr.wls_equivalent_weights_for_mean(regr_train,
                                                             weights[w_nonzero].reshape(-1, 1),
                                                             y=None,
                                                             x_eval=regr_eval,
                                                             int_dtype=int_dtype,
                                                             out_dtype=out_dtype,
                                                             weights_eval=ba_data.weights_eval,
                                                             return_residuals=False,
                                                             )
    weights_ba = weights_ba.reshape(-1)                                   # pylint: disable=E1101

    sum_weighs_ba = np.sum(weights_ba)
    # Eleminate negative weights & recompute positive weights
    if np.any(weights_ba < 0.0) and pos_weights_only:
        weights_ba = mcf_ewr.project_to_simplex(weights_ba, sum_weighs_ba, dtype=out_dtype,)
    # Make sure outgoing weights have same sum as incoming weights
    if  np.abs(sum_weighs_ba) > zero_tol:
        weights_ba *= (np.sum(weights) / sum_weighs_ba)
    # Adjust the positive weights
    weights_all = np.zeros_like(weights, dtype=weights_ba.dtype)
    weights_all[w_nonzero] = weights_ba

    return weights_all


def get_ba_data_train(idx_sub: pd.Index,
                      idx: pd.Index,
                      prog: ArrayLike = None,
                      prop: ArrayLike = None,
                      x_ba: ArrayLike = None,
                      ) -> tuple[ArrayLike, ArrayLike, ArrayLike,]:
    """Select rows of data that correspond to idx_sub."""
    if prog is None:
        prog_dat = None
    else:
        prog_dat = pd.DataFrame(prog, index=idx).loc[idx_sub].to_numpy(dtype=prog.dtype, copy=False)
    if prop is None:
        prop_dat = None
    else:
        prop_dat = pd.DataFrame(prop, index=idx).loc[idx_sub].to_numpy(dtype=prop.dtype, copy=False)
    if x_ba is None:
        x_ba_dat = None
    else:
        x_ba_dat = pd.DataFrame(x_ba, index=idx).loc[idx_sub].to_numpy(dtype=x_ba.dtype, copy=False)

    return prog_dat, prop_dat, x_ba_dat


# Can be used for training and prediction
def compute_scores_x(mcf_: 'ModifiedCausalForest',
                     train_df: pd.DataFrame,
                     predict_df: pd.DataFrame,
                     ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute prognostic and propensity scores and get x_values."""
    var_x_type = mcf_.var_x_type

    use_prop_score = mcf_.p_ba_cfg.use_prop_score
    use_prog_score = mcf_.p_ba_cfg.use_prog_score
    use_x = mcf_.p_ba_cfg.use_x

    # Estimate the prognostic and/or propensity score in the train_df data and
    # predict their values in the predict_df data

    # Get the ordered and unordered variables
    names_ordered = [name for name, type_ in var_x_type.items() if type_ in [0]]
    names_unordered = [name for name, type_ in var_x_type.items() if type_ in [1, 2]]
    if use_prop_score or use_prog_score:
        prog_score, prop_score = get_scores(mcf_, train_df, predict_df,
                                            names_ordered,  names_unordered
                                            )
    else:
        prog_score = prop_score = None

    if use_x:
        x_np = get_x(mcf_, predict_df, names_ordered,  names_unordered)
    else:
        x_np = None
    # Ensure that we return only 2D arrays (progscore is 2D automatically)
    if prog_score is not None and prog_score.ndim == 1:
        prog_score = prog_score.reshape(-1, 1)
    if x_np is not None and x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)

    return prog_score, prop_score, x_np


def get_x(mcf_: 'ModifiedCausalForest',
          data_df: pd.DataFrame,
          names_ordered: list[str],
          names_unordered: list[str],
          ) -> NDArray[Any]:
    """Get x values als numpy array (dummies for ordered variables)."""
    x_name_ba = mcf_.var_cfg.x_name_ba
    names_ba_ordered = [name for name in x_name_ba if name in names_ordered]
    names_ba_unordered = [name for name in x_name_ba if name in names_unordered]

    if names_ba_ordered:
        x_ba_o = data_df[names_ba_ordered].to_numpy(dtype=np.float32)
    else:
        x_ba_o = None

    if names_ba_unordered:
        x_ba_u_df = data_df[names_ba_unordered].astype('category')
        x_ba_u_df = pd.get_dummies(data=x_ba_u_df,
                                   columns=names_ba_unordered,
                                   prefix=names_ba_unordered,
                                   drop_first=True,
                                   dtype='int8',
                                   )
        x_ba_u = x_ba_u_df.values
    else:
        x_ba_u = None

    if names_ba_ordered and names_ba_unordered:
        return np.concatenate((x_ba_o, x_ba_u.astype(np.float32)), axis=1)

    if names_ba_ordered and not names_ba_unordered:
        return x_ba_o

    return x_ba_u


def get_scores(mcf_: 'ModifiedCausalForest',
               train_df: pd.DataFrame,
               predict_df: pd.DataFrame,
               names_ordered: list[str],
               names_unordered: list[str],
               ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute the prognostic and the propensity scores."""
    # This procedure maybe optimized for doing training and prediction only
    # Not yet fully thought through to the end
    # train = train_df is not None
    # predict = predict_df is not None
    d_name, d_values, no_of_treat = get_treat_info(mcf_)
    y_tree_name = mcf_.var_cfg.y_tree_name
    y_tree_name_unc = mcf_.var_cfg.y_tree_name_unc
    obs_bigdata = mcf_.int_cfg.obs_bigdata
    boot = mcf_.cf_cfg.boot
    lc_cfg, gen_cfg, p_ba_cfg = mcf_.lc_cfg, mcf_.gen_cfg, mcf_.p_ba_cfg
    zero_tol = mcf_.int_cfg.zero_tol
    with_output = gen_cfg.with_output

    use_prop_score = mcf_.p_ba_cfg.use_prop_score
    use_prog_score = mcf_.p_ba_cfg.use_prog_score

    d_predict_np, d_in_predict_df = None, False
    y_predict_np, y_in_predict_df = None, False

    if names_unordered:
        # categorize the unordered variables
        # if predict:
        predict_df[names_unordered] = predict_df[names_unordered].astype('category')
        if use_prop_score or use_prog_score:  # and train:
            train_df[names_unordered] = train_df[names_unordered].astype('category')

    # Get the numpy x and y data
    if names_ordered:
        # if predict:
        x_predict_o_df = predict_df[names_ordered]
        # if train:
        x_train_o_df = train_df[names_ordered]
    else:
        x_predict_o_df = x_train_o_df = None

    if names_unordered:
        # if predict:
        x_predict_u_df = pd.get_dummies(data=predict_df[names_unordered],
                                        columns=names_unordered,
                                        prefix=names_unordered,
                                        dtype=int,
                                        )
        # if train:
        x_train_u_df = pd.get_dummies(data=train_df[names_unordered],
                                      columns=names_unordered,
                                      prefix=names_unordered,
                                      dtype=int,
                                      )
        # if predict and train:
        # Make sure predict has exactly the same columns as train
        x_predict_u_df = x_predict_u_df.reindex(columns=x_train_u_df.columns, fill_value=0)
    else:
        x_predict_u_df = x_train_u_df = None

    if names_ordered and names_unordered:
        # if train:
        x_train_np = pd.concat([x_train_u_df, x_train_o_df], axis=1).to_numpy(dtype=np.float32)
        #if predict:
        x_predict_np = pd.concat([x_predict_u_df, x_predict_o_df], axis=1,
                                 ).to_numpy(dtype=np.float32)
    elif names_ordered and not names_unordered:
        # if train:
        x_train_np = x_train_o_df.to_numpy(dtype=np.float32)
        # if predict:
        x_predict_np = x_predict_o_df.to_numpy(dtype=np.float32)
    else:
        # if train:
        x_train_np = x_train_u_df.to_numpy(dtype=np.float32)
        # if predict:
        x_predict_np = x_predict_u_df.to_numpy(dtype=np.float32)
    # if train:
    d_train_np = train_df[d_name].to_numpy(dtype=np.float32)

    if with_output: # and predict:
        d_in_predict_df = d_name[0] in predict_df.columns
        if d_in_predict_df:  # later used to compute out-of-sample fit measures
            d_predict_np = predict_df[d_name].to_numpy(dtype=np.float32)

    # obs_predict = 0 if predict else x_predict_np.shape[0]
    obs_predict = x_predict_np.shape[0]

    # Use same parameters for both classifier estimations
    params_rfc = {"n_estimators": boot,
                  "max_features": "sqrt",
                  "bootstrap": True,
                  "oob_score": False,
                  "n_jobs": gen_cfg.mp_parallel,
                  "random_state": 42,
                  "verbose": False,
                  'min_samples_split': 5,
                  }
    if use_prog_score:
        if with_output:
            txt = '\nComputing prognostic scores for bias adjustment'
            print_mcf(gen_cfg, txt, summary=False)

        # if predict:
        prog_score_np = np.empty((obs_predict, no_of_treat), dtype=np.float32)

        #   if train:
        y_name = y_tree_name_unc if lc_cfg.yes else y_tree_name
        y_train_np = train_df[y_name].to_numpy(dtype=np.float32)

        # Only prognostic score for variable used to build forest used

        if with_output: # and predict:
            y_in_predict_df = y_name[0] in predict_df.columns
            if y_in_predict_df:
                y_predict_np = predict_df[y_name].to_numpy(dtype=np.float32)

        # Create a model, either a RFclassifier or a regression based method
        # if train:
        y_discrete = np.unique(y_train_np).shape[0] < 10

        if y_discrete:
            rfc_all = RandomForestClassifier(**params_rfc)
        # else:
        #     rfc_all = RandomForestRegressor(**params)

        # Train for each individual treatment
        for idx, d_treatment in enumerate(d_values):

            # Train on the treatment group
            # if train:
            treatments = np.isclose(d_train_np, d_treatment, rtol=zero_tol, atol=zero_tol).squeeze()
            x_tmp_np = x_train_np[treatments].copy()
            y_tmp_np = y_train_np[treatments].squeeze().copy()
            # Get a new clean instance
            if y_discrete:
                obj = deepcopy(rfc_all)
                # train the model
                obj.fit(x_tmp_np, y_tmp_np)
                # predict on the full prediction dataset
                prog_score_np[:, idx] = obj.predict(x_predict_np)
                transform_x = False
            else:
                if p_ba_cfg.cv_k is None:
                    p_ba_cfg.cv_k = pa_ba_update_train(p_ba_cfg.cv_k, x_tmp_np.shape[0],)
                (estimator, params, txt_sel, _, transform_x, txt_mse
                 ) = best_regression(x_tmp_np,  y_tmp_np.ravel(), estimator=p_ba_cfg.estimator,
                                     boot=boot, seed=123456, max_workers=gen_cfg.mp_parallel,
                                     test_share=0, cross_validation_k=p_ba_cfg.cv_k,
                                     obs_bigdata=obs_bigdata,
                                     )
                if transform_x:
                    _, x_train, x_pred = scale(x_tmp_np.copy(), x_predict_np.copy())
                else:
                    x_train, x_pred = x_tmp_np, x_predict_np

                obj = regress_instance(estimator, params)
                if obj is None:
                    mean = np.average(y_tmp_np.ravel())
                    prog_score_np[:, idx] = mean
                else:
                    obj.fit(x_train, y_tmp_np.ravel())
                    prog_score_np[:, idx] = obj.predict(x_pred)

            if with_output and y_in_predict_df and d_in_predict_df:
                treatments_p = np.isclose(d_predict_np, d_treatment, atol=zero_tol, rtol=zero_tol
                                          ).squeeze()
                x_predict = x_pred if transform_x else x_predict_np
                if obj is not None:
                    score = obj.score(x_predict[treatments_p], y_predict_np[treatments_p],)
                else:
                    score = 0
                if y_discrete:
                    txt = (f'\nOut-of-sample accuracy of prognostic score '
                           f'of treatment {d_treatment: d}: {score:.2%}'
                           '\n' + '- ' * 50
                           )
                else:
                    txt = (txt_sel + txt_mse +
                           '\nOut-of-sample R2 of prognostic score of '
                           f'treatment {d_treatment: d}: {score:.2%}'
                           '\n' + '- ' * 50
                           )
                print_mcf(gen_cfg, txt, summary=False)
    else:
        prog_score_np = None

    if use_prop_score:
        if with_output:
            txt = '\nComputing propensity scores for bias adjustment'
            print_mcf(gen_cfg, txt, summary=False)
        # Always use a classifier for the discrete treatments
        classif = RandomForestClassifier(**params_rfc)

        # Train on the training data
        classif.fit(x_train_np, d_train_np.ravel())

        # Predict the probabilities with the features of the prediction data
        # prop_all = classif.predict_proba(x_predict_np).astype(np.float32)
        prop_all = predict_proba_aligned(classif, x_predict_np, d_values,
                                         dtype=np.float32, zero_tol=zero_tol,
                                         )
        prop_score_np = prop_all[:, 1:]
        if with_output and d_in_predict_df:
            score = classif.score(x_predict_np, d_predict_np)
            txt = (f'\nOut-of-sample accuracy of propensity score: {score:.2%} '
                   '(score method of RandomForestClassifier by scikit-learn)'
                   )
            print_mcf(gen_cfg, txt, summary=False)

        # First category skipped becaused probabilities add up to 1
        prop_score_np = prop_all[:, 1:]
    else:
        prop_score_np = None

    return prog_score_np, prop_score_np


def compute_ba_arrays(mcf_: 'ModifiedCausalForest', *,
                      train_df: pd.DataFrame,
                      predict_df: pd.DataFrame,
                      ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Compute numpy arrays used for bias adjustment before forest building."""
    # TODO --> auslagern in anderes File!!
    # Note: For large data, this is done before forests are computed in
    #       folds. Therefore, we need to save the index to be able to
    #       relate the data to the correct folds in the prediction method
    #       (if needed)
    # These values will be stored in the respective forest_dic defined below
    # TODO save instances if prediction data is used
    p_ba_cfg = mcf_.p_ba_cfg
    gen_cfg = mcf_.gen_cfg
    prog_r, prop_r, x_r = compute_scores_x(mcf_, train_df=train_df, predict_df=predict_df)
    # TODO save means if prediction data is used
    prog_r, prop_r = standardize_col(prog_r), standardize_col(prop_r)
    x_r = standardize_col(x_r)

    # Define the evaluation points
    match p_ba_cfg.adj_method:
        case 'zeros':
            p_ba_cfg.x_ba_eval = None
            p_ba_cfg.prog_score_eval = None
            p_ba_cfg.prop_score_eval = None
        case 'obs' | 'w_obs':
            p_ba_cfg.x_ba_eval = standardize_col(x_r)
            p_ba_cfg.prog_score_eval = standardize_col(prog_r)
            p_ba_cfg.prop_score_eval = standardize_col(prop_r)
            # TODO here add NEW case if prediction data is used
        case _:
            raise ValueError('Invalid Bias Adjustment Method')

    if gen_cfg.any_eff:
        # Change role of samples
        prog_a, prop_a, x_a = compute_scores_x(mcf_, train_df=predict_df, predict_df=train_df)
        prog_a, prop_a, x_a = standardize_col(prog_a), standardize_col(prop_a), standardize_col(x_a)
    else:
        prog_a = prop_a = x_a = None

    return prog_r, prop_r, x_r, prog_a, prop_a, x_a
