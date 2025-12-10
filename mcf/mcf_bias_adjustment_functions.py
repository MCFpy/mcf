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

from mcf.mcf_data_functions import get_treat_info
from mcf.mcf_print_stats_functions import print_mcf
from mcf.mcf_estimation_generic_functions import (
    best_regression, scale, regress_instance
    )
from mcf.mcf_init_update_helper_functions import pa_ba_update_train

type ArrayLike = NDArray[Any] | None

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def demean_col(x_in: ArrayLike) -> ArrayLike:
    """Substract row mean from columns of numpy array."""
    return None if x_in is None else (x_in - np.mean(x_in, axis=0))


def get_weights_eval_ba(weights: NDArray[np.floating],
                        no_of_treat: int,
                        zero_tol: float = 1e-15,
                        ) -> NDArray[np.floating]:
    """Get weights for weighted evaluation distribution."""
    w_dim, shape = weights.ndim, weights.shape
    # Initialise weight matrix
    size = (shape[0], shape[2]) if w_dim == 3 else shape[1]
    weights_eval = np.zeros(size, dtype=weights.dtype)
    # Share of Number of treated to be used for reweighthing weights
    nonzero = (np.abs(weights) >= zero_tol).mean(axis=-1, dtype=weights.dtype)

    for t_idx in range(no_of_treat):
        # Add the weights, reweighted by relative treatment share
        if w_dim == 3:
            weights_eval += (weights[:, t_idx, :]
                             / nonzero[:, t_idx].reshape(-1, 1)
                             )
        else:
            weights_eval += weights[t_idx, :] / nonzero[t_idx]

    return weights_eval


@dataclass(slots=True, kw_only=True)
class BaData:
    """This holds the data used in ba_adjustments within each fold."""
    x_ba_train: ArrayLike
    prog_sc_train: ArrayLike
    prop_sc_train: ArrayLike
    x_ba_eval: ArrayLike
    prog_sc_eval: ArrayLike
    prop_sc_eval: ArrayLike
    weights_eval: ArrayLike


def get_ba_data_prediction(weights_dic: dict, p_ba_cfg: Any) -> Any:
    """Get data for bias adjustment as instance of named tuple."""
    ba_data = BaData(
        x_ba_train=weights_dic['x_ba_dat'],
        prog_sc_train=weights_dic['prog_dat_np'],
        prop_sc_train=weights_dic['prop_dat_np'],
        x_ba_eval=p_ba_cfg.x_ba_eval,
        prog_sc_eval=p_ba_cfg.prog_score_eval,
        prop_sc_eval=p_ba_cfg.prop_score_eval,
        weights_eval = None  # If needed, it will be changed outside
        )
    return ba_data


# TODO: In the future: Create CUDA version
def bias_correction_wols(weights: NDArray[np.floating],
                         ba_data: Any,
                         int_dtype: DTypeLike = np.float64,
                         out_dtype: DTypeLike = np.float64,
                         pos_weights_only: bool = False,
                         w_index: ArrayLike = None,
                         zero_tol: float = 1e-15
                         ) -> NDArray[np.floating]:
    """Adjust the weights to account for the bias correction."""
    def regressors_add(regr_train: list[NDArray[Any]],
                       regr_eval: list[NDArray[Any]],
                       x_train: NDArray[Any],
                       x_eval: NDArray[Any],
                       ) -> tuple[list[NDArray[Any]],
                                  list[NDArray[Any]],
                                  ]:
        """Build the regressor matrices."""
        regr_train = np.concatenate((regr_train, x_train), axis=1)
        if x_eval is None or np.all(x_eval == 0):
            regr_eval = np.concatenate((regr_eval, np.zeros_like(x_train)),
                                       axis=1,
                                       )
        else:
            regr_eval = np.concatenate((regr_eval, x_eval), axis=1)

        return regr_train, regr_eval

    # Build regressor matrix  (all regressors are 2D arrays, if they exist)
    w_nonzero = np.abs(weights) > zero_tol  # Use only rows with nonzero weights
    #                             otherwise they may come from a different
    #                             treatment
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

        regr_train, regr_eval = regressors_add(regr_train, regr_eval,
                                               x_ba_train,
                                               ba_data.x_ba_eval,
                                               )
    if ba_data.prog_sc_train is not None:
        if w_index is None:
            prog_sc_train = ba_data.prog_sc_train[w_nonzero]
        else:
            prog_sc_train = ba_data.prog_sc_train[w_index][w_nonzero]

        regr_train, regr_eval = regressors_add(regr_train, regr_eval,
                                               prog_sc_train,
                                               ba_data.prog_sc_eval,
                                               )
    if ba_data.prop_sc_train is not None:
        if w_index is None:
            prop_sc_train = ba_data.prop_sc_train[w_nonzero]
        else:
            prop_sc_train = ba_data.prop_sc_train[w_index][w_nonzero]

        regr_train, regr_eval = regressors_add(regr_train, regr_eval,
                                               prop_sc_train,
                                               ba_data.prop_sc_eval,
                                               )
    # run WOLS & obtain weights from this regression
    weights_ba = wls_equivalent_weights_for_mean(
        regr_train,
        weights[w_nonzero].reshape(-1, 1),
        regr_eval,
        int_dtype=int_dtype,
        out_dtype=out_dtype,
        weights_eval=ba_data.weights_eval,
        )
    weights_ba = weights_ba.reshape(-1)

    sum_weighs_ba = np.sum(weights_ba)
    # Eleminate negative weights & recompute positive weights
    if np.any(weights_ba < 0.0) and pos_weights_only:
        weights_ba = project_to_simplex(weights_ba,
                                        sum_weighs_ba,
                                        dtype=out_dtype,
                                        )
    # Make sure outgoing weights have same sum as incoming weights
    if  np.abs(sum_weighs_ba) > zero_tol:
        weights_ba *= (np.sum(weights) / sum_weighs_ba)
    # Adjust the positive weights
    weights_all = np.zeros_like(weights, dtype=weights_ba.dtype)
    weights_all[w_nonzero] = weights_ba

    return weights_all


def project_to_simplex(weight_in: NDArray[np.floating],
                       sum_out: float | np.floating,
                       dtype: DTypeLike = np.float64,
                       ) -> NDArray[np.floating]:
    """
    Euclidean projection of weight_in onto the simplex
    {a >= 0, sum(a) = sum_out}.
    O(n log n) via sorting (Duchi et al., 2008).
    """
    weight_in = np.asarray(weight_in, dtype=dtype)
    if sum_out <= 0:  # Do not do anything
        return weight_in

    u = np.sort(weight_in)[::-1]
    cssv = np.cumsum(u) - sum_out
    rho = np.nonzero(u - cssv / (np.arange(weight_in.size) + 1) > 0)[0]
    if rho.size == 0:
        # all entries project to zero except we must hit the sum z; fall back
        theta = (u.sum() - sum_out) / weight_in.size
    else:
        rho = rho[-1]
        theta = cssv[rho] / (rho + 1.0)
    weight_out = weight_in - theta
    weight_out[weight_out < 0.0] = 0.0

    return weight_out


def wls_equivalent_weights_for_mean(x: NDArray[Any],
                                    w: NDArray[np.floating],
                                    x_eval: ArrayLike = None,
                                    int_dtype: DTypeLike = np.float64,
                                    out_dtype: DTypeLike = np.float64,
                                    weights_eval: ArrayLike = None,
                                    ) -> NDArray[np.floating]:
    """Compute new weights for mean."""
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_eval is None:
        x_eval = np.asarray(x, dtype=int_dtype)

    return proj.weights_matrix_for_mean(x_eval,
                                        avg_w=weights_eval,
                                        int_dtype=int_dtype,
                                        out_dtype=out_dtype,
                                        )


class WLSProjector:
    """
    Equivalent-weights projector for WLS with sampling weights.
    x must already include the intercept column.

    For any x_new, yhat = w_eq(x_new) @ y, where
    w_eq(x_new) = W X (X' W X)^{-1} x_new.
    """
    def __init__(self,
                 x: NDArray[Any],
                 w: NDArray[Any],
                 dtype: DTypeLike = np.float64,
                 ):
        # accept float32 input, work in float64 internally
        x64 = np.ascontiguousarray(np.asarray(x, dtype=dtype))
        w64 = np.asarray(w, dtype=dtype)
        self.x = x64
        self.w = w64

        wx = x64 * w64.reshape(-1, 1)
        g = x64.T @ wx  # (p, p)
        try:
            self._l = np.linalg.cholesky(g)   # g = l l^T
            self._use_chol = True
        except np.linalg.LinAlgError:   # If not of full rank, use pseudoinverse
            self._use_chol = False
            self._g_pinv = np.linalg.pinv(g)

    def _solve_g(self,
                 b: NDArray[np.floating],
                 dtype: DTypeLike = np.float64,
                 ) -> NDArray[np.floating]:
        b = np.asarray(b, dtype=dtype)
        if self._use_chol:
            y = np.linalg.solve(self._l, b)

            return np.linalg.solve(self._l.T, y)

        return self._g_pinv @ b


    def weights_matrix_for_mean(self,
                     x_new: NDArray[Any],
                     avg_w: ArrayLike = None,
                     int_dtype: DTypeLike = np.float64,
                     out_dtype: DTypeLike = np.float64,
                     ) -> NDArray[Any]:
        """
        Equivalent weights (n,) that produce the average prediction over rows
        of x_new. If avg_w is given, it uses a weighted average over rows
        (avg_w >= 0, sums to 1).
        """
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            xbar = xn
        else:
            if avg_w is None:
                xbar = xn.mean(axis=0)
            else:
                a = np.asarray(avg_w, dtype=int_dtype)
                a = a / a.sum()                      # normalize
                xbar = a @ xn                        # weighted mean row
        u = self._solve_g(xbar, dtype=int_dtype)     # (p,)
        z = self.x @ u                              # (n,)

        return (self.w * z.reshape(-1, 1)).astype(out_dtype, copy=False)


    def weights_for(self,
                    x_new: NDArray[Any],
                    int_dtype: DTypeLike = np.float64,
                    out_dtype: DTypeLike = np.float64,
                    ) -> NDArray[np.floating]:
        """
        Compute weights.
        CURRENTLY NOT USED.
        """
        x_new = np.asarray(x_new, dtype=int_dtype)       # (p,)
        u = self._solve_g(x_new, dtype=int_dtype)        # (p,)
        z = self.x @ u                                  # (n,)

        return (self.w * z).astype(out_dtype, copy=False)


    def predict(self,
                y: NDArray[Any],
                x_new: NDArray[Any],
                int_dtype: DTypeLike = np.float64,
                out_dtype: DTypeLike = np.float64,
                ) -> NDArray[np.floating]:
        """
        Predict the outcome as weighted mean.
        CURRENTLY NOT USED.
        """
        y = np.asarray(y, dtype=int_dtype)
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            w_eq = self.weights_for(xn,
                                    int_dtype=int_dtype,
                                    out_dtype=int_dtype,
                                    )
            return np.array(w_eq @ y, dtype=out_dtype)
        # batch
        u = self._solve_g(xn.T, dtype=int_dtype)         # (p, m)
        z = self.x @ u                                   # (n, m)
        wz = z * self.w.reshape(-1, 1)                   # (n, m)

        return (wz.T @ y).astype(out_dtype, copy=False)  # (m,)

    def predict_mean(self,
                     y: NDArray[Any],
                     x_new: NDArray[Any],
                     avg_w: ArrayLike = None,
                     int_dtype: DTypeLike = np.float64,
                     out_dtype: DTypeLike = np.float64,
                     ) -> NDArray[Any]:
        """
        Scalar average prediction over x_new (or weighted average with avg_w).
        CURRENTLY NOT USED.
        """
        y64 = np.asarray(y, dtype=int_dtype)
        w_eq_mean = self.weights_matrix_for_mean(x_new,
                                                 avg_w=avg_w,
                                                 int_dtype=int_dtype,
                                                 out_dtype=int_dtype,
                                                 )
        return np.array(w_eq_mean @ y64, dtype=out_dtype)

    def weights_matrix(self,
                       x_new: NDArray[Any],
                       int_dtype: DTypeLike = np.float64,
                       out_dtype: DTypeLike = np.float64,
                       ) -> NDArray[np.floating]:
        """
        Compute new weights to compute average prediction.
        CURRENTLY NOT USED.
        """
        xn = np.asarray(x_new, dtype=int_dtype)
        if xn.ndim == 1:
            xn = xn.reshape(1, -1)
        u = self._solve_g(xn.T, dtype=int_dtype)        # (p, m)
        z = self.x @ u                                  # (n, m)

        return (z * self.w.reshape(-1, 1)).T.astype(out_dtype, copy=False)

    def projector_matrix(self,
                         int_dtype: DTypeLike = np.float64,
                         out_dtype: DTypeLike = np.float64,
                         ) -> NDArray[np.floating]:
        """
        returns s = W X (X' W X)^{-1}  so that w_eq(x_new) = s @ x_new.
        useful when you’ll form many equivalent-weight vectors.
        CURRENTLY NOT USED.
        """
        p = self.x.shape[1]
        inv_g = self._solve_g(np.eye(p, dtype=int_dtype))    # (p, p) via solves
        s = (self.x * self.w.reshape(-1, 1)) @ inv_g         # (n, p)

        return s.astype(out_dtype, copy=False)


def wls_equivalent_weights(x: NDArray[Any],
                           w: NDArray[np.floating],
                           x_new: ArrayLike = None,
                           int_dtype: DTypeLike = np.float64,
                           out_dtype: DTypeLike = np.float64,
                           ) -> NDArray[np.floating]:
    """
    Compute new weights.
    CURRENTLY NOT USED.
    """
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)

    return proj.weights_matrix(x_new, out_dtype=out_dtype)


def wls_predict_fast(x: NDArray[Any],
                     w: NDArray[np.floating],
                     y: NDArray[Any],
                     x_new: NDArray[Any] | None = None,
                     int_dtype: DTypeLike = np.float64,
                     out_dtype: DTypeLike = np.float64,
                     ) -> NDArray[Any]:
    """
    Do fast predictions.
    CURRENTLY NOT USED.
    """
    # avoids materializing the (m x n) weights matrix — faster/leaner for big n
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)

    return proj.predict(y, x_new, out_dtype=out_dtype)


def wls_weights_and_pred(x: NDArray[Any],
                         w: NDArray[np.floating],
                         y: NDArray[Any],
                         x_new: ArrayLike = None,
                         int_dtype: DTypeLike = np.float64,
                         out_dtype: DTypeLike = np.float64,
                         ) -> tuple[NDArray[np.floating], NDArray[Any]]:
    """
    Compute predictions and update weights.
    CURRENTLY NOT USED.
    """
    proj = WLSProjector(x, w, dtype=int_dtype)
    if x_new is None:
        x_new = np.asarray(x, dtype=int_dtype)
    w_eq = proj.weights_matrix(x_new, out_dtype=out_dtype)     # shape (m, n)
    yhat = (w_eq @ np.asarray(y, dtype=int_dtype)).astype(out_dtype, copy=False)

    return w_eq, yhat


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
        prog_dat = pd.DataFrame(prog, index=idx).loc[idx_sub].to_numpy(
            dtype=prog.dtype, copy=False
            )
    if prop is None:
        prop_dat = None
    else:
        prop_dat = pd.DataFrame(prop, index=idx).loc[idx_sub].to_numpy(
            dtype=prop.dtype, copy=False
            )
    if x_ba is None:
        x_ba_dat = None
    else:
        x_ba_dat = pd.DataFrame(x_ba, index=idx).loc[idx_sub].to_numpy(
            dtype=x_ba.dtype, copy=False
            )
    return prog_dat, prop_dat, x_ba_dat


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
    names_unordered = [name for name, type_ in var_x_type.items()
                       if type_ in [1, 2]
                       ]
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
        x_ba_o = data_df[names_ba_ordered].values.astype(np.float32)
    else:
        x_ba_o = None

    if names_ba_unordered:
        x_ba_u_df = data_df[names_ba_unordered].astype('category')
        x_ba_u_df = pd.get_dummies(
            data=x_ba_u_df,
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
    d_name, d_values, no_of_treat = get_treat_info(mcf_)
    y_tree_name = mcf_.var_cfg.y_tree_name
    y_tree_name_unc = mcf_.var_cfg.y_tree_name_unc
    obs_bigdata = mcf_.int_cfg.obs_bigdata
    boot = mcf_.cf_cfg.boot
    lc_cfg, gen_cfg, p_ba_cfg = mcf_.lc_cfg, mcf_.gen_cfg, mcf_.p_ba_cfg
    with_output = gen_cfg.with_output

    use_prop_score = mcf_.p_ba_cfg.use_prop_score
    use_prog_score = mcf_.p_ba_cfg.use_prog_score

    d_predict_np, d_in_predict_df = None, False
    y_predict_np, y_in_predict_df = None, False

    if names_unordered:
        # categorize the unordered variables
        predict_df[names_unordered] = predict_df[names_unordered].astype(
            'category')
        if use_prop_score or use_prog_score:
            train_df[names_unordered] = train_df[names_unordered].astype(
                'category')

    # Get the numpy x and y data
    if names_ordered:
        x_predict_o_df = predict_df[names_ordered]
        x_train_o_df = train_df[names_ordered]
    else:
        x_predict_o_df = x_train_o_df = None
    if names_unordered:
        x_predict_u_df = pd.get_dummies(
            data=predict_df[names_unordered],
            columns=names_unordered,
            prefix=names_unordered,
            dtype=int,
            )
        x_train_u_df = pd.get_dummies(
            data=train_df[names_unordered],
            columns=names_unordered,
            prefix=names_unordered,
            dtype=int,
            )
        # Make sure predict has exactly the same columns as train
        x_predict_u_df = x_predict_u_df.reindex(columns=x_train_u_df.columns,
                                                fill_value=0)
    else:
        x_predict_u_df = x_train_u_df = None

    if names_ordered and names_unordered:
        x_train_np = pd.concat([x_train_u_df, x_train_o_df],
                               axis=1,
                               ).values.astype(np.float32)
        x_predict_np = pd.concat([x_predict_u_df, x_predict_o_df],
                                 axis=1,
                                 ).values.astype(np.float32)
    elif names_ordered and not names_unordered:
        x_train_np = x_train_o_df.values.astype(np.float32)
        x_predict_np = x_predict_o_df.values.astype(np.float32)
    else:
        x_train_np = x_train_u_df.values.astype(np.float32)
        x_predict_np = x_predict_u_df.values.astype(np.float32)

    d_train_np = train_df[d_name].values.astype(np.float32)

    if with_output:
        d_in_predict_df = d_name[0] in predict_df.columns
        if d_in_predict_df:  # later used to compute out-of-sample fit measures
            d_predict_np = predict_df[d_name].values.astype(np.float32)

    obs_predict = x_predict_np.shape[0]

    # Use same parameters for both classifier estimations
    params_rfc = {
        "n_estimators": boot,
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

        prog_score_np = np.empty((obs_predict, no_of_treat), dtype=np.float32)

        y_name = y_tree_name_unc if lc_cfg.yes else y_tree_name
        # Only prognostic score for variable used to build forest used
        y_train_np = train_df[y_name].values.astype(np.float32)
        if with_output:
            y_in_predict_df = y_name[0] in predict_df.columns
            if y_in_predict_df:
                y_predict_np = predict_df[y_name].values.astype(np.float32)

        # Create a model, either a RFclassifier or a regression based method
        y_discrete = np.unique(y_train_np).shape[0] < 10

        if y_discrete:
            rfc_all = RandomForestClassifier(**params_rfc)
        # else:
        #     rfc_all = RandomForestRegressor(**params)

        # Train for each individual treatment
        for idx, d_treatment in enumerate(d_values):

            # Train on the treatment group
            treatments = (d_train_np == d_treatment).squeeze()
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
                    p_ba_cfg.cv_k = pa_ba_update_train(p_ba_cfg.cv_k,
                                                       x_tmp_np.shape[0],
                                                       )
                (estimator, params, txt_sel, _, transform_x, txt_mse
                 ) = best_regression(
                    x_tmp_np,  y_tmp_np.ravel(), estimator=p_ba_cfg.estimator,
                    boot=boot, seed=123456, max_workers=gen_cfg.mp_parallel,
                    test_share=0, cross_validation_k=p_ba_cfg.cv_k,
                    obs_bigdata=obs_bigdata
                    )
                if transform_x:
                    _, x_train, x_pred = scale(
                        x_tmp_np.copy(), x_predict_np.copy()
                        )
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
                treatments_p = (d_predict_np == d_treatment).squeeze()
                x_predict = x_pred if transform_x else x_predict_np

                score = obj.score(x_predict[treatments_p],
                                  y_predict_np[treatments_p],)
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
        prop_all = classif.predict_proba(x_predict_np).astype(np.float32)
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
