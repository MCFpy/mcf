"""Created on Thu Oct 31 10:08:06 2024.

Contains IV specific functions.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_gate_functions as mcf_gate
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_weight_functions as mcf_w

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


# Compute weights of reduced form & 1st stage
def get_weights_iv_local(mcf_: 'ModifiedCausalForest',
                         mcf_1st: 'ModifiedCausalForest',
                         mcf_redf: 'ModifiedCausalForest',
                         forest_1st_dic: dict,
                         forest_redf_dic: dict,
                         iate_1st_dic: dict,
                         iate_eff_1st_dic: dict,
                         data_df: DataFrame,
                         round_: str,
                         local_effects: bool = True,
                         no_1st_weights: bool = False,
                         ) -> tuple[dict, dict, dict]:
    """Get IV adjusted weights."""
    if no_1st_weights:
        weights_1st_dic = None
    else:
        weights_1st_dic = mcf_w.get_weights_mp(
            mcf_1st, data_df, forest_1st_dic, round_ == 'regular'
            )
    weights_redf_dic = mcf_w.get_weights_mp(
        mcf_redf, data_df, forest_redf_dic, round_ == 'regular'
        )

    # Scale reduced form weights by 1st stage effects
    if local_effects:
        effect_dic = iate_1st_dic if round_ == 'regular' else iate_eff_1st_dic

        iate_1st = np.squeeze(effect_dic['y_pot'], axis=-1)  # Reduce to 2D
        iate_d_z = iate_1st[:, 1] - iate_1st[:, 0]
        weights_local_dic = scale_weights_1st_stage(
            iate_d_z,
            weights_redf_dic,
            mcf_.int_cfg.weight_as_sparse,
            )
    else:
        weights_local_dic = {}

    return weights_1st_dic, weights_redf_dic, weights_local_dic


def scale_weights_1st_stage(effect_1st: NDArray[Any],
                            weights_redf_dic: dict,
                            weight_as_sparse: bool
                            ) -> dict:
    """Obtain weights for final IV estimation (local and global)."""
    if weight_as_sparse:
        # Note: Rows of weights add to one here (before rescaling by 1st stage)
        effect_1st_inv = 1 / effect_1st.reshape(-1, 1)
        weights = [weight.multiply(effect_1st_inv).tocsr()
                   for weight in weights_redf_dic['weights']
                   ]
    else:
        w_redf = weights_redf_dic['weights']
        ate_estimation = len(effect_1st) < len(w_redf)
        weights = [[[None] * 2] * len(w_redf[0])] * len(w_redf)
        for obs_i, weight_i in enumerate(w_redf):
            for treat_j, w_treat_j in enumerate(weight_i):
                weights[obs_i][treat_j][0] = w_treat_j[0].copy()  # Indices
                if ate_estimation:
                    weights[obs_i][treat_j][1] = w_treat_j[1] / effect_1st
                else:
                    weights[obs_i][treat_j][1] = (w_treat_j[1]
                                                  / effect_1st[obs_i]
                                                  )
    weights_scaled_dic = deepcopy(weights_redf_dic)
    weights_scaled_dic['weights'] = weights

    return weights_scaled_dic


def iate_1st_stage_all_folds_rounds(mcf_: 'ModifiedCausalForest',
                                    mcf_1st: 'ModifiedCausalForest',
                                    data_df: DataFrame,
                                    only_one_fold_one_round: bool
                                    ) -> tuple[dict, dict]:
    """Compute first stage IATEs across all folds and rounds."""
    iate_1st_dic = iate_eff_1st_dic = None
    for fold in range(mcf_.cf_cfg.folds):
        for round_ in mcf_.cf_cfg.est_rounds:
            # Get relevant forests
            if only_one_fold_one_round:
                forest_1st_dic = mcf_1st.forest[fold][0]
            else:
                forest_1st_dic = deepcopy(
                    mcf_1st.forest[fold][0 if round_ == 'regular' else 1])
            if mcf_.gen_cfg.with_output and mcf_.gen_cfg.verbose:
                print(f'\n\nWeight maxtrix (1st stage) {fold+1} /',
                      f'{mcf_.cf_cfg.folds} forests, {round_}')
            weights_1st_dic = mcf_w.get_weights_mp(
                mcf_1st, data_df, forest_1st_dic, round_ == 'regular')

            (y_pot_f, _, _, _, txt_w_f) = mcf_iate.iate_est_mp(
                 mcf_, weights_1st_dic, None, round_ == 'regular',
                 iv_scaling=True)
            if round_ == 'regular':
                y_pot_iate_1st_f = y_pot_f.copy()
                iate_1st_dic = mcf_est.aggregate_pots(
                    mcf_1st, y_pot_iate_1st_f, None, txt_w_f,
                    iate_1st_dic, fold, title='IATE (1st stage')
            else:
                iate_eff_1st_dic = mcf_est.aggregate_pots(
                    mcf_1st, (y_pot_iate_1st_f + y_pot_f) / 2, None, txt_w_f,
                    iate_eff_1st_dic, fold, title='IATE eff (1st stage)')

    return iate_1st_dic, iate_eff_1st_dic


def bala_1st_redf(instances: tuple['ModifiedCausalForest',
                                   'ModifiedCausalForest',
                                   'ModifiedCausalForest'],
                  weights: tuple[dict, dict, dict],
                  bala_1st_dic: dict,
                  bala_redf_dic: dict,
                  data_df: DataFrame,
                  fold: int,
                  ) -> tuple[dict, dict]:
    """Perform balancing tests for 1st stage and reduced form."""
    _, mcf_1st, mcf_redf = instances
    _, weights_1st_dic, weights_redf_dic = weights
    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_1st, data_df, weights_1st_dic, balancing_test=True,
        )
    # Aggregate Balancing results over folds
    bala_1st_dic = mcf_est.aggregate_pots(
        mcf_1st, y_pot_f, y_pot_var_f, txt_w_f, bala_1st_dic,
        fold, title='Reduced form balancing check: ')

    (_, y_pot_f, y_pot_var_f, txt_w_f) = mcf_ate.ate_est(
        mcf_redf, data_df, weights_redf_dic, balancing_test=True,
        )
    # Aggregate Balancing results over folds
    bala_redf_dic = mcf_est.aggregate_pots(
        mcf_redf, y_pot_f, y_pot_var_f, txt_w_f, bala_redf_dic,
        fold, title='Reduced form balancing check: ')

    return bala_1st_dic, bala_redf_dic


def ate_iv(instances: tuple['ModifiedCausalForest', 'ModifiedCausalForest',
                            'ModifiedCausalForest'],
           weights: tuple[dict, dict, dict],
           late_global_dic: dict,
           late_local_dic: dict,
           ate_1st_dic: dict,
           ate_redf_dic: dict,
           data_df: DataFrame,
           fold: int,
           global_effects: bool = True,
           local_effects: bool = True,
           ) -> tuple[NDArray[Any], NDArray[Any], dict, dict,
                      NDArray[Any], dict,
                      NDArray[Any], dict
                      ]:
    """Compute LATE, reduced form and first stage."""
    # Unpack inputs
    mcf_, mcf_1st, mcf_redf = instances
    weights_local_dic, weights_1st_dic, weights_redf_dic = weights

    # 1st stage for current fold
    (w_ate_1st, y_pot_1st_f, y_pot_var_1st_f, txt_w_1st_f
     ) = mcf_ate.ate_est(mcf_1st, data_df, weights_1st_dic, iv=False,
                         )
    # Reduced form for current fold
    (w_ate_redf, y_pot_redf_f, y_pot_var_redf_f, txt_w_redf_f
     ) = mcf_ate.ate_est(mcf_redf, data_df, weights_redf_dic, iv=False,
                         )
    # Aggregate ATEs over folds
    ate_1st_dic = mcf_est.aggregate_pots(
        mcf_1st, y_pot_1st_f, y_pot_var_1st_f, txt_w_1st_f, ate_1st_dic, fold,
        title='ATE (1st stage)'
        )
    ate_redf_dic = mcf_est.aggregate_pots(
        mcf_redf, y_pot_redf_f, y_pot_var_redf_f, txt_w_redf_f, ate_redf_dic,
        fold, title='ATE (reduced form)'
        )

    if global_effects:
        ate_1st = y_pot_1st_f[:, 1] - y_pot_1st_f[:, 0]
        weights_global_dic = scale_weights_1st_stage(
            ate_1st,
            weights_redf_dic,
            mcf_.int_cfg.weight_as_sparse,
            )
        (w_late_global, y_pot_f_global, y_pot_var_f_global, txt_w_f_global
         ) = mcf_ate.ate_est(mcf_, data_df, weights_global_dic, iv=True,
                             )
        # Aggregate ATEs over folds
        late_global_dic = mcf_est.aggregate_pots(
            mcf_,
            y_pot_f_global, y_pot_var_f_global,
            txt_w_f_global, late_global_dic, fold,
            title='LATE(global)'
            )
    else:
        w_late_global = late_global_dic = None

    if local_effects:
        (w_late_local, y_pot_f_local, y_pot_var_f_local, txt_w_f_local
         ) = mcf_ate.ate_est(mcf_, data_df, weights_local_dic, iv=True,
                             )
        # Aggregate ATEs over folds
        late_local_dic = mcf_est.aggregate_pots(
            mcf_,
            y_pot_f_local, y_pot_var_f_local,
            txt_w_f_local, late_local_dic, fold,
            title='LATE(local)'
            )
    else:
        w_late_local = late_local_dic = None

    return (w_late_global, w_late_local,
            late_global_dic, late_local_dic,
            w_ate_1st, ate_1st_dic,
            w_ate_redf, ate_redf_dic
            )


def bgate_iv(mcf_: 'ModifiedCausalForest',
             data_df: DataFrame,
             weights: tuple[dict, dict, dict],
             # w_late_global: npt.NDArray[Any],
             w_late_local: NDArray[Any],
             lbgate_global_dic: dict,
             lbgate_local_dic: dict,
             lbgate_m_late_global_dic: dict,
             lbgate_m_late_local_dic: dict,
             fold: int,
             iv_tuple: tuple = None,
             gate_type: str = 'BGATE',
             title: str = 'LBGATE',
             global_effects: bool = False,
             local_effects: bool = True,
             ) -> tuple[dict, dict, dict, dict, dict, dict, str]:
    """Compute BGATE with instrumental variables."""
    # weights_local_dic, weights_1st_dic, weights_redf_dic = weights
    weights_local_dic, _, _ = weights
    txt_b = ''
    if global_effects:
        # weights_global_dic = None
        # (y_pot_lbgate_f, y_pot_var_lbgate_f, y_pot_mlate_lbgate_f,
        #  y_pot_mlate_var_lbgate_f, lbgate_est_global_dic, txt_w_f, txt_b,
        #  ) = bgate_est(mcf_, data_df, weights_global_dic, w_late_global, None,
        #                gate_type=gate_type, iv_tuple=iv_tuple,
        #                # iv_global_effects=True
        #                )
        # lbgate_global_dic = mcf_est.aggregate_pots(
        #     mcf_, y_pot_lbgate_f, y_pot_var_lbgate_f, txt_w_f,
        #     lbgate_global_dic, fold, pot_is_list=True, title=title+' (local)'
        #     )
        # if y_pot_mlate_lbgate_f is not None:
        #     lbgate_m_late_global_dic = mcf_est.aggregate_pots(
        #         mcf_, y_pot_mlate_lbgate_f, y_pot_mlate_var_lbgate_f, txt_w_f,
        #         lbgate_m_late_global_dic, fold, pot_is_list=True,
        #         title=title + ' minus LATE (local)'
        #         )
        raise NotImplementedError('LBGATEs, LCGATEs, and LGATEs are not '
                                  'implemented for global compliers.')
    lbgate_est_global_dic = None

    if local_effects:
        (y_pot_lbgate_f, y_pot_var_lbgate_f, y_pot_mlate_lbgate_f,
         y_pot_mlate_var_lbgate_f, lbgate_est_local_dic, txt_w_f, txt_b,
         ) = mcf_gate.bgate_est(mcf_, data_df, weights_local_dic, w_late_local,
                                None, gate_type=gate_type, iv_tuple=iv_tuple,
                                # iv_global_effects=False
                                )
        lbgate_local_dic = mcf_est.aggregate_pots(
            mcf_, y_pot_lbgate_f, y_pot_var_lbgate_f, txt_w_f,
            lbgate_local_dic, fold, pot_is_list=True, title=title+' (local)'
            )
        if y_pot_mlate_lbgate_f is not None:
            lbgate_m_late_local_dic = mcf_est.aggregate_pots(
                mcf_, y_pot_mlate_lbgate_f, y_pot_mlate_var_lbgate_f, txt_w_f,
                lbgate_m_late_local_dic, fold, pot_is_list=True,
                title=title + ' minus LATE (local)'
                )
    else:
        lbgate_est_local_dic = None

    return (lbgate_global_dic, lbgate_local_dic,
            lbgate_m_late_global_dic, lbgate_m_late_local_dic,
            lbgate_est_global_dic, lbgate_est_local_dic,
            txt_b
            )


def gate_iv(mcf_: 'ModifiedCausalForest',
            data_df: DataFrame,
            weights: tuple[dict, dict, dict],
            # w_late_global: npt.NDArray[Any],
            w_late_local: NDArray[Any],
            lgate_global_dic: dict,
            lgate_local_dic: dict,
            lgate_m_late_global_dic: dict,
            lgate_m_late_local_dic: dict,
            fold: int,
            title: str = 'LGATE',
            global_effects: bool = False,
            local_effects: bool = True,
            ) -> tuple[dict, dict, dict, dict, dict, dict,]:
    """Compute GATE with instrumental variables."""
    # weights_local_dic, weights_1st_dic, weights_redf_dic = weights
    weights_local_dic, _, _ = weights
    if global_effects:
        # weights_global_dic = None
        # (y_pot_lgate_f, y_pot_var_lgate_f, y_pot_mlate_lgate_f,
        #  y_pot_mlate_var_lgate_f, lgate_est_global_dic, txt_w_f, txt_b,
        #  ) = mcf_gate.gate_est(mcf_, data_df, weights_global_dic,
        #               w_late_global, None,
        #               gate_type=gate_type, iv_tuple=iv_tuple,
        #               # iv_global_effects=True
        #               )
        # lgate_global_dic = mcf_est.aggregate_pots(
        #     mcf_, y_pot_lgate_f, y_pot_var_lgate_f, txt_w_f,
        #     lgate_global_dic, fold, pot_is_list=True, title=title+' (local)'
        #     )
        # if y_pot_mlate_lbgate_f is not None:
        #     lgate_m_late_global_dic = mcf_est.aggregate_pots(
        #         mcf_, y_pot_mlate_lgate_f, y_pot_mlate_var_lgate_f, txt_w_f,
        #         lgate_m_late_global_dic, fold, pot_is_list=True,
        #         title=title + ' minus LATE (local)'
        #         )
        raise NotImplementedError('LBGATEs, LCGATEs, and LGATEs are not '
                                  'implemented for global compliers.')
    lgate_est_global_dic = None

    if local_effects:
        (y_pot_lgate_f, y_pot_var_lgate_f, y_pot_mlate_lgate_f,
         y_pot_mlate_var_lgate_f, lgate_est_local_dic, txt_w_f
         ) = mcf_gate.gate_est(mcf_, data_df, weights_local_dic, w_late_local,
                               iv=True
                               )

        lgate_local_dic = mcf_est.aggregate_pots(
            mcf_, y_pot_lgate_f, y_pot_var_lgate_f, txt_w_f,
            lgate_local_dic, fold, pot_is_list=True, title=title+' (local)'
            )
        if y_pot_mlate_lgate_f is not None:
            lgate_m_late_local_dic = mcf_est.aggregate_pots(
                mcf_, y_pot_mlate_lgate_f, y_pot_mlate_var_lgate_f, txt_w_f,
                lgate_m_late_local_dic, fold, pot_is_list=True,
                title=title + ' minus LATE (local)'
                )
    else:
        lgate_est_local_dic = None

    return (lgate_global_dic, lgate_local_dic,
            lgate_m_late_global_dic, lgate_m_late_local_dic,
            lgate_est_global_dic, lgate_est_local_dic
            )


def iate_iv(mcf_: 'ModifiedCausalForest',
            weights_dic: dict,
            iate_dic: dict,
            iate_m_ate_dic: dict,
            iate_eff_dic: dict,
            w_ate: NDArray[Any],
            y_pot_iate_f: list | NDArray[Any],
            round_: bool,
            fold: int,
            iv: bool = True,
            title: str = 'LIATE',
            ) -> tuple[dict, dict, dict, list | NDArray[Any]]:
    """Compute IATEs."""
    (y_pot_f, y_pot_var_f, y_pot_m_ate_f, y_pot_m_ate_var_f,
     txt_w_f) = mcf_iate.iate_est_mp(
         mcf_, weights_dic, w_ate, round_ == 'regular', iv=True)
    if round_ == 'regular':
        y_pot_iate_f = y_pot_f.copy()
        y_pot_varf = (None if y_pot_var_f is None else y_pot_var_f.copy())
        iate_dic = mcf_est.aggregate_pots(
            mcf_, y_pot_iate_f, y_pot_varf, txt_w_f, iate_dic,
            fold, title=title)
        if y_pot_m_ate_f is not None:
            titel2 = title + ' minus LATE' if iv else title + ' minus ATE'
            iate_m_ate_dic = mcf_est.aggregate_pots(
                mcf_, y_pot_m_ate_f, y_pot_m_ate_var_f, txt_w_f, iate_m_ate_dic,
                fold, title=titel2)
    else:
        y_pot_eff = (y_pot_iate_f + y_pot_f) / 2
        iate_eff_dic = mcf_est.aggregate_pots(
            mcf_, y_pot_eff, None, txt_w_f, iate_eff_dic, fold,
            title=title + ' eff'
            )
    return iate_dic, iate_m_ate_dic, iate_eff_dic, y_pot_iate_f
