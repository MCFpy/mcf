"""Created on Thu Oct 31 10:08:06 2024.

Contains IV specific functions.

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np

from mcf import mcf_estimation_functions as mcf_est
from mcf import mcf_iate_functions as mcf_iate
from mcf import mcf_weight_functions as mcf_w


# Compute weights of reduced form & 1st stage
def get_weights_late(self, mcf_1st, mcf_redf, forest_1st_dic, forest_redf_dic,
                     iate_1st_dic, iate_eff_1st_dic, data_df, round_):
    """Get IV adjusted weights."""
    weights_1st_dic = mcf_w.get_weights_mp(
        mcf_1st, data_df, forest_1st_dic, round_ == 'regular')
    weights_redf_dic = mcf_w.get_weights_mp(
        mcf_redf, data_df, forest_redf_dic, round_ == 'regular')

    # Scale reduced form weights by 1st stage effects
    if round_ == 'regular':
        weights_dic = get_iv_weights(self, weights_redf_dic, iate_1st_dic)
    else:
        weights_dic = get_iv_weights(self, weights_redf_dic, iate_eff_1st_dic)

    return weights_1st_dic, weights_redf_dic, weights_dic


def get_iv_weights(self, weights_redf_dic, iate_1st_dic):
    """Compute IV weights by scaling reduced form by 1st stage."""
    weights_dic = deepcopy(weights_redf_dic)
    iate_1st = np.squeeze(iate_1st_dic['y_pot'], axis=-1)  # Reduce to 2D
    iate_d_z = iate_1st[:, 1] - iate_1st[:, 0]

    # Scale weights by 1st effect
    if self.int_dict['weight_as_sparse']:
        iate_d_z_inv = 1 / iate_d_z.reshape(-1, 1)
        weights = [weight.multiply(iate_d_z_inv).tocsr()
                   for weight in weights_redf_dic['weights']]
    else:
        weights = [weight / iate_d_z for weight in weights_redf_dic['weights']]

    weights_dic['weights'] = weights
    return weights_dic


def get_1st_stage_iate(self, mcf_1st, data_df, only_one_fold_one_round):
    """Compute first stage IATEs."""
    iate_1st_dic = iate_eff_1st_dic = None
    for fold in range(self.cf_dict['folds']):
        for round_ in self.cf_dict['est_rounds']:
            # Get relevant forests
            if only_one_fold_one_round:
                forest_1st_dic = mcf_1st.forest[fold][0]
            else:
                forest_1st_dic = deepcopy(
                    mcf_1st.forest[fold][0 if round_ == 'regular' else 1])
            if self.int_dict['with_output'] and self.int_dict['verbose']:
                print(f'\n\nWeight maxtrix (1st stage) {fold+1} /',
                      f'{self.cf_dict["folds"]} forests, {round_}')
            weights_1st_dic = mcf_w.get_weights_mp(
                mcf_1st, data_df, forest_1st_dic, round_ == 'regular')

            (y_pot_f, _, _, _, txt_w_f) = mcf_iate.iate_est_mp(
                 self, weights_1st_dic, None, round_ == 'regular',
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
