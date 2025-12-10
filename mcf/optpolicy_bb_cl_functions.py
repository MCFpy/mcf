"""
Provide functions for allocations based on classifers applied to Black-Box.

Created on Sun Jul 16 14:03:58 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from typing import TYPE_CHECKING

from numpy import column_stack
from pandas import DataFrame

from mcf.mcf_print_stats_functions import print_mcf
from mcf import mcf_estimation_generic_functions as mcf_est_g
from mcf import optpolicy_data_functions as op_data


if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


def bps_classifier_allocation(optp_: 'OptimalPolicy',
                              data_df: DataFrame,
                              allocation_df: DataFrame,
                              seed: int = 234356
                              ) -> tuple[DataFrame, dict, str]:
    """Compute various Black-Box allocations and return to main programme."""
    alloc_name_bb = allocation_df.columns

    x_dat_scaled_np, x_name, scaler = op_data.prepare_data_for_classifiers(
        data_df, optp_.var_cfg, scaler=None)
    txt_all = ''
    alloc_all, scikit_obj_all, alloc_name_all, results_dic = [], [], [], {}
    for alloc_name in alloc_name_bb:
        d_np = allocation_df[alloc_name].to_numpy()
        method, params, lables, score = mcf_est_g.best_classifier(
            x_dat_scaled_np, d_np, seed=seed, boot=1000, max_workers=None,
            test_share=0.25)
        txt = mcf_est_g.printable_output(lables, score)
        scikit_obj = mcf_est_g.classif_instance(method, params)
        scikit_obj.fit(x_dat_scaled_np, d_np)
        alloc = scikit_obj.predict(x_dat_scaled_np)

        alloc_all.append(alloc)
        scikit_obj_all.append((scaler, scikit_obj))
        alloc_name_all.append('bps_classif_' + alloc_name)
        txt_all += f'\n{alloc_name+":":40s} ' + txt

    # alloc_all_np = concatenate([alloc.reshape(-1, 1) for alloc in alloc_all],
    #                            axis=1)
    alloc_all_np = column_stack(alloc_all)

    allocations_df = DataFrame(data=alloc_all_np, columns=alloc_name_all)
    results_dic['scikit_obj_dict'] = dict(zip(alloc_name_all, scikit_obj_all))
    optp_.bps_class_dict['scikit_obj_dict'] = results_dic['scikit_obj_dict']
    optp_.bps_class_dict['x_name_train'] = x_name

    if optp_.gen_cfg.with_output:
        print_mcf(optp_.gen_cfg, txt_all, summary=True)

    return allocations_df, results_dic, txt_all


def bps_class_prediction_only(optp_: 'OptimalPolicy',
                              data_df: DataFrame
                              ) -> tuple[DataFrame, str]:
    """Predicts allocation based on new data for the classifiers."""
    allocs_dict = optp_.bps_class_dict['scikit_obj_dict']
    scaler = next(iter(allocs_dict.values()))[0]  # Getting scaler from for dict
    x_dat_scaled_np, _, _ = op_data.prepare_data_for_classifiers(
        data_df, optp_.var_cfg, scaler=scaler,
        x_name_train=optp_.bps_class_dict['x_name_train'])
    alloc_all, alloc_name_all = [], []
    for key, (_, scikit_obj) in allocs_dict.items():
        # scikit_obj = value[1]
        alloc_np = scikit_obj.predict(x_dat_scaled_np)
        alloc_all.append(alloc_np)
        alloc_name_all.append(key)

    alloc_all_np = column_stack(alloc_all)
    allocations_df = DataFrame(data=alloc_all_np, columns=alloc_name_all)
    allocation_txt = ''

    return allocations_df, allocation_txt
