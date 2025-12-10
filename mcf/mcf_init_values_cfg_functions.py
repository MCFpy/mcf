"""
Created on Tue Nov 25 06:09:30 2025.

@author: MLechner
# -*- coding: utf-8 -*-

Contains functions that set default values and check user inputs.
"""
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from mcf.mcf_general_sys import get_fig_path


def p_ba_adjust_method(adj_method: Any) -> str:
    """Check if valid or default is used."""
    VALID_METHODS = ('zeros', 'observables', 'weighted_observables')

    match adj_method:
        case None:   # Default
            adj_method_n = 'w_obs'
        case 'zeros':
            adj_method_n = 'zeros'
        case 'observables':
            adj_method_n = 'obs'
        case 'weighted_observables':
            adj_method_n = 'w_obs'
        case _:
            raise ValueError(
                'Invalid inference method for bias adjustment specified: '
                f'{adj_method}. '
                f'\nValid methods are: {" ".join(VALID_METHODS)}'
                )
    return adj_method_n


def p_cb_normalize(choice_based_probs: list[Real] | tuple[Real, ...],
                   no_of_treat: int
                   ) -> list[Real]:
    """Normalize the choice based probabilities."""
    pcb = np.array(choice_based_probs)
    pcb = pcb / np.sum(pcb) * no_of_treat

    return pcb.tolist()


def iv_agg_method_fct(iv_aggregation_method):
    """Check if a valid method is used."""
    VALID_IV = ('local', 'global')
    DEFAULT_IV = ('local', 'global')
    if isinstance(iv_aggregation_method, (str, int, float,)):
        if iv_aggregation_method in VALID_IV:
            ivm_v = (iv_aggregation_method,)
        else:
            raise ValueError(
                f'{iv_aggregation_method} is not a valid method for IV '
                f'estimation. Valid methods are {" ".join(VALID_IV)}.'
                )
    elif isinstance(iv_aggregation_method, (list, tuple)):
        if not set(iv_aggregation_method) == set(VALID_IV):
            raise ValueError(
                f'{iv_aggregation_method} is not a valid method for IV '
                f'estimation. Valid methods are {" ".join(VALID_IV)}.'
                )
        ivm_v = tuple(iv_aggregation_method)
    else:
        ivm_v = DEFAULT_IV

    return ivm_v

def get_directories_for_output(outpath: Path,
                               with_output: bool,
                               qiate: bool = False,
                               cbgate: bool = None,
                               bgate: bool = None,
                               ) -> dict[str, Path]:
    """Define paths needed to save figures and data."""
    paths_v = get_fig_path({}, outpath, 'ate_iate', with_output)
    paths_v = get_fig_path(paths_v, outpath, 'gate', with_output)
    if qiate:
        paths_v = get_fig_path(paths_v, outpath, 'qiate', with_output)
    if cbgate:
        paths_v = get_fig_path(paths_v, outpath, 'cbgate', with_output)
    if bgate:
        paths_v = get_fig_path(paths_v, outpath, 'bgate', with_output)

    return paths_v
