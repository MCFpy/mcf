"""
Created on Tue Nov 25 06:09:30 2025.

@author: MLechner
# -*- coding: utf-8 -*-

Contains functions that set default values and check user inputs.
"""
from __future__ import annotations

from numbers import Real
from pathlib import Path
from platform import system
from typing import Any

import numpy as np

try:
    import torch  # type: ignore[import]
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]

try:
    import ray  # type: ignore[import]
except (ImportError, OSError):
    ray = None  # type: ignore[assignment]

from mcf.mcf_general_sys import get_fig_path


def determine_cv_k(obs):
    """Determine value of cross-validation based on number of observations."""
    match obs:
        case _ if obs < 100_000: return 5
        case _ if obs < 250_000: return 4
        case _ if obs < 500_000: return 3
        case _:                  return 2


def gen_tv_valid_cv_k(cv_k):
    """Check if value for CV for treatment versions is valid."""
    match cv_k:
        case None:
            return None    # Will be set later based on training data
        case k if isinstance(k, int) and k > 1:
            return k
        case _:
            raise ValueError('Cross-validation parameter for treatment '
                             'versions must be None or an integer > 1.'
                              f'The specified value {cv_k} is not allowed.'
                              )


def gen_tv_valid_estimators(estimator):
    """Check if estimators for treatment versions are valid."""
    ALLOWED_ESTIMATORS = ('ridge', 'ols',)     # pylint: disable=C0103
    match estimator:
        case None:  # Default
            return 'ridge'
        case est if est in ALLOWED_ESTIMATORS:
            return estimator
        case _:
            raise ValueError('Estimator for Treatment Variants must be one of the '
                             f'following: {" ".join(ALLOWED_ESTIMATORS)}. The specified '
                             f'estimator {estimator} is not allowed.'
                             )


def gen_tv_valid_specification(specification):
    """Check if regressor specifications for treatment versions are valid."""
    ALLOWED_SPECS = ('interacted', 'separable',)    # pylint: disable=C0103
    match specification:
        case None:  # Default
            return 'interacted'
        case est if est in ALLOWED_SPECS:
            return specification
        case _:
            raise ValueError('Specification for Treatment Variants must be one of the following: '
                             f'{" ".join(ALLOWED_SPECS)}. {specification} is not allowed.'
                             )


def p_ba_adjust_method(adj_method: Any, yes=True) -> str:
    """Check if valid or default is used."""
    if not yes:
        return 'w_obs'

    VALID_METHODS = ('zeros', 'train_obs', 'weighted_train_obs', 'pred_obs', # pylint: disable=C0103
                     )

    match adj_method:
        case None:   # Default
            return 'w_obs'
        case 'zeros':
            return 'zeros'
        case 'train_obs':
            return 'obs'
        case 'weighted_train_obs':
            return 'w_obs'
        case 'pred_obs':
            raise NotImplementedError('pred_obs not yet available for bias adjustment.')
        case _:
            raise ValueError('Invalid inference method for bias adjustment specified: '
                             f'{adj_method}. '
                             f'\nValid methods are: {" ".join(VALID_METHODS)}'
                             )


def p_cb_normalize(choice_based_probs: list[Real] | tuple[Real, ...],
                   no_of_treat: int
                   ) -> list[Real]:
    """Normalize the choice based probabilities."""
    pcb = np.array(choice_based_probs)
    pcb = pcb / np.sum(pcb) * no_of_treat

    return pcb.tolist()


def iv_agg_method_fct(iv_aggregation_method: Any) -> tuple:
    """Check if a valid method is used."""
    VALID_IV = ('local', 'global')     # pylint: disable=C0103
    DEFAULT_IV = ('local', 'global')   # pylint: disable=C0103
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


def inconsistencies(self) -> None:
    """Stop execution if user provided specification is not allowed."""
    if self.int_cfg.low_memory_predict and self.p_cfg.qiate:
        raise NotImplementedError('When _int_low_memory_predict is used, QIATEs cannot be '
                                  'computed. Either set _int_low_memory_predict or p_qiate '
                                  'to False.'
                                   )
    if self.int_cfg.low_memory_predict and self.p_cfg.iate_m_ate:
        raise NotImplementedError('When _int_low_memory_predict is used, the difference between '
                                  'IATE(x) and the ATE cannot be computed. Either set '
                                  '_int_low_memory_predict or iate_m_ate to False.'
                                  )
    if self.gen_cfg.d_type == 'continuous' and self.p_cfg.qiate:
        raise NotImplementedError('QIATEs are not yet implemented for continuous treatments.')

    if self.gen_cfg.d_type == 'continuous' and self.p_cfg.choice_based_sampling:
        raise NotImplementedError('No choice based sample with continuous treatments.')

    if self.gen_cfg.d_type == 'continuous' and self.p_ba_yes:
        raise NotImplementedError('No bias adjustment with continuous treatments.')

    if self.p_ba_cfg.yes and self.p_cfg.qiate:
        raise NotImplementedError('No bias adjustment is incompatible with estimating QIATEs.')

    if self.p_ba_cfg.yes and self.p_cfg.cluster_std:
        raise NotImplementedError('Bias adjustment when computing clustered standard errors '
                                  'is not yet implemented.'
                                  )
    if self.p_ba_cfg.yes and self.gen_cfg.weighted:
        raise NotImplementedError('Bias adjustment for weighetd estimation is not yet implemented.')

    if torch is None and self.int_cfg.cuda:
        raise RuntimeError('Cuda is used, but importing torch failed.')

    if ray is None and (self.int_cfg.mp_use_old_ray or self.int_cfg.mp_backend == 'ray'):
        txt = 'Ray is used, but importing ray failed. '
        if self.int_cfg.mp_use_old_ray:
            txt += 'Set _int_mp_use_old_ray = False and set _int_mp_backend = "joblib".'
        else:
            txt += 'Use joblib instead (_int_mp_backend = "joblib").'
        raise RuntimeError(txt)

    if (not self.int_cfg.mp_use_old_ray and system() == 'Windows'
        and self.int_cfg.mp_backend == 'joblib' and self.gen_cfg.mp_parallel > 60
            ):
        raise RuntimeError('Joblib on Windows cannot handle more than 60 parallel processes. Use '
                           'Ray instead or reduce gen_mp_parallel to no more than 60 '
                           '(the default setting does not account for this hardware feature).'
                           )


def inconsistencies_train(self, *, iv: bool=False) -> None:
    """Stop execution if user provided specification is not allowed for IV."""
    if iv and self.int_cfg.low_memory_predict:
        raise NotImplementedError('IV cannot be used in combination with _int_low_memory_predict.'
                                  'Set _int_low_memory_predict to False.'
                                  )


def inconsistencies_sens(self) -> None:
    """Stop execution if user provided specification is not allowed for sensitivity analysis."""
    if self.fs_cfg.yes:
        raise NotImplementedError('Sensitivity analysis does not (yet) run with feature selection')
