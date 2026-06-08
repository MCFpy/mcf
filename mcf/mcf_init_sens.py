"""
Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for initialising the parameters of the programme.
@author: MLechner
-*- coding: utf-8 -*-
"""
from dataclasses import dataclass
from pandas import DataFrame

from mcf.mcf_init_update_helper import isinstance_scalar


@dataclass(slots=True, kw_only=True)
class SensCfg:
    """Parameters of sensitivity (post-estimation) analysis."""

    cbgate: bool = False
    bgate: bool = False
    gate: bool = False
    iate: bool = False
    iate_se: bool = False
    reference_population: int | float | None = None
    cv_k: int = 5
    replications: int = 2
    scenarios: tuple[str, ...] = ('basic',)

    @classmethod
    def from_args(cls,
                  p_cfg,  # transient; not stored
                  *,
                  bgate: bool | None = None,
                  cbgate: bool | None = None,
                  cv_k: int | float | None = None,
                  gate: bool | None = None,
                  iate: bool | None = None,
                  iate_se: bool | None = None,
                  replications: int | float | None = 2,
                  scenarios: str | tuple[str, ...] | list[str] | None = None,
                  reference_population: int | float | None = None,
                  iate_df: DataFrame | None = None,
                  ) -> 'SensCfg':
        """Get input and normalize parameters."""
        # type checks (preserve original semantics)
        if cv_k is not None and not isinstance_scalar(cv_k):
            raise TypeError('Number of folds for cross-validation must be integer, float or None')
        if replications is not None and not isinstance_scalar(replications):
            raise TypeError('Number of replication must be integer, float or None')
        if scenarios is not None and not isinstance(scenarios, (list, tuple, str)):
            raise TypeError('Names of scenarios must be string or None')
        if cbgate is not None and not isinstance(cbgate, bool):
            raise TypeError('cbgate must be boolean or None')
        if bgate is not None and not isinstance(bgate, bool):
            raise TypeError('bgate must be boolean or None')
        if gate is not None and not isinstance(gate, bool):
            raise TypeError('gate must be boolean or None')
        if iate is not None and not isinstance(iate, bool):
            raise TypeError('iate must be boolean or None')
        if iate_se is not None and not isinstance(iate_se, bool):
            raise TypeError('iate_se must be boolean or None')
        if reference_population is not None and not isinstance_scalar(reference_population):
            raise TypeError('reference_population must be a Number or None')

        cbgate_b = cbgate is True
        bgate_b = bgate is True
        gate_b = (gate is True) or cbgate_b or bgate_b
        iate_b = True if iate_df is not None else (iate is True)
        iate_se_b = iate_se is True

        if cbgate_b and not getattr(p_cfg, 'cbgate', False):
            raise ValueError(
                'p_cbgate must be set to True if sens_cbgate is True'
                )
        if gate_b and not getattr(p_cfg, 'bgate', False):
            raise ValueError('p_bgate must be set to True if sens_bgate is True')

        refpop_v = None if reference_population is None else reference_population
        cv_k_v = 5 if cv_k is None or cv_k < 0.5 else round(cv_k)
        replications_v = 2 if replications is None or replications < 0.5 else round(replications)
        match scenarios:
            case None:
                scens_v = ('basic',)
            case str() as s:
                scens_v = (s,)
            case _:
                raise NotImplementedError(f'Sensitivity scenario {scenarios!r} not implemented')
        eligible_scenarios = ('basic',)
        wrong_scenarios = [scen for scen in scens_v
                           if scen not in eligible_scenarios
                           ]
        if wrong_scenarios:
            raise ValueError(f'{wrong_scenarios} {"are" if len(wrong_scenarios) > 1 else "is"} '
                             'ineligable for sensitivity analysis.'
                             )
        return cls(cbgate=cbgate_b,
                   bgate=bgate_b,
                   gate=gate_b,
                   iate=iate_b,
                   iate_se=iate_se_b,
                   reference_population=refpop_v,
                   cv_k=cv_k_v,
                   replications=replications_v,
                   scenarios=scens_v,
                   )
