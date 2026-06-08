"""
Created on Wed Feb  4 08:40:24 2026.

# -*- coding: utf-8 -*-

@author: MLechner

"""
from typing import TYPE_CHECKING

from mcf.mcf_print_stats import print_mcf, string_dc

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicyVersion
    from mcf.mcf_init import GenCfg


def print_dic_values_all_optp_version(optpv_: 'OptimalPolicyVersion',
                                      gen_cfg: 'GenCfg',
                                      summary_dic: bool = False,
                                      line_length: int = 100,
                                      ) -> None:
    """Print the dataclasses."""
    txt = ('\n' + '-' * line_length + '\nAdditional parameters for version estimation'
           + '\n' + '-' * line_length
           )
    print_mcf(gen_cfg, txt, summary=summary_dic)
    print_dic_values_optpversion(optpv_, gen_cfg, summary=summary_dic)


def print_dic_values_optpversion(optp_: 'OptimalPolicyVersion',
                                 gen_cfg: 'GenCfg',
                                 summary: bool = False,
                                 ) -> None:
    """Print values of dictionaries that determine module."""
    dc_list = [optp_.version_cfg,]
    label_list = ['version_cfg',]

    print_str_list = [string_dc(dc, label) for dc, label in zip(dc_list, label_list)]
    print_str = '\n'.join(print_str_list)

    print_mcf(gen_cfg, print_str, summary=summary)

    print_mcf(gen_cfg, '\n', summary=summary)
