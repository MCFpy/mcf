User Guide
=======================

This user guide provides a comprehensive walkthrough on how to use the main features of the **mcf**. It includes several example scripts to help users, regardless of their technical expertise, to navigate and use the **mcf** package effectively.

Modified Causal Forest
----------------------

.. toctree::
    :maxdepth: 1
    :numbered:

    user_guide/estimation
    user_guide/data_cleaning
    user_guide/weights_clusters
    user_guide/common_support
    user_guide/feature_selection
    user_guide/post_estimation_diagnostics
    user_guide/experimental_features
    user_guide/computational_speed

Optimal Policy
--------------

.. toctree::
    :maxdepth: 1
    :numbered:

    user_guide/optimal-policy_example

Example scripts
---------------

We provide several example scripts `in our example folder on GitHub <https://github.com/MCFpy/mcf/tree/main/examples>`__. Below you also find the direct links to these scripts. 

**Modified Causal Forest**

- `Minimal example <https://github.com/MCFpy/mcf/blob/main/examples/mcf_min_parameters.py>`__
- `Minimal example that uses the same data for training and prediction <https://github.com/MCFpy/mcf/blob/main/examples/mcf_training_prediction_data_same.py>`__
- `Full example with all parameters used <https://github.com/MCFpy/mcf/blob/main/examples/mcf_all_parameters.py>`__

**Optimal Policy**

- `Minimal example <https://github.com/MCFpy/mcf/blob/main/examples/optpolicy_min_parameters.py>`__
- `Minimal example that jointly estimates a Modified Causal Forest and an Optimal Policy Tree <https://github.com/MCFpy/mcf/blob/main/examples/mcf_optpol_combined.py>`__
- `Full example with all parameters used <https://github.com/MCFpy/mcf/blob/main/examples/optpolicy_all_parameters.py>`__
- `Example on fairscores method <https://github.com/MCFpy/mcf/blob/main/examples/optpolicy_fairness.py>`__
