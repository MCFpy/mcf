Python API
==========

Overview of classes 
-------------------

.. autosummary::
    ~mcf.mcf_main.ModifiedCausalForest
    ~mcf.optpolicy_main.OptimalPolicy
    ~mcf.reporting.McfOptPolReport

Modified Causal Forest
----------------------

.. currentmodule:: mcf.mcf_main

.. autoclass:: ModifiedCausalForest

Methods
^^^^^^^

.. autosummary:: 
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.sensitivity


Optimal Policy
--------------

.. currentmodule:: mcf.optpolicy_main

.. autoclass:: OptimalPolicy

Methods
^^^^^^^

.. autosummary:: 
    ~OptimalPolicy.solve
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.print_time_strings_all_steps


Reporting
---------

.. currentmodule:: mcf.reporting

.. autoclass:: McfOptPolReport

Methods
^^^^^^^

.. autosummary:: 
    ~McfOptPolReport.report


Example Data function
---------------------

.. currentmodule:: mcf.example_data_functions

.. autofunction:: example_data

.. autosummary:: 
    ~example_data_functions.example_data
