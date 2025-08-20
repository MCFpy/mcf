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

..
   HACK -- Also list the methods in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary:: 
      :toctree:
      ModifiedCausalForest.train
      ModifiedCausalForest.predict
      ModifiedCausalForest.analyse
      ModifiedCausalForest.sensitivity

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

..
   HACK -- Also list the methods in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary:: 
      :toctree:
      OptimalPolicy.solve
      OptimalPolicy.allocate
      OptimalPolicy.evaluate
      OptimalPolicy.evaluate_multiple
      OptimalPolicy.print_time_strings_all_steps

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

..
   HACK -- Also list the methods in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary:: 
      :toctree:
      McfOptPolReport.report

.. autosummary:: 
    ~McfOptPolReport.report


Example Data function
---------------------

.. currentmodule:: mcf.example_data_functions
.. autoclass:: example_data

   .. autosummary:: 
      :toctree:
      example_data_functions.example_data

.. autosummary:: 
    ~example_data
