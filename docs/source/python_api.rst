Python API
==========

Overview of classes 
-------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mcf.mcf_main.ModifiedCausalForest
    mcf.optpolicy_main.OptimalPolicy
    mcf.reporting.McfOptPolReport


Modified Causal Forest
----------------------

.. currentmodule:: mcf.mcf_main
.. autoclass:: ModifiedCausalForest
   :members:
   :undoc-members:
   :show-inheritance:

Methods
^^^^^^^

.. autosummary:: 
   :toctree: generated/
   :nosignatures:

   ModifiedCausalForest.train
   ModifiedCausalForest.predict
   ModifiedCausalForest.analyse
   ModifiedCausalForest.sensitivity


Optimal Policy
--------------

.. currentmodule:: mcf.optpolicy_main
.. autoclass:: OptimalPolicy
   :members:
   :undoc-members:
   :show-inheritance:

Methods
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   OptimalPolicy.solve
   OptimalPolicy.allocate
   OptimalPolicy.evaluate
   OptimalPolicy.evaluate_multiple
   OptimalPolicy.print_time_strings_all_steps


Reporting
---------

.. currentmodule:: mcf.reporting
.. autoclass:: McfOptPolReport
   :members:
   :undoc-members:
   :show-inheritance:

Methods
^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   McfOptPolReport.report

