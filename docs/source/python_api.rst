Python API
==========

.. currentmodule:: mcf

Overview of classes 
-------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ModifiedCausalForest
    OptimalPolicy
    McfOptPolReport


Modified Causal Forest
----------------------

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
