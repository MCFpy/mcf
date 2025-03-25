Python API
==========


Overview of classes 
-------------------

..  
    If you add a '~' before a reference like `mymodule.MyClass.mymethod`, the
    link text will then only show 'mymethod' which is often desirable.
.. autosummary::
    ~mcf_functions.ModifiedCausalForest
    ~optpolicy_functions.OptimalPolicy
    ~reporting.McfOptPolReport


Modified Causal Forest
----------------------

.. currentmodule:: mcf_functions

.. autoclass:: ModifiedCausalForest
    :members:
    :undoc-members:
    :show-inheritance:

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

.. currentmodule:: optpolicy_functions

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
      OptimalPolicy.fairscores
      OptimalPolicy.solve
      OptimalPolicy.allocate
      OptimalPolicy.evaluate
      OptimalPolicy.evaluate_multiple
      OptimalPolicy.print_time_strings_all_steps


.. autosummary:: 
    ~OptimalPolicy.fairscores
    ~OptimalPolicy.solve
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.print_time_strings_all_steps


Reporting
--------------

.. currentmodule:: reporting

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

.. currentmodule:: example_data_functions

.. autofunction:: example_data

   .. autosummary:: 
      :toctree:
      example_data_functions.example_data

.. autosummary:: 
    ~example_data_functions.example_data
