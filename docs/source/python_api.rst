Python API
==========


Overview of classes 
-------------------

..  
    If you add a '~' before a reference like `mymodule.MyClass.mymethod`, the
    link text will then only show 'mymethod' which is often desirable.
.. autosummary::
    ~mcf_main.ModifiedCausalForest
    ~optpolicy_main.OptimalPolicy
    ~reporting.McfOptPolReport


Modified Causal Forest
----------------------

.. currentmodule:: mcf_main

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
      ModifiedCausalForest.train_iv
      ModifiedCausalForest.predict
      ModifiedCausalForest.predict_different_allocations
      ModifiedCausalForest.predict_iv
      ModifiedCausalForest.analyse
      ModifiedCausalForest.sensitivity

.. autosummary:: 
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.train_iv
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.predict_different_allocations
    ~ModifiedCausalForest.predict_iv
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.sensitivity


Optimal Policy
--------------

.. currentmodule:: optpolicy_main

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
      OptimalPolicy.allocate
      OptimalPolicy.evaluate
      OptimalPolicy.evaluate_multiple
      OptimalPolicy.estrisk_adjust
      OptimalPolicy.solvefair
      OptimalPolicy.solve
      OptimalPolicy.print_time_strings_all_steps
      OptimalPolicy.winners_losers


.. autosummary:: 
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.estrisk_adjust
    ~OptimalPolicy.solvefair
    ~OptimalPolicy.solve
    ~OptimalPolicy.print_time_strings_all_steps
    ~OptimalPolicy.winners_losers


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

