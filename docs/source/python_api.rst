Python API
===========

Classes 
--------

..  
    If you add a '~' before a reference like `mymodule.MyClass.mymethod`, the
    link text will then only show 'mymethod' which is often desirable.
.. autosummary::
    ~mcf_mini.ModifiedCausalForest
    ~optpol_mini.OptimalPolicy

-----

Modified Causal Forest
--------------

.. currentmodule:: mcf_mini

.. autoclass:: ModifiedCausalForest

   .. rubric:: Methods overview

   .. autosummary::
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.blinder_iates
    ~ModifiedCausalForest.sensitivity

Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will still even when commented out, generate the separate
   page that can be cross-referenced.

   .. autosummary::
      :toctree:

      ~ModifiedCausalForest.blind_dict
      ~ModifiedCausalForest.cf_dict
      ~ModifiedCausalForest.cs_dict

.. autosummary:: ~ModifiedCausalForest.blind_dict
.. autosummary:: ~ModifiedCausalForest.cf_dict
.. autosummary:: ~ModifiedCausalForest.cs_dict

Methods
^^^^^^^^^^

.. autosummary:: ModifiedCausalForest.train

.. automethod:: ModifiedCausalForest.predict
.. automethod:: ModifiedCausalForest.analyse
.. automethod:: ModifiedCausalForest.blinder_iates
.. automethod:: ModifiedCausalForest.sensitivity


Optimal Policy
---------------------------

.. currentmodule:: optpol_mini

.. autoclass:: OptimalPolicy

   .. rubric:: Methods overview

   .. autosummary::
    ~OptimalPolicy.solve
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.print_time_strings_all_steps
    
    
Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will still even when commented out, generate the separate
   page that can be cross-referenced.

   .. autosummary::
      :toctree:
      ~OptimalPolicy.int_dict
      ~OptimalPolicy.dc_dict

.. autosummary:: ~OptimalPolicy.int_dict
.. autosummary:: ~OptimalPolicy.dc_dict

Methods
^^^^^^^^^^

.. Hack to have solve and allocate on separate page?
    .. autosummary:: 
        :toctree:
        OptimalPolicy.solve
        OptimalPolicy.allocate

.. automethod:: OptimalPolicy.evaluate
.. automethod:: OptimalPolicy.evaluate_multiple
.. automethod:: OptimalPolicy.print_time_strings_all_steps
