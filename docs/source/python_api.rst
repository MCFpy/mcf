Python API
===========


Overview of classes 
--------

..  
    If you add a '~' before a reference like `mymodule.MyClass.mymethod`, the
    link text will then only show 'mymethod' which is often desirable.
.. autosummary::
    ~mcf_mini.ModifiedCausalForest
    ~optpol_mini.OptimalPolicy


Modified Causal Forest
--------------

.. currentmodule:: mcf_mini

.. autoclass:: ModifiedCausalForest


Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

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
      ModifiedCausalForest.blinder_iates
      ModifiedCausalForest.sensitivity

.. autosummary:: ~ModifiedCausalForest.train
.. autosummary:: ~ModifiedCausalForest.predict
.. autosummary:: ~ModifiedCausalForest.analyse
.. autosummary:: ~ModifiedCausalForest.blinder_iates
.. autosummary:: ~ModifiedCausalForest.sensitivity


Optimal Policy
---------------------------

.. currentmodule:: optpol_mini

.. autoclass:: OptimalPolicy


Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary::
      :toctree:
      OptimalPolicy.int_dict
      OptimalPolicy.dc_dict

.. autosummary:: ~OptimalPolicy.int_dict
.. autosummary:: ~OptimalPolicy.dc_dict

Methods
^^^^^^^^^^

..
   HACK -- Also list the methods in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary:: 
      :toctree:
      ~OptimalPolicy.solve
      ~OptimalPolicy.allocate
      ~OptimalPolicy.evaluate
      ~OptimalPolicy.evaluate_multiple
      ~OptimalPolicy.print_time_strings_all_steps

.. autosummary:: ~OptimalPolicy.solve
.. autosummary:: ~OptimalPolicy.allocate
.. autosummary:: ~OptimalPolicy.evaluate
.. autosummary:: ~OptimalPolicy.evaluate_multiple
.. autosummary:: ~OptimalPolicy.print_time_strings_all_steps
