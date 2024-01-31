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


Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary::
      :toctree:

      ModifiedCausalForest.blind_dict
      ModifiedCausalForest.cf_dict
      ModifiedCausalForest.cs_dict
      ModifiedCausalForest.ct_dict
      ModifiedCausalForest.int_dict
      ModifiedCausalForest.dc_dict
      ModifiedCausalForest.fs_dict
      ModifiedCausalForest.forest
      ModifiedCausalForest.gen_dict
      ModifiedCausalForest.p_dict
      ModifiedCausalForest.post_dict
      ModifiedCausalForest.sens_dict
      ModifiedCausalForest.time_strings
      ModifiedCausalForest.var_dict
      ModifiedCausalForest.var_x_type
      ModifiedCausalForest.var_x_values

.. autosummary:: 
    ~ModifiedCausalForest.blind_dict
    ~ModifiedCausalForest.cf_dict
    ~ModifiedCausalForest.cs_dict
    ~ModifiedCausalForest.ct_dict
    ~ModifiedCausalForest.int_dict
    ~ModifiedCausalForest.dc_dict
    ~ModifiedCausalForest.fs_dict
    ~ModifiedCausalForest.forest
    ~ModifiedCausalForest.gen_dict
    ~ModifiedCausalForest.p_dict
    ~ModifiedCausalForest.post_dict
    ~ModifiedCausalForest.sens_dict
    ~ModifiedCausalForest.time_strings
    ~ModifiedCausalForest.var_dict
    ~ModifiedCausalForest.var_x_type
    ~ModifiedCausalForest.var_x_values

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
      ModifiedCausalForest.blinder_iates
      ModifiedCausalForest.sensitivity

.. autosummary:: 
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.blinder_iates
    ~ModifiedCausalForest.sensitivity


Optimal Policy
--------------

.. currentmodule:: optpolicy_functions

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
      OptimalPolicy.gen_dict
      OptimalPolicy.other_dict
      OptimalPolicy.pt_dict
      OptimalPolicy.rnd_dict
      OptimalPolicy.time_strings
      OptimalPolicy.var_dict
      OptimalPolicy.var_x_type
      OptimalPolicy.var_x_values

.. autosummary:: 
    ~OptimalPolicy.int_dict
    ~OptimalPolicy.dc_dict
    ~OptimalPolicy.gen_dict
    ~OptimalPolicy.other_dict
    ~OptimalPolicy.pt_dict
    ~OptimalPolicy.rnd_dict    
    ~OptimalPolicy.time_strings
    ~OptimalPolicy.var_dict
    ~OptimalPolicy.var_x_type    
    ~OptimalPolicy.var_x_values

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
      OptimalPolicy.print_dic_values_all_optp
      OptimalPolicy.print_dic_values_optp

.. autosummary:: 
    ~OptimalPolicy.solve
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.print_time_strings_all_steps
    ~OptimalPolicy.print_dic_values_all_optp
    ~OptimalPolicy.print_dic_values_optp


Reporting
--------------

.. currentmodule:: reporting

.. autoclass:: McfOptPolReport


Attributes
^^^^^^^^^^

..
   HACK -- Also list the attributes in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary::
      :toctree:
      McfOptPolReport.xxxx


.. autosummary:: 
    ~McfOptPolReport.xxxx


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

