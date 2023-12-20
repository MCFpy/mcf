Python API
===========


Classes 
--------

.. autosummary::
    double_ml_data.DoubleMLData
    mcf_mini.ModifiedCausalForest
    optpol_mini.OptimalPolicy


dml example API
-----------------

.. currentmodule:: double_ml_data

.. autoclass:: DoubleMLData
   
   .. rubric:: Methods

   .. autosummary::
    ~DoubleMLData.from_arrays
    ~DoubleMLData.set_x_d
   
   .. rubric:: Attributes

   .. autosummary::
      ~DoubleMLData.all_variables
      ~DoubleMLData.binary_outcome
      ~DoubleMLData.binary_treats
      ~DoubleMLData.d
      ~DoubleMLData.d_cols
      ~DoubleMLData.data
      ~DoubleMLData.force_all_x_finite
      ~DoubleMLData.n_coefs
      ~DoubleMLData.n_instr
      ~DoubleMLData.n_obs
      ~DoubleMLData.n_treat
      ~DoubleMLData.t
      ~DoubleMLData.t_col
      ~DoubleMLData.use_other_treat_as_covariate
      ~DoubleMLData.x
      ~DoubleMLData.x_cols
      ~DoubleMLData.y
      ~DoubleMLData.y_col
      ~DoubleMLData.z
      ~DoubleMLData.z_cols
   

.. automethod:: DoubleMLData.from_arrays

.. automethod:: DoubleMLData.set_x_d


mcf mini API
--------------

.. currentmodule:: mcf_mini

.. autoclass:: ModifiedCausalForest

   .. rubric:: Methods

   .. autosummary::
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.blinder_iates
    ~ModifiedCausalForest.sensitivity




Attributes
^^^^^^^^^^

.. autosummary:: ModifiedCausalForest.blind_dict
.. autosummary:: ModifiedCausalForest.cf_dict
.. autosummary:: ModifiedCausalForest.cs_dict

Methods
^^^^^^^^^^

.. automethod:: ModifiedCausalForest.train
.. automethod:: ModifiedCausalForest.predict
.. automethod:: ModifiedCausalForest.analyse
.. automethod:: ModifiedCausalForest.blinder_iates
.. automethod:: ModifiedCausalForest.sensitivity


optimal policy mini API
---------------------------

.. currentmodule:: optpol_mini

.. autoclass:: OptimalPolicy

   .. rubric:: Methods

   .. autosummary::
    ~OptimalPolicy.solve
    ~OptimalPolicy.allocate
    ~OptimalPolicy.evaluate
    ~OptimalPolicy.evaluate_multiple
    ~OptimalPolicy.print_time_strings_all_steps
    
    
Attributes
^^^^^^^^^^

.. autosummary:: OptimalPolicy.int_dict
.. autosummary:: OptimalPolicy.dc_dict


Methods
^^^^^^^^^^

.. automethod:: OptimalPolicy.solve
.. automethod:: OptimalPolicy.allocate
.. automethod:: OptimalPolicy.evaluate
.. automethod:: OptimalPolicy.evaluate_multiple
.. automethod:: OptimalPolicy.print_time_strings_all_steps