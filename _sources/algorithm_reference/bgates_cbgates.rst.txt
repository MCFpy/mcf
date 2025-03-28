CBGATEs and BGATEs
==================

Balanced Group Average Treatment Effects (:math:`\textrm{BGATE's}`) and Causal Balanced Group Average Treatment Effects (:math:`\textrm{CBGATE's}`) have been introduced by `Bearth & Lechner (2024) <https://browse.arxiv.org/abs/2401.08290>`_.

The :math:`\textrm{BGATE}` can be used to estimate :math:`\textrm{ATE's}` for different groups (:math:`\textrm{GATE's}`), while accounting for differences in other covariates, i.e.

.. math::
   BGATE(m,l;x) = \mathbb{E} \bigg[ \mathbb{E} \big[ Y^1 - Y^0 \big\vert Z=z, W=w \big]\bigg]

Here, :math:`Z` is a single feature of :math:`X` and :math:`W` denotes a subgroup of features of :math:`X` excluding :math:`Z`. :math:`z` is a possible value of the variable of interest :math:`Z`. 

The :math:`\textrm{BGATE}` partially overcomes the attribution problem related to a simple :math:`\textrm{GATE}`, where other relevant variables may confound effect heterogeneity.
Furthermore, the Causal Balanced Group Average Treatment Effect (:math:`\textrm{CBGATE}`) makes a causal interpretation of the :math:`\textrm{BGATE}` possible, when all variables other than the heterogeneity variable :math:`Z` are balanced and further asssumptions discussed in `Bearth & Lechner (2024) <https://browse.arxiv.org/abs/2401.08290>`_ hold. Hence, both :math:`\textrm{CBGATE}` and the plain-vanilla :math:`\textrm{GATE}` are limiting cases of the :math:`\textrm{BGATE}`.

Algorithmically, the :math:`\textrm{BGATE}` and the :math:`\textrm{CBGATE}` are implemented as follows:

1. Draw a random sample from the prediction data.
2. Keep the heterogeneity and balancing variables.
3. Replicate the data from step 2 :math:`n_z` times, where :math:`n_z` denotes the cardinality of the heterogeneity variable of interest. In each :math:`n_z` fold, set :math:`Z` to a specific value.
4. Draw the nearest neighbours of each observation in the prediction data in terms of the balancing variables and the heterogeneity variable. If there is a tie, the algorithm chooses one randomly.
5. Form a new sample with all selected neighbours.
6. Compute :math:`\textrm{GATE's}` and their standard errors.

One should note that this procedure only happens in the prediction part using the previously trained forest. This implementation differs from `Bearth & Lechner (2024) <https://browse.arxiv.org/abs/2401.08290>`_ estimation approach. They use double/debiased machine learning to estimate the parameters of interest.

To turn on the :math:`\textrm{BGATE}` , set ``p_bgate`` to True. To turn on the :math:`\textrm{CBGATE}`, set ``p_cbgate`` to True. The balancing variables :math:`W` have to be specified in ``var_x_name_balance_bgate``.


Below you find a list of the main parameters which are related to the :math:`\textrm{BGATE's}` and :math:`\textrm{CBGATE's}`. Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details or additional parameters. 


Parameters 
------------------------

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``var_x_name_balance_bgate``
     - This parameter, which can be a string or a list of strings, specifies the variables that the GATEs should be balanced on. It's only relevant if p_bgate is set to True. When a BGATE is computed, the distribution of these specified variables remains constant, ensuring that the effect of the treatment is estimated in a balanced manner across these variables. This helps to control for potential confounding effects that these variables might have on the treatment effect. If set to None, the program defaults to using the other heterogeneity variables (specified in var_z) for balancing. This means that the GATEs will be balanced across the distribution of these heterogeneity variables. 
   * - ``p_bgate``
     - Activates the estimation of a Balanced Group Average Treatment Effect (BGATE). 
   * - ``p_cbgate``
     - Enables the estimation of a GATE that is balanced in all other features. 
   * - ``p_gate_no_evalu_points``
     - Determines the number of evaluation points for discretized variables in (C)BGATE estimation. The default value is 50.
   * - ``p_bgate_sample_share``
     - Used to speed up the program as the (C)BGATE estimation is CPU intensive. 


Examples
~~~~~~~~~

.. code-block:: python

   from mcf.example_data_functions import example_data
   from mcf.mcf_functions import ModifiedCausalForest
   
   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()
   
   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       var_z_name_list=["x_cont0", "x_cont1"],
       # Variables to balance the GATEs on
       var_x_name_balance_bgate=["x_cont0", "x_cont1"],
       # Estimate a balanced GATE in selected features
       p_bgate=True,
       # Random samples to speed up the programme
       p_bgate_sample_share = None
   )
   
   my_mcf.train(training_df)
   results, _ = my_mcf.predict(prediction_df)


.. code-block:: python

   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_unord0", "x_cont0", "x_ord1"],
       var_z_name_list=["x_cont0"],
       # Estimate a GATE that is balanced in all other features
       p_cbgate=True
   )
   
   my_mcf.train(training_df)
   results, _ = my_mcf.predict(prediction_df)
