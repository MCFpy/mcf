Fairness Extensions in Optimal Policy
===========================================

The :py:class:`~optpolicy_functions.OptimalPolicy` class in the **mcf** includes experimental features for fairness adjustments, accessible through the ``fairscores`` method. 
These features are designed to ensure that policy scores are fair with respect to certain protected variables. 
The fairness adjustments are based on the work by Bearth, Lechner, Mareckova, and Muny (2024).

This method can be configured using several parameters to control the type and extent of fairness adjustments. 

Example
~~~~~~~~~
To use the fairness adjustments, configure the :py:class:`~optpolicy_functions.OptimalPolicy` class with the appropriate parameters and call the ``fairscores`` method on your data. This will return a DataFrame with adjusted policy scores that account for fairness considerations.

.. code-block:: python

    import os
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_functions import OptimalPolicy
    
    # Generate data
    training_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=1000,
        obs_x_iate=1000,
        no_features=5,
        no_treatments=2,
        seed=12345,
        type_of_heterogeneity='WagerAthey',
        descr_stats=True
    )
    
    # Define parameters for OptimalPolicy
    params = {
        'var_d_name': 'treat',
        'var_polscore_name': ('y_pot0', 'y_pot1'),
        'var_x_name_ord': ('x_ord0', 'x_ord1'),
        'var_protected_name_ord': ('x_ord0'),  # Specify at least one protected feature
        'gen_outfiletext': "OptPolicy_Simple_Example",
        'gen_outpath': os.getcwd() + '/outputOPT'
    }
    
    # Initialize and adjust fairness scores
    myoptp_fair = OptimalPolicy(**params)
    training_fair_df, fairscore_names, _, _ = myoptp_fair.fairscores(training_df.copy(), data_title='training')
    
    # Update params to use the fair scores
    params['var_polscore_name'] = tuple(fairscore_names)
    
    # Solve for optimal allocation rule
    alloc_train_df, _, _ = myoptp_fair.solve(training_fair_df.copy(), data_title='training fair')
    
    # Evaluate the allocation
    results_eva_train, _ = myoptp_fair.evaluate(alloc_train_df, training_fair_df.copy(), data_title='training fair')

Note
------
These features are experimental and may require further testing and validation. For more details or additional parameters, please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` documentation.
