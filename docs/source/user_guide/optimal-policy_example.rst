Learning an optimal policy
==========================

Different types of policy rules
-------------------------------

The Modified Causal Forest refines treatment assignment mechanisms by defining the objective of an assignment rule. Its :py:class:`~optpolicy_main.OptimalPolicy` class offers two different methods to compute assignment algorithms, namely the policy tree and the best-scores methods. The policy tree method is based on the work of `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ , modifying their approach with respect to policy score calculation, definition of constraints, and handling of multi-valued features. In contrast, the best-score method simply assigns units to the treatment with the highest individual potential outcome. The **mcf** algorithm compares the two methods with random treatment allocations to demonstrate the increase in reward.

The following sections demonstrate how to implement these methods for policy learning. The :doc:`Algorithm reference <../algorithm_reference/optimal-policy_algorithm>` provides more details on the computational algorithms.

Policy Trees
------------

Generating data
~~~~~~~~~~~~~~~

The code below creates artificial example data for training and prediction. 

.. code-block:: python

    import os
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

    # Creating data.
    training_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=1000, obs_x_iate=1000, no_treatments=3)

Estimating an optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we initialize an instance by calling the :py:class:`~optpolicy_main.OptimalPolicy` class to estimate an optimal policy tree of depth two.

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=0,
                           gen_outpath=os.getcwd() + '/out')

Estimating sequentially optimal policy trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, we can create an instance to estimate sequentially optimal policy trees. This may lead to a reduction in runtime by, for instance, creating two sequentially optimal trees of depth 2+1 instead of a single optimal tree of depth 3. 

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=1,
                           gen_outpath=os.getcwd() + '/out')

Estimating a constrained optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code block creates a constrained optimal policy tree, considering a constraint with respect to the treatment shares.

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=0,
                           other_max_shares=(0.2, 0.8, 0),
                           gen_outpath=os.getcwd() + '/out')

Solve, allocate, and evaluate methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After initializing a class instance, we use it to solve for an optimal allocation rule, to allocate units to treatment states, and to evaluate the allocations with potential outcome data. 

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    alloc_train_df, _, _ = myoptp.solve(training_df, data_title='training')
    results_eva_train, _ = myoptp.evaluate(alloc_train_df, training_df,
                                           data_title='training')
    alloc_pred_df, _ = myoptp.allocate(prediction_df, 
                                       data_title='prediction')
    results_eva_pred, _ = myoptp.evaluate(alloc_pred_df, prediction_df,
                                          data_title='prediction')

Reporting
~~~~~~~~~

Finally, the code creates a PDF report. Please note that the program saves by default information like summary statistics and leaf information for the policy tree in a folder in the current working directory. 

.. code-block:: python

    # Creating a PDF report.
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + 'policy tree')
    my_report.report()

Best Policy Scores
------------------

The following code demonstrates how to obtain a policy rule based on the best-score method. 
                       
Estimating a policy rule using the best-score method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    import os
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

.. code-block:: python

    # Creating data.
      training_df, prediction_df, name_dict = example_data(
          obs_y_d_x_iate=1000, obs_x_iate=1000, no_treatments=3)

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='best_policy_score',
                          var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                          var_x_name_ord=('x_cont0', 'x_ord0'),
                          gen_outpath=os.getcwd() + '/out')

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    alloc_train_df, _, _ = myoptp.solve(training_df, data_title='training')
    results_eva_train, _ = myoptp.evaluate(alloc_train_df, training_df,
                                           data_title='training')
    alloc_pred_df, _ = myoptp.allocate(prediction_df,
                                       data_title='prediction')
    results_eva_pred, _ = myoptp.evaluate(alloc_pred_df, prediction_df,
                                      data_title='prediction')

.. code-block:: python

    # Creating a PDF report.
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + 'best_policy_score')
    my_report.report()
