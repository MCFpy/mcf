.. _getting-started:

Getting started
=======================

This guide will walk you through using the **mcf** package to:

- estimate heterogeneous treatment effects using the Modified Causal Forest.
- learn an optimal policy rule based on a Policy Tree.


Example data
^^^^^^^^^^^^^^^^

First, we will use the :py:func:`~example_data_functions.example_data` function to generate synthetic datasets for training and prediction. This functions creates training (``training_df``) and prediction (``prediction_df``) DataFrames with a specified number of observations, features, and treatments, and allows for different heterogeneity types (``'linear'``, ``'nonlinear'``, ``'quadratic'``, ``'WagerAthey'``). The function also returns ``name_dict``, a dictionary containing the names of variable groups. You can define some features of the generated data by using the following parameters:

- ``obs_y_d_x_iate`` , the number of observations for the training data 
- ``obs_x_iate`` , the number of observations for the prediction data
- ``no_features`` , the number of features of different type to generate
- ``no_treatments`` , the number of treatments
- ``type_of_heterogeneity`` , different types of heterogeneity

For more details, visit the :doc:`python_api`. 

By default, the :py:func:`~example_data_functions.example_data` produces 1000 observations for both training and prediction, with 20 features, and 3 treatments. Let us change this slightly and generate 1500 training and prediction observations for 10 features and 3 treatments.

.. code-block:: python

    from mcf.example_data_functions import example_data
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data(
                                            obs_y_d_x_iate=1500,
                                            obs_x_iate=1500,
                                            no_features=10,
                                            no_treatments=3)
    
Estimating heterogeneous treatment effects
------------------------------------------

To estimate a Modified Causal Forest, we use the :py:class:`~mcf_functions.ModifiedCausalForest` class of the **mcf** package. To create an instance of the :py:class:`~mcf_functions.ModifiedCausalForest` class, we need to specify the name of

- at least one outcome variable through the ``var_y_name`` parameter
- the treatment variable through the ``var_d_name`` parameter
- ordered features through ``var_x_name_ord`` and/or unordered features through ``var_x_name_unord``

as follows:

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_functions import ModifiedCausalForest
    from mcf.optpolicy_functions import OptimalPolicy
    from mcf.reporting import McfOptPolReport
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    # Create an instance of the Modified Causal Forest model
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",  # Outcome variable
        var_d_name="treat",    # Treatment variable
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],  # Ordered covariates
        var_x_name_unord=["x_unord0"],  # Unordered covariate
        _int_show_plots=False  # Disable plots for faster performance
    )

Frequently used parameters
--------------------------

Below you find a selected list of optional parameters that are often used to initialize a Modified Causal Forest. For a more detailed description of these parameters, please refer to the documentation of :py:class:`~mcf_functions.ModifiedCausalForest`.

.. dropdown:: Commonly used optional parameters

    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | Parameter                        | Description                                                                                                       |
    +==================================+===================================================================================================================+
    | ``cf_boot``                      | Number of Causal Trees. Default: 1000.                                                                            |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``p_atet``                       | If True, :math:`\textrm{ATE's}` are also computed by treatment status (:math:`\textrm{ATET's}`). Default: False.  |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_list``              | Ordered feature(s) with many values used for :math:`\textrm{GATE}` estimation.                                    |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_ord``               | Ordered feature(s) with few values used for :math:`\textrm{GATE}` estimation.                                     |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_unord``             | Unordered feature(s) used for :math:`\textrm{GATE}` estimation.                                                   |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``p_gatet``                      | If True, :math:`\textrm{GATE's}` are also computed by treatment status (:math:`\textrm{GATET's}`). Default: False.|
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_ord``     | Ordered feature(s) always used in splitting decision.                                                             |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_unord``   | Unordered feature(s) always used in splitting decision.                                                           |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_y_tree_name``              | Outcome used to build trees. If not specified, the first outcome in ``y_name`` is selected for building trees.    |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_id_name``                  | Individual identifier.                                                                                            |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+

Accessing and customizing output location
------------------------------------------

The **mcf** package generates a number of standard outputs for your convenience. After initializing a Modified Causal Forest, the package will create an output folder where these results are stored.
Any method you are using, returns the location of these output files as last return (the reporting method returns the full file name of the pdf file in addition). 
Manually, you can find the location of the output folder by accessing the ``outpath`` entry of the ``gen_dict`` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_dict["outpath"]

We recommend you specify your preferred location for the output folder using the ``gen_outpath`` parameter of the class :py:class:`~mcf_functions.ModifiedCausalForest`.

Training a Modified Causal Forest
-----------------------------------

Next we will train the Modified Causal Forest on the ``train_mcf_df`` data using the :py:meth:`~mcf_functions.ModifiedCausalForest.train` method:

.. code-block:: python

    my_mcf.train(training_df)

Now we are ready to estimate heterogeneous treatment effects on the ``pred_mcf_train_pt_df`` data using the :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method.

.. code-block:: python

    results = my_mcf.predict(prediction_df)


Accessing results
~~~~~~~~~~~~~~~~~

The simplest way to get an overview of your results is to read the PDF-report that is generated by the class :py:class:`~reporting.McfOptPolReport`:

.. code-block:: python

    mcf_report = McfOptPolReport(mcf=my_mcf, outputfile='Modified-Causal-Forest_Report')
    mcf_report.report()


You can also access all the results programmatically. Here's how to do it:

The :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method returns a ``results`` tuple. This includes:

- All estimates.

.. code-block:: python

    results[0]

- A string with the path to the location of the results.

.. code-block:: python

    results[1]

The former contains a dictionary with the estimation results. To get an overview, start by extracting the dictionary:

.. code-block:: python

    results_dict = results[0]

Now, we can have a look at the keys of the dictionary:

.. code-block:: python

    results_dict.keys()

By default, the average treatment effects (:math:`\textrm{ATE's}`) as well as the individualized average treatment effects (:math:`\textrm{IATE's}`) are estimated. If these terms do not sound familiar, :doc:`here <user_guide/estimation>` you can learn more about the different kinds of heterogeneous treatment effects.

In the multiple treatment setting there is more than one :math:`\textrm{ATE}` to consider. The following entry of the ``results_dict`` dictionary lists the estimated treatment contrasts:

.. code-block:: python

    ate_array = results_dict.get('ate')
    print("Average Treatment Effect (ATE):\n", ate_array)

For instance, if you have treatment levels 0, 1, and 2, you will see an entry of the form [[[0.1, 0.3, 0.5]]]. Here, the first entry, 0.1, specifies the treatment contrast between treatment level 1 and treatment level 0. The second entry, 0.3, specifies the treatment contrast between treatment level 2 and treatment level 0. The third entry specifies the treatment contrast between level 1 and 2.

In the same way, you can access and print the standard errors of the respective :math:`\textrm{ATE's}` by running:

.. code-block:: python

    ate_se_array = results_dict.get('ate_se')
    print("\nStandard Error of ATE:\n", ate_se_array)

The estimated :math:`\textrm{IATE's}`, together with the locally centered and uncentered potential outcomes, are stored as columns of a Pandas DataFrame that you have access to from the extracted ``results_dict`` dictionary. If you do not know the variable names of your specific estimation in advance, have a look at the keys of this dictionary:

.. code-block:: python

    results_dict.get('iate_data_df').keys()

You can access these elements all at once or independently in the following ways:

.. code-block:: python

    # access all at once (the full DataFrame)
    df = results_dict['iate_data_df']

    # access only the IATEs
    df_iate = df.loc[:, df.columns.str.endswith('_iate') ]  

    # centered potential outcomes
    df_po_centered = df.loc[:, (df.columns.str.endswith('pot')) &
                                             ~df.columns.str.endswith('un_lc_pot')]
    # uncentered potential outcomes
    df_po_uncentered = df.loc[:, df.columns.str.endswith('un_lc_pot')]


To illustrate this, let us build on the previous example with three treatment levels, 0, 1, and 2. The columns ``outcome_lc0_pot``, ``outcome_lc1_pot``, and ``outcome_lc2_pot`` represent the *predicted* potential outcomes under the respective treatment level. You can extract these, for example, using:

.. code-block:: python

    results_dict.get('iate_data_df')['outcome_lc0_pot']

The columns ``outcome_lc1vs0_iate``, ``outcome_lc2vs0_iate``, and ``outcome_lc2vs1_iate`` give you the estimated :math:`\textrm{IATE's}`. As above, these columns contrast the respective treatment levels.

.. code-block:: python

    results_dict.get('iate_data_df')['outcome_lc1vs0_iate']




Note that, if you specify the methods as in the provided example files, you have access to all the elements discussed above directly from the ``results`` tuple. For example,

.. code-block:: python

    # use the .predict() method as shown in the example files
    results, _ = my_mcf.predict(prediction_df)

    # access a potential outcome
    results.get('iate_data_df')['outcome_lc1vs0_iate']


Here, ``results`` essentially plays the same role as ``results_dict`` explained previously. These are two equivalent ways to access your results.


Post-estimation
---------------

You can use the :py:meth:`~mcf_functions.ModifiedCausalForest.analyse` method to investigate a number of post-estimation plots. These plots are also exported to the previously created output folder:

.. code-block:: python

    my_mcf.analyse(results)

    
Learning an optimal policy rule
-------------------------------

Let's explore how to learn an optimal policy rule using the :py:class:`~optpolicy_functions.OptimalPolicy` class of the **mcf** package. To get started we need a Pandas DataFrame that holds the estimated potential outcomes (also called policy scores), the treatment variable and the features on which we want to base the decision tree.

As you may recall, we estimated the potential outcomes in the previous section. They are stored as columns in the ``iate_data_df`` entry of the results dictionary:

.. code-block:: python

    print(results["iate_data_df"].head())

The column names are explained in the ``iate_names_dic`` entry of the results dictionary. The uncentered potential outcomes are stored in columns with the suffix ``_un_lc_pot``.

.. code-block:: python

    print(results["iate_names_dic"])

Now that we understand this, we are ready to build an Optimal Policy Tree. To do so, we need to create an instance of class :py:class:`~optpolicy_functions.OptimalPolicy` where we set the ``gen_method`` parameter to "policy tree" and provide the names of

- the treatment through the ``var_d_name`` parameter
- the potential outcomes through the ``var_polscore_name`` parameter
- ordered and/or unordered features used to build the policy tree using the ``var_x_name_ord`` and ``var_x_name_unord`` parameter respectively

as follows:

.. code-block:: python

    # Create an instance of the OptimalPolicy class:
    my_optimal_policy = OptimalPolicy(
        var_d_name="treat",
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        var_x_name_unord=["x_unord0"],
        gen_method="best_policy_score", 
        pt_depth_tree_1=2
        )


Note that the ``pt_depth_tree_1`` parameter specifies the depth of the (first) policy tree. For demonstration purposes we set it to 2. In practice, you should choose a larger value which will increase the computational burden. See the :doc:`User guide <user_guide/optimal-policy_example>` and the :doc:`Algorithm reference <algorithm_reference/optimal-policy_algorithm>` for more detailed explanations.

Accessing results
~~~~~~~~~~~~~~~~~

After initializing an Optimal Policy Tree, the **mcf** package will automatically create an output folder. This folder will contain a number of standard outputs for your convenience. You can find the location of this folder in your console output. Alternatively, you can manually specify the folder location using the ``gen_outpath`` parameter.


Fit an Optimal Policy Tree
----------------------------

To find the Optimal Policy Tree, we use the :py:meth:`~optpolicy_functions.OptimalPolicy.solve` method, where we need to supply the pandas DataFrame holding the potential outcomes, treatment variable and the features:

.. code-block:: python

    train_pt_df = results["iate_data_df"]
    alloc_train_df, _, _ = my_optimal_policy.solve(training_df, data_title='training')

The returned DataFrame contains the optimal allocation rule for the training data.

.. code-block:: python

    print(alloc_train_df)

Next, we can use the :py:meth:`~optpolicy_functions.OptimalPolicy.evaluate` method to evaluate this allocation rule. This will return a dictionary holding the results of the evaluation. As a side-effect, the DataFrame with the optimal allocation is augmented with columns that contain the observed treatment and a random allocation of treatments.

.. code-block:: python

    results_eva_train, _ = my_optimal_policy.evaluate(alloc_train_df, training_df,
                                           data_title='training')

    print(results_eva_train)

Overview of results
~~~~~~~~~~~~~~~~~~~~~

A great way to get an overview of the results is to read the PDF-report that can be generated using the class :py:class:`~reporting.McfOptPolReport`:

.. code-block:: python

    policy_tree_report = McfOptPolReport(
        optpol = my_policy_tree,
        outputfile = 'Optimal-Policy_Report'
        )
    policy_tree_report.report()

Additionally, you can access the results programmatically. The `report` attribute of your optimal policy object is a dictionary containing the results. Here's how you can access a specific element:

.. code-block:: python

    dictionary_of_results = my_optimal_policy.report
    print(dictionary_of_results.keys())
    evaluation_list = dictionary_of_results['evalu_list']
    print("Evaluation List: ", evaluation_list)

Finally, it is straightforward to apply our Optimal Policy Tree to new data. To do so, we simply apply the :py:meth:`~optpolicy_functions.OptimalPolicy.allocate` method
to the DataFrame holding the potential outcomes, treatment variable and the features for the data that was held out for evaluation:

.. code-block:: python

    alloc_pred_df, _ = my_optimal_policy.allocate(prediction_df, data_title='prediction')

To evaluate this allocation rule, again apply the :py:meth:`~optpolicy_functions.OptimalPolicy.allocate` method similar to above.

.. code-block:: python

    results_eva_pred, _ = my_optimal_policy.evaluate(alloc_pred_df, prediction_df,
                                      data_title='prediction')

    print(results_eva_pred)

Next steps
----------

The following are great sources to learn even more about the **mcf** package:

- The :doc:`user_guide` offers explanations on additional features of the mcf package and provides several example scripts.
- Check out the :doc:`python_api` for details on interacting with the mcf package.
- The :doc:`algorithm_reference` provides a technical description of the methods used in the package.
