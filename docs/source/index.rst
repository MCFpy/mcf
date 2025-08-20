Modified Causal Forests
=======================

Welcome to the documentation of **mcf**, the Python package implementing the Modified Causal Forest introduced by `Lechner (2018) <https://doi.org/10.48550/arXiv.1812.09487>`_. This package allows you to estimate heterogeneous treatment effects for binary and multiple treatments from experimental or observational data. Additionally, it allows to learn optimal policy allocations.

If you're new to the **mcf** package, we recommend following these steps:

- `Installation Guide`_: Learn about the installation procedure for your system.
- `Usage Example`_: Explore a simple example to see how to apply the **mcf** to your data.
- :doc:`getting_started`: Dive into a more detailed example.

For further information:

- :doc:`user_guide`: Explore further features of the package and example scripts.
- :doc:`python_api`: Get to know the details on how to interact with the package.
- :doc:`algorithm_reference`: Learn about the technical background of the methods applied in the package.

.. _installation-guide:

Installation Guide
------------------

The current **mcf** version is compatible with Python 3.12. For the installation you can proceed in different ways. 

You can install the package from PyPI using:

.. code-block:: bash

    pip install mcf

For a smooth experience without conflicts with other packages, use a virtual environment based on conda. You can manage conda environments either via the command line or a graphical interface. 
The command line offers a compatible solution for all operating systems, making it our recommended choice. However, the graphical interface is more user-friendly.

If you prefer the command line, install conda as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__. Next open your Anaconda Prompt (Windows) or terminal (macOS and Linux) and do the following:

1. Set up and activate a conda environment named *mcf-env*:

  .. code-block:: bash

      conda create -n mcf-env

  .. code-block:: bash

      conda activate mcf-env

2. Install Python 3.12:

  .. code-block:: bash

      conda install Python="3.12"

3. Install **mcf** in this environment using pip:

  .. code-block:: bash

      pip install mcf

If you prefer a graphical interface, do the following:

1. Install Anaconda distribution including Anaconda navigator from `here <https://docs.anaconda.com/free/navigator/install/>`__.

2. Set up an environment as described `here <https://docs.anaconda.com/free/navigator/getting-started/#managing-environments>`__ and make sure you choose Python=3.12.5 for your environment.

3. Install the **mcf** package by using pip install in your IDE console:

  .. code-block:: bash

      pip install mcf

An alternative to the third step, installing the **mcf** package, is to use `this <https://docs.anaconda.com/free/navigator/getting-started/#managing-packages>`__ guide. It is recommended to prioritize ``conda install`` for package installations before using ``pip install``.

**Note (1)**, if you plan to use Spyder as your IDE on a Windows machine, make sure to execute ``conda install spyder`` before proceeding with ``pip install mcf``. This reduces the risk of errors during installation.

.. _usage-example:

Usage Example
-------------

We use the :py:func:`~example_data_functions.example_data` function to generate synthetic datasets for training and prediction to showcase an application of the :py:class:`~mcf_main.ModifiedCausalForest`. 

.. code-block:: python

    import numpy as np
    import pandas as pd
    
    from mcf.example_data_functions import example_data
    from mcf import ModifiedCausalForest
    from mcf import OptimalPolicy
    from mcf import McfOptPolReport
    
    
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
    
    # Train the Modified Causal Forest on the training data
    my_mcf.train(training_df)
    # Predict treatment effects using the model on prediction data
    results = my_mcf.predict(prediction_df)
    
    # The `results` object is a tuple with two elements:
    # 1. A dictionary containing all estimates
    results[0]
    # 2. A string with the path to the results location
    results[1] 
       
    # Extract the dictionary of estimates
    results_dict = results[0]
    
    # Access the Average Treatment Effect (ATE)
    ate_array = results_dict.get('ate')
    print("Average Treatment Effect (ATE):\n", ate_array)
    
    # Access the Standard Error of the ATE
    ate_se_array = results_dict.get('ate_se')
    print("\nStandard Error of ATE:\n", ate_se_array)
    
    # Access the Individualized Treatment Effects (IATE)
    iate_array = results_dict.get('iate')
    print("\nIndividualized Treatment Effects (IATE):\n", iate_array)
    
    # Access the DataFrame of Individualized Treatment Effects
    iate_df = results_dict.get('iate_data_df')
    print("\nDataFrame of Individualized Treatment Effects:\n", iate_df)
    
    
    # Create an instance of the OptimalPolicy class:
    my_optimal_policy = OptimalPolicy(
        var_d_name="treat",
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        var_x_name_unord=["x_unord0"]
        )
    
    # Learn an optimal policy rule using the predicted potential outcomes
    alloc_train_df, _, _ = my_optimal_policy.solve(training_df, data_title='training')
    
    # Evaluate the optimal policy rule on the training data:
    results_eva_train, _ = my_optimal_policy.evaluate(alloc_train_df, training_df,
                                               data_title='training')
    
    # Allocate observations to treatment state using the prediction data
    alloc_pred_df, _ = my_optimal_policy.allocate(prediction_df, data_title='prediction')
    
    # Evaluate allocation with potential outcome data.
    results_eva_pred, _ = my_optimal_policy.evaluate(alloc_pred_df, prediction_df,
                                              data_title='prediction')
        
    # Allocation DataFrame for the training set
    print(alloc_train_df)
    
    # Produce a PDF-report that summarises the results
    my_report = McfOptPolReport(mcf=my_mcf, 
                                optpol=my_optimal_policy,
                                outputfile='mcf_report')
    my_report.report()

**Note (2)**, to check the version of the **mcf** module used to create an instance, you can additionally run the following code:

.. code-block:: python
    
    print(my_mcf.__version__)


Source code and contributing
-----------------------------

The Python source code is available on `GitHub <https://github.com/MCFpy/mcf>`_. 
If you have questions, want to report bugs, or have feature requests, please use the `issue tracker <https://github.com/MCFpy/mcf/issues>`__.

References
----------
**Conceptual foundation**:

- Lechner M (2018). **Modified Causal Forests for Estimating Heterogeneous Causal Effects**. `Read Paper <https://doi.org/10.48550/arXiv.1812.09487>`__
- Lechner M, Mareckova J (2022). **Modified Causal Forest**. `Read Paper <https://doi.org/10.48550/arXiv.2209.03744>`__
- Lechner M, Bearth N (2024). **Causal Machine Learning for Moderation Effects**. `Read Paper <https://arxiv.org/abs/2401.08290>`__

**Algorithm demonstrations**:

- Bodory H, Busshoff H, Lechner M (2022). **High Resolution Treatment Effects Estimation: Uncovering Effect Heterogeneities with the Modified Causal Forest**. *Entropy*. 24(8):1039. `Read Paper <https://doi.org/10.3390/e24081039>`__
- Bodory H, Mascolo F, Lechner M (2024). **Enabling Decision Making with the Modified Causal Forest: Policy Trees for Treatment Assignment**. *Algorithms*. 17(7):318. `Read Paper <https://doi.org/10.3390/a17070318>`__

**Simulations**:

- Lechner M, Mareckova J (2024). **Comprehensive Causal Causal Machine Learning**. `Read Paper <https://doi.org/10.48550/arXiv.2405.10198>`__

**Applications in diverse fields**:

- Audrino F, Chassot J, Huang C, Knaus M, Lechner M, Ortega JP (2024). **How does post-earnings announcement affect firms’ dynamics? New evidence from causal machine learning**. *Journal of Financial Econometrics*. 22(3), 575–604. `Read paper <https://academic.oup.com/jfec/article/22/3/575/6640191>`__

- Burlat H (2024). **Everybody’s got to learn sometime? A causal machine learning evaluation of training programmes for jobseekers in France**. *Labour Economics*. In Press. Paper 102573. `Read paper <https://doi.org/10.1016/j.labeco.2024.102573>`__

- Cockx B, Michael L, Joost B (2023). **Priority to unemployed immigrants? A causal machine learning evaluation of training in Belgium**. *Labour Economics*. 80(102306). `Read paper <https://www.sciencedirect.com/science/article/pii/S0927537122001968>`__

- Handouyahia A, Rikhi T, Awad G, Aouli E (2024). **Heterogeneous causal effects of labour market programs: A machine learning approach**. *Proceedings of Statistics Canada Symposium 2022*. `Read paper <https://www150.statcan.gc.ca/n1/pub/11-522-x/2022001/article/00017-eng.pdf>`__

- Heiniger S, Koeniger W, Lechner M (2024). **The heterogeneous response of real estate prices during the Covid-19 pandemic**. *Journal of the Royal Statistical Society Series A: Statistics in Society*, 00, 1–24. `Read paper <https://doi.org/10.1093/jrsssa/qnae078>`__

- Hodler R, Lechner M, and Raschky P (2023). **Institutions and the Resource Course: New Insights from Causal Machine Learning**. *PLoS ONE*. 18(6): e0284968. `Read paper <https://doi.org/10.1371/journal.pone.0284968>`__

- Zhu M (2023). **The Effect of Political Participation of Chinese Citizens on Government Satisfaction: Based on Modified Causal Forest**. *Procedia Computer Science*. 221, 1044–1051. `Read paper <https://linkinghub.elsevier.com/retrieve/pii/S187705092300844X>`__

License
-------

**mcf** is distributed under the `MIT License <https://github.com/MCFpy/mcf?tab=MIT-1-ov-file#readme>`__.

.. toctree::
   :hidden:

   getting_started.rst
   user_guide.rst
   algorithm_reference.rst
   python_api.rst
   FAQ.rst
   changelog.rst
