Modified Causal Forests
=======================

Welcome to the documentation of **mcf**, the Python package implementing the Modified Causal Forest introduced by `Lechner (2018) <https://doi.org/10.48550/arXiv.1812.09487>`_. This package allows you to estimate heterogeneous treatment effects for binary and multiple treatments from experimental or observational data. Additionally, mcf offers the capability to learn optimal policy allocations.

If you're new to the **mcf** package, we recommend following these steps:

- `Installation Guide`_: Learn how to install mcf on your system.
- `Usage Example`_: Explore a simple example to quickly understand how to apply mcf to your data.
- :doc:`getting_started`: Dive into a more detailed example to get a better feel for working with mcf.

For those seeking further information:

- The :doc:`user_guide` offers explanations on additional features of the mcf package and provides several example scripts.
- Check out the :doc:`python_api` for details on interacting with the mcf package.
- The :doc:`algorithm_reference` provides a technical description of the methods used in the package.


Installation Guide
------------------

The current version of **mcf** is compatible with **Python 3.11**. You can install mcf from PyPI using:

.. code-block:: bash

    pip install mcf

For a smoother experience and to avoid conflicts with other packages, we strongly recommend using a virtual environment based on **conda**. 

You can manage conda environments either via the command line or a graphical interface. 
The command line offers a compatible solution for all operating systems, making it our recommended choice. However, the graphical interface is more user-friendly.

If you prefer to use the command line, first install conda as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__. Next follow the steps below in your Anaconda Prompt (Windows) or terminal (macOS and Linux):

1. Set up and activate a conda environment named *mcf-env*:

  .. code-block:: bash

      conda create -n mcf-env

  .. code-block:: bash

      conda activate mcf-env

2. Install Python **3.11**:

  .. code-block:: bash

      conda install Python="3.11"

3. Finally, install **mcf** in this environment using **pip**:

  .. code-block:: bash

      pip install mcf



If you prefer a graphical interface, you can:

1. Install Anaconda distribution including Anaconda navigator by downloading it `here <https://docs.anaconda.com/free/navigator/install/>`__.

2. Set up an environment, follow the guide `here <https://docs.anaconda.com/free/navigator/getting-started/#managing-environments>`__ and make sure you choose **Python=3.11.8** for your environment.

3. Install the **mcf** package by using pip install in your IDE console:

  .. code-block:: bash

      pip install mcf

An alternative to the step above is to install the **mcf** package using this guide `here <https://docs.anaconda.com/free/navigator/getting-started/#managing-packages>`__.

Note: It is recommended to prioritize ``conda install`` for package installations before using ``pip install``. On a Windows machine, if you plan to use Spyder as your IDE, make sure to execute ``conda install spyder`` before proceeding with ``pip install mcf`` to reduce the risk of errors during installation.


.. _usage-example:

Usage Example
-------------

To demonstrate how to use **mcf**, let's simulate some data and apply the Modified Causal Forest:

.. code-block:: python

    import numpy as np
    import pandas as pd

    from mcf import ModifiedCausalForest
    from mcf import OptimalPolicy
    from mcf import McfOptPolReport

    def simulate_data(n: int, seed: int) -> pd.DataFrame:
        """
        Simulate data with a binary treatment 'd', outcome 'y', unordered control
        variable 'female' and two ordered controls 'x1', 'x2'.

        Parameters:
        - n (int): Number of observations in the simulated data.
        - seed (int): Seed for the random number generator.

        Returns:
        pd.DataFrame: Simulated data in a Pandas DataFrame.

        """
        rng = np.random.default_rng(seed)

        d = rng.integers(low=0, high=1, size=n, endpoint=True)
        female = rng.integers(low=0, high=1, size=n, endpoint=True)
        x_ordered = rng.normal(size=(n, 2))
        y = (x_ordered[:, 0] +
            x_ordered[:, 1] * (d == 1) +
            0.5 * female +
            rng.normal(size=n))

        data = {"y": y, "d": d, "female": female}

        for i in range(x_ordered.shape[1]):
            data["x" + str(i + 1)] = x_ordered[:, i]

        return pd.DataFrame(data)

    df = simulate_data(n=100, seed=1234)

    # Create an instance of class ModifiedCausalForest:
    my_mcf = ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        var_x_name_unord=["female"],
        _int_show_plots=False
    )

    # Train the Modified Causal Forest on the simulated data and predict treatment
    # effects in-sample:
    my_mcf.train(df)
    results = my_mcf.predict(df)

    # The 'results' dictionary contains the estimated treatment effects:
    print(results.keys())

    print(results["ate"])  # Average Treatment Effect (ATE)
    print(results["ate_se"])  # Standard Error (SE) of the ATE

    # DataFrame with Individualized Treatment Effects (IATE) and potential outcomes
    print(results["iate_data_df"])


    # Create an instance of class OptimalPolicy:
    my_optimal_policy = OptimalPolicy(
        var_d_name="d",
        var_polscore_name=["Y_LC0_un_lc_pot", "Y_LC1_un_lc_pot"],
        var_x_name_ord=["x1", "x2"],
        var_x_name_unord=["female"]
        )

    # Learn an optimal policy rule using the predicted potential outcomes
    alloc_df = my_optimal_policy.solve(results["iate_data_df"])

    # Evaluate the optimal policy rule on the simulated data:
    my_optimal_policy.evaluate(alloc_df, results["iate_data_df"])

    # Compare the optimal policy rule to the observed and a random allocation:
    print(alloc_df)

    # Produce a PDF-report that summarises the most important results
    my_report = McfOptPolReport(mcf=my_mcf, optpol=my_optimal_policy,
                                outputfile='mcf_report')
    my_report.report()

For a more detailed example, see the :doc:`getting_started` section.

Source code and contributing
-----------------------------

The Python source code is available on `GitHub <https://github.com/MCFpy/mcf>`_. If you have questions, want to report bugs, or have feature requests, please use the `issue tracker <https://github.com/MCFpy/mcf/issues>`__.

References
----------

- Bodory H, Busshoff H, Lechner M. **High Resolution Treatment Effects Estimation: Uncovering Effect Heterogeneities with the Modified Causal Forest**. *Entropy*. 2022; 24(8):1039. `Read Paper <https://doi.org/10.3390/e24081039>`__

- Lechner M. **Modified Causal Forests for Estimating Heterogeneous Causal Effects**. 2018. `Read Paper <https://doi.org/10.48550/arXiv.1812.09487>`__

- Lechner M, Mareckova J. **Modified Causal Forest**. 2022. `Read Paper <https://doi.org/10.48550/arXiv.2209.03744>`__

License
-------

**mcf** is distributed under the `MIT License <https://github.com/MCFpy/mcf?tab=MIT-1-ov-file#readme>`__.

.. toctree::
   :hidden:

   getting_started.rst
   user_guide.rst
   algorithm_reference.rst
   python_api.rst
   changelog.rst
