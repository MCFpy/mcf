Modified Causal Forests
=======================

Welcome to the documentation of ``mcf``, the Python package implementing the Modified Causal Forest introduced by `Lechner (2019) <https://doi.org/10.48550/arXiv.1812.09487>`_. This package allows you to estimate heterogeneous treatment effects for binary and multiple treatments from experimental or observational data. Additionally, ``mcf`` offers the capability to learn optimal policy allocations based on decision trees.

If you're new to the ``mcf`` package, we recommend following these steps:

- `Installation Guide`_: Learn how to install ``mcf`` on your system.
- `Usage Example`_: Explore a simple example to quickly understand how to apply ``mcf`` to your data.
- :doc:`getting_started`: Dive into a more detailed example to get a better feel for working with ``mcf``.

For those seeking more detailed information:

- The :doc:`user_guide` offers detailed explanations on specific topics.
- Check out the :doc:`python_api` for extra information on interacting with the ``mcf`` package.
- The :doc:`algorithm_reference` provides a technical description of the methods used in the package.


Installation Guide
------------------

The current version of ``mcf`` is compatible with Python **3.11**. You can install ``mcf`` from PyPI using:

.. code-block:: bash

    pip install mcf

For a smoother experience and to avoid conflicts with other packages, we strongly recommend using a virtual environment based on ``conda``. Follow the steps below in your Anaconda Prompt (Windows) or terminal (macOS and Linux):

1. Install ``conda`` as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__.

2. Set up and activate a conda environment named *mcf-env*:

  .. code-block:: bash

      conda create -n mcf-env

  .. code-block:: bash

      conda activate mcf-env

3. Install Python **3.11**:

  .. code-block:: bash

      conda install Python="3.11"

4. Finally, install ``mcf`` in this environment using `pip`:

  .. code-block:: bash

      pip install mcf

Note: It is recommended to prioritize ``conda install`` for package installations before using ``pip install``. On a Windows machine, if you plan to use Spyder as your IDE, make sure to execute ``conda install spyder`` before proceeding with ``pip install mcf`` to reduce the risk of errors during installation.

.. _usage-example:

Usage Example
-------------

To demonstrate how to use ``mcf``, let's simulate some data and apply the Modified Causal Forest:

.. code-block:: python

    import numpy as np
    import pandas as pd

    from mcf import ModifiedCausalForest

    def simulate_data(n):
        """
        Simulate data with a binary treatment 'd', outcome 'y', unordered control
        variable 'female' and three ordered controls 'x1', 'x2', 'x3'.

        Parameters:
        - n (int): Number of observations in the simulated data.

        Returns:
        pd.DataFrame: Simulated data in a Pandas DataFrame.

        """
        d = np.random.choice([0, 1], n, replace=True)
        female = np.random.choice([0, 1], n, replace=True)
        x_ordered = np.random.normal(size=(n, 3))
        y = (x_ordered[:, 0] +
            x_ordered[:, 1] * (d == 1) +
            x_ordered[:, 2] * (d == 2) +
            0.5 * female +
            np.random.normal(size=n))

        data = {"y": y, "d": d, "female": female}

        for i in range(x_ordered.shape[1]):
            data["x" + str(i + 1)] = x_ordered[:, i]

        return pd.DataFrame(data)

    np.random.seed(123)
    df = simulate_data(n=500)

    # Create an instance of class ModifiedCausalForest:
    modified_causal_forest_model = ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2", "x3"],
        var_x_name_unord=["female"],
        _int_show_plots=False
    )

    # Train the Modified Causal Forest on simulated data and predict treatment
    # effects in-sample:
    modified_causal_forest_model.train(df)
    results = modified_causal_forest_model.predict(df)

    # The 'results' dictionary contains the estimated treatment effects:
    print(results.keys())

    print(results["ate"])  # Average Treatment Effect (ATE)
    print(results["ate_se"])  # Standard Error (SE) of the ATE

For a more detailed example, see the **Getting Started** section.

Source code and contributing
-----------------------------

The Python source code is available on `GitHub <https://github.com/MCFpy/mcf>`_. If you have questions, want to report bugs, or have feature requests, please use the `issue tracker <https://github.com/MCFpy/mcf/issues>`__.

References
----------

- Bodory H, Busshoff H, Lechner M. **High Resolution Treatment Effects Estimation: Uncovering Effect Heterogeneities with the Modified Causal Forest**. *Entropy*. 2022; 24(8):1039. `Read Paper <https://doi.org/10.3390/e24081039>`__

- Lechner M. **Modified Causal Forests for Estimating Heterogeneous Causal Effects**. 2019. `Read Paper <https://doi.org/10.48550/arXiv.1812.09487>`__

License
-------

``mcf`` is distributed under the `MIT License <https://github.com/MCFpy/mcf?tab=MIT-1-ov-file#readme>`__.
