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

Choose your Python version before creating the environment. Depending on your operating system and whether you prefer using Ray or Joblib for parallel processing, the following configurations are possible:

- **Ray**: install Python 3.12 on Windows, or Python 3.13 on Linux or Mac.
- **Joblib**: install Python 3.14 on any operating system.

By default, **mcf** uses Joblib when Ray is unavailable and for smaller adjusted training
samples on Windows (>250'000). Otherwise, it uses Ray. Further guidance is available in
:ref:`computational-speed`.

For the installation you can proceed in different ways.

You can install the package from PyPI using:

.. code-block:: bash

    pip install mcf

For a smooth experience without conflicts with other packages, use a virtual environment based on conda. You can manage conda environments either via the command line or a graphical interface.
The command line offers a compatible solution for all operating systems, making it our recommended choice. However, the graphical interface is more user-friendly. As an alternative, you may also create a new virtual environment directly from the ``environment.yml`` file available in our GitHub repository.

If you prefer the command line, install conda as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__. Next open your Anaconda Prompt (Windows) or terminal (macOS and Linux) and do the following:

1. Set up and activate a conda environment named *mcf-env*:

.. code-block:: bash

    conda create -n mcf-env

.. code-block:: bash

    conda activate mcf-env

2. Install your chosen Python version. For example, for Python 3.12:

.. code-block:: bash

  conda install Python="3.12"

3. Install **mcf** in this environment using pip:

.. code-block:: bash

  pip install mcf

If you prefer a graphical interface, do the following:

1. Download Anaconda Distribution including Anaconda Navigator from the Anaconda Documentation `here <https://www.anaconda.com/docs/main>`__ and install the software on your device.

2. Set up an environment as described in the Environments Page of the Anaconda Documentation and make sure you choose the compatible Python version indicated at the beginning of this installation guide.

3. Install the **mcf** package by using pip install in your IDE console:

.. code-block:: bash

  pip install mcf

Alternative ways of installing packages are shown in the Packages Page of the Anaconda Documentation. It is recommended to prioritize ``conda install`` for package installations before using ``pip install``.

.. note::

    If you plan to use Spyder as your IDE on a Windows machine, make sure to
    execute ``conda install spyder`` before proceeding with ``pip install mcf``.
    This reduces the risk of errors during installation.


.. _usage-example:

Usage Example
-------------

We use the :py:func:`~example_data.example_data` function to generate
synthetic data for a combined effect-estimation and policy-learning example.
First, we train a :py:class:`~mcf_main.ModifiedCausalForest`. We then use its
estimated potential outcomes as scores for learning and evaluating a policy
tree with :py:class:`~optpolicy_main.OptimalPolicy`.

The example keeps three samples separate:

- ``train_mcf_df`` is used to train the forest.
- ``pred_mcf_train_pt_df`` is used to predict scores and learn the policy tree.
- ``evaluate_pt_df`` is used to evaluate the learned policy on observations
  used neither to train the forest nor to learn the policy tree.


.. code-block:: python

    from pathlib import Path

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

    # Generate forest-training data and a separate prediction sample.
    train_mcf_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=3000, obs_x_iate=3000
    )

    # Split prediction data into policy-learning and evaluation samples.
    prediction_df = prediction_df.sample(frac=1, random_state=42)
    split = len(prediction_df) // 2
    pred_mcf_train_pt_df = prediction_df.iloc[:split].copy()
    evaluate_pt_df = prediction_df.iloc[split:].copy()

    out = Path.cwd() / 'mcf_tutorial'

    # Train the forest.
    mcf = ModifiedCausalForest(
        var_y_name='outcome', var_d_name='treat',
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        var_x_name_unord=['x_unord0'],
        gen_outpath=out / 'effects', _int_show_plots=False
    )
    mcf.train(train_mcf_df)

    # Predict potential outcomes for policy learning.
    train_results = mcf.predict(pred_mcf_train_pt_df)
    data_train_pt = train_results['iate_data_df'].copy()
    score_names = train_results['iate_names_dic'][0][
        'names_y_pot_uncenter'
    ]

    # Learn the policy tree using estimated, uncentered potential outcomes.
    policy = OptimalPolicy(
        gen_method='policy_tree', var_d_name='treat',
        var_polscore_name=score_names,
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        pt_depth_tree_1=2, pt_depth_tree_2=0,
        gen_outpath=out / 'policy', _int_show_plots=False
    )
    fit, data_train_pt = policy.solve(data_train_pt)
    policy.evaluate(fit['allocation_df'], data_train_pt)

    # Predict scores and evaluate the policy on the separate sample.
    test_results = mcf.predict(evaluate_pt_df)
    oos_df = test_results['iate_data_df'].copy()
    allocation = policy.allocate(oos_df)
    evaluation = policy.evaluate(allocation['allocation_df'], oos_df)

    # Produce a PDF report.
    report = McfOptPolReport(
        mcf=mcf, optpol=policy, outputpath=out, outputfile='Tutorial'
    )
    pdf_path = report.report()

.. note::

    To check the version of the **mcf** module used to create an instance,
    you can additionally run the following code:

    .. code-block:: python

        print(mcf.__version__)



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

- Lechner M, Mareckova J (2024). **Comprehensive Causal Machine Learning**. `Read Paper <https://doi.org/10.48550/arXiv.2405.10198>`__

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
