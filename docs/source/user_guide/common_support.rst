.. _common-support:

Common support
==============

Common support is crucial in estimating heterogeneous treatment effects. Loosely speaking, it requires that the distributions of the covariates overlap across all treatment arms. The :py:class:`~mcf_functions.ModifiedCausalForest` class provides several options to check for and enforce common support.

Common support checks and adjustments are performed before any causal effects are estimated. You can control the type of common support adjustment with the parameter ``cs_type`` of the class :py:class:`~mcf_functions.ModifiedCausalForest`. If you set ``cs_type`` to 0, there is no common support adjustment.

If you set ``cs_type`` to 1 or 2, common support is enforced based on propensity scores that are estimated with classification forests [1]_. The Modified Causal Forest will then remove all observations whose propensity scores lie outside certain cut-off probabilities. For a value of 1, which is the default, the cut-off probabilities are determined automatically by the **mcf** package. For a value of 2, you can specify the cut-off probabilities yourself using the parameter ``cs_min_p``: Any observation with a propensity score :math:`P(D = m| X)` of less than or equal to ``cs_min_p`` - for at least one treatment arm - will be removed from the data set.

When common support adjustments are enabled, the **mcf** package will display standard common support plots to help you understand the distribution of propensity scores across treatment arms. These plots are also saved in the output folder that the **mcf** package generates. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord="x",
        cs_type=1
    )
    my_mcf.gen_dict["outpath"]

The common support plots will be stored in the subfolder `common_support`.

------

.. [1] Out of bag predictions are used to avoid overfitting.


Advanced options
----------------

Common support criteria become more restrictive with an increasing number of treatments. The parameter ``cs_adjust_limits`` allows you to reduce this restrictiveness by offsetting the cut-off limits. The upper cut-off will be multiplied by :math:`1 + \text{cs_adjust_limits}` and the lower cut-off will be multiplied by :math:`1 - \text{cs_adjust_limits}`. By default ``cs_adjust_limits`` has a value of :math:`(\text{number of treatments} - 2) \times 0.05`.

The parameter ``cs_max_del_train`` allows you to specify a maximum share of observations in the training data set that are allowed to be dropped to enforce common support. If this threshold is exceeded, the program will terminate and raise a corresponding exception. By default, an exception will be raised if more than 50% of the observations are dropped. In this case, you should consider using a more balanced input data set.

The parameter ``cs_quantil`` allows you to deviate from the default cut-off probabilities when ``cs_type`` is set to 1, which are based on min-max rules. If ``cs_quantil`` is set to a value of less than 1, the respective quantile is used to determine the upper and lower cut-off probabilities: Concretely, observations will be dropped if for at least one treatment :math:`m` the propensity score :math:`P(D = m| X)` lies outside the interval :math:`[q_{\text{cs_quantil}}, q_{\text{1-cs_quantil}}]`, where :math:`q_{\alpha}` denotes the :math:`\alpha`-quantile of the propensity scores :math:`\{P(D_i = m| X_i)\}_{i=1}^n`.

Parameter overview
------------------

Below is an overview of the above mentioned parameters related to common support adjustments in the class :py:class:`~mcf_functions.ModifiedCausalForest`:  

+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter            | Description                                                                                                                                                                                                                                                                            |
+======================+========================================================================================================================================================================================================================================================================================+
| ``cs_type``          | If set to 0, there is no common support adjustment. For a value of 1, common support is enforced by automatically chosen cut-off probabilities. For a value of 2, you can specify the cut-off probabilities yourself using the parameter ``cs_min_p``. Default: 1.                     |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cs_min_p``         | Only relevant if ``cs_type`` is set to 2. Observations are removed if for at least one treatment the propensity score is less then or equal to ``cs_min_p``. Default: 0.01.                                                                                                            |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cs_adjust_limits`` | Only relevant if ``cs_type`` is set to 1 or 2. The upper cut-off is multiplied by :math:`1 + \text{cs_adjust_limits}` and the lower cut-off is multiplied by :math:`1 - \text{cs_adjust_limits}`. Default: :math:`(\text{number of treatments} - 2) \times 0.05`.                      |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cs_max_del_train`` | Only relevant if ``cs_type`` is set to 1 or 2. Raises an exception if the share of observations that are dropped to enforce common support exceeds ``cs_max_del_train``. Default: 0.5.                                                                                                 |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``cs_quantil``       | Only relevant if ``cs_type`` is set to 1. If ``cs_quantil`` is set to a value less than 1, the respective quantile is used to determine the upper and lower cut-off probabilities. If set to 1, the cut-off probabilities are chosen automatically based on min-max rules. Default: 1. |
+----------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details.

Examples
------------------

.. code-block:: python

    from mcf import ModifiedCausalForest

    ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Turn common support adjustments off:
        cs_type=0
    )

    ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Use automatic common support adjustments
        cs_type=1,
        # Offset the cut-off limits: Multiply the upper cut-off by 1.1 and the
        # lower cut-off by 0.9:
        cs_adjust_limits=0.1,
        # Raise an exception if more than 25% of the observations are dropped:
        cs_max_del_train=0.25
    )

    ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Use common support adjustments and specify cut-off probabilities manually:
        cs_type=2,
        cs_min_p=0.05
    )
