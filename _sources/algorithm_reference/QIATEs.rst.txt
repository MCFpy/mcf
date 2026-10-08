QIATEs
======

Definition
----------

A Quantile Individualized Average Treatment Effect (QIATE) is a quantile of the
conditional mean treatment-effect function across the target population:

.. math::

    \tau_{m,l}(x) &= E[Y^m-Y^l\mid X=x], \\
    \operatorname{QIATE}(q;m,l)
    &= \inf\{t:\Pr(\tau_{m,l}(X)\leq t)\geq q\}.

The distribution is over :math:`X` in the prediction population; at a fixed :math:`x`, the conditional mean effect is a scalar.

Interpretation
--------------

QIATEs describe heterogeneity in mean effects explained by observed features.
They are neither quantiles of unobserved realized individual effects nor
differences between marginal potential-outcome quantiles.

Implementation
--------------

Estimated IATEs are sorted separately for each outcome and treatment
comparison. With ``p_qiate_smooth=True``, QIATE estimation combines
observations with similar ranks using kernel weights. Without smoothing,
it uses the observation closest to the requested quantile.
``p_qiate_bias_adjust`` controls a separate adjustment for ranking bias.

``p_qiate_no_of_quantiles`` sets the number of evenly spaced quantile
levels to estimate. The median is included when this number is odd.
You can inspect the selected levels using ``mymcf.p_cfg.qiate_quantiles``.

- ``p_qiate_m_mqiate`` returns QIATE(q) minus QIATE(0.5) as ``qiate_mmed``,
  with ``qiate_mmed_se`` when requested.
- ``p_qiate_m_opp`` returns QIATE(q) minus QIATE(1-q) as ``qiate_mopp``,
  with ``qiate_mopp_se``, including levels above 0.5.

QIATEs require discrete treatments, ``_int_low_memory_predict=False``,
``p_ba=False`` and no treatment versions.

Example
-------

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate artificial data
    training_df, prediction_df, name_dict = example_data(
        no_treatments=2,
        obs_y_d_x_iate=2000,
        obs_x_iate=2000,
        no_effect=False
    )

    mymcf = ModifiedCausalForest(
        var_d_name=name_dict['d_name'],
        var_y_name=name_dict['y_name'],
        var_x_name_ord=name_dict['x_name_ord'],
        var_x_name_unord=name_dict['x_name_unord'],
        # QIATE specific parameters
        p_qiate=True,
        p_ba=False,
        p_qiate_se=True,
        p_qiate_m_mqiate=True,
        p_qiate_m_opp=True,
        p_qiate_no_of_quantiles=None,
        p_qiate_smooth=None,
        p_qiate_smooth_bandwidth=None,
        p_qiate_bias_adjust=None,
        _int_low_memory_predict=False
    )

    mymcf.train(training_df)
    results = mymcf.predict(prediction_df)




    print(mymcf.p_cfg.qiate_quantiles)
    print(results['qiate'])
    print(results['qiate_mmed'])
    print(results['qiate_mopp'])
