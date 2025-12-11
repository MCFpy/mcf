QIATEs
======

The **Quantile Individualized Average Treatment Effect (QIATE)** was introduced in 
Kutz and Lechner (2025).

Definition
-------------

.. math::

   QIATE(\alpha; d, d', x) := q^{\alpha}(\text{IATE}(d, d', x)) \\
   = q^{\alpha} \left( \mathbb{E}[Y(d) - Y(d') \mid X = x] \right).

Here,

.. math::

   \text{IATE}(d, d', x) = \mathbb{E}[Y(d) - Y(d') \mid X = x]

denotes the IATE contrasting treatments 
:math:`d` and :math:`d'` for individual :math:`i`.  

:math:`q^\alpha := \inf \{ z \in \mathbb{R} \mid F(z) \geq \alpha \}` is the 
:math:`\alpha`-th quantile of a cumulative distribution function :math:`F` of a 
random variable :math:`Z`, such that :math:`F(z) = P(Z \leq z)`.

Interpretation
-----------------

The QIATE estimates the :math:`\alpha`-th quantile of the IATE distribution, focusing 
on the part of treatment effect variation that can be explained by observed characteristics 
(*the actionable component of heterogeneity*).  
This enables researchers to investigate how treatment effects differ across the explainable 
part of the distribution.

Implementation
-----------------

The QIATE is implemented as follows:

#. Estimate the IATE using the **mcf**.
#. Sort the estimated :math:`\widehat{IATE}` and determine their relative position or 
   rank :math:`z \in [0,1]`.
#. For each relative position :math:`z_i` estimate the QIATE as a continuous GATE.  
   To account for uncertainty in the ranking we smooth the weights using 
   **Nadaraya–Watson kernel regression**.

Example
-------

.. code-block:: python

    from mcf.example_data_functions import example_data
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
        p_qiate_se=True,
        p_qiate_m_mqiate=True,
        p_qiate_m_opp=True,
        p_qiate_no_of_quantiles=None,
        p_qiate_smooth=None,
        p_qiate_smooth_bandwidth=None,
        p_qiate_bias_adjust=None
    )

    mymcf.train(training_df)
    results = mymcf.predict(prediction_df)



