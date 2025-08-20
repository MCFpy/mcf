QIATEs
======

The **Quantile Individualized Average Treatment Effect (QIATE)** was introduced in 
Kutz and Lechner (2025).

Definition
----------

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
--------------

The QIATE estimates the :math:`\alpha`-th quantile of the IATE distribution, focusing 
on the part of treatment effect variation that can be explained by observed characteristics 
(*the actionable component of heterogeneity*).  
This enables researchers to investigate how treatment effects differ across the explainable 
part of the distribution.

Implementation
--------------

The QIATE is implemented as follows:

#. Estimate the IATE using the **mcf**.
#. Sort the estimated :math:`\widehat{IATE}` and determine their relative position or 
   rank :math:`z \in [0,1]`.
#. For each relative position :math:`z_i` estimate the QIATE as a continuous GATE.  
   To account for uncertainty in the ranking we smooth the weights using 
   **Nadaraya–Watson kernel regression**.
