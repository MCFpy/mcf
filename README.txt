![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mcf)
[![PyPI Downloads](https://static.pepy.tech/badge/mcf)](https://pepy.tech/projects/mcf)
![PyPI Downloads](https://img.shields.io/pypi/dm/mcf)

# mcf – Modified Causal Forest in Python

**mcf** is a Python package implementing the Modified Causal Forest (MCF) methodology introduced by Lechner (2018) for estimating heterogeneous causal effects.

It provides a flexible framework for causal machine learning with support for binary and multiple treatments in both experimental and observational data settings.

In addition to treatment effect estimation, **mcf** enables data-driven policy learning through optimal treatment allocation rules based on estimated potential outcomes.

---

## Documentation and Maintenance

- Documentation and website: https://mcfpy.github.io/mcf/#/
- Bug tracker: https://github.com/mcfpy/mcf/issues

---

## Main Features

The **mcf** package provides two core components for causal machine learning:

### Modified Causal Forest

The `ModifiedCausalForest` class implements a flexible tree-based framework for estimating heterogeneous treatment effects. It supports:

- Estimation of and inference for individualized treatment effects (IATEs) and aggregates as the (group) average treatment effects (GATEs, ATEs)
- Binary and multiple treatment settings
- Experimental and observational data
- Flexible covariate specification (ordered and unordered variables)

The object-oriented design provides a unified workflow for model training, prediction, and extraction of causal estimates.

---

### Optimal Policy Learning

The `OptimalPolicy` class enables data-driven treatment assignment by learning decision rules that maximize a reward function. It provides:

- Learning of optimal treatment allocation rules
- Policy evaluation on training and test data
- Support for multiple treatment alternatives and policy constraints

This allows translation of policy scores into actionable decision rules for optimal policy design.

---

## Citation

The implementation builds on the following literature:

- Lechner (2019): *Modified Causal Forests for Estimating Heterogeneous Causal Effects*

If you use **mcf** in your work, please cite:

> Lechner, M. (2019). Modified Causal Forests for Estimating Heterogeneous Causal Effects. arXiv:1812.09487.

```bibtex
@misc{lechner2019modifiedcausalforest,
  title        = {Modified Causal Forests for Estimating Heterogeneous Causal Effects},
  author       = {Michael Lechner},
  year         = {2019},
  eprint       = {1812.09487},
  archivePrefix = {arXiv},
  primaryClass = {econ.EM},
  url          = {https://arxiv.org/abs/1812.09487}
}

---

## License

This project is distributed under the MIT License.
