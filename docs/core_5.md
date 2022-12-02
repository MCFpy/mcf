# Practical hints

## Speed ups

### Discretizing continuous GATE (group average treatment effect) features

You can pass the maximum number of categories for continuous GATE features to the keyword argument [max_cats_z_vars](./core_6.md#max_cats_z_vars). 

| Argument                                       | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [max_cats_z_vars](./core_6.md#max_cats_z_vars) | Maximum number of categories for continuous GATE features. The default value is *N*<sup>0.3</sup>, where *N* denotes the number of observations. |

### Ordered features

Ordinal, ordered features will be dealt with in the same (standard) fashion as continuous features.

### Unordered features

Unordered (categorical) features are treated in specific ways that depend on the number of categories.

| Categories  | Program processing                                           |
| ----------- | ------------------------------------------------------------ |
| 2           | Reclassified by the program as ordered and treated that way. |
| More than 2 | At each split, a new feature will be created that contains the leaf means of the dependent variable for particular categories. The split is then determined using this new feature like an ordered one instead of the categorical feature directly. For features with many categories, this is computationally much more efficient than creating all possible splitting groups. This procedure is proposed in [Chou (1991)](https://ieeexplore.ieee.org/document/88569) and in a prediction context in [Hastie, Tibshirani, and Friedman (2009, 2nd edition, p. 310)](https://www.springer.com/de/book/9780387848570). If this procedure is to be avoided, then do not specify unordered features but recode them so that they can be used like ordered features (e.g. dummies). |

### Sparse arrays

For small datasets, internally storing the forest weights in dense instead of sparse matrices may reduce the program runtime as well as drastically lower the demand on the installed RAM. The type of this two-dimensional array can be specified in [weight_as_sparse](./core_6.md#weight_as_sparse).

| Argument                                         | Description                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [weight_as_sparse](./core_6.md#weight_as_sparse) | *True* if the weights are internally stored in sparse matrices, otherwise *False*. The default is *True*. |


### Misc 

You can decrease running time by executing code directly in the terminal.

