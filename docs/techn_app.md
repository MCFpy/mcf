# Technical Appendix

## Parallelization

From a computational perspective, several tasks naturally admit parallelization. One prime example is forest building as trees are grown independently from one another. Depending on the particular application and operating system (OS), you may wish to opt for different modes of parallelization. The program hosts the subsequent solutions:

1. parallelization via the Ray module
2. parallelization via the concurrent-futures module

The subsequent discussion is meant to provide a rough guidance, which mode of parallelization to choose.

To begin with, note that in non Unix-based OS, memory may pose a significant bottleneck.  When the forest building is parallelized over the trees, the data to determine the splits is copied in each process. If the data is large, this may cause the program to crash. The number of parallel workers therefore should be guided by the number of logical processors, RAM, and data size.  The program includes a routine to determine the optimal number of workers for multiprocessing. This routine takes into account possible memory limitations to ensure that the system does not crash. The thus identified optimal number of workers can be manually overwritten.

We have found that Ray scales better in terms of CPU utilization than the concurrent-futures module; however, for small samples the concurrent-futures option outperforms Ray.

|Argument | Description |
|---------|-------------|
|[mp_parallel](./core_6.md#mp_parallel) |Number of parallel processes. |
|[mp_with_ray](./core_6.md#mp_with_ray) |Implement parallelization via Ray |
|[mp_ray_objstore_multiplier](./core_6.md#mp_ray_objstore_multiplier)|Increase internal default values for the Ray object store above 1 if programme crashes because object store is full.|

Note that, in the current Ray version 1.3 results from the Ray futures are not evicted once they fall out of scope. This induces the program to crash eventually as the local object store is full of objects. We correct for this behavior explicitly via the general purpose function `auto_garbage_collect(pct=80.0)`. This routine deletes objects (to some extent) if memory used is larger than 80% of total available memory.



## Technical details on feature selection

### Defaults

To reduce the runtime, the program builds a single forest only for feature selection.  The defaults of the tuning parameters for tree building are:

### Tuning the forest for feature selection

| Argument                                     | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| [boot](./core_6.md#boot)                     | Number of trees. The default is 1000.                        |
| [m_grid](./core_6.md#m_grid)                 | Number of features for each split. Grid value = 1, where the number of features drawn at each new split of a tree is equal to the total number of features times the average of [m_min_share](./core_6.md#m_min_share) and [m_max_share](./core_6.md#m_max_share). |
| [alpha_reg_grid](./core_6.md#alpha_reg_grid) | Alpha regularization parameter for tree pruning. Grid value = 1, where the value of the alpha regularization parameter for tree pruning is specified in [alpha_reg_min](./core_6.md#alpha_reg_min). |
| [grid_n_min](./core_6.md#grid_n_min)         | Minimum leaf size. Grid value = 1, where the minimum leaf size is defined in [n_min_min](./core_6.md#n_min_min). |

### Feature importance statistics (FIS)

FIS are computed for each single feature in percent:
$$
FIS = \frac{MSE_{OOB}^{X^{k,rand}}}{MSE_{OOB}^{ref}}*100-100.
$$
$MSE_{OOB}$ indicates the objective function of the forest (without the penalty), obtained by comparing the observed and predicted out-of-bag outcomes. The numerator $MSE_{OOB}^{X^{k,rand}}$ is computed after randomizing the values of one single feature $X^k$ across observations, such that the feature is no longer informative, for $k=1,\cdots,K,$ where $K$ is the total number of features. The denominator $MSE_{OOB}^{ref}$ refers to the reference forest, without changing any feature values.  

### Grouping

The program creates groups of features and computes feature importance statistics for each group (FISG). The groups are put together by sorting the single features according to their FIS in ascending order and splitting them into approximately equally sized portions. FISG are obtained after randomizing simultaneously all features in a particular group.

The program then aggregates the groups and calculates feature importance statistics for each aggregated group (FISaG). The first aggregated group is the group with the lowest FISG, and the next ones always merge the group with the lowest FISG not yet assigned to an aggregated group to the previous one. The best performing group with the highest FISG is not used for aggregation. The program then discards all features, whose values of FIS, FISG and FISaG are jointly below the user-defined importance threshold [fs_rf_threshold](./core_6.md#fs_rf_threshold).

The following table presents the number of groups. It shows that the number of (aggregated) groups depends on the number of features.

### Relation between number of features and (aggregated) groups

| Number of features (M) | Number of groups | Number of aggregated groups |
| ---------------------- | ---------------- | --------------------------- |
| $M >=100$              | 20               | 19                          |
| $20<=M<100$            | 10               | 9                           |
| $10<=M<20$             | 5                | 4                           |
| $4<=M<10$              | 2                | 0                           |
| $M<4$                  | 0                | 0                           |

## Technical details on inference

### Tuning parameters for conducting inference

| Argument                                   | Description                                                  |
| ------------------------------------------ | ------------------------------------------------------------ |
| [cond_var_flag](./core_6.md#cond_var_flag) | *True* if variance estimation is based on a variance decomposition of weighted conditional  means $\hat{w}_i\mu_{Y \vert \hat{W}} (\hat{w}_i)$ and variances $\hat{w}_i\sigma^2_{Y \vert \hat{W}} (\hat{w}_i)$. False if variance estimation simply builds on the sum of variances of weighted outcomes. The default is *True*. |
| [knn_flag](./core_6.md#knn_flag)           | *True* if $\mu_{Y \vert \hat{W}} (\hat{w}_i)$ and $\sigma^2_{Y \vert \hat{W}} (\hat{w}_i)$ are estimated by the *k-NN* estimator, *False* if they are estimated by the Nadaraya-Watson kernel estimator. The default is *False*. Calling this argument requires [cond_var_flag](./core_6.md#cond_var_flag) to be *True*. |
| [knn_min_k](./core_6.md#knn_min_k)         | Minimum number of neighbors for *k-NN* estimation. The default is 10. |
| [knn_const](./core_6.md#knn_const)         | Multiplier in the asymptotic expansion formula defining the number of nearest neighbors for the *k-NN* estimator, as indicated below. The default is 1. |
| [nw_bandw](./core_6.md#nw_bandw)           | Multiplier in Silverman's optimal bandwidth rule for the Nadaraya-Watson estimator. The default is 1. |
| [nw_kern_flag](./core_6.md#nw_kern_flag)   | Type of kernel for the Nadaraya-Watson estimator. Epanechnikov kernel = 1, Gaussian kernel = 2. The default is 1. |
| [se_boot_ate](./core_6.md#se_boot_ate)     | Integer. Number of replications to estimate the bootstrap standard error of ATEs. Bootstrapping is only activated for more than 99 replications. The default is *False* (no bootstrapping). |
| [se_boot_gate](./core_6.md#se_boot_gate)   | Integer. Number of replications to estimate the bootstrap standard error of GATEs. Bootstrapping is only activated for more than 99 replications. The default is *False* (no bootstrapping). |
| [se_boot_iate](./core_6.md#se_boot_iate)   | Integer. Number of replications to estimate the bootstrap standard error of IATEs. Bootstrapping is only activated for more than 99 replications. The default is *False* (no bootstrapping). |
| [panel_data](./core_6.md#panel_data)       | *True* for panel data, otherwise *False*. The default is *False*. |
| [cluster_std](./core_6.md#cluster_std)     | *True* for clustered standard errors, otherwise *False*. The default is *False*. |

### Tuning the *k-NN* estimator

For *k-NN* estimation, an important tuning parameter for estimating the weighted conditional means $\hat{w}_i\mu_{Y \vert \hat{W}} (\hat{w}_i)$ and variances $\hat{w}_i\sigma^2_{Y \vert \hat{W}} (\hat{w}_i)$  is the number of nearest neighbors (*k*). By default, *k* is obtained by the following asymptotic expansion formula:
$$
k = \max\bigg(C, \Big\lfloor{2 \cdot \kappa \cdot N^{0.5}\Big\rceil\bigg)}.
$$

The constant $C$ stands for the minimum number of neighbors and is specified in [knn_min_k](./core_6.md#knn_min_k). The rounding operator $\lfloor\rceil$ rounds its input argument to the nearest integer, $\kappa$  is a multiplier defined in [knn_const](./core_6.md#knn_const), and $N$ denotes the number of observations. [Bodory, Camponovo, Huber, and Lechner (2020)](https://www.tandfonline.com/doi/abs/10.1080/07350015.2018.1476247?journalCode=ubes20) provide further information on the implementation of the *k-NN* estimator for weights-based inference.

### Tuning the Nadaraya-Watson estimator

The Nadaraya-Watson estimator can be applied with two different kernels, either with the Epanechnikov kernel or the Gaussian kernel (to be specified in [nw_kern_flag](./core_6.md#nw_kern_flag)). The Silverman rule is used for selecting the kernel bandwidth $h$:
$$
h = C_h \cdot N^{-\frac{1}{5}} \cdot \min \bigg( \sigma_{\hat{W}},
                    \frac{IQR_{\hat{W}}}{1.349}
                    \bigg).
$$
The kernel-specific constant $C_h$ can be varied in [nw_bandw](./core_6.md#nw_bandw). The number of observations is denoted by $N$, and the statistics $\sigma_{\hat{W}}$ and $IQR_{\hat{W}}$ stand for the standard deviation and interquartile range of the estimated weights $\hat{W}$, respectively.

## Technical details on *k-Means* clustering

The following table presents the default for the number of *k-Means* clusters specified in [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups).

### *k-Means* clusters

| Number of observations in prediction data ($N$) | Number of clusters                                  |
| ----------------------------------------------- | --------------------------------------------------- |
| $N < 10^4$                                      | $5$                                                 |
| $N > 10^5$                                      | $10$                                                |
| $10^4 >= N  <= 10^5$                            | $5 + \Big\lfloor{\frac{N}{2 \cdot 10^4}\Big\rceil}$ |

## Technical details on marginal treatment effects

### Random subsets of the data for AMTEs

AMTEs are based on random subsets of the data if $\frac{N}{N_{EP}}>10$, where $N$ denotes the number of observations in the prediction data set and $N_{EP}$ stand for the number of evaluation points specified by [gmate_no_evaluation_points](./core_6.md#gmate_no_evaluation_points). The size of the random subsets $N_{SS}$ is defined by:
$$
N_{SS} = \frac{C_{SS}}{N_{EP}}.
$$


You can specify the constant $C_{SS} \in (0,1]$ by assigning it to the input argument [gmate_sample_share](./core_6.md#gmate_sample_share). By default, $C_{SS}$ is defined as a function of $N$:

| $N$ (number of observations for the prediction data) | $C_{SS}$ (value of the constant)            |
| ---------------------------------------------------- | ------------------------------------------- |
| $N<10^3$                                             | 1                                           |
| $N>=10^3$                                            | $\frac{10^3 + (N-10^{3})^{\frac{3}{4}}}{N}$ |
