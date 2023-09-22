# Python API


```python
ModifiedCausalForest(cf_boot=cf_boot,
                       lc_yes=lc_yes,
                       p_atet=p_atet,
                       p_gatet_flag=p_gatet_flag,
                       var_d_name=var_d_name,
                       var_x_name_always_in_ord=var_x_name_always_in_ord,
                       var_x_name_always_in_unord=x_name_always_in_unord,
                       var_x_name_ord=x_name_ord,
                       var_x_name_remain_ord=var_x_name_remain_ord,
                       var_x_name_remain_unord=var_x_name_remain_unord,
                       var_x_name_unord=var_x_name_unord,
                       var_y_name=var_y_name,
                       var_z_name_list=var_z_name_list,
                       var_z_name_ord=var_z_name_ord,
                       var_z_name_unord=var_z_name_unord,)
```


## Variable Names

- <a id="var_d_name"><strong>var_d_name</strong></a>
	* Name of treatment. Pass over as string.

- <a id="var_x_name_always_in_ord"><strong>var_x_name_always_in_ord</strong></a>
  * Names of ordered variables, which should be always included when deciding on the next split. Pass over as a list of strings.

- <a id="var_x_name_always_in_unord"><strong>var_x_name_always_in_unord</strong></a>
  * Names of unordered variables, which should be always included when deciding on the next split. Pass over as a list of strings.

- <a id="var_x_name_ord"><strong>var_x_name_ord</strong></a>
  *	Names of ordered features. Pass over as a list of strings.

- <a id="var_x_name_unord"><strong>var_x_name_unord</strong></a>
  * Names of unordered features. Pass over as a list of strings.

- <a id="var_x_name_remain_ord"><strong>var_x_name_remain_ord</strong></a>
  *	Names of unordered features. Pass over as a list of strings.

- <a id="var_x_name_remain_unord"><strong>var_x_name_remain_unord</strong></a>
  * Names of variables excluded from preliminary feature selection. The default is None. Pass over as a list of strings.

- <a id="var_y_name"><strong>var_y_name</strong></a>
  * Name(s) of outcome variable(s). Pass over as a list of strings.

- <a id="var_z_name_list"><strong>var_z_name_list</strong></a>
  * Names of variables for heterogeneity analysis relevant to group treatment effects. Pass over as a list of strings.

- <a id="var_z_name_ord"><strong>var_z_name_ord</strong></a>
  * Names of ordered variables to define policy-relevant heterogeneity. Pass over as a list of strings.

- <a id="var_z_name_unord"><strong>var_z_name_unord</strong></a>
  * Names of unordered variables to define policy-relevant heterogeneity. Pass over as a list of strings.

## Parameters

- <a id="cf_boot"><strong>cf_boot</strong></a>
  * Number of subsampling replications. Default is 1000.

- <a id="lc_yes"><strong>lc_yes</strong></a>
  * If True, local centering is deployed. The default is True.

- <a id="p_atet"><strong>p_atet</strong></a>
  * If True, the average effects are estimated by treatment group. The default is False.

- <a id="p_gatet"><strong>p_gatet</strong></a>
  * If set to True, GATEs and GATETs are computed. The default is False.
