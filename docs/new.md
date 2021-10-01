# Release Updates

## Version 0.1.1

### Bug fixes
- Minor bug fixes concerning [_with_output](./core_6.md#_with_output), [_smaller_sample](./core_6.md#_smaller_sample), (A,AM)GATE/IATE-ATE plots, and the sampling weights.

### What's new
- k-Means cluster indicator for the IATEs saved in file with IATE predictions.
- Evaluation points of GATE figures are included in the output csv-file.
- Exception raised if choice based sampling is activated and there is no treatment information in predictions file.
- New defaults for [random_thresholds](./core_6.md#random_thresholds); by default the value is set to 20 percent of the square-root of the number of training observations. 
- Stabilizing ``ray`` by deleting references to object store and tasks
- The function ``ModifiedCausalForest()`` returns now ATE, standard error (SE) of the ATE, GATE, SE of the GATE, IATE, SE of the IATE, and the name of the file with the predictions.


## Version 0.1.0

### Bug fixes
 - Bug fix for dealing with missings.
 - Bug fixes for problems computing treatment effects for treatment populations.
 - Bug fixes for the use of panel data and clustering.

### What's New
- [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups) can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment specific statistics to detect common support issues.
- Improved statistics for common support analysis.

### Experimental
- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
