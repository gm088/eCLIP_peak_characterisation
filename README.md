## Description

Characterisation of subgroups of genomic regions based on underlying features.

* Features are transformed and scaled using custom transformation functions, and scikit-learn functions.
* Transformed and scaled data are subjected to [UMAP](https://umap-learn.readthedocs.io/).
* The UMAP projection of the data points are then used to define clusters.

## Overview

feature distribution after transformation and scaling
[feature distribution after transformation and scaling](./outputs/feature_posttransform_scaling.pdf)

feature distribution on UMAP projection
[feature distribution on UMAP projection](./outputs/feature_vals.pdf)

define 4 clusters
[define 4 clusters](./outputs/clusters.pdf)

feature distribution of example cluster
[feature distribution of example cluster](./outputs/feature_dist_clus_3.pdf)

