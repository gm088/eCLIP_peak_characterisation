## Description

Characterisation of subgroups of genomic regions based on underlying features.
Background and goal:

* Features are transformed and scaled using custom transformation functions, and scikit-learn functions.
* Transformed and scaled data are subjected to [UMAP](https://umap-learn.readthedocs.io/).
* The UMAP projection of the data points are then used to define clusters.

## Overview

<img src="./outputs/eCLIP_char.png" alt="schematic" height="300" width="1000"/>
