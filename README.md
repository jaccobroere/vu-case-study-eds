# Repository: Cancer subtype classification using shrinking methods on gene expression data
### Authors: Jacco Broere, Caspar Hentenaar, Bas Willemsen

This repository contains the code to reproduce the results in our report.
For this report we implemented general SPCA and an SPCA variant for gene expression data as described in [Zou et al. (2006)](https://www-jstor-org.vu-nl.idm.oclc.org/stable/27594179). Using these transformations classifiers were built to accurately classify different cancer subtypes on gene expression datasets. For classification a logistic regression and a tree boosting method using LightGBM were employed, hyperparameter tuning was done using the Optuna framework.

The two implemented SPCA variants are made available in `spca_classes.py` and can be used as SKLearn transformers, in a pipeline, or standalone.
Note that general SPCA, or `EnetSPCA` as implemented here is hardly runnable for datasets with more than 3000 features $(p)$ as computational complexity increases in $O(p^3)$

Gene expression data is obtained from [this](https://github.com/kivancguckiran/microarray-data) repo.
The written report is available [here](https://github.com/jaccobroere/vu-case-study-eds/blob/main/paper/Case_Study_EDS.pdf)

This project was done during the course: Case Study for Econometrics and Data Science (2023) at the VU.
