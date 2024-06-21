# SIAL: Statistical Inference After Learning


## Overview
 
`sial` is a python package for conducting statistical inference on feature importance in machine learning. It is written on the basis of [`scikit-learn`](https://scikit-learn.org/stable/index.html). Hence, in principle, any regressor/classifier learned by `scikit-learn` can be inferred by `sial`.

`sial` provides two types of `Inferer` for making inferences: `CIT` for conditional independence test and `RIT` for risk invariance test. `CIT` can implement the holdout randomization test (HRT; Tansey et al., 2022), residual permutation test (RPT; Huang, 2024), and conditional predictive impact (CPI; Watson & Wright, 2021). On the other hand, `RIT` can conduct leave-one-covariate-out (LOCO; Lei et al., 2018) and plug-in estimation (PIE; Williamson et al., 2023). In addition, `sial` also constructs a `Crosser` class. With the help of `Crosser`, cross-fitting and $p$-value combination can be easily implemented to improve the stability and even statistical power of the CITs and RITs.

## Installation
`sial` can be installed via `pip`:
```
pip install sial-pkg
```

## Tutorial

Three Jupyter notebooks are available to learn the use of `sial`:

1. [Conditional Independence Tests](https://github.com/aipsylab/sial/blob/main/tutorial/tutorial_1_cit.ipynb)
2. [Risk Invariance Tests](https://github.com/aipsylab/sial/blob/main/tutorial/tutorial_2_rit.ipynb)
3. [Cross-Fitting and $p$-Value Combination](https://github.com/aipsylab/sial/blob/main/tutorial/tutorial_3_crosser.ipynb)

These notebooks assume that readers have experiences in using `scikit-learn`.

## References

Huang, P.-H. (2024). Residual Permutation Tests for Feature Importance in Machine Learning. [Manuscript submitted for publication].

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523), 1094–1111. doi: 10.1080/01621459.2017.1307116

Tansey, W., Veitch, V., Zhang, H., Rabadan, R., & Blei, D. M. (2022). The holdout randomization test for feature selection in black box models. Journal of Computational and Graphical Statistics, 31(1), 151–162. doi: 10.1080/10618600.2021.1923520

Watson, D., & Wright, M. (2021). Testing conditional independence in supervised learning algorithms. Machine Learning, 110, 1-23. doi: 10.1007/s10994-021-06030-6

Williamson, B. D., Gilbert, P. B., Simon, N. R., & Carone, M. (2023). A general framework for inference on algorithm-agnostic variable importance. Journal of the American Statistical Association, 118(543), 1645–1658. doi: 10.1080/01621459.2021.2003200