# mlr3keras
An extension for `mlr3` to enable using various `keras` models as learners.

[![Build Status](https://travis-ci.org/mlr-org/mlr3keras.svg?branch=master)](https://travis-ci.org/mlr-org/mlr3keras)
<!--
[![Build status](https://ci.appveyor.com/api/projects/status/m2tuhgdxo8is0nv0?svg=true)](https://ci.appveyor.com/project/mlr-org/mlr3keras)
[![CRAN](https://www.r-pkg.org/badges/version/mlr3)](https://cran.r-project.org/package=mlr3keras)
[![codecov](https://codecov.io/gh/mlr-org/mlr3/branch/master/graph/badge.svg)](https://codecov.io/gh/mlr-org/mlr3)
-->

## Status

`mlr3keras` is in very early stages of development, and currently under development.

Functionality is therefore experimental and we do not guarantee *correctness*, *safety* or *stability*.

It builds on top of the (awesome) R packages `reticulate`, `tensorflow` and `keras`.

Comments, discussion and issues/bug reports and PR's are **highly** appreciated.

If you want to **contribute**, please propose / discuss adding functionality in an issue in order to avoid unnecessary or duplicate work.

## Installation:

```r
# Install from GitHub
remotes::install_github("mlr-org/mlr3keras")
```

**Troubleshooting:**

If you encounter problems using the correct python versions, see [here](https://rstudio.github.io/reticulate/articles/versions.html).

`mlr3keras` is currently tested and works using the python packages `keras (2.3.1)` and `tensorflow (2.0.0)`.

## Usage

`mlr3keras` currently exposes three `Learners` for regression and classification respectively.

* **(Regr|Classif)Keras**:   A generic wrapper that allows to supply a custom keras architecture as
                         a hyperparameter.
* **(Regr|Classif)KerasFF**: A fully-connected feed-forward Neural Network with entity embeddings
* **(Regr|Classif)TabNet**: An implementation of `TabNet` (c.f. Sercan, A. and Pfister, T. (2019): TabNet).

Learners can be used for `training` and `prediction` as follows:

```r
  # Instantiate Learner
  lrn = LearnerClassifKerasFF$new()

  # Set Learner Hyperparams
  lrn$param_set$values$epochs = 50
  lrn$param_set$values$layer_units = 12

  # Train and Predict
  lrn$train(mlr_tasks$get("iris"))
  lrn$predict(mlr_tasks$get("iris"))
```

The [vignette](https://github.com/mlr-org/mlr3keras/blob/master/vignettes/fist_steps.Rmd) has some examples on how to use some of the functionality introduces in `mlr3keras`.

## Design

This package's purpose for now is to understand the design-decisions required to make `keras` \ `tensorflow` work
with `mlr3` **and** flexible enough for users.

Several design decisions are not made yet, so input is highly appreciated.


## Installation

```r
remotes::install_github("mlr-org/mlr3keras")
```

## Resources

* There is a [book](https://mlr3book.mlr-org.com/) on `mlr3` and its ecosystem, but it is still unfinished.
* [Reference Manual](https://mlr3.mlr-org.com/reference/)
* [Extension packages](https://github.com/mlr-org/mlr3/wiki/Extension-Packages).
* [useR2019 talks](https://github.com/mlr-org/mlr-outreach/tree/master/2019_useR)
