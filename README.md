# mlr3keras
An extension for `mlr3` to enable using various `keras` models as learners.

[![tic](https://github.com/mlr-org/mlr3keras/workflows/tic/badge.svg?branch=master)](https://github.com/mlr-org/mlr3keras/actions)

## Status

`mlr3keras` is in very early stages, and currently under development.
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

`mlr3keras` is currently tested and works using the python packages `keras (2.4)` and `tensorflow (2.3.1)`.


### Setting up mlr3keras with anaconda
One possible workflow for working with mlr3keras is described below.
While (1.) and (2.) are one-time setup steps, (3.) now has to be called everytime mlr3keras is loaded.

> *Note from the author:*
> The workflow described below is something that works for me personally, as I have to switch between versions and projects often. 
> It is described for the user, as I personally find it useful. It assumes, the R packages `keras`, `tensorflow` and `reticulate` are installed.
> In order to load mlr3keras I now have to execute an additional one additional line (see 3.), but version management is heavily simplified.

1. Install Miniconda

```r
# Execute and restart R afterwards
reticulate::install_miniconda()
```

2. Install a mlr3keras conda environment together with `keras` and `tensorflow`

```r
# Execute and restart R afterwards
reticulate::conda_create(
  envname = "mlr3keras",
  packages = "pandas",
  python_version = "3.8"
)
keras::install_keras("conda", tensorflow="2.3.1", envname="mlr3keras")
```

3. Loading the mlr3keras package

```r
reticulate::use_condaenv("mlr3keras")
library(mlr3keras)
```


## Usage

`mlr3keras` currently exposes three `Learners` for regression and classification respectively.

| Learner | Details | Reference |
|---|---|---|
| [regr/classif.keras]()   | A generic wrapper that allows to supply a custom keras architecture as a hyperparameter.| --  |
| [regr/classif.kerasFF]() | A fully-connected feed-forward Neural Network with entity embeddings                    |  Guo et al. (2016) Entity Embeddings for Categorical Variables |
| [regr/classif.tabNet]()  | An implementation of `TabNet`                      | Sercan, A. and Pfister, T. (2019): TabNet |
| [regr/classif.smlp]()    | Shaped MLP as described in Configuration Space 1*  | Zimmer, L. et al. (2020): Auto PyTorch Tabular |
| [regr/classif.smlp2]()   | Shaped MLP as described in Configuration Space 2* | Zimmer, L. et al. (2020): Auto PyTorch Tabular |

* with some slight changes, namely no Shake-Shake, Shake-Drop, Mixup Training.
and added Entity Embeddings for categorical variables.

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

The [vignette](https://github.com/mlr-org/mlr3keras/blob/master/vignettes/mlr3keras.Rmd) has some examples on how to use some of the functionality introduces in `mlr3keras`.

## Design

This package's purpose for now is to understand the design-decisions required to make `keras` \ `tensorflow` work
with `mlr3` **and** flexible enough for users.

Several design decisions are not made yet, so input is highly appreciated.


### Current Design and Scope

**Design**

The goal of the project is to expose keras *models* as mlr3 learners.
A keras model in this context should be understood as the combination of

  - model architecture:
    ```r
    keras_model_sequential() %>%
    ... %>%
    layer_activation(...)
    ```

  - training procedure:
    ```r
    model$compile(...)
    model$fit(...)
    ```

  - All hyperparameters that control the steps:
    - architecture hyperparams (dropout, neurons / filters, activation functions, ...)
    - optimizer choice and hyperparams (learning rate, ...)
    - fit hyperparams (epochs, callbacks, ...)

Some important caveats:
- Architectures are often data-dependent, e.g. require correct number of input / output neurons.
  As a result, in `mlr3keras`, the architecture is a function of the incoming training data.
  In `mlr3keras`, this is abstracted via `KerasArchitecture`:
  See `KerasArchitectureFF` for an example.
  This Architecture is initialized with a `build_arch_fun` which given the `task` and a
  set of hyperparameters constructs & compiles the architecture.

- Depending on the architecture, different data-formats are required for `x` (features) and `y` (target)
  (e.g. a matrix for a feed-forward NN, a list of features if we use embeddings, ...)
  To accomodate this, each architecture comes with an `x_transform` and a `y_transform`
  method, which are called on the features and target respectively before passing those on to
  `fit(...)`.


**Scope**
The current scope for `mlr3keras` is to support deep learning on different kinds of **tabular** data. In the future,
we aim to extend this to other data modalities, but as of yet, work on this has not started.


In an initial version, we aim to support two types of models:

- Pre-defined architectures:
  In many cases, we just want to try out and tune architectures that have already been successfully
  used in other contexts (LeNet, ResNet, TabNet). We aim to implement / make those accessible
  for simplified tuning and fast iteration.
  Example: `LearnerClassifTabNet$new()`.

- Fully custom architectures:
  Some operations require completely new architectures. We aim to allow users to supply custom architectures
  and tune hyperparameters of those. This can be done via `KerasArchitectureCustom` by providing a
  function that builds the model given a `Task` and a set of hyperparameters.

All architectures can be parametrized and tuned using the `mlr3tuning` library.




Open Issues:
- Custom Optimizers / Losses / Activations
  - As `keras` and it's ecosystem is constantly growing, the interface needs to be flexible and
    highly adaptable. We try to solve this using a `reflections` mechanism:
    `keras_reflections` that stores possible values for exchangable parts of an architecture.
    New methods can now be added by adding to the respective reflection.
- More data types:
  - Currently `mlr3` does not support features such as images / audio / ..., therefore `mlr3keras` is
    not yet applicable for image classification and other related tasks. We aim to make this possible in the future!
    A minor road block here is to find a way to not read images to memory in R but directly load from disk
    to avoid additional overhead.
  - It is unclear how efficient data preprocessing for images etc. can be paired with `mlr3pipelines`,
    yet we hope this can be solved at some point in the future.
- More task types:
  - Currently `mlr3` focusses heavily on standard classification and regression. Many Deep Learning tasks
    require slight extensions (image annotation, bounding boxes, object detection, ... ) of those existing
    data containers (`Task`, `Prediction`, ...).


## Installation

```r
remotes::install_github("mlr-org/mlr3keras")
```

## Resources

* There is a [book](https://mlr3book.mlr-org.com/) on `mlr3` and its ecosystem, but it is still unfinished.
* [Reference Manual](https://mlr3.mlr-org.com/reference/)
* [Extension packages](https://github.com/mlr-org/mlr3/wiki/Extension-Packages).
* [useR2019 talks](https://github.com/mlr-org/mlr-outreach/tree/master/2019_useR)
