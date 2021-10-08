# mlr3keras 0.1.3

Now works with `keras > 2.3` and `tensorflow > 2.3`.

## Generators
* Re-worked generators, now use a python implementation.
  This required re-designing generator constructors and so on.

## Learners
* Shaped MLP 1 & 2 Learners for Regression and Classification
* mlr3keras can now deal with images via the new `KerasCNN` learner.
* Added a deep-wide architecture inspired by Ericson et al., 2020 AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.

## Bugfixes
* TabNet now defaults to `num_layers = 1L` for `stack = TRUE`.


# mlr3keras 0.1.2

## Learners
* TabNet and FeedForward can now deal with factor / ordered / character features
* FeedForward Keras Models now default to "embeddings" for factor features

# mlr3keras 0.1.1

## General
* KerasArchitecture:
  Introduced new abstraction for architectures.
  This should rarely be visible to users but makes stuff easier to extend.

## Learners
* Add regression learner for a custom model
* Add regression learner for parametrized feedforward model
* Add stacked and unstacked tabnet classification and regression learner


# mlr3keras 0.1.0

* Initial prototype.
