#' @import data.table
#' @import keras
#' @import paradox
#' @import mlr3misc
#' @import mlr3
#' @import checkmate
#' @importFrom R6 R6Class
#' @importFrom stats setNames
#' @importFrom tensorflow tf
#' @description
#' A package that connects mlr3 to keras.
"_PACKAGE"

#' re-export k and tf functions
#' @noRd
k  = utils::getFromNamespace("keras", "keras")
tf = utils::getFromNamespace("tf", "tensorflow")


#' @title Reflections mechanism for keras
#'
#' @details
#' Used to store / extend available hyperparameter levels for options used throughout keras,
#' e.g. the available 'loss' for a given Learner.
#'
#' @format [environment].
#' @export
keras_reflections = new.env(parent = emptyenv()) # nocov

register_mlr3 = function() { # nocov start
  # Add Learners
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  x$add("classif.keras", LearnerClassifKeras)
  x$add("regr.keras", LearnerRegrKeras)
  x$add("classif.kerasff", LearnerClassifKerasFF)
  x$add("regr.kerasff", LearnerRegrKerasFF)
  x$add("classif.tabnet", LearnerClassifTabNet)
  x$add("regr.tabnet", LearnerRegrTabNet)
  x$add("classif.smlp", LearnerClassifShapedMLP)
  x$add("regr.smlp", LearnerRegrShapedMLP)
  x$add("classif.smlp2", LearnerClassifShapedMLP2)
  x$add("regr.smlp2", LearnerRegrShapedMLP2)
  x$add("classif.kerascnn", LearnerClassifKerasCNN)

  local({
    keras_reflections$loss = list(
        classif = c("binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy"),
        regr = c("cosine_proximity", "cosine_similarity", "mean_absolute_error", "mean_squared_error",
          "poison", "squared_hinge", "mean_squared_logarithmic_error")
      )
  })
}

.onLoad = function(libname, pkgname) {
  reticulate::configure_environment(pkgname)
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  register_mlr3()
  setHook(packageEvent("mlr3", "onLoad"), function(...) register_mlr3(), action = "append")
}

.onUnload = function(libpath) {
  event = packageEvent("mlr3", "onLoad")
  hooks = getHook(event)
  pkgname = vapply(hooks, function(x) environment(x)$pkgname, NA_character_)
  setHook(event, hooks[pkgname != "mlr3keras"], action = "replace")
}

# silence R CMD check for callbacks:
utils::globalVariables("model") # nocov end
