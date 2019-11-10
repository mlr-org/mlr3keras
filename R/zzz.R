#' @import data.table
#' @import keras
#' @import paradox
#' @import mlr3misc
#' @import checkmate
#' @importFrom R6 R6Class
#' @importFrom mlr3 mlr_learners LearnerClassif LearnerRegr assert_task
#' @importFrom stats setNames
#' @description
#' More learners are available in the `mlr3learners` repository on Github (\url{https://github.com/mlr3learners}).
#' There also is a wiki page listing all currently available custom learners (\url{https://github.com/mlr-org/mlr3learners/wiki/Extra-Learners}).
#' A guide on how to create custom learners is covered in the book: \url{https://mlr3book.mlr-org.com}.
#' Feel invited to contribute a missing learner to the \CRANpkg{mlr3} ecosystem!
"_PACKAGE"


register_mlr3 = function() {
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  x$add("classif.kerasff", LearnerClassifKerasFF)
  x$add("classif.keras", LearnerClassifKeras)
}

.onLoad = function(libname, pkgname) {
  # nocov start
  register_mlr3()
  setHook(packageEvent("mlr3", "onLoad"), function(...) register_mlr3(), action = "append")
} # nocov end

.onUnload = function(libpath) {
  # nocov start
  event = packageEvent("mlr3", "onLoad")
  hooks = getHook(event)
  pkgname = vapply(hooks, function(x) environment(x)$pkgname, NA_character_)
  setHook(event, hooks[pkgname != "mlr3keras"], action = "replace")
} # nocov end
