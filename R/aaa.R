#' @title Reflections mechanism for keras
#'
#' @details
#' Used to store / extend available hyperparameter levels for options used throughout keras,
#' e.g. the available 'loss' for a given Learner.
#'
#' @format [environment].
#' @export
keras_reflections = new.env(parent = emptyenv()) # nocov