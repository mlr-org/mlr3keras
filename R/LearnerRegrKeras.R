#' @title Keras Neural Network with custom architecture (Regression)
#'
#' @usage NULL
#' @aliases mlr_learners_regr.keras
#' @format [R6::R6Class()] inheriting from [mlr3::LearnerRegr].
#'
#' @section Construction:
#' ```
#' LearnerRegrKeras$new()
#' mlr3::mlr_learners$get("regr.keras")
#' mlr3::lrn("regr.keras")
#' ```
#'
#' @description
#' Neural Network using Keras and Tensorflow.
#' This learner allows for supplying a custom architecture.
#' Calls [keras::fit()] from package \CRANpkg{keras}.
#'
#' Parameters:
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `model`: A compiled keras model.
#' * `class_weight`: needs to be a named list of class-weights
#'   for the different classes numbered from 0 to c-1 (for c classes).
#'   ```
#'   Example:
#'   wts = c(0.5, 1)
#'   setNames(as.list(wts), seq_len(length(wts)) - 1)
#'   ```
#' * `callbacks`: A list of keras callbacks.
#'   See `?callbacks`.
#'
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name regr.keras
#' @examples
#'  # Define a model
#'  library(keras)
#'  model = keras_model_sequential() %>%
#'  layer_dense(units = 12L, input_shape = 10L, activation = "relu") %>%
#'  layer_dense(units = 12L, activation = "relu") %>%
#'  layer_dense(units = 1L, activation = "linear") %>%
#'    compile(optimizer = optimizer_sgd(),
#'      loss = "mean_squared_error",
#'      metrics = "mean_squared_error")
#'  # Create the learner
#'  learner = LearnerRegrKeras$new()
#'  learner$param_set$values$model = model
#'  learner$train(mlr3::mlr_tasks$get("mtcars"))
#' @export
LearnerRegrKeras = R6::R6Class("LearnerRegrKeras",
  inherit = LearnerRegr,
  public = list(
    architecture = NULL,
    initialize = function(
        id = "regr.keras",
        predict_types = c("response"),
        feature_types = c("integer", "numeric"),
        properties = character(),
        packages = "keras",
        man = "mlr3keras::mlr_learners_regr.keras",
        architecture = KerasArchitectureCustomModel$new()
      ) {
      self$architecture = assert_class(architecture, "KerasArchitecture")
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 100L, lower = 0L, tags = "train"),
        ParamUty$new("model", tags = c("train")),
        ParamUty$new("class_weight", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamLgl$new("low_memory", default = FALSE, tags = c("train")),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict"))
      ))
      ps$values = list(epochs = 100L, callbacks = list(), validation_split = 1/3, batch_size = 128L, low_memory = FALSE, verbose = 0L)
      ps = ParamSetCollection$new(list(ps, self$architecture$param_set))
      super$initialize(
        id = assert_character(id, len = 1),
        param_set = ps,
        predict_types = assert_character(predict_types),
        feature_types = assert_character(feature_types),
        properties = assert_character(properties),
        packages = assert_character(packages),
        man = assert_character(man)
      )
      # Set y_transform
      self$architecture$set_transform("y", function(target, pars) {as.numeric(target)})
    },

    train_internal = function(task) {
      pars = self$param_set$get_values(tags = "train")
      model = self$architecture$get_model(task, pars)
      # Custom transformation depending on the model.
      # Could be generalized at some point.
      features = task$data(cols = task$feature_names)
      target = task$data(cols = task$target_names)[[task$target_names]]

      if (!pars$low_memory) {
        x = self$architecture$transforms$x(features, pars)
        y = self$architecture$transforms$y(target, pars)
        history = invoke(keras::fit,
          object = model,
          x = x,
          y = y,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          batch_size = as.integer(pars$batch_size),
          validation_split = pars$validation_split,
          verbose = as.integer(pars$verbose),
          callbacks = pars$callbacks)
      } else {

        generators = make_train_valid_generators(
          task = task,
          x_transform = function(features) self$architecture$transforms$x(features, pars = pars),
          y_transform = function(target) self$architecture$transforms$y(target, pars),
          validation_split = pars$validation_split,
          batch_size = pars$batch_size)

        history = invoke(keras::fit_generator,
          object = model,
          generator = generators$train_gen,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          steps_per_epoch = generators$train_steps,
          validation_data = generators$valid_gen,
          validation_steps = generators$valid_steps,
          verbose = pars$verbose,
          callbacks = pars$callbacks)
      }
      return(list(model = model, history = history))
    },

    predict_internal = function(task) {
      pars = self$param_set$get_values(tags = "predict")

      features = task$data(cols = task$feature_names)
      newdata = self$architecture$transforms$x(features, pars)
      pars = pars[intersect(names(pars), self$keras_predict_pars)]

      if (self$predict_type == "response") {
        p = invoke(self$model$model$predict, x = newdata, .args = pars)
        PredictionRegr$new(task = task, response = drop(p))
      }
    },
    save = function(filepath) {
      assert_path_for_output(filepath)
      if (is.null(self$model)) stop("Model must be trained before saving")
      keras::save_model_hdf5(self$model$model, filepath)
    },
    load_model_from_file = function(filepath) {
      assert_file_exists(filepath)
      self$state$model$model = keras::load_model_hdf5(filepath)
    },
    plot = function() {
      if (is.null(self$model)) stop("Model must be trained before saving")
      plot(self$model$history)
    },
    keras_predict_pars = c("batch_size", "verbose")
  )
)
