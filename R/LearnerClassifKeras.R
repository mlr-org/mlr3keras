#' @title Keras Neural Network with custom architecture (Classification)
#'
#' @usage NULL
#' @aliases mlr_learners_classif.keras
#' @format [R6::R6Class()] inheriting from [mlr3::LearnerClassif].
#'
#' @section Construction:
#' ```
#' LearnerClassifKeras$new()
#' mlr3::mlr_learners$get("classif.keras")
#' mlr3::lrn("classif.keras")
#' ```
#'
#' @description
#' Neural Network using Keras and Tensorflow.
#' This learner allows for supplying a custom architecture.
#' Calls [keras::fit] from package \CRANpkg{keras}.
#'
#' Parameters:\cr
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `model`: A compiled keras model.
#' * `class_weight`: needs to be a named list of class-weights
#'   for the dierent classes numbered from 0 to c-1 (for c classes).
#'   ```
#'   Example:
#'   wts = c(0.5, 1)
#'   setNames(as.list(wts), seq_len(length(wts)) - 1)
#'   ```
#' * `callbacks`: A list of keras callbacks.
#'   See `?callbacks`.
#'
#'
#' @template seealso_learner
#' @templateVar learner_name classif.keras
#' @examples
#'  # Define a model
#'  library(keras)
#'  model = keras_model_sequential() %>%
#'  layer_dense(units = 12L, input_shape = 4L, activation = "relu") %>%
#'  layer_dense(units = 12L, activation = "relu") %>%
#'  layer_dense(units = 3L, activation = "softmax") %>%
#'    compile(optimizer = optimizer_sgd(),
#'      loss = "categorical_crossentropy",
#'      metrics = "accuracy")
#'  # Create the learner
#'  learner = LearnerClassifKeras$new()
#'  learner$param_set$values$model = model
#'  learner$train(mlr3::mlr_tasks$get("iris"))
#' @export
LearnerClassifKeras = R6::R6Class("LearnerClassifKeras", inherit = LearnerClassif,
  public = list(
    architecture = NULL,
    initialize = function(architecture = KerasArchitectureCustomModel$new()) {
      self$architecture = assert_class(architecture, "KerasArchitecture")
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamUty$new("model", tags = c("train")),
        ParamUty$new("class_weight", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict"))
      ))
      ps$values = list(epochs = 30L, callbacks = list(), validation_split = 1/3, batch_size = 128L)
      ps = ParamSetCollection$new(list(ps, self$architecture$param_set))
      super$initialize(
        id = "classif.keras",
        param_set = ps,
        predict_types = c("response", "prob"),
        feature_types = c("integer", "numeric"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.keras"
      )
    },

    train_internal = function(task) {
      pars = self$param_set$get_values(tags = "train")

      model = self$architecture$get_model(task, pars)
      # Custom transformation depending on the model.
      # Could be generalized at some point.
      features = task$data(cols = task$feature_names)
      
      x = self$architecture$transforms$x(features, pars)
      y = self$architecture$transforms$y(task, pars, model)

      history = invoke(keras::fit,
        object = model,
        x = x,
        y = y,
        epochs = as.integer(pars$epochs),
        class_weight = pars$class_weight,
        batch_size = pars$batch_size,
        validation_split = pars$validation_split,
        verbose = pars$verbose,
        callbacks = pars$callbacks)
      return(list(model = model, history = history, class_names = task$class_names))
    },

    predict_internal = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      
      features = task$data(cols = task$feature_names)
      newdata = self$architecture$transforms$x(features, pars)

      if (inherits(self$model$model, "keras.engine.sequential.Sequential")) {
        if (self$predict_type == "response") {
          p = invoke(keras::predict_classes, self$model$model, x = newdata, .args = pars)
          p = factor(self$model$class_names[p + 1])
          PredictionClassif$new(task = task, response = drop(p))
        } else if (self$predict_type == "prob") {
          prob = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pars)
          if (ncol(prob) == 1L) prob = cbind(1-prob, prob)
          colnames(prob) = task$class_names
          PredictionClassif$new(task = task, prob = prob)
        }
      } else {
        p = invoke(self$model$model$predict, x = newdata, .args = pars)
        if (self$predict_type == "response") {
          p = factor(self$model$class_names[apply(p, 1, which.max)])
          PredictionClassif$new(task = task, response = drop(p))
        } else if (self$predict_type == "prob") {
          if (ncol(p) == 1L) p = cbind(1-p, p)
          colnames(p) = task$class_names
          PredictionClassif$new(task = task, prob = p)
        }
      }
    }
  )
)
