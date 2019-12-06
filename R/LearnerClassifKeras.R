#' @title Keras Neural Network with custom architecture
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
    transforms = list(),
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamUty$new("model", tags = c("train")),
        ParamUty$new("class_weight", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict")),
        ParamLgl$new("low_memory", default=FALSE, tags = c("train", "predict"))
      ))
      ps$values = list(epochs = 30L, callbacks = list(),
        validation_split = 1/3, batch_size = 128L, low_memory=FALSE)

      super$initialize(
        id = "classif.keras",
        param_set = ps,
        predict_types = c("response", "prob"),
        feature_types = c("integer", "numeric"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.keras"
      )

      x_transform = function(features, pars) {
        as.matrix(features)
      }
      y_transform = function(target, pars) {
        y = to_categorical(as.integer(target) - 1)
        if (pars$model$loss == "binary_crossentropy") y = y[, 1, drop = FALSE]
        return(y)
      }
      self$set_transform("x", x_transform)
      self$set_transform("y", y_transform)
    },

    train_internal = function(task) {
      pars = self$param_set$get_values(tags = "train")
      assert_class(pars$model, "keras.engine.training.Model")

      if(!is.null(pars$low_memory) && pars$low_memory) {
        gen <- make_data_generator(
          task = task,
          batch_size = batch_size,
          x_transform = function(x) {self$transforms$x(x, pars)},
          y_transform = function(y) {self$transforms$y(y, pars)}
        )
        
        history = invoke(keras::fit_generator,
                         object = pars$model,
                         generator = gen,
                         epochs = as.integer(pars$epochs),
                         class_weight = pars$class_weight, 
                         steps_per_epoch = floor(task$nrow / pars$batch_size),
                         # not implemented: validation_data can be set up using validation split and mlr
                         verbose = pars$verbose,
                         callbacks = pars$callbacks)
        
      } else {
        features <- task$data(cols = task$feature_names)
        target = task$data(cols = task$target_names)[[task$target_names]]
        
        x = self$transforms$x(features, pars) 
        y = self$transforms$y(target, pars)
  
        history = invoke(keras::fit,
                         object = pars$model,
                         x = x,
                         y = y,
                         epochs = as.integer(pars$epochs),
                         class_weight = pars$class_weight,
                         batch_size = pars$batch_size,
                         validation_split = pars$validation_split,
                         verbose = pars$verbose,
                         callbacks = pars$callbacks)
      }
      return(list(model = pars$model, history = history, target_labels = task$class_names))
    },

    predict_internal = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      
      if(!is.null(pars$low_memory) && pars$low_memory) {
        pars["low_memory"] <- NULL # Do not pass to keras
        pars["batch_size"] <- NULL # Generator does not take batch size
        
        gen <- make_data_generator(
          task = task,
          batch_size = batch_size,
          x_transform = function(x) {self$transforms$x(x, pars)}
        )
        prob = invoke(keras::predict_generator, self$model$model, generator = gen, .args = pars)
        PredictionClassif$new(task = task, prob = prob)
      } else {
        pars["low_memory"] <- NULL # Do not pass to keras
        
        features <- task$data(cols = task$feature_names)
        newdata = self$transforms$x(features)
        
        if (self$predict_type == "response") {
          p = invoke(keras::predict_classes, self$model$model, x = newdata, .args = pars)
          p = factor(self$model$target_labels[p + 1])
          PredictionClassif$new(task = task, response = drop(p))
        } else {
          prob = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pars)
          if (ncol(prob) == 1L) prob = cbind(1-prob, prob)
          colnames(prob) = task$class_names
          PredictionClassif$new(task = task, prob = prob)
        }
      }
    },

    set_transform = function(name, transform) {
      self$transforms[[name]] = transform
    }
  ),
)
