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
#' * `class_weights`: needs to be a named list of class-weights 
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
#'  model = keras_model_sequential() %>%
#'  layer_dense(units = 12L, input_shape = 4L, activation = "relu") %>%
#'  layer_dense(units = 12L, activation = "relu") %>%
#'  layer_dense(units = 3L, activation = "softmax") %>%
#'    compile(optimizer = optimizer_sgd(),
#'      loss = "categorical_crossentropy",
#'      metrics = "accuracy")
#'  # Create the learner
#'  learner = LearnerClassifKeras$new()
#'  learner$param_set$values = list(model = model)
#'  learner$train(mlr_tasks$get("iris"))
#' @export
LearnerClassifKeras = R6::R6Class("LearnerClassifKeras", inherit = LearnerClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamUty$new("model", tags = c("train")),
        ParamUty$new("class_weights", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 2/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict"))  
      ))
      ps$values = list(epochs = 30L, callbacks = list(),
        validation_split = 2/3, batch_size = 128L)

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
      data = as.matrix(task$data(cols = task$feature_names))
      target = task$data(cols = task$target_names)

      assert_class(pars$model, "keras.engine.training.Model")

      y = to_categorical(as.integer(target[[task$target_names]]) - 1)
      
      history = invoke(keras::fit, 
        object = pars$model,
        x = data,
        y = y,
        epochs = as.integer(pars$epochs),
        class_weights = pars$class_weights,
        batch_size = pars$batch_size,
        validation_split = pars$validation_split,
        verbose = pars$verbose,
        callbacks = pars$callbacks)
      return(list(model = pars$model, history = history, target_labels = task$class_names))   
    },

    predict_internal = function(task) {

      pars = self$param_set$get_values(tags = "predict")
      newdata = as.matrix(task$data(cols = task$feature_names))

      if (self$predict_type == "response") {
        p = invoke(keras::predict_classes, self$model$model, x = newdata, .args = pars)
        p = factor(self$model$target_labels[p + 1])
        PredictionClassif$new(task = task, response = drop(p))
      } else {
        prob = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pars)
        colnames(prob) = task$class_names
        PredictionClassif$new(task = task, prob = prob)
      }
    }   
  )
)