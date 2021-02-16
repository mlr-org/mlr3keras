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
#' Calls [keras::fit()] from package \CRANpkg{keras}.
#'
#' Parameters:
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `model`: A compiled keras model suited for the task.
#' * `class_weight`: A named list of class-weights
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
    initialize = function(
        id = "classif.keras",
        predict_types = c("response", "prob"),
        feature_types = c("integer", "numeric"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.keras",
        architecture = KerasArchitectureCustomModel$new()
      ) {
      self$architecture = assert_class(architecture, "KerasArchitecture")
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 100L, lower = 0L, tags = "train"),
        ParamUty$new("model", tags = c("train")),
        ParamUty$new("class_weight", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict", "predict_fun")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamLgl$new("low_memory", default=FALSE, tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict", "predict_fun"))
      ))
      ps$values = list(epochs = 100L, callbacks = list(), validation_split = 1/3, batch_size = 128L, low_memory = FALSE, verbose=0L)

      super$initialize(
        id = assert_character(id, len = 1),
        param_set = ParamSetCollection$new(list(ps, self$architecture$param_set)),
        predict_types = assert_character(predict_types),
        feature_types = assert_character(feature_types),
        properties = assert_character(properties),
        packages = assert_character(packages),
        man = assert_character(man)
      )

      # Set y_transform: use to_categorical, if goal is binary crossentropy drop 2nd column.
      self$architecture$set_transform("y",
        function(target, pars, model_loss) {
          if (is.data.frame(target)) {
            target = unlist(target)
          }
          y = to_categorical(as.integer(target) - 1, num_classes = length(levels(target)))
          if (model_loss == "binary_crossentropy") y = y[, 1, drop = FALSE]
          return(y)
        }
      )
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
      if (is.null(self$model)) stop("Model must be trained before plotting")
      plot(self$model$history)
    },
    lr_find = function(task, epochs = 5L, lr_min = 10^-4, lr_max = 0.8, batch_size = 128L) {
      data = find_lr(self$clone(), task, epochs, lr_min, lr_max, batch_size)
      plot_find_lr(data)
    }
  ),

  private = list(
    .train = function(task) {
      pars = self$param_set$get_values(tags = "train")

      # Construct / Get the model depending on task and hyperparams.
      model = self$architecture$get_model(task, pars)

      # Custom transformation depending on the model.
      # Could be generalized at some point.
      rows = sample(task$row_roles$use)
      features = task$data(cols = task$feature_names, rows = rows)
      target = task$data(cols = task$target_names, rows = rows)[[task$target_names]]

      # Either fit directly on data or create a generator and fit from there
      if (!pars$low_memory) {

        x = self$architecture$transforms$x(features, pars)
        y = self$architecture$transforms$y(target, pars, model_loss = model$loss)

        history = invoke(keras::fit,
          object = model,
          x = x,
          y = y,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          batch_size = as.integer(pars$batch_size),
          validation_split = pars$validation_split,
          verbose = as.integer(pars$verbose),
          callbacks = pars$callbacks,
          shuffle = TRUE
        )

      } else {

        generators = make_train_valid_generators(
          task = task,
          x_transform = function(features) self$architecture$transforms$x(features, pars = pars),
          y_transform = function(target) self$architecture$transforms$y(target, pars = pars, model_loss = model$loss),
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
      return(list(model = model, history = history, class_names = task$class_names))
    },

    .predict = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      features = task$data(cols = task$feature_names)
      newdata = self$architecture$transforms$x(features, pars)
      pf_pars = self$param_set$get_values(tags = "predict_fun")
      if (inherits(self$model$model, "keras.engine.sequential.Sequential")) {
        p = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pf_pars)
      } else {
        p = invoke(predict, self$model$model, x = newdata, .args = pf_pars)
      }
      fixup_target_levels_prediction_classif(p, task, self$predict_type)
    }
  )
)

#' Fix target levels
#' @param prob [`numeric`]\cr
#'   The prediction to fix levels for.
#' @param task [`Task`]\cr
#'   The [`Task`] to create prediction from.
#' @param out [`character`]\cr
#'   Output type, either "response" or "prob".
#' @return A [`PredictionClassif`]
fixup_target_levels_prediction_classif = function(prob, task, out = "response") {
  if (ncol(prob) == 1L) prob = cbind(prob, 1 - prob)
  colnames(prob) = task$class_names

  if (out == "response") {
    argmx = apply(prob, 1, which.max)
    # Binary response with positive class:
    if (length(task$class_names) == 2) {
      if (all(levels(task$data()[[task$target_names]]) != task$class_names)) {
        argmx  = 3 - argmx
      }
    }
    response = factor(task$class_names[argmx])
    PredictionClassif$new(task = task, prob = NULL, response = response)
  } else if (out == "prob") {
    # Binary response with positive class:
    if (length(task$class_names) == 2) {
      if (all(levels(task$data()[[task$target_names]]) != task$class_names)) {
        prob = 1 - prob
      }
    }
    PredictionClassif$new(task = task, prob = prob, response = NULL)
  }
}
