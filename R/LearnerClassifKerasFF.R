#' @title Keras Feed Forward Neural Network
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasff
#' @format [R6::R6Class()] inheriting from [mlr3::LearnerClassif].
#'
#' @section Construction:
#' ```
#' LearnerClassifKerasFF$new()
#' mlr3::mlr_learners$get("classif.kerasff")
#' mlr3::lrn("classif.kerasff")
#' ```
#'
#' @description
#' Feed Forward Neural Network using Keras and Tensorflow.
#' This learner builds and compiles the keras model from the hyperparameters in `param_set`,
#' and does not require a supplied and compiled model.
#'
#' Calls [keras::fit] from package \CRANpkg{keras}.
#' Layers are set up as follows:
#' * The inputs are connected to a `layer_dropout`, applying the `input_dropout`.
#'   Afterwards, each `layer_dense()` is followed by a `layer_activation`, and
#'   depending on hyperparameters by a `layer_batch_normalization` and or a
#'   `layer_dropout` depending on the architecture hyperparameters.
#'   This is repeated `length(layer_units)` times, i.e. one
#'   'dense->activation->batchnorm->dropout' block is appended for each `layer_unit`.
#'
#' Parameters:\cr
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `layer_units`: An integer vector storing the number of units in each
#'   consecutive layer. `layer_units = c(32L, 32L, 32L)` results in a 3 layer
#'   network with 32 neurons in each layer.
#'   Can be `integer(0)`, in which case we fit a (multinomial) logistic regression model.
#'
#' * `initializer`: Weight and bias initializer.
#'   ```
#'   "glorot_uniform"  : initializer_glorot_uniform(seed)
#'   "glorot_normal"   : initializer_glorot_normal(seed)
#'   "he_uniform"      : initializer_he_uniform(seed)
#'   "..."             : see `??keras::initializer`
#'   ```
#'
#' * `optimizer`: Some optimizers and their arguments can be found below.\cr
#'   Inherits from `tensorflow.python.keras.optimizer_v2`.
#'   ```
#'   "sgd"     : optimizer_sgd(lr, momentum, decay = decay),
#'   "rmsprop" : optimizer_rmsprop(lr, rho, decay = decay),
#'   "adagrad" : optimizer_adagrad(lr, decay = decay),
#'   "adam"    : optimizer_adam(lr, beta_1, beta_2, decay = decay),
#'   "nadam"   : optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
#'   ```
#'
#' * `regularizer`: Regularizer for keras layers:
#'   ```
#'   "l1"      : regularizer_l1(l = 0.01)
#'   "l2"      : regularizer_l2(l = 0.01)
#'   "l1_l2"   : regularizer_l1_l2(l1 = 0.01, l2 = 0.01)
#'   ```
#'
#' * `class_weights`: needs to be a named list of class-weights
#'   for the different classes numbered from 0 to c-1 (for c classes).
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
#' @templateVar learner_name classif.kerasff
#' @template example
#' @export
LearnerClassifKerasFF = R6::R6Class("LearnerClassifKerasFF", inherit = LearnerClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamUty$new("layer_units", default = c(32, 32, 32), tags = "train"),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("input_dropout", lower = 0, upper = 1, tags = "train"),
        ParamUty$new("class_weights", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict"))
      ))
      ps$values = list(epochs = 30L, activation = "relu",
       layer_units = c(32, 32, 32),
       initializer = initializer_glorot_uniform(),
       optimizer = optimizer_sgd(10^-3),
       regularizer = regularizer_l1_l2(),
       use_batchnorm = TRUE,
       use_dropout = TRUE, dropout = 0, input_dropout = 0,
       callbacks = list(),
       validation_split = 1/3, batch_size = 128L)

      super$initialize(
        id = "classif.kerasff",
        param_set = ps,
        predict_types = c("response", "prob"),
        feature_types = c("integer", "numeric"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.kerasff"
      )
    },

    train_internal = function(task) {
      pars = self$param_set$get_values(tags = "train")
      data = as.matrix(task$data(cols = task$feature_names))
      target = task$data(cols = task$target_names)
      y = to_categorical(as.integer(target[[task$target_names]]) - 1)

      input_shape = ncol(data)
      target_labels = task$class_names
      output_shape = length(target_labels)

      model = self$model_from_pars(pars, input_shape, output_shape)
      model %>% compile(
        optimizer = pars$optimizer,
        loss = "categorical_crossentropy",
        metrics = "accuracy"
      )

      history = invoke(keras::fit,
        object = model,
        x = data,
        y = y,
        epochs = as.integer(pars$epochs),
        class_weights = pars$class_weights,
        batch_size = pars$batch_size,
        validation_split = pars$validation_split,
        verbose = pars$verbose,
        callbacks = pars$callbacks)
      return(list(model = model, history = history, target_labels = target_labels))
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
    },

    model_from_pars = function(pars, input_shape, output_shape) {
      assert_integerish(pars$layer_units, lower = 1)

      # Not sure whether we should check like this, this breaks with different keras versions
      # assert_class(pars$initializer, "tensorflow.python.ops.init_ops_v2.Initializer")
      # assert_class(pars$regularizer, "keras.regularizers.Regularizer")
      # assert_class(pars$optimizer,   "keras.optimizer_v2.optimizer_v2.OptimizerV2")

      model = keras_model_sequential()
      if (pars$use_dropout) model = model %>% layer_dropout(pars$input_dropout, input_shape = input_shape)

      # Build hidden layers
      for (i in seq_len(length(pars$layer_units))) {
        model = model %>%
          layer_dense(
            units = pars$layer_units[i],
            input_shape = input_shape,
            kernel_regularizer = pars$regularizer,
            kernel_initializer = pars$initializer,
            bias_regularizer = pars$regularizer,
            bias_initializer = pars$initializer
          ) %>%
          layer_activation(pars$activation)
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        # Dense -> Act -> [BN] -> [Dropout]
        if (pars$use_batchnorm) model = model %>% layer_batch_normalization()
        if (pars$use_dropout) model = model %>% layer_dropout(pars$dropout)
      }
      # Output layer
      model = model %>% layer_dense(units = output_shape, activation = 'softmax')
    }
  )
)
