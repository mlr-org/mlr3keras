#' @title Keras Feed Forward Neural Network
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasff
#' @format [R6::R6Class()] inheriting from [mlr3::LearnerClassif].
#'
#' @section Construction:
#' ```
#' LearnerClassifKerasff$new()
#' mlr3::mlr_learners$get("classif.kerasff")
#' mlr3::lrn("classif.kerasff")
#' ```
#'
#' @description
#' Feed Forward Neural Network using Keras and Tensorflow.
#' Calls [keras::fit] from package \CRANpkg{keras}.
#' 
#' Parameters:\cr
#' Most of the parameters can be obtained from the `keras` 
#' documentation. Some exceptions are documented here
#' * `initializer`: An object of class `tensorflow.python.ops.init_ops_v2.Initializer`.
#'   Keras initializers start with 'initializer_...'
#' * `optimizer`: Some optimizers and their arguments can be found below.\cr
#'   Inherits from `tensorflow.python.keras.optimizer_v2`.
#'   ```
#'   "sgd"     : optimizer_sgd(lr, momentum, decay = decay),
#'   "rmsprop" : optimizer_rmsprop(lr, rho, decay = decay),
#'   "adagrad" : optimizer_adagrad(lr, decay = decay),
#'   "adam"    : optimizer_adam(lr, beta_1, beta_2, decay = decay),
#'   "nadam"   : optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
#'   ```
#' * `regularizer`: Inherits from `tensorflow.python.keras.regularizers`.
#' 
#' * `class_weights`: needs to be a named list of class-weights 
#'   for the different classes numbered from 0 to c-1 (for c classes).
#'   ```
#'   Example:
#'   wts = c(0.5, 1)
#'   setNames(as.list(wts), seq_len(length(wts)) - 1)
#'   ```
#' 
#' @template seealso_learner
#' @templateVar learner_name classif.kerasff
#' @template example
#' @export
LearnerClassifKerasff = R6::R6Class("LearnerClassifKerasff", inherit = LearnerClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu")),
        ParamInt$new("early_stopping_patience", lower = 0L, default = 2L, tags = "train"),
        ParamUty$new("class_weights", default = list(), tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 2/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict")),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict"))  
      ))
      ps$values = list(epochs = 30L, activation = "relu",
       initializer = initializer_glorot_uniform(),
       optimizer = optimizer_sgd(10^-3),
       regularizer = regularizer_l1_l2(),
       validation_split = 2/3, batch_size = 128L)

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

      input_shape = ncol(data)
      target_labels = task$class_names
      output_shape = length(target_labels)

      # # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
      # # Dense -> Act -> [BN] -> [Dropout]

      # callbacks = c()
      # if (early_stopping_patience > 0)
      #   callbacks = c(callbacks, callback_early_stopping(monitor = 'val_loss', patience = early_stopping_patience))
      # if (learning_rate_scheduler)
      #   callbacks = c(callback_learning_rate_scheduler(function(epoch, lr) {lr * 1/(1 * epoch)}))

      layers = 3
      units_layers = rep(12, 3) # c(units_layer1, units_layer2, units_layer3, units_layer4)

      model = keras_model_sequential()
      # if (batchnorm_dropout == "dropout")
      #   model = model %>% layer_dropout(input_dropout_rate, input_shape = input_shape)

      for (i in seq_len(layers)) {
        model = model %>%
          layer_dense(
            units = units_layers[i],
            input_shape = input_shape,
            kernel_regularizer = pars$regularizer,
            kernel_initializer = pars$initializer,
            bias_regularizer = pars$regularizer,
            bias_initializer = pars$initializer
          ) %>%
          layer_activation(pars$activation)
        # if (batchnorm_dropout == "batchnorm") model = model %>% layer_batch_normalization()
        # if (batchnorm_dropout == "dropout") model = model %>% layer_dropout(dropout_rate)
      }
      model = model %>% layer_dense(units = output_shape, activation = 'softmax')

      model %>% compile(
        optimizer = pars$optimizer,
        loss = "categorical_crossentropy",
        metrics = c('accuracy')
      )

      y = to_categorical(as.integer(target[[task$target_names]]) - 1)
      
      history = invoke(keras::fit, 
        object = model,
        x = data,
        y = y,
        epochs = as.integer(pars$epochs),
        class_weights = pars$class_weights,
        batch_size = pars$batch_size,
        validation_split = pars$validation_split,
        verbose = pars$verbose)
        # callbacks = callbacks)
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
    }
  )
)