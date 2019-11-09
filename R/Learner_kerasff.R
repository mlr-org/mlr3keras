#' @title Keras Feed Forward Neural Network
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasff
#' @format [R6::R6Class()] inheriting from [mlr3::LearnerClassif].
#'
#' @section Construction:
#' ```
#' LearnerClassifkerasff$new()
#' mlr3::mlr_learners$get("classif.kerasff")
#' mlr3::lrn("classif.kerasff")
#' ```
#'
#' @description
#' Feed Forward Neural Network using Keras and Tensorflow.
#' Calls [keras::fit] from package \CRANpkg{keras}.
#'
#' @export
LearnerClassifkerasff = R6::R6Class("LearnerClassifkerasff", inherit = LearnerClassif,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamInt$new("early_stopping_patience", lower = 0L, default = 2L),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 2/3)
      ))
      ps$values = list(epochs = 30L, validation_split = 2/3)

      super$initialize(
        id = "classif.kerasff",
        param_set = ps,
        predict_types = c("response", "prob"),
        feature_types = c("integer", "numeric"),
        properties = c("weights", "twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3learners::mlr_learners_classif.kerasff"
      )
    },

    train_internal = function(task) {

      require("keras")
      pars = self$param_set$get_values(tags = "train")
      data = as.matrix(task$data(cols = task$feature_names))
      target = task$data(cols = task$target_names)

      if ("weights" %in% task$properties) {
        pars$weights = task$weights$weight
      }

      input_shape = ncol(data)
      target_labels = tsk$class_names
      output_shape = length(target_labels)

      # # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
      # # Dense -> Act -> [BN] -> [Dropout]
      # regularizer = regularizer_l1_l2(l1 = l1_reg_layer, l2 = l2_reg_layer)
      # initializer = switch(init_layer,
      #   "glorot_normal" = initializer_glorot_normal(),
      #   "glorot_uniform" = initializer_glorot_uniform(),
      #   "he_normal" = initializer_he_normal(),
      #   "he_uniform" = initializer_he_uniform()
      # )
      # optimizer = switch(optimizer,
      #   "sgd" = optimizer_sgd(lr, momentum, decay = decay),
      #   "rmsprop" = optimizer_rmsprop(lr, rho, decay = decay),
      #   "adagrad" = optimizer_adagrad(lr, decay = decay),
      #   "adam" = optimizer_adam(lr, beta_1, beta_2, decay = decay),
      #   "nadam" = optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
      # )

      regularizer = NULL
      initializer = "glorot_uniform"
      act_layer = "relu"
      optimizer = "sgd"

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
            kernel_regularizer = regularizer,
            kernel_initializer = initializer,
            bias_regularizer = regularizer,
            bias_initializer = initializer)
        model = model %>% layer_activation(act_layer)
        # if (batchnorm_dropout == "batchnorm") model = model %>% layer_batch_normalization()
        # if (batchnorm_dropout == "dropout") model = model %>% layer_dropout(dropout_rate)
      }
      model = model %>% layer_dense(units = output_shape, activation = 'softmax')

      model %>% compile(
        optimizer = optimizer,
        loss = "categorical_crossentropy",
        metrics = c('accuracy')
      )

      y = to_categorical(as.integer(target[[task$target_names]]) - 1)
      
      history = invoke(model$fit, 
        x = data,
        y = y,
        epochs = as.integer(pars$epochs),
        batch_size = 128L,
        validation_split = pars$validation_split)
        # callbacks = callbacks)
      return(list(model = model, history = history, target_labels = target_labels))   
    },

    predict_internal = function(task) {

      pars = self$param_set$get_values(tags = "predict")
      newdata = as.matrix(task$data(cols = task$feature_names))

      if (self$predict_type == "response") {
        p = invoke(predict_classes, self$model$model, x = newdata, .args = pars)
        p = factor(self$model$target_labels[p + 1])
        PredictionClassif$new(task = task, response = drop(p))
      } else {
        prob = invoke(predict, self$model$model, x = newdata, .args = pars)
        if (length(model$target_labels) > 2L) {
          prob = prob[, , 1L]
        }
        colnames(prob) = task$class_names
        PredictionClassif$new(task = task, prob = prob)
      }
    }
  )
)