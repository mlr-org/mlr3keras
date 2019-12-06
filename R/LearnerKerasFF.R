#' @title Keras Feed Forward Neural Network for Classification
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasff
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifKerasFF$new()
#' mlr3::mlr_learners$get("classif.kerasff")
#' mlr3::lrn("classif.kerasff")
#' ```
#' 
#' @template kerasff_description
#' @template seealso_learner
#' @templateVar learner_name classif.kerasff
#' @template example
#' @export
LearnerClassifKerasFF = R6::R6Class("LearnerClassifKerasFF",
  inherit = LearnerClassifKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamUty$new("layer_units", default = c(32, 32, 32), tags = "train"),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu", "linear")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("input_dropout", lower = 0, upper = 1, tags = "train"),
        ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",
          levels = c("binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy")),
        ParamFct$new("output_activation", levels = c("softmax", "linear", "sigmoid"), tags = "train"),
        ParamUty$new("metrics", tags = "train")
      ))
      ps$values = list(
        activation = "relu",
        layer_units = c(32, 32, 32),
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_adam(lr = 3*10^-4),
        regularizer = regularizer_l1_l2(),
        use_batchnorm = FALSE,
        use_dropout = FALSE, dropout = 0, input_dropout = 0,
        loss = "categorical_crossentropy",
        metrics = "accuracy",
        output_activation = "softmax"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_keras_ff_model,  param_set = ps)
      super$initialize(architecture = arch)
    }
  )
)

#' @title Keras Feed Forward Neural Network for Regression
#'
#' @usage NULL
#' @aliases mlr_learners_regr.kerasff
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerRegrKeras].
#' @section Construction:
#' ```
#' LearnerRegrKerasFF$new()
#' mlr3::mlr_learners$get("regr.kerasff")
#' mlr3::lrn("regr.kerasff")
#' ```
#' @template kerasff_description
#' @template seealso_learner
#' @templateVar learner_name classif.kerasff
#' @template example
#' @export
LearnerRegrKerasFF = R6::R6Class("LearnerRegrKerasFF",
  inherit = LearnerRegrKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamUty$new("layer_units", default = c(32, 32, 32), tags = "train"),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu", "linear")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("input_dropout", lower = 0, upper = 1, tags = "train"),
        ParamFct$new("loss", default = "mean_squared_error", tags = "train",
          levels = c("cosine_proximity", "cosine_similarity", "mean_absolute_error", "mean_squared_error",
            "poison", "squared_hinge", "mean_squared_logarithmic_error")),
        ParamFct$new("output_activation", levels = c("linear", "sigmoid"), tags = "train"),
        ParamUty$new("metrics", default = "mean_squared_logarithmic_error", tags = "train")
      ))
      ps$values = list(
        activation = "relu",
        layer_units = c(32, 32, 32),
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_adam(lr = 3*10^-4),
        regularizer = regularizer_l1_l2(),
        use_batchnorm = FALSE,
        use_dropout = FALSE, dropout = 0, input_dropout = 0,
        loss = "mean_squared_error",
        metrics = "mean_squared_logarithmic_error",
        output_activation = "linear"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_keras_ff_model,  param_set = ps)
      super$initialize(architecture = arch)
    }
  )
)

#' Builds a Keras Feed Forward Neural Network 
#' @param pars [`list`] \cr
#'   A list of parameter values from the Learner(Regr|Classif)KerasFF param_set.
#' @param input_shape [`integer(1)`] \cr
#'   Number of input units.@return A compiled keras model
#' @param output_shape  [`integer(1)`] \cr
#'   Number of output classes. Always 1L for regression.
#' @template kerasff_description
#' @return A compiled keras model
build_keras_ff_model = function(task, pars) {

  # Get input and output shape for model
  input_shape = task$ncol - 1L
  if (inherits(task, "TaskRegr")) {
    output_shape = 1L
  } else if (inherits(task, "TaskClassif")) {
    output_shape = length(task$class_names)
    if (pars$loss == "binary_crossentropy") {
      if (length(output_shape) > 2L) stop("binary_crossentropy loss is only available for binary targets")
      output_shape = 1L
    }
  }

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
  if (output_shape == 1L)
    model = model %>% layer_dense(units = output_shape, activation = "sigmoid")
  else
    model = model %>% layer_dense(units = output_shape, activation = pars$output_activation)

  model %>% compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
}