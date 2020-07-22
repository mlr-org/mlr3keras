#' @title Keras Feed Forward Neural Network for Classification: Shaped MLP
#'
#' @usage NULL
#' @aliases mlr_learners_classif.smlp
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifShapedMLP$new()
#' mlr3::mlr_learners$get("classif.smlp")
#' mlr3::lrn("classif.smlp")
#' ```
#'
#' @template shaped_mlp_1_description
#' @template shaped_mlp_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name classif.smlp
#' @template example
#' @export
LearnerClassifShapedMLP = R6::R6Class("LearnerClassifShapedMLP",
  inherit = LearnerClassifKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamLgl$new("use_embedding", default = TRUE, tags = c("train", "predict")),
        ParamDbl$new("embed_dropout", default = 0, lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("embed_size", default = NULL, lower = 1, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamInt$new("n_max", default = 128L, tags = "train", lower = 1, upper = Inf),
        ParamInt$new("n_layers", default = 2L, tags = "train", lower = 1, upper = Inf),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu", "linear")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("input_dropout", lower = 0, upper = 1, tags = "train"),
        ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",  levels = keras_reflections$loss$classif),
        ParamFct$new("output_activation", levels = c("softmax", "linear", "sigmoid"), tags = "train"),
        ParamUty$new("metrics", tags = "train")
      ))
      ps$values = list(
        use_embedding = FALSE, embed_dropout = 0, embed_size = NULL,
        activation = "relu",
        n_max = 128L,
        n_layers = 2L,
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_sgd(lr = 3*10^-4, momentum = 0.9),
        regularizer = regularizer_l1_l2(),
        use_batchnorm = FALSE,
        use_dropout = TRUE, dropout = 0, input_dropout = 0,
        loss = "categorical_crossentropy",
        metrics = "accuracy",
        output_activation = "softmax"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_shaped_mlp, param_set = ps)
      super$initialize(
        feature_types = c("integer", "numeric", "factor", "ordered"),
        man = "mlr3keras::mlr_learners_classif.smlp",
        architecture = arch
      )
    }
  )
)

#' @title Keras Feed Forward Neural Network for Regression: Shaped MLP
#'
#' @usage NULL
#' @aliases mlr_learners_regr.smlp
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerRegrKeras].
#' @section Construction:
#' ```
#' LearnerRegrShapedMLP$new()
#' mlr3::mlr_learners$get("regr.smlp")
#' mlr3::lrn("regr.smlp")
#' ```
#'
#' @template shaped_mlp_1_description
#' @template shaped_mlp_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name regr.smlp
#' @template example
#' @export
LearnerRegrShapedMLP = R6::R6Class("LearnerRegrShapedMLP",
  inherit = LearnerRegrKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamLgl$new("use_embedding", default = TRUE, tags = c("train", "predict")),
        ParamDbl$new("embed_dropout", default = 0, lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("embed_size", default = NULL, lower = 1, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamInt$new("n_max", default = 128L, tags = "train", lower = 1, upper = Inf),
        ParamInt$new("n_layers", default = 2L, tags = "train", lower = 1, upper = Inf),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu", "linear")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("input_dropout", lower = 0, upper = 1, tags = "train"),
        ParamFct$new("loss", default = "mean_squared_error", tags = "train", levels = keras_reflections$loss$regr),
        ParamFct$new("output_activation", levels = c("linear", "sigmoid"), tags = "train"),
        ParamUty$new("metrics", default = "mean_squared_logarithmic_error", tags = "train")
      ))
      ps$values = list(
        use_embedding = TRUE, embed_dropout = 0,  embed_size = NULL,
        activation = "relu",
        n_max = 128L,
        n_layers = 2L,
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_sgd(lr = 3*10^-4, momentum = 0.9),
        regularizer = regularizer_l1_l2(),
        use_batchnorm = FALSE,
        use_dropout = TRUE, dropout = 0, input_dropout = 0,
        loss = "mean_squared_error",
        metrics = "mean_squared_logarithmic_error",
        output_activation = "linear"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_shaped_mlp, param_set = ps)
      super$initialize(
        feature_types = c("integer", "numeric", "factor", "ordered"),
        man = "mlr3keras::mlr_learners_regr.smlp",
        architecture = arch
      )
    }
  )
)


# Shaped MLP as used in Zimmer et al. Auto Pytorch Tabular (2020)
# and proposed by https://mikkokotila.github.io/slate.
#
# Implements 'Search Space 1' from Zimmer et al. Auto Pytorch Tabular (2020)
# (https://arxiv.org/abs/2006.13799)
build_shaped_mlp = function(task, pars) {

  if ("factor" %in% task$feature_types$type && !pars$use_embedding)
    stop("Factor features are only available with use_embedding = TRUE!")

  # Get input and output shape for model
  input_shape = list(task$ncol - 1L)
  if (inherits(task, "TaskRegr")) {
    output_shape = 1L
  } else if (inherits(task, "TaskClassif")) {
    output_shape = length(task$class_names)
    if (pars$loss == "binary_crossentropy") {
      if (length(output_shape) > 2L) stop("binary_crossentropy loss is only available for binary targets")
      output_shape = 1L
    }
  }

  if (pars$use_embedding) {
    embd = make_embedding(task, pars$embed_size, pars$embed_dropout)
    model = embd$layers
  } else {
    model = keras_model_sequential()
  }

  # Build hidden layers
  n_neurons_layer = pars$n_max

  for (i in seq_len(pars$n_layers)) {
    model = model %>%
      layer_dense(
        units = n_neurons_layer,
        input_shape = if (i == 1) input_shape else NULL,
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
     n_neurons_layer = ceiling(n_neurons_layer - (pars$n_max - output_shape) / (pars$n_layers - 1L))
  }
  # Output layer
  if (output_shape == 1L)
    model = model %>% layer_dense(units = output_shape, activation = "sigmoid")
  else
    model = model %>% layer_dense(units = output_shape, activation = pars$output_activation)

  if (pars$use_embedding) model = keras_model(inputs = embd$inputs, outputs = model)

  model %>% compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
}
