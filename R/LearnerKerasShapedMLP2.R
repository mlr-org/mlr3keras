#' @title Keras Feed Forward Neural Network for Classification: Shaped MLP 2
#'
#' Currently does not allow Shake-Shake, Shake-Drop or Mixup training as well as SVD on sparse matrices.
#'
#' @usage NULL
#' @aliases mlr_learners_classif.smlp2
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifShapedMLP2$new()
#' mlr3::mlr_learners$get("classif.smlp2")
#' mlr3::lrn("classif.smlp2")
#' ```
#' @template shaped_mlp_2_description
#' @template shaped_mlp_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name classif.smlp2
#' @template example
#' @export
LearnerClassifShapedMLP2 = R6::R6Class("LearnerClassifShapedMLP2",
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
        n_layers = 3L,
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_sgd(lr = 3*10^-4, momentum = 0.9),
        regularizer = regularizer_l1_l2(),
        dropout = 0, input_dropout = 0,
        loss = "categorical_crossentropy",
        metrics = "accuracy",
        output_activation = "softmax"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_shaped_mlp2, param_set = ps)
      super$initialize(
        feature_types = c("integer", "numeric", "factor", "ordered"),
        man = "mlr3keras::mlr_learners_classif.smlp2",
        architecture = arch
      )
      self$param_set$values$callbacks = c(self$param_set$values$callbacks, cb_lr_scheduler_cosine_anneal())
    }
  )
)

#' @title Keras Feed Forward Neural Network for Regression: Shaped MLP 2
#'
#' Currently does not allow Shake-Shake, Shake-Drop or Mixup training as well as SVD on sparse matrices.
#'
#' @usage NULL
#' @aliases mlr_learners_regr.smlp2
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerRegrKeras].
#' @section Construction:
#' ```
#' LearnerRegrShapedMLP2$new()
#' mlr3::mlr_learners$get("regr.smlp2")
#' mlr3::lrn("regr.smlp2")
#' ```
#' @template shaped_mlp_2_description
#' @template shaped_mlp_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name regr.smlp2
#' @template example
#' @export
LearnerRegrShapedMLP2 = R6::R6Class("LearnerRegrShapedMLP2",
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
        n_layers = 3L,
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_sgd(lr = 3*10^-4, momentum = 0.9),
        regularizer = regularizer_l1_l2(),
        dropout = 0, input_dropout = 0,
        loss = "mean_squared_error",
        metrics = "mean_squared_logarithmic_error",
        output_activation = "linear"
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_shaped_mlp2, param_set = ps)
      super$initialize(
        feature_types = c("integer", "numeric", "factor", "ordered"),
        man = "mlr3keras::mlr_learners_regr.smlp2",
        architecture = arch
      )
      self$param_set$values$callbacks = c(self$param_set$values$callbacks, cb_lr_scheduler_cosine_anneal())
    }
  )
)

# Shaped MLP as used in Zimmer et al. Auto Pytorch Tabular (2020)
# and proposed by https://mikkokotila.github.io/slate.
#
# Implements 'Search Space 2' from Zimmer et al. Auto Pytorch Tabular (2020)
# (https://arxiv.org/abs/2006.13799) with some extensions.
build_shaped_mlp2 = function(task, pars) {

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
      pars$output_activation = "sigmoid"
    }
  }

  if (pars$use_embedding) {
    embd = make_embedding(task, pars$embed_size, pars$embed_dropout)
    model = embd$layers
    input = embd$inputs
  } else {
    model = input = layer_input(shape = input_shape)
  }

  # Build hidden layers
  n_neurons_layer = pars$n_max

  for (i in seq_len(pars$n_layers - 1L)) {
    if (i == 1L) {
      # First layer is just a dense Layer
      model = model %>% layer_dense(
        units = n_neurons_layer,
        kernel_regularizer = pars$regularizer,
        kernel_initializer = pars$initializer,
        bias_regularizer = pars$regularizer,
        bias_initializer = pars$initializer
      )
    } else {
      # browser()
      # For layers >= 2, add residual blocks
      res_block = model %>% layer_batch_normalization() %>%
        layer_activation(pars$activation) %>%
        layer_dense(
          units = n_neurons_layer,
          kernel_regularizer = pars$regularizer,
          kernel_initializer = pars$initializer,
          bias_regularizer = pars$regularizer,
          bias_initializer = pars$initializer
        ) %>%
        layer_batch_normalization() %>%
        layer_activation(pars$activation) %>%
        layer_dropout(pars$dropout) %>%
        layer_dense(
          units = n_neurons_layer,
          kernel_regularizer = pars$regularizer,
          kernel_initializer = pars$initializer,
          bias_regularizer = pars$regularizer,
          bias_initializer = pars$initializer
        )

      skip_block = model %>% layer_dense(
        units = n_neurons_layer,
        kernel_regularizer = pars$regularizer,
        kernel_initializer = pars$initializer,
        bias_regularizer = pars$regularizer,
        bias_initializer = pars$initializer
      )
      model = layer_add(list(res_block, skip_block))
    }
    # Compute n_neurons in next layer
    n_neurons_prev = n_neurons_layer
    n_neurons_layer = ceiling(n_neurons_layer - (pars$n_max - output_shape) / (pars$n_layers - 1L))
  }

  # Output layer (BN -> Act -> Dense)
  model = model %>%
        layer_batch_normalization() %>%
        layer_activation(pars$activation) %>%
        layer_dense(units = output_shape, activation = pars$output_activation)

  model = keras_model(inputs = input, outputs = model)

  model %>% compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
}
