#' @title Keras TabNet Neural Network for Classification
#'
#' Implementation of "TabNet" from the paper TabNet: Attentive Interpretable Tabular Learning (Sercan, Pfister, 2019).
#' See https://arxiv.org/abs/1908.07442 for details.
#'
#' @usage NULL
#' @aliases mlr_learners_classif.tabnet
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifTabNet$new()
#' mlr3::mlr_learners$get("classif.tabnet")
#' mlr3::lrn("classif.tabnet")
#' ```
#' @template tabnet_description
#' @template learner_methods
#' @references Sercan, A. and Pfister, T. (2019): TabNet. \url{https://arxiv.org/abs/1908.07442}.
#' @template seealso_learner
#' @templateVar learner_name classif.tabnet
#' @template example
#' @export
LearnerClassifTabNet = R6::R6Class("LearnerClassifTabNet",
  inherit = LearnerClassifKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("embed_size", default = NULL, lower = 1L, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamLgl$new("stacked", default = FALSE, tags = "train"),
        ParamInt$new("num_layers", lower = 1, upper = Inf, default = 1L, tags = "train"),
        ParamDbl$new("batch_momentum", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("relaxation_factor", lower = 0, upper = Inf, default = 1.5, tags = "train"),
        ParamDbl$new("sparsity_coefficient", lower = 0, upper = 1, default = 10^-5, tags = "train"),
        ParamInt$new("num_decision_steps", lower = 1L, upper = Inf, default = 5L, tags = "train"),
        ParamInt$new("feature_dim", lower = 1L, upper = Inf, default = 64L, tags = "train"),
        ParamInt$new("output_dim", lower = 1L, upper = Inf, default = 64L, tags = "train"),
        ParamInt$new("num_groups", lower = 1L, upper = Inf, default = 2L, tags = "train"),
        ParamDbl$new("epsilon", lower = 0, upper = 1, default = 10^-5, tags = "train"),
        ParamFct$new("norm_type", levels = c("group", "batch"), default = "group", tags = "train"),
        ParamInt$new("virtual_batch_size", lower = 1L, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",
          levels = c("categorical_crossentropy", "sparse_categorical_crossentropy")),
        ParamUty$new("optimizer", default = "optimizer_adam(3*10^-4)", tags = "train"),
        ParamUty$new("metrics", default = "accuracy", tags = "train")
      ))
      ps$add_dep("num_layers", "stacked", CondEqual$new(TRUE))
      ps$values = list(
        embed_size = NULL,
        stacked = FALSE,
        batch_momentum = 0.98,
        relaxation_factor = 1.0,
        sparsity_coefficient = 10^-5,
        num_decision_steps = 2L,
        output_dim = 4L,
        feature_dim = 4L,
        epsilon = 10^-5,
        norm_type = "group",
        num_groups = 2L,
        virtual_batch_size = NULL,
        optimizer = optimizer_adam(lr = 3*10^-4),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
      )
      arch = KerasArchitectureTabNet$new(build_arch_fn = build_keras_tabnet, param_set = ps)
      super$initialize(
        id = "classif.tabnet",
        feature_types = c("integer", "numeric", "factor", "logical"),
        packages = c("keras", "tensorflow", "reticulate"),
        man = "mlr3keras::mlr_learners_classif.tabnet",
        architecture = arch)
      self$param_set$values$validation_split = 0 # Does not to work with tf_data.
    }
  )
)


#' @title Keras TabNet Neural Network for Regression
#'
#' Implementation of "TabNet" from the paper TabNet: Attentive Interpretable Tabular Learning (Sercan, Pfister, 2019).
#' See https://arxiv.org/abs/1908.07442 for details.
#'
#' @usage NULL
#' @aliases mlr_learners_regr.tabnet
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerRegrKeras].
#'
#' @section Construction:
#' ```
#' LearnerRegrTabNet$new()
#' mlr3::mlr_learners$get("regr.tabnet")
#' mlr3::lrn("regr.tabnet")
#' ```
#' @template tabnet_description
#' @template learner_methods
#' @references Sercan, A. and Pfister, T. (2019): TabNet. \url{https://arxiv.org/abs/1908.07442}.
#' @template seealso_learner
#' @templateVar learner_name regr.tabnet
#' @template example
#' @export
LearnerRegrTabNet = R6::R6Class("LearnerRegrTabNet",
  inherit = LearnerRegrKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamInt$new("embed_size", default = NULL, lower = 1L, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamLgl$new("stacked", default = FALSE, tags = "train"),
        ParamInt$new("num_layers", lower = 1, upper = Inf, default = 1L, tags = "train"),
        ParamDbl$new("batch_momentum", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("relaxation_factor", lower = 0, upper = Inf, default = 1.5, tags = "train"),
        ParamDbl$new("sparsity_coefficient", lower = 0, upper = 1, default = 10^-5, tags = "train"),
        ParamInt$new("num_decision_steps", lower = 1L, upper = Inf, default = 5L, tags = "train"),
        ParamInt$new("feature_dim", lower = 1L, upper = Inf, default = 64L, tags = "train"),
        ParamInt$new("output_dim", lower = 1L, upper = Inf, default = 64L, tags = "train"),
        ParamInt$new("num_groups", lower = 1L, upper = Inf, default = 2L, tags = "train"),
        ParamDbl$new("epsilon", lower = 0, upper = 1, default = 10^-5, tags = "train"),
        ParamFct$new("norm_type", levels = c("group", "batch"), default = "group", tags = "train"),
        ParamInt$new("virtual_batch_size", lower = 1L, upper = Inf, tags = "train", special_vals = list(NULL)),
        ParamUty$new("optimizer", default = "optimizer_adam(3*10^-4)", tags = "train"),
        ParamFct$new("loss", default = "mean_squared_error", tags = "train",
          levels = c("cosine_proximity", "cosine_similarity", "mean_absolute_error", "mean_squared_error",
            "poison", "squared_hinge", "mean_squared_logarithmic_error")),
        ParamUty$new("metrics", default = "mean_squared_logarithmic_error", tags = "train")
      ))
      ps$add_dep("num_layers", "stacked", CondEqual$new(TRUE))
      ps$values = list(
        embed_size = NULL,
        stacked = FALSE,
        batch_momentum = 0.98,
        relaxation_factor = 1.0,
        sparsity_coefficient = 10^-5,
        num_decision_steps = 2L,
        output_dim = 4L,
        feature_dim = 4L,
        epsilon = 10^-5,
        norm_type = "group",
        num_groups = 2L,
        virtual_batch_size = NULL,
        optimizer = optimizer_adam(lr = 3*10^-4),
        loss = "mean_squared_error",
        metrics = "mean_squared_logarithmic_error"
      )

      arch = KerasArchitectureTabNet$new(build_arch_fn = build_keras_tabnet, param_set = ps)
      super$initialize(
        id = "regr.tabnet",
        feature_types = c("integer", "numeric"),
        packages = c("keras", "tensorflow", "reticulate"),
        man = "mlr3keras::mlr_learners_regr.tabnet",
        architecture = arch)
      self$param_set$values$validation_split = 0 # Does not to work with tf_data.
    }
  )
)


#' @title Keras TabNet architecture
#' @rdname KerasArchitecture
#' @family KerasArchitectures
#' @export
KerasArchitectureTabNet = R6::R6Class("KerasArchitectureTabNet",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn, x_transform, y_transform, param_set) {
      x_transform = function(features, pars) {
        x = lapply(names(features), function(x) {
          x = features[, get(x)]
          if (is.numeric(x) || is.integer(x)) {
            as.matrix(as.numeric(x))
          } else if (is.logical(x)) {
            as.matrix(as.integer(x))
          } else {
            as.matrix(x)
          }})
        names(x) = names(features)
        return(x)
      }
      super$initialize(build_arch_fn = build_arch_fn, x_transform = x_transform, param_set = param_set)
    }
  )
)


build_keras_tabnet = function(task, pars) {
  requireNamespace("reticulate")
  requireNamespace("tensorflow")
  if(!reticulate::py_module_available("tabnet")) {
    stop("Python module tabnet is not available. In order to install it use
      keras::install_keras(extra_packages = c('tensorflow-hub', 'tabnet==0.1.4.1')).")
  }
  tabnet = reticulate::import("tabnet")
  feature_columns = make_tf_feature_cols(task, pars$embed_size)
  tabnet_param_names = c("feature_dim", "output_dim", "num_decision_steps", "relaxation_factor",
    "sparsity_coefficient", "virtual_batch_size", "norm_type", "num_groups")
  if (pars$stacked) tabnet_param_names = c(tabnet_param_names, "num_layers")

  if (inherits(task, "TaskClassif")) {
    if (pars$stacked) clf = tabnet$StackedTabNetClassifier
    else clf = tabnet$TabNetClassifier
    model = invoke(clf,
      num_features = get_tf_num_features(task, pars),
      feature_columns = feature_columns, num_classes = length(task$class_names),
      .args = pars[tabnet_param_names])
  } else if (inherits(task, "TaskRegr")) {
    if (pars$stacked) regr = tabnet$StackedTabNetRegressor
    else regr = tabnet$TabNetRegressor
    model = invoke(regr,
      num_features = get_tf_num_features(task, pars),
      feature_columns = feature_columns, num_regressors = 1L,
      .args = pars[tabnet_param_names])
  }

  model %>% compile(
    loss = pars$loss,
    optimizer = pars$optimizer,
    metrics = pars$metrics
  )
}

# Create a tf$feature_column according to the type of a Task's column.
# FIXME: This covers only the most basic feature types, needs to be extended.
make_tf_feature_column = function(id, type, args) {
  if (type %in% c("numeric", "integer")) tensorflow::tf$feature_column$numeric_column(id)
  else if (type == "logical") {
    tensorflow::tf$feature_column$indicator_column(
      tensorflow::tf$feature_column$categorical_column_with_identity(id, num_buckets = 2L)
    )
  }
  else if (type %in% c("factor", "character", "ordered")) {
    if (is.null(args$embed_size)) args$embed_size = get_default_embed_size(args$levels[[id]])
    tensorflow::tf$feature_column$embedding_column(
      tensorflow::tf$feature_column$categorical_column_with_vocabulary_list(id, args$levels[[id]]),
      dimension = args$embed_size
    )
  }
}

get_tf_num_features = function(task, pars) {
  dims = pmap_int(task$feature_types, function(id, type, levels) {
    switch(type,
      "numeric" = 1L,
      "integer" = 1L,
      "logical" = 2L,
      "factor" = {
        if (is.null(pars$embed_size))
          get_default_embed_size(levels[[1]][[id]])
        else pars$embed_size
      },
      "character" = {
        if (is.null(pars$embed_size))
          get_default_embed_size(levels[[1]][[id]])
        else pars$embed_size
      },
      "ordered" = {
        if (is.null(pars$embed_size))
          get_default_embed_size(levels[[1]][[id]])
        else pars$embed_size
      }
  )}, list(task$levels()))
  sum(dims)
}

make_tf_feature_cols = function(task, embed_size = NULL) {
  assert_r6(task, "Task")
  feature_columns = pmap(.f = make_tf_feature_column, .x = task$feature_types, list(levels = task$levels(), embed_size = embed_size))
}

get_default_embed_size = function(levels) {
    # As a default we use the fast.ai heuristic
    min(600L, round(1.6 * length(levels)^0.56))
}

