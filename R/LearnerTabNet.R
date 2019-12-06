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
#' 
#' @references Sercan, A. and Pfister, T. (2019): TabNet. \url{https://arxiv.org/abs/1908.07442}.
#' @description
#' Most of the parameters can be obtained from the paper.
#' Some exceptions are documented here.
#' * `output_dim`: Dimensions of the pen-ultimate layer(s).
#' * `feature_dim`: Dimension of the intermediate feature representations.
#' @template seealso_learner
#' @templateVar learner_name classif.tabnet
#' @template example
#' @export
LearnerClassifTabNet = R6::R6Class("LearnerClassifTabNet",
  inherit = LearnerClassifKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamDbl$new("batch_momentum", lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("relaxation_factor", lower = 0, upper = Inf, default = 1, tags = "train"),
        ParamDbl$new("sparsity_coefficient", lower = 0, upper = 1, default = 10^-5, tags = "train"),
        ParamInt$new("num_decision_steps", lower = 1L, upper = Inf, default = 2L, tags = "train"),
        ParamInt$new("feature_dim", lower = 1L, upper = Inf, default = 4L, tags = "train"),
        ParamInt$new("output_dim", lower = 1L, upper = Inf, default = 4L, tags = "train"),
        ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",
          levels = c("categorical_crossentropy", "sparse_categorical_crossentropy")),
        ParamUty$new("optimizer", default = "optimizer_adam(3*10^-4)", tags = "train"),
        ParamUty$new("metrics", default = "accuracy", tags = "train")
      ))
      ps$values = list(
        batch_momentum = 0.98,
        relaxation_factor = 1.0,
        sparsity_coefficient = 10^-5,
        num_decision_steps = 2L,
        output_dim = 4L,
        feature_dim = 4L,
        optimizer = optimizer_adam(lr = 3*10^-4),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
      )
      arch = KerasArchitectureTabNet$new(build_arch_fn = build_keras_tabnet, param_set = ps)
      super$initialize(architecture = arch)
      self$param_set$values$validation_split = 0 # Does not to work with tf_data.
    }
  )
)


#' @title Keras TabNet architecture
#' @rdname KerasArchitecture
#' @export
KerasArchitectureTabNet = R6::R6Class("KerasArchitectureTabNet",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn, x_transform, y_transform, param_set) {
      x_transform = function(task, pars) {
        x = lapply(task$feature_names, function(x) { as.matrix(task$data(cols = x))})
        names(x) = task$feature_names
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
      keras::install_keras(extra_packages = c('tensorflow-hub', 'tabnet==0.1.4.1').")
  }
  
  tabnet = reticulate::import("tabnet")
  feature_columns = make_tf_feature_cols(task)

  model = tabnet$TabNetClassifier(feature_columns, num_classes = length(task$class_names),
    feature_dim = pars$feature_dim, output_dim = pars$output_dim, 
    num_decision_steps = pars$num_decision_steps,
    relaxation_factor = pars$relaxation_factor,
    sparsity_coefficient= pars$sparsity_coefficient,
    batch_momentum = pars$batch_momentum,
    # FIXME: Understand / Read more about the parameters below.
    virtual_batch_size = NULL, norm_type = 'group', num_groups = 1)

  model %>% compile(
    loss = pars$loss,
    optimizer = pars$optimizer,
    metrics = pars$metrics
  )
}

# FIXME: This covers only the most basic feature types, needs to be extended.
make_tf_feature_cols = function(task) {
  assert_class(task, "Task")
  make_feature_column = function(id, type) {
    if (type == "numeric") tensorflow::tf$feature_column$numeric_column(id)
    else if (type == "integer") tensorflow::tf$feature_column$numeric_column(id)
    else tensorflow::tf$feature_column$categorical_column_with_vocabulary_list(id, task$levels(id)[[1]])
  }
  feature_columns = pmap(.f = make_feature_column, .x = task$feature_types)
}
