#' @title Generic Learner for 'tfestimators'
#'
#' @usage NULL
#' @aliases mlr_learners_classif.tfestimator
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifTfEstimator$new()
#' mlr3::mlr_learners$get("classif.tfestimator")
#' mlr3::lrn("classif.tfestimator")
#' ```
#'
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name classif.tfestimator
#' @template example
#' @export
LearnerClassifTfEstimator = R6::R6Class("LearnerClassifTfEstimator",
  inherit = LearnerClassif,
  public = list(
    initialize = function(id = "classif.tfestimator") {
      ps = ParamSet$new(list(
        ParamUty$new("estimator", tags = "train"),
        ParamUty$new("optimizer", default = NULL, tags = "train"),
        ParamInt$new("epochs", default = 30L, lower = 1L, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict"))
      ))
      ps$values = list(optimizer = NULL, batch_size = 128L, epochs = 30L)
      # arch = KerasArchitectureTfEstimator$new(build_arch_fn = build_keras_tabnet, param_set = ps)
      super$initialize(
        id = assert_character(id, len = 1),
        param_set = ps,
        predict_types = c("response", "prob"),,
        feature_types = c("integer", "numeric", "factor", "logical"),
        properties = c("twoclass", "multiclass"),
        packages = "tensorflow",
        man = "mlr3keras::mlr_learners_classif.tfestimator"
      )
    },

    train_internal = function(task, pars) {
      pars = self$param_set$get_values(tags = "train")
      feature_columns = make_tf_feature_cols(task)

      pars$estimator(feature_columns = feature_columns, optimizer = pars$optimizer)

      input_fn = tfestimators::input_fn(
        object = as.data.frame(task$data()),
        features = task$feature_names,
        response = task$target_names,
        batch_size = 128L,
        num_epochs = 3L
      )

      pars$estimator$train(input_fn = input_fn(pars$estimator))

    }
  )
)

#' @title Keras TfEstimator architecture
#' @rdname KerasArchitecture
#' @family KerasArchitectures
#' @export
KerasArchitectureTfEstimator = R6::R6Class("KerasArchitectureTfEstimator",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn, x_transform, y_transform, param_set) {
      x_transform = function(features, pars) {
        x = lapply(names(features), function(x) {
          x = features[, get(x)]
          if (is.numeric(x) || is.integer(x)) {
            as.matrix(as.numeric(x))
          } else if (is.factor(x)) {
            as.list(map(x, as.character))
          } else {
            as.matrix(as.integer(x))
          }
        })
        names(x) = names(features)
        # x = tf$data$Dataset$from_tensor_slices(as.list(features))
        return(x)
      }
      super$initialize(build_arch_fn = build_arch_fn, x_transform = x_transform, param_set = param_set)
    }
  )
)

build_tfestimator = function(task, pars) {
  requireNamespace("tfestimators")
  feature_columns = make_tf_feature_cols(task)
  tabnet_param_names = c("feature_dim", "output_dim", "num_decision_steps", "relaxation_factor",
    "sparsity_coefficient", "virtual_batch_size", "norm_type", "num_groups")
  if (pars$stacked) tabnet_param_names = c(tabnet_param_names, "num_layers")

  if (inherits(task, "TaskClassif")) {
    if (pars$stacked) clf = tabnet$StackedTabNetClassifier
    else clf = tabnet$TabNetClassifier
    model = invoke(clf,
      feature_columns = feature_columns, num_classes = length(task$class_names),
      .args = pars[tabnet_param_names])
  } else if (inherits(task, "TaskRegr")) {
    if (pars$stacked) regr = tabnet$StackedTabNetRegressor
    else regr = tabnet$TabNetRegressor
    model = invoke(regr,
      feature_columns = feature_columns, num_regressors = 1L,
      .args = pars[tabnet_param_names])
  }
}
