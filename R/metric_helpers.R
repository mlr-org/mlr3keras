#' Create a keras metric from a [mlr3::Measure].
#' @param measure `character`
#' @export
metric_custom_mlr3 = function(measure) {
  if (!inherits(measure, "Measure")) {
    measure = msr(measure)
  }
  custom_metric(measure$id, keras_reflections$metrics_from_mlr3[[measure$task_type]])
}

# Enter functions per `task_type` into dict.
keras_reflections$metrics_from_mlr3 = list(
  "classif" = function(y_true, y_pred) {
    y_true = factor(k_argmax(y_true), levels = y_true$shape[[2]])
    colnames(y_pred) = levels(y_true)
    p = PredictionClassif$new(
      row_ids = seq_along(y_true),
      truth = y_true,
      prob = y_pred
    )
    measure$score(p)
  },
  "regr" = function(y_true, y_pred) {
    p = PredictionClassif$new(
      row_ids = seq_along(y_true),
      truth = y_true,
      prob = y_pred
    )
    measure$score(p)
   }
)
