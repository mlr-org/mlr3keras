#' Plot learning rate
#'
#' @description
#' Trains the model for few epochs, iteratively increasing learning rate.
#' Tries to provide insight with respect to the choice of a learning rate by iteratively increasing
#' learning rates from lr_min and lr_max, recording performance gains
#' (c.f.  Leslie N. Smith: Cyclical Learning Rates for Training Neural Networks 2015).
#'
#' 'find_lr' is also available via `Learner$lr_find(...)`.
#'
#' @param learner [`Learner`]\cr
#'   An mlr3 [`Learner`] from mlr3keras.
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param epochs [`integer`]\cr
#'   Number of epochs to train for. Defaults to 5.
#' @param lr_min [`numeric`]\cr
#'   Minimum learning rate to try. Defaults to 1e-4
#' @param lr_max [`numeric`]\cr
#'   Maximum learning rate to try. Defaults to 0.8
#' @param batch_size [`numeric`]\cr
#'   Batch size. Defaults to 128.
#' @export
find_lr = function(learner, task, epochs = 5, lr_min = 10^-4, lr_max = 0.8, batch_size = 128L) { # nocov start
  assert_learner(learner)
  assert_task(task)
  learner$param_set$values$epochs = assert_integerish(epochs)
  learner$param_set$values$batch_size = assert_integerish(batch_size)
  batches = epochs * ceiling(task$nrow / batch_size)
  if (inherits(task, "TaskClassif"))
    metric = "accuracy"
  else
    metric = "mean_squared_logarithmic_error"
  metrics = LogMetrics$new(log_metric = metric)
  lr_log = SetLogLR$new(batches = batches)
  learner$param_set$values$callbacks = list(lr_log, metrics)
  learner$train(task)

  data.frame(
    lr_hist = lr_log$lr_hist,
    log_metric = metrics$log_metric,
    log_metric_value = metrics$log_metric_value,
    loss = metrics$loss
  )
} # nocov end

# Plot data from 'find_lr'
plot_find_lr = function(data) {
  assert_data_frame(data)
  ggplot2::ggplot(data) +
    ggplot2::geom_line(
      ggplot2::aes_string(x = "lr_hist", y = "loss"), color = "darkblue"
    ) +
    ggplot2::scale_x_log10() +
    ggplot2::theme_bw() +
    ggplot2::xlab("Learning Rate (log-scale)") +
    ggplot2::ylab("Loss")
}