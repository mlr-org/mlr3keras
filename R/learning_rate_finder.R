# Implementation of a learning rate finder.
# This is a wrapper around find_lr, that creates the model
# and returns the learning rate.
# FIXME: This needs to be exported and get some docs.
lr_finder = function(learner, task, epochs = 5, lr_min = 10^-4, lr_max = 0.8, batch_size = 128L) { # nocov start
  assert_learner(learner)
  assert_task(task)
  learner$param_set$values$epochs = assert_integerish(epochs)
  learner$param_set$values$batch_size = assert_integerish(batch_size)
  batches = epochs * ceiling(task$nrow / batch_size)
  metrics = LogMetrics$new()
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
