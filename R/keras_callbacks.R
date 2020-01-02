#' `cb_es`: Early stopping callback
#' @param patience [`integer`]\cr
#'   Number of iterations without improvement to wait before stopping.
#' @rdname callbacks
#' @export
cb_es = function(patience = 3L) {
  assert_int(patience, lower = 0L)
  callback_early_stopping(monitor = 'val_loss', patience = 3)
}

#' `cb_lrs`: Learning rate scheduler callback
#' @rdname callbacks
#' @export
cb_lr_scheduler = function() {
  callback_learning_rate_scheduler(function(epoch, lr) {lr * 1/(1 * epoch)})
}

#' `cb_tb`: Tensorboard callback
#' @rdname callbacks
#' @export
cb_tensorboard = function() {
  callback_tensorboard()
}

#' `cb_lr_log`: Learning rate logger callback
#' @rdname callbacks
#' @export
cb_lr_log = function() {
  lr_hist = numeric()
  callback_lr_log = function(batch, logs){
    lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
  }
  callback_lambda(on_batch_begin=callback_lr_log)
}


#' `LogMetrics`: Batch-wise Metrics Logger Callback
#' @rdname callbacks
#' @export
LogMetrics = R6::R6Class("LogMetrics",
  inherit = KerasCallback,
  public = list(
    loss = NULL,
    log_metric = NULL,
    log_metric_value = NULL,
    initialize = function(log_metric = "accuracy") {
      self$log_metric = assert_string(log_metric)
    },
    on_batch_end = function(batch, logs = list()) {
      self$loss = c(self$loss, logs[["loss"]])
      self$log_metric_value = c(self$log_metric_value, logs[[self$log_metric]])
    }
  )
)

#' `SetLogLR`: Batch-wise Metrics Setter Callback
#' @rdname callbacks
#' @export
SetLogLR = R6::R6Class("SetLogLR",
  inherit = KerasCallback,
  public = list(
    batches = NULL,
    lr_min = NULL,
    lr_max = NULL,
    iter = 0,
    lr_hist = c(),
    iter_hist = c(),
    initialize = function(batches = 100, lr_min = 0.0001, lr_max = 0.1) {
      self$lr_min = assert_number(lr_min, lower = 0)
      self$lr_max = assert_number(lr_max, lower = 0)
      self$batches = assert_int(batches, lower = 1L)
    },
    on_batch_begin = function(batch, logs = list()) {
      l_rate = seq(0, 15, length.out = self$batches)
      l_rate = (l_rate - min(l_rate)) / max(l_rate)
      l_rate = self$lr_min + l_rate * (self$lr_max - self$lr_min)
      self$iter = self$iter + 1
      LR = l_rate[self$iter] # if number of iterations > l_rate values, make LR constant to last value
      if (is.na(LR)) LR = l_rate[length(l_rate)] #nocov
      k_set_value(self$model$optimizer$lr, LR)
      self$lr_hist = c(self$lr_hist, k_get_value(self$model$optimizer$lr))
      self$iter_hist = c(self$iter_hist, k_get_value(self$model$optimizer$iterations))
    }
  )
)
