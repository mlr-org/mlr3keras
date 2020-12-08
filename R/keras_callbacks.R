#' `cb_es`: Early stopping callback
#' @param monitor [`character`]\cr
#'   Quantity to be monitored.
#' @param patience [`integer`]\cr
#'   Number of iterations without improvement to wait before stopping.
#' @rdname callbacks
#' @export
cb_es = function(monitor = 'val_loss', patience = 3L) {
  assert_character(monitor, len = 1)
  assert_int(patience, lower = 0L)
  callback_early_stopping(monitor = monitor, patience = patience)
}

#' Learning rate scheduler callback: cosine annealing
#'
#' For more information see:
#' Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983.
#'
#' Closed form:
#'  \eqn{\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
#'          \cos(\frac{T_{cur}}{T_{max}}\pi))}
#'
#' @param eta_max [`numeric`]\cr
#'   Max  learning rate.
#' @param T_max [`integer`]\cr
#'   Reset learning rate every T_max epochs.  Default 10.
#' @param T_mult [`integer`]\cr
#'   Multiply T_max by T_mult every T_max iterations. Default 2.
#' @param M_mult [`numeric`]\cr
#'   Decay learning rate by factor 'M_mult' after each learning rate reset.
#' @param eta_min [`numeric`]\cr
#'   Minimal learning rate.
#'
#' @rdname callbacks
#' @export
cb_lr_scheduler_cosine_anneal = function(eta_max = 0.01, T_max = 10.0, T_mult = 2.0, M_mult = 1.0, eta_min = 0.0) {
  callback_learning_rate_scheduler(
    k$experimental$CosineDecayRestarts(eta_max, T_max, t_mul = T_mult, m_mul = M_mult, alpha = eta_min)
  )
}


#' Learning rate scheduler callback: exponential decay
#'
#' @rdname callbacks
#' @export
cb_lr_scheduler_exponential_decay = function() {
  callback_learning_rate_scheduler(function(epoch, lr) {
    lr * 1.0/(1.0 * epoch)})
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
#' Note, that the specified metric must be additionally supplied to
#' 'keras::compile' or as a hyperparameter of kerasff in
#' order to be trackable during training.
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
