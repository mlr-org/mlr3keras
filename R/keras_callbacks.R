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
cb_lrs = function() {
  callback_learning_rate_scheduler(function(epoch, lr) {lr * 1/(1 * epoch)})
}

#' `cb_tb`: Tensorboard callback
#' @rdname callbacks
#' @export
cb_tb = function() {
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
