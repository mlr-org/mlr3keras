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

