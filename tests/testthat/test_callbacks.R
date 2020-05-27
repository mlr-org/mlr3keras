context("Callbacks")

test_that("callback early stopping", {
  skip_on_os("solaris")
  # It is hard to reliably test whether callback works without long training
  lrn = mlr_learners$get("classif.kerasff")
  lrn$param_set$values$epochs = 25L
  lrn$param_set$values$layer_units = 2L
  lrn$param_set$values$callbacks = list(cb_es(patience = 3))
  lrn$train(mlr_tasks$get("iris"))
  expect_true(lrn$model$history$params$epochs > length(lrn$model$history$metrics))
  k_clear_session()
})

test_that("Callbacks can be initialized", {
  x = cb_es()
  expect_class(x, "keras.callbacks.Callback")
  x = cb_lr_scheduler()
  expect_class(x, "keras.callbacks.Callback")
  x = cb_tensorboard()
  expect_class(x, "keras.callbacks.Callback")
})
