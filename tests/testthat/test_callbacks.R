context("Callbacks")

test_that("callback early stopping", {
  skip_on_os("solaris")
  # It is hard to reliably test whether callback works without long training
  lrn = mlr_learners$get("classif.kerasff")
  lrn$param_set$values$epochs = 25L
  lrn$param_set$values$layer_units = 2L
  lrn$param_set$values$callbacks = list(cb_es(3))
  lrn$train(mlr_tasks$get("iris"))
  expect_true(lrn$model$history$params$epochs > length(lrn$model$history$metrics))
  k_clear_session()
})

# test_that("callback lr scheduler and history", {
#   # It is hard to reliably test whether callback works without long training
#   lrn = mlr_learners$get("classif.kerasff")
#   lrn$param_set$values$epochs = 5L
#   lrn$param_set$values$layer_units = 2L
#   lrn$param_set$values$callbacks = list(cb_lrs(), cb_lr_log())
#   lrn$train(mlr_tasks$get("iris"))
#   assert_true(lrn$model$history$params$epochs > length(lrn$model$history$metrics))
#   k_clear_session()
# })
