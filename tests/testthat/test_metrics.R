test_that("mlr3 classification metric", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  learner$param_set$values$metrics = metric_custom_mlr3("classif.acc")
  expect_learner(learner)

  learner$param_set$values$epochs = 3L
  learner$param_set$values$layer_units = integer()
  learner$train(mlr_tasks$get("iris"))

  expect_list(learner$state)
  expect_list(learner$state$model)
  prd = learner$predict(mlr_tasks$get("iris"))
  expect_class(prd, "PredictionClassif")
  k_clear_session()
})

