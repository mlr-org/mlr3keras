context("kerasff")

test_that("autotest", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  expect_learner(learner)

  learner$param_set$values$epochs = 3L
  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})


test_that("can fit logistic regression", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
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


test_that("works with pipelines", {
  skip_on_os("solaris")
  skip_if_not(require("mlr3pipelines"))

  po_enc = PipeOpEncode$new()
  po_lrn = PipeOpLearner$new(lrn("classif.kerasff"))
  po_lrn$param_set$values$epochs = 3L
  po_lrn$param_set$values$layer_units = integer()

  pipe = po_enc %>>% po_lrn
  expect_true(!pipe$is_trained)
  pipe$train(mlr_tasks$get("pima"))
  expect_true(pipe$is_trained)
  prd = pipe$predict(mlr_tasks$get("pima"))
  expect_class(prd[[1]], "PredictionClassif")

  po_lrn = PipeOpLearner$new(lrn("classif.kerasff"))
  po_lrn$param_set$values$epochs = 3L
  po_lrn$param_set$values$layer_units = c(10, 5)
  pipe = po_enc %>>% po_lrn
  expect_true(!pipe$is_trained)
  pipe$train(mlr_tasks$get("pima"))
  expect_true(pipe$is_trained)
  prd = pipe$predict(mlr_tasks$get("pima"))
  expect_class(prd[[1]], "PredictionClassif")

  k_clear_session()
})

