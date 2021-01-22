test_that("autotest classif tabnet", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))
  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest classif stacked tabnet", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))
  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$stacked = TRUE
  learner$param_set$values$num_layers = 2L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest regr tabnet", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))
  learner = LearnerRegrTabNet$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("test tabnet on pima", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))
  skip_if_not(require("mlr3pipelines"))
  po_enc = PipeOpImputeMedian$new()
  po_lrn = PipeOpLearner$new(lrn("classif.tabnet"))
  po_lrn$param_set$values$epochs = 3L
  pipe = po_enc %>>% po_lrn
  expect_true(!pipe$is_trained)
  pipe$train(mlr_tasks$get("pima"))
  expect_true(pipe$is_trained)
  prd = pipe$predict(mlr_tasks$get("pima"))
  expect_class(prd[[1]], "PredictionClassif")
})
