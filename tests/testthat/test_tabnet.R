context("tabnet")

test_that("autotest classif tabnet", {
  skip_if_not(reticulate::py_module_available("tabnet"))
  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})