test_that("autotest classif smlp", {
  skip_on_os("solaris")
  learner = LearnerClassifShapedMLP$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})


test_that("autotest regr smlp", {
  skip_on_os("solaris")
  learner = LearnerRegrShapedMLP$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})
