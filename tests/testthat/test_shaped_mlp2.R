test_that("autotest classif smlp 2", {
  skip_on_os("solaris")
  learner = LearnerClassifShapedMLP2$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})


test_that("autotest regr smlp 2", {
  skip_on_os("solaris")
  learner = LearnerRegrShapedMLP2$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})
