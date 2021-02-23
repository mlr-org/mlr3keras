test_that("autotest classif deep_wide", {
  skip_on_os("solaris")
  learner = LearnerClassifKerasDeepWide$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest regr deep_wide", {
  skip_on_os("solaris")
  learner = LearnerRegrKerasDeepWide$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$use_embedding = TRUE
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
  k_clear_session()
})