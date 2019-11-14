context("kerasff")

test_that("autotest", {
  learner = mlr3::lrn("classif.kerasff")
  expect_learner(learner)

  skip_on_os("solaris")
  learner$param_set$values$epochs = 3L
  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})
