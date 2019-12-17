context("tfestimators")

test_that("autotest classification works", {
  skip_on_os("solaris")
  skip_if_not(require("tensorflow"))
  learner = LearnerClassifTfEstimator$new()
  learner$param_set$values = list(estimator = tf$estimator$LinearClassifier)
  # expect_learner(learner)

  tsk = mlr_tasks$get("german_credit")
  tsk = tsk$select(c("age", "amount"))
  learner$train(tsk)
  k_clear_session()
})
