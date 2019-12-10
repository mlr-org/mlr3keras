context("conversion to tf_feature_cols works")

test_that("classif tabnet with logical features", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))


  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  expect_true("logical" %in% learner$feature_types)
  tsk = mlr_tasks$get("german_credit")
  tsk$select(c("age", "amount", "foreign_worker"))
  learner$train(tsk)

  learner$train(mlr_tasks$get("zoo"))
  k_clear_session()
})


