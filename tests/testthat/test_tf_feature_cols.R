context("conversion to tf_feature_cols works")

test_that("classif tabnet with logical features", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))


  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$num_groups = 1L
  expect_learner(learner)
  expect_true("logical" %in% learner$feature_types)
  tsk = mlr_tasks$get("german_credit")
  tsk$select(c("age", "amount", "foreign_worker"))
  learner$train(tsk)

  learner$train(mlr_tasks$get("zoo"))
  k_clear_session()
})


# require("tensorflow")
# require("keras")
# fts = list(
#   tensorflow::tf$feature_column$numeric_column("age"),
#   tensorflow::tf$feature_column$numeric_column("amount"),
#   tensorflow::tf$feature_column$embedding_column(
#     tensorflow::tf$feature_column$categorical_column_with_vocabulary_list("foreign_worker", list("yes", "no"), default_value = "no"),
#     dimension = 2L
#   )
# )


# library(tfdatasets)
# hearts_dataset <- tensor_slices_dataset(hearts)
# spec <- feature_spec(hearts_dataset, target ~ .)