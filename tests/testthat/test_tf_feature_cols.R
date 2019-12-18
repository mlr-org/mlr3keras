context("conversion to tf_feature_cols works")

test_that("classif tabnet with logical or factor features", {
  skip_on_os("solaris")
  skip_if_not(reticulate::py_module_available("tabnet"))

  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$num_groups = 1L
  expect_learner(learner)
  expect_true("logical" %in% learner$feature_types)
  expect_true("factor" %in% learner$feature_types)
  tsk = mlr_tasks$get("german_credit")
  tsk$select(c("age", "amount", "foreign_worker", "job"))
  learner$train(tsk)
  prd = learner$predict(tsk)
  expect_r6(prd, "Prediction")

  learner = LearnerClassifTabNet$new()
  learner$param_set$values$epochs = 3L
  learner$param_set$values$num_groups = 1L
  tsk = mlr_tasks$get("zoo")
  tsk$select(c("legs", "aquatic"))
  learner$train(tsk)
  prd = learner$predict(tsk)
  expect_r6(prd, "Prediction")
  k_clear_session()
})


test_that("tf_feature_cols returns list", {
  map(mlr_tasks$keys(), function(id) {
    tsk = mlr_tasks$get(id)
    lst = make_tf_feature_cols(tsk)
    map(lst, assert_class, "tensorflow.python.feature_column.feature_column._FeatureColumn")
    expect_list(lst, len = tsk$ncol - 1L)
    expect_integerish(get_tf_num_features(tsk, pars = list(embed = NULL)), lower = tsk$ncol -1L, upper = Inf)
    # FIXME: We could check for expected number of features etc. here.
  })
})

test_that("get_default_embed_size", {
  expect_integerish(get_default_embed_size(as.factor(letters[seq_len(1)])), lower = 2L)
  expect_integerish(get_default_embed_size(as.factor(letters[seq_len(2)])), lower = 2L)
  expect_integerish(get_default_embed_size(as.factor(letters[seq_len(5)])), lower = 2L)
  expect_integerish(get_default_embed_size(as.factor(letters[seq_len(26)])), lower = 2L)
  expect_integerish(get_default_embed_size(as.factor(1:10^5)), lower = 2L, upper = 600L)
})

