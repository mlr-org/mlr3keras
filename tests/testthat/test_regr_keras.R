context("keras regression custom model")

test_that("autotest regression custom model", {
  skip_on_os("solaris")
  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 2L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 1L, activation = "linear") %>%
    compile(optimizer = optimizer_adam(lr = 10e-3),
      loss = "mean_squared_error",
      metrics = "mean_squared_logarithmic_error")
  learner = LearnerRegrKeras$new()
  learner$param_set$values$model = model
  learner$param_set$values$epochs = 3L
  expect_learner(learner)

  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest low memory generator", {
  skip_on_os("solaris")
  model = keras_model_sequential() %>%
    layer_dense(units = 12L, input_shape = 2L, activation = "relu") %>%
    layer_dense(units = 12L, activation = "relu") %>%
    layer_dense(units = 1L, activation = "linear") %>%
    compile(optimizer = optimizer_adam(lr = 10e-3),
            loss = "mean_squared_error",
            metrics = "mean_squared_logarithmic_error")
  learner = LearnerRegrKeras$new()
  learner$param_set$values$model = model
  learner$param_set$values$low_memory=TRUE
  learner$param_set$values$epochs = 3L
  expect_learner(learner)

  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest low memory zero validation_split", {
  skip_on_os("solaris")
  model = keras_model_sequential() %>%
    layer_dense(units = 12L, input_shape = 2L, activation = "relu") %>%
    layer_dense(units = 12L, activation = "relu") %>%
    layer_dense(units = 1L, activation = "linear") %>%
    compile(optimizer = optimizer_adam(lr = 10e-3),
            loss = "mean_squared_error",
            metrics = "mean_squared_logarithmic_error")
  learner = LearnerRegrKeras$new()
  learner$param_set$values$model = model
  learner$param_set$values$low_memory=TRUE
  learner$param_set$values$validation_split=0
  learner$param_set$values$epochs = 3L
  expect_learner(learner)

  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})

# ----------------------------------------------------------------------------------------
context("keras regression feed forward model")

test_that("autotest feed forward", {
  skip_on_os("solaris")
  learner = LearnerRegrKerasFF$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)")
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("Learner methods", {
  fp = tempfile(fileext = ".h5")
  lrn = lrn("regr.kerasff", epochs = 3L)
  expect_error(lrn$plot())
  expect_error(lrn$save(fp))
  lrn$train(mlr_tasks$get("mtcars"))

  # Saving to h5
  lrn$save(fp)
  expect_file_exists(fp)
  unlink(fp)

  # Plotting
  p = lrn$plot()
  expect_class(p, "ggplot")
  k_clear_session()
})
