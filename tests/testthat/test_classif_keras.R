context("keras custom model")

test_that("autotest binary", {

  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 2L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 2L, activation = "softmax") %>%
    compile(optimizer = optimizer_sgd(),
      loss = "categorical_crossentropy",
      metrics = c("accuracy"))
  learner = LearnerClassifKeras$new()
  learner$param_set$values = list(model = model)
  learner$param_set$values$epochs = 2L
  expect_learner(learner)

  skip_on_os("solaris")
  result = run_autotest(learner, exclude = "(feat_single|sanity|multiclass)")
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("autotest multiclass", {

  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 2L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 3L, activation = "softmax") %>%
    compile(optimizer = optimizer_sgd(),
      loss = "categorical_crossentropy",
      metrics = c("accuracy"))
  learner = LearnerClassifKeras$new()
  learner$param_set$values = list(model = model)
  learner$param_set$values$epochs = 2L
  expect_learner(learner)

  skip_on_os("solaris")
  result = run_autotest(learner, exclude = "(feat_single|sanity|binary)")
  expect_true(result, info = result$error)
  k_clear_session()
})
