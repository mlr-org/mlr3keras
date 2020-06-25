context("set_seeds")

test_that("make reproducable outputs", {
  skip_if_not(tensorflow::tf_version() < "2.1", "R Generators only work for tensorflow < 2.1")
  skip_on_os("solaris")

  build_model = function() {
    mlr3keras_set_seeds()
    keras_model_sequential() %>%
      layer_dense(units = 12L, input_shape = 10L, activation = "relu") %>%
      layer_dense(units = 12L, activation = "relu") %>%
      layer_dense(units = 1L, activation = "linear") %>%
      compile(optimizer = optimizer_adam(lr = 10e-3),
              loss = "mean_squared_error",
              metrics = "mean_squared_logarithmic_error")
}

  predict_mt <- function() {
    learner = LearnerRegrKeras$new()
    learner$param_set$values$model = build_model()
    learner$param_set$values$low_memory = TRUE
    learner$param_set$values$epochs = 3L
    learner$train(tsk("mtcars"))
    learner$predict(tsk("mtcars"))
  }

  results <- lapply(1:2, function(x) predict_mt())
  expect_true(identical(results[[2]]$data, results[[1]]$data))

  k_clear_session()
})
