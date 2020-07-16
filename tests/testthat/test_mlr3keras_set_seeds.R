context("set_seeds")

test_that("random.random reproducible", {
  skip_on_os("solaris")
  # random seed python:
  mlr3keras_set_seeds(random_seed = TRUE)
  a = reticulate::import("random")$random()
  mlr3keras_set_seeds(random_seed = TRUE)
  expect_equal(a, reticulate::import("random")$random())

})

test_that("numpy.random reproducible", {
  skip_on_os("solaris")
  # random seed python:
  mlr3keras_set_seeds(python_seed = TRUE)
  a = reticulate::import("numpy")$random$rand()
  mlr3keras_set_seeds(python_seed = TRUE)
  expect_equal(a, reticulate::import("numpy")$random$rand())
})

test_that("tf.random reproducible", {
  skip_on_os("solaris")
  tf = reticulate::import("tensorflow")
  # random seed python
  mlr3keras_set_seeds(tensorflow_seed = TRUE)
  a = tf$keras$backend$eval(tf$random$uniform(list(1L)))
  mlr3keras_set_seeds(tensorflow_seed = TRUE)
  expect_equal(a, tf$keras$backend$eval(tf$random$uniform(list(1L))))
  k_clear_session()
})


test_that("make reproducible outputs", {
  skip_on_os("solaris")

  build_model = function() {
    mlr3keras_set_seeds(3)
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
    learner$param_set$values$epochs = 3L
    learner$train(tsk("mtcars"))
    learner$predict(tsk("mtcars"))
  }

  results <- lapply(1:2, function(x) predict_mt())
  expect_true(identical(results[[2]]$data, results[[1]]$data))

  k_clear_session()
})
