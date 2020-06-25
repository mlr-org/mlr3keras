context("Resampling works for keras models")

test_that("can be trained with cv3", {
  skip_on_os("solaris")
  # Build model
  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 4L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 3L, activation = "softmax") %>%
    compile(optimizer = optimizer_sgd(),
      loss = "categorical_crossentropy",
      metrics = c("accuracy"))
  learner = LearnerClassifKeras$new()
  learner$param_set$values$model = model
  learner$param_set$values$epochs = 2L

  # Resample
  rsm = rsmp("cv", folds = 3)
  rr = resample(mlr_tasks$get("iris"), learner, rsm, store_models = TRUE)
  expect_class(rr, "ResampleResult")
  expect_numeric(rr$aggregate(msr("classif.acc")), lower = 0, upper = 1)
  k_clear_session()
})

test_that("tuning works without pipelines", {
  skip_on_os("solaris")
  skip_if_not(require("mlr3tuning"))

  # Build model
  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 4L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 3L, activation = "softmax") %>%
    compile(optimizer = optimizer_sgd(),
      loss = "categorical_crossentropy",
      metrics = c("accuracy"))
  learner = LearnerClassifKeras$new()
  learner$param_set$values$model = model
  learner$param_set$values$epochs = 2L

  # Parameter Set
  param_set = ParamSet$new(list(
      paradox::ParamInt$new("epochs", lower = 1L, upper = 3L)))
  param_set$trafo = function(x, param_set) {
      x$epochs = ceiling(exp(x$epochs))
      return(x)
  }

  # Tuning Params
  task = mlr_tasks$get("iris")
  resampling = rsmp("holdout")
  measure = msr("classif.ce")
  tuner = tnr("grid_search", resolution = 2)
  terminator = term("evals", n_evals = 2)
  instance = TuningInstance$new(
    task = task,
    learner = learner,
    resampling = resampling,
    measures = measure,
    search_space = param_set,
    terminator = terminator
  )
  tuner$optimize(instance)
  out = instance$result$params
  assert_class(instance, "TuningInstance")
  assert_class(instance$archive, "Archive")
  assert_list(out)
  assert_integerish(out$epochs, lower = 1, upper = 30)
  k_clear_session()
})
