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

test_that("autotest binary low memory", {
  
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
  learner$param_set$values$low_memory = TRUE
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


test_that("autotest multiclass low memory", {
  
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
  learner$param_set$values$low_memory = TRUE  
  expect_learner(learner)
  
  skip_on_os("solaris")
  result = run_autotest(learner, exclude = "(feat_single|sanity|binary)")
  expect_true(result, info = result$error)
  k_clear_session()
})

test_that("can fit with binary_crossentropy", {
  skip_if_not(require("mlr3pipelines"))
  po_imp = PipeOpImputeMedian$new()
  po_lrn = PipeOpLearner$new(lrn("classif.keras"))
  model = keras_model_sequential() %>%
  layer_dense(units = 12L, input_shape = 8L, activation = "relu") %>%
  layer_dense(units = 12L, activation = "relu") %>%
  layer_dense(units = 1L, activation = "sigmoid") %>%
    compile(optimizer = optimizer_adam(3*10^-4),
      loss = "binary_crossentropy",
      metrics = "accuracy")
  po_lrn$param_set$values$model = model
  po_lrn$param_set$values$epochs = 10L
  pipe = po_imp %>>% po_lrn
  pipe$train(mlr_tasks$get("pima"))

  expect_list(pipe$state)
  expect_list(pipe$pipeops$classif.keras$state$model)
  prd = pipe$predict(mlr_tasks$get("pima"))
  expect_class(prd[[1]], "PredictionClassif")
  expect_true(is.null(prd[[1]]$prob))

  pipe$pipeops$classif.keras$learner$predict_type = "prob"
  prd2 = pipe$predict(mlr_tasks$get("pima"))
  expect_class(prd2[[1]], "PredictionClassif")
  expect_matrix(prd2[[1]]$prob, nrows = 768L, ncols = 2L)
  expect_true(all(prd[[1]]$response == prd2[[1]]$response))
  k_clear_session()
})

