test_that("Keras CNN", {
  skip_on_os("solaris")
  skip_on_os("windows") # FIXME: Remove this at some point!
  dir = system.file(file.path("extdata", "images"), package = "mlr3keras")
  dt = imagepathdf_from_imagenet_dir(dir)
  expect_file_exists(dt$image)
  t = TaskClassif$new(id = "internal", backend = dt, target="class")
  # Learner
  l = LearnerClassifKerasCNN$new()
  l$param_set$values$epochs = 12L
  l$param_set$values$optimizer = optimizer_rmsprop()
  l$param_set$values$application = application_mobilenet
  l$param_set$values$validation_fraction = 0
  suppressWarnings(l$train(t))
  prd = suppressWarnings(l$predict(t))
  expect_learner(l)
  expect_true(!is.null(l$state))
  expect_prediction(prd)
})
