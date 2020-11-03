context("Resampling works for keras models")

test_that("can be trained with cv3", {
  skip_on_os("solaris")
  # Build model
  t = tsk("iris")
  po = PipeOpAutoencoder$new()
  po$train(list(t))
  po$predict(list(t))
})