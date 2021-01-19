test_that("test data generator", {
  skip_on_os("solaris")
  library("reticulate")

  # Create a generator from the task
  t = mlr_tasks$get("iris")
  g = make_generator_from_task(t, shuffle = FALSE, batch_size = 20L)
  batch_1 = g$`__getitem__`(0L)

  expect_true(all(map_lgl(batch_1, function(x) nrow(x) == 20L)))

  x_20 = as.matrix(t$data()[1:20, 2:5])
  # Only equal up to 10^-6 (numerics) due to py -> R conversion
  expect_true(sum(abs(batch_1[[1]] - x_20)) < 1e-5)

  y_20 = to_categorical(as.integer(t$data()[[1]]) - 1)[1:20,]
  expect_true(all(batch_1[[2]] == y_20))

})
