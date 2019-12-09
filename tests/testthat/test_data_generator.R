context("test data generator")

test_that("test data generator", {
  skip_on_os("solaris")
  library("reticulate")

  # Create a generator from the task
  tsk = mlr_tasks$get("iris")

  gen = py_iterator(
    make_data_generator(tsk, batch_size = 73), # Make generator
    completed = NULL)

  batch1 = generator_next(gen) # Get next batch

  expect_equal(length(batch1[[1]]), 4) # 4 features
  expect_equal(length(batch1[[1]]$Sepal.Width), 73) # batch_size records
  expect_equal(length(batch1[[2]]), 73) # batch_size records

  batch2 = generator_next(gen) # Get next batch

  expect_equal(length(batch2[[1]]), 4) # 4 features
  expect_equal(length(batch2[[1]]$Sepal.Width), 73) # batch_size records
  expect_equal(length(batch2[[2]]), 73) # batch_size records

  batch3 = generator_next(gen) # Get last batch

  expect_equal(length(batch3[[1]]), 4) # 4 features
  expect_equal(length(batch3[[1]]$Sepal.Width), 150 - 73 - 73) # remaining records
  expect_equal(length(batch3[[2]]), 150 - 73 - 73) # remaining records

  batch4 = generator_next(gen) # Get next batch - reset

  expect_equal(length(batch4[[1]]), 4) # 4 features
  expect_equal(length(batch4[[1]]$Sepal.Width), 73) # batch_size records
  expect_equal(length(batch4[[2]]), 73) # batch_size records

})
