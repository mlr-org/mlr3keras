context("lr finder")

test_that("lr finder works", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  expect_learner(learner)
  data = find_lr(learner, mlr_tasks$get("iris"), batch_size = 25, epochs = 30)
  expect_data_frame(data)
  p = plot_find_lr(data)
  expect_true(inherits(p, "ggplot"))
})


test_that("lr finder works", {
  skip_on_os("solaris")
  learner = mlr3::lrn("regr.kerasff")
  expect_learner(learner)
  data = find_lr(learner, mlr_tasks$get("mtcars"), batch_size = 25, epochs = 30)
  expect_data_frame(data)
  p = plot_find_lr(data)
  expect_true(inherits(p, "ggplot"))
})

