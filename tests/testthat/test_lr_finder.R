context("lr finder")

test_that("lr finder works", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  expect_learner(learner)
  lr_finder(learner, mlr_tasks$get("iris"), batch_size = 25, epochs = 30)
  # expect_class(p, "ggplot")
  k_clear_session()
})

