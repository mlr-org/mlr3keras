context("kerasff")

test_that("classif kerasff works multiclass", {
  library(mlr3)
  tsk = mlr_tasks$get("iris")
  lrn = LearnerClassifkerasff$new()
  lrn$train(tsk)
  lrn$predict(tsk)
})

test_that("classif kerasff works binaryclass", {
  library(mlr3)
  tsk = mlr_tasks$get("german_credit")$select(c("age", "amount"))
  lrn = LearnerClassifkerasff$new()
  lrn$train(tsk)
  lrn$predict(tsk)
})