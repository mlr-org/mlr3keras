context("global learner methods")

test_that("save/plot/serialize works kerasff", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  tsk = mlr_tasks$get("iris")
  learner$train(tsk)
  tmpdir = tempfile()
  learner$save(tmpdir)
  expect_directory_exists(tmpdir)
  unlink(tmpdir, force = TRUE)
  prd = learner$predict(tsk)
  expect_class(prd, "Prediction")

  p = learner$plot()
  expect_class(p, "ggplot")
  expect_true(1 %in% p$data$epoch)

  tmprds = tempfile(fileext = ".RDS")
  saveRDS(learner, tmprds)
  lrn2 = readRDS(tmprds)
  expect_file_exists(tmprds)
  expect_learner(lrn2)
  expect_list(lrn2$model)
  # FIXME: This currently breaks, as
  # lrn2$model$model is a null-pointer
  # prd2 = lrn2$predict(tsk)
  unlink(tmprds, force = TRUE)
  k_clear_session()
})
