context("global learner methods")

test_that("save/plot/serialize works kerasff", {
  skip_on_os("solaris")
  learner = mlr3::lrn("classif.kerasff")
  tsk = mlr_tasks$get("iris")
  learner$train(tsk)
  tmpdir = tempfile()
  learner$save(tmpdir)
  assert_directory_exists(tmpdir)
  unlink(tmpdir, force = TRUE)
  prd = learner$predict(tsk)

  p = learner$plot()
  assert_class(p, "ggplot")
  assert_true(1 %in% p$data$epoch)

  tmprds = tempfile(fileext = ".RDS")
  saveRDS(learner, tmprds)
  lrn2 = readRDS(tmprds)
  assert_file_exists(tmprds)
  assert_learner(lrn2)
  assert_list(lrn2$model)
  # FIXME: This currently breaks, as
  # lrn2$model$model is a null-pointer
  # prd2 = lrn2$predict(tsk)
  unlink(tmprds, force = TRUE)
  k_clear_session()
})
