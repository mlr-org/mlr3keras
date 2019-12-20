context("global learner methods")

test_that("save/plot/serialize works kerasff", {
  skip_on_os("solaris")

  tmphd5 = tempfile(fileext = ".hd5")
  tmprds = tempfile(fileext = ".rds")

  # Can be saved
  learner = mlr3::lrn("classif.kerasff")
  tsk = mlr_tasks$get("iris")
  learner$train(tsk)
  learner$save(tmphd5)
  expect_file_exists(tmphd5)
  prd = learner$predict(tsk)
  expect_class(prd, "Prediction")

  # Plot works before serialization
  p = learner$plot()
  expect_class(p, "ggplot")
  expect_true(1 %in% p$data$epoch)

  # Learner can be serialized
  saveRDS(learner, tmprds)
  expect_file_exists(tmprds)

  # And read back in
  lrn2 = readRDS(tmprds)
  expect_learner(lrn2)
  expect_list(lrn2$model)

  # We can also load model again and predict
  lrn2$load_model_from_file(tmphd5)
  prd2 = lrn2$predict(tsk)
  expect_class(prd2, "Prediction")
  expect_true(all(prd$response == prd2$response))

  # Plot works before serialization
  p2 = lrn2$plot()
  expect_class(p2, "ggplot")
  expect_true(1 %in% p2$data$epoch)

  unlink(tmphd5, force = TRUE)
  unlink(tmprds, force = TRUE)
  k_clear_session()
})
