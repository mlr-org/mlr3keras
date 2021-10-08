test_that("imagepathdf_from_imagenet_dir", {
  skip_on_os("solaris")
  dir = system.file(file.path("extdata", "images"), package = "mlr3keras")
  dt = df_from_imagenet_dir(dir)
  expect_file_exists(dt$image)
  t = TaskClassif$new(id = "internal", backend = dt, target="class")
  ftdt = t$feature_types
  expect_true(ftdt$id == "image")
  expect_task(t)
  expect_data_table(dt, min.rows=3L, max.rows=3L)
  expect_file_exists(dt$image[1])
  expect_file_exists(dt$image[2])

  t$set_col_roles("image", "uri")
})
