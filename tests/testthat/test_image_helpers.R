context("Image Helpers")

test_that("imagepathdf_from_imagenet_dir", {
  skip_on_os("solaris")
  dir = system.file("extdata/images", package = "mlr3keras")
  dt = imagepathdf_from_imagenet_dir(dir)
  t = TaskClassif$new(id = "internal", backend = dt, target="class")
  ftdt = t$feature_types
  expect_true(ftdt$id == "image")
  expect_true(ftdt$type == "imagepath")
  expect_task(t)
  expect_data_table(dt, types = c("factor", "imagepath"), min.rows=3L, max.rows=3L)
  expect_file_exists(dt$image[1])
  expect_file_exists(dt$image[2])
})
