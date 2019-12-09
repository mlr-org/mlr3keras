context("entity embedding")

test_that("entity embedding works for all tasks", {
  skip_on_os("solaris")
  for (k in mlr_tasks$keys()) {
    task = mlr3::mlr_tasks$get(k)
    embds = make_embedding(task)
    expect_list(embds, len = 2L, names = "named")
    dt = task$feature_types[type %in% c("character", "factor", "ordered"), ]
    expect_true(length(embds$inputs) == nrow(dt) + 1)
    expect_class(embds$layers, "tensorflow.tensor")
    map(embds$inputs, expect_class, "tensorflow.tensor")
  }
  k_clear_session()
})

test_that("entity embedding works for all tasks", {
  for (k in mlr_tasks$keys()) {
    task = mlr3::mlr_tasks$get(k)
    embds = reshape_task_embedding(task)
    expect_list(embds, len = 2L, names = "named")
    dt = task$feature_types[type %in% c("character", "factor", "ordered"), ]
    expect_true(length(embds$fct_levels) == nrow(dt))
    expect_true(length(embds$fct_levels) == length(embds$data) - 1)
  }
})
