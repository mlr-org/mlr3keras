#' Create a data generator for a task
#'
#' Creates a data generator for a mlr3 task.
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param training
#'   If true, shuffles the data, includes y, and loops infinitely.
#'   Training=FALSE is currently not used.
#' @param batch_size [`numeric`]\cr
#'   Batch size.
#' @param filter_ids [`integer`]\cr
#'   Id's to filter.
#' @param x_transform [`function`]\cr
#'   Function used to transform data to a keras input format for features.
#' @param y_transform [`function`]\cr
#'   Function used to transform data to a keras input format for the response.
#' @param ... [`any`]\cr
#'   Further arguments passed on to `x_transform` and `y_transform`
#' @examples
#' require("keras")
#' tsk = mlr3::mlr_tasks$get("iris")
#' gen = reticulate::py_iterator(make_data_generator(tsk))
#' data = generator_next(gen) # Get next batch
#' @export
make_data_generator = function(
  task,
  training = TRUE,
  batch_size = 128,
  filter_ids = NULL,
  x_transform = function(x) x,
  y_transform = function(y) y
  ) {

  row_ids = task$row_roles$use

  if (!is.null(filter_ids)) {
    row_ids = intersect(row_ids, filter_ids)
  }

  # Sample from task$row_roles$use which are the usable row ids of the task
  if (training) {
    order_records = sample(row_ids)
  } else {
    order_records = row_ids
  }

  start = 1L
  function() {
    end = min(start + batch_size - 1, length(row_ids))

    # Get records before transform
    features = task$data(rows=order_records[start:end], cols = task$feature_names)
    target = task$data(rows=order_records[start:end], cols = task$target_names)[[task$target_names]]

    # Update slice for next run
    start <<- start + batch_size
    if (start >= length(row_ids) && training) {
      # The generator is expected to loop over its data indefinitely during training.
        order_records <<- sample(row_ids)
        start <<- 1
    }

    # Output in right format for both generators
    if (training) {
      list(
        x_transform(features),
        y_transform(target)
      )
    } else {
      x_transform(features)
    }
  }
}

#' Create train / validation data generators from a task and params
#'
#' Creates a data generator for a mlr3 task.
#' @param
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param x_transform [`function`]\cr
#'   Function used to transform data to a keras input format for features.
#' @param y_transform [`function`]\cr
#'   Function used to transform data to a keras input format for the response.
#' @param validation_split [`numeric(1)`]\cr
#'   Fraction of data to use for validation.
#' @param batch_size [`integer(1)`]\cr
#'   Batch_size for the generators.
#' @export
make_train_valid_generators = function(task, x_transform, y_transform, validation_split = 1/3, batch_size = 128L) {
  rho = rsmp("holdout", ratio = 1 - pars$validation_split)
  rho$instantiate(task)

  train_gen = make_data_generator(
    task = task,
    batch_size = batch_size,
    filter_ids = rho$train_set(1),
    x_transform = x_transform,
    y_transform = y_transform
  )
  train_steps = ceiling(length(rho$train_set(1)) / batch_size)

  if (pars$validation_split > 0) {
    valid_gen = make_data_generator(
      task = task,
      batch_size = batch_size,
      filter_ids = rho$test_set(1),
      x_transform = x_transform,
      y_transform = y_transform
    )
    valid_steps = ceiling(length(rho$test_set(1)) / batch_size)
  } else {
    valid_gen = NULL
    valid_steps = NULL
  }

  list(train_gen = train_gen, valid_gen = valid_gen, train_steps = train_steps, valid_steps = valid_steps)
}
