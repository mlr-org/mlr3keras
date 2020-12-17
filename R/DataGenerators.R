#' Make a DataGenerator from a data.frame or data.table
#'
#' Creates a Python Class that internally iterates over the data.
#' @param dt [`data.frame`|`data.table`] \cr
#'   Data container to iterate over.
#' @param x_cols [`character`] \cr
#'   Names of features to be used. Defaults to all but y_cols.
#' @param y_cols [`character`] \cr
#'   Target variable. Automatically converted to one-hot if "y_cols_to_categorical" is TRUE.
#' @param generator [`Python Object`] \cr
#'   A generator as e.g. obtained from `keras::image_data_generator`.
#'   Used for consistent train-test splits.
#' @param batch_size [`integer`] \cr
#'   Batch size.
#' @param shuffle [`logical`] \cr
#'   Should data be shuffled?
#' @param seed [`integer`] \cr
#'   Set a seed for shuffling data.
#' @param y_cols_to_categorical [`logical`] \cr
#'   Should target be converted to one-hot representation?
#' @export
make_generator_from_dataframe = function(dt, x_cols=NULL, y_cols, generator = keras::image_data_generator(), batch_size=32L, shuffle=TRUE, seed=1L, y_cols_to_categorical = TRUE, subset=NULL) {
  python_path <- system.file("python", package = "mlr3keras")
  generators <- reticulate::import_from_path("generators", path = python_path)
  if (is.null(x_cols)) {
    x_cols = setdiff(names(dt), y_cols)
  }
  x = as.matrix(dt[, x_cols, with = FALSE])
  if (y_cols_to_categorical) {
    y = keras::to_categorical(as.integer(dt[[y_cols]]) - 1, num_classes = length(levels(dt[[y_cols]])))
  } else {
    y = as.matrix(dt[, y_cols, with = FALSE])
  }
  generators$Numpy2DArrayIterator(x, y, generator, batch_size=as.integer(batch_size), shuffle=assert_flag(shuffle),seed=as.integer(seed), subset=subset)
}

#' Make a DataGenerator from a [mlr3::Task]
#'
#' Creates a Python Class that internally iterates over the data.
#' @param task [`Task`] \cr
#'   Data container to iterate over.
#' @param generator [`Python Object`] \cr
#'   A generator as e.g. obtained from `keras::image_data_generator`.
#'   Used for consistent train-test splits.
#' @param batch_size [`integer`] \cr
#'   Batch size.
#' @param shuffle [`logical`] \cr
#'   Should data be shuffled?
#' @param seed [`integer`] \cr
#'   Set a seed for shuffling data.
#' @param y_cols_to_categorical [`logical`] \cr
#'   Should target be converted to one-hot representation? Defaults to `TRUE`.
#' @export
make_generator_from_task = function(task, generator = keras::image_data_generator(), batch_size=32L, shuffle=TRUE, seed=1L, y_cols_to_categorical = TRUE, subset=NULL) {
  assert_task(task)
  dt = task$data()
  make_generator_from_dataframe(
    dt = dt, x_cols = task$feature_names, y_cols = task$target_names,
    generator = generator, batch_size = batch_size, shuffle = shuffle, seed = seed,
     y_cols_to_categorical = y_cols_to_categorical, subset = subset
  )
}

#' Make a DataGenerator that merges multiple DataGenerators into one.
#'
#' @description
#' Creates a Python Class that internally iterates over the data.
#' Generators can e.g. come from `make_generator_from_dataframe` or
#' from `keras::flow_images_from_dataframe`.
#' Need to have equal batch size, and optionally equal seeds if shuffling is used.
#'
#' Returns batches of the following form:
#' list(
#'   list(X1_batch, X2_batch),
#'   Y_batch
#' )
#' or where `Y_batch` is [Y1_batch, Y2_batch] if both `gen._y` are `TRUE`, else just
#' e.g. Y1_batch (default).
#'
#' @param gen1 [`DataGenerator`] \cr
#'   First Data Generator to  loop over.
#' @param gen2 [`DataGenerator`] \cr
#'   Second Data Generator to  loop over.
#' @param gen1_y [`logical`] \cr
#'   Should target variable of `gen1` be returned? Default `TRUE`.
#' @param gen2_y [`logical`] \cr
#'   Should target variable of `gen2` be returned? Default `FALSE`.
#' @export
combine_generators = function(gen1, gen2, gen1_y = TRUE, gen2_y = FALSE) {
  assert_flag(gen1_y)
  assert_flag(gen2_y)
  python_path <- system.file("python", package = "mlr3keras")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGenerator(gen1, gen2, gen1_y, gen2_y)
}