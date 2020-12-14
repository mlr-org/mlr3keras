#' Make a DataGenerator from a data.frame or data.table
#'
#' Creates a Python Class that internally iterates over the data.
#' @param dt [`data.frame`|`data.table`] \cr
#'   Data container to iterate over.
#' @param x_cols [`character`] \cr
#'   Names of features to be used. Defaults to all but y_cols.
#' @param x_cols [`character`] \cr
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
make_generator_from_dataframe = function(dt, x_cols=NULL, y_cols, generator, batch_size=32L, shuffle=TRUE, seed=1L, y_cols_to_categorical = TRUE) {
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
  generators$Numpy2DArrayIterator(x, y, generator, batch_size=batch_size, shuffle=shuffle,seed=seed)
}

#' Make a DataGenerator that merges multiple DataGenerators into one.
#'
#' Creates a Python Class that internally iterates over the data.
#' Generators can e.g. come from `make_generator_from_dataframe` or
#' from `keras::flow_images_from_dataframe`.
#' Need to have equal batch size, and optionally equal seeds if shuffling is used.
#' @param gen1 [`DataGenerator`] \cr
#'   First Data Generator to  loop over.
#' @param gen2 [`DataGenerator`] \cr
#'   Second Data Generator to  loop over.
#' @export
combine_generators = function(gen1, gen2) {
  python_path <- system.file("python", package = "mlr3keras")
  generators <- import_from_path("generators", path = python_path)
  generators$CombinedGenerator(gen1, gen2)
}