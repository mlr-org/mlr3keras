#' Create the embedding for a dataset.
#'
#' Creates an input for each categorical var, concatenates those,
#' Adds batch-norm to continuous vars etc.
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param embed_size [`numeric`]\cr
#'   A numeric, either a single number determining the embedding size for
#'   all categorical features, a named vector, determining the embedding
#'   the size for each feature individually or `NULL`: Use a heuristic.
#'   The heuristic is `round(1.6 * n_cat^0.56)` where n_cat is the number of levels.
#' @param embed_dropout [`numeric`]\cr
#'   Dropout fraction for the embedding layer.
#'
#' @references Guo, Berkhan, 2016 Entity Embeddings of Categorical Variables
#'
#' @examples
#' task = mlr3::mlr_tasks$get("boston_housing")
#' make_embedding(task)
#' @return A `list` of input tensors and `layer`: the concatenated embeddings.
#' @export
make_embedding = function(task, embed_size = NULL, embed_dropout = 0) {
  # stop("This is not yet working")
  assert_task(task)
  assert_numeric(embed_size, null.ok = TRUE)
  assert_number(embed_dropout)

  typedt = task$feature_types
  data = as.matrix(task$data(cols = task$feature_names))
  target = task$data(cols = task$target_names)

  embed_vars = typedt[typedt$type %in% c("ordered", "factor", "character"),]$id
  n_cont = nrow(typedt) - length(embed_vars)

  # Embeddings for categorical variables: for each categorical:
  # - create a layer_input
  # - create an embedding
  # - apply dropout
  embds = list()
  if (length(embed_vars) > 0) {
    embds = map(.f = function(feat_name) {
      x = data[,feat_name]
      if (is.factor(x)) n_cat = length(levels(x)) else n_cat = length(unique(x))
      # Use heuristic from fast.ai https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py
      # or a user supplied value
      if (length(embed_size) >= 2) embed_size = embed_size[feat_name]
      if (length(embed_size) == 0) embed_size = min(600L, round(1.6 * n_cat^0.56))
      input = layer_input(shape = 1, dtype = "int32", name = feat_name)
      layers = input %>%
      layer_embedding(input_dim = as.numeric(n_cat), output_dim = as.numeric(embed_size),
        input_length = 1L, name = paste0("embed_", feat_name),
        embeddings_initializer = initializer_he_uniform()) %>%
      layer_dropout(embed_dropout, input_shape = as.numeric(embed_size)) %>%
      layer_flatten()
      return(list(input = input, layers = layers))
    }, embed_vars)
  }

  # Layer for the continuous variables
  # - apply batchnorm
  if (n_cont > 0) {
    input = layer_input(shape = n_cont, dtype = "float32", name = "continuous")
    layers = input %>% layer_batch_normalization(input_shape = n_cont, axis = 1)
    embds = c(embds, list(cont = list(input = input, layers = layers)))
  }

  # Concatenate all layers
  if (length(embds) >= 2)
    layers = layer_concatenate(unname(lapply(embds, function(x) x$layers)))
  else
    layers = unname(embds[[1]]$layers)
   return(list(inputs = lapply(embds, function(x) x$input), layers = layers))
}

#' Reshape a Task for use with entity embeddings.
#'
#' @param task [`Task`]\cr
#'   A mlr3 [`Task`].
#'
#' @description
#' * `logical` variables are treated as integers and converted to
#'    either 0 or 1.
#' * `continuous` variables are stored in a matrix "continuous"
#' * `categorical` variables are integer encoded and stored
#'   as a single list element each.
#' @examples
#' task = mlr3::mlr_tasks$get("boston_housing")
#' reshape_task_embedding(task)
#' @family reshape_task_embedding
#' @return A `list` with slots `data`:the reshaped data and `fct_levels`: the levels corresponding to each factor feature.
#' @export
reshape_task_embedding = function(task) {
  assert_task(task)
  data = task$data(cols = task$feature_names)
  reshape_data_embedding(data)
}

#' Reshape data for use with entity embeddings.
#'
#' @param data [`data.table`]\cr
#'   data.table containing the features (without target variable).
#'
#' @description
#' * `logical` variables are treated as integers and converted to
#'    either 0 or 1.
#' * `continuous` variables are stored in a matrix "continuous"
#' * `categorical` variables are integer encoded and stored
#'   as a single list element each.
#' @family reshape_task_embedding
#' @export
reshape_data_embedding = function(data) {
  assert_data_table(data)

  types = map_chr(data, function(x) class(x)[[1]])
  embed_vars = names(types)[types %in% c("ordered", "factor")]

  fct_levels = NULL
  if (length(embed_vars) > 0)
    fct_levels = map(as.list(data[, embed_vars, with = FALSE]), function(x) levels(x))
  out_data = list()
  if (length(embed_vars)  > 0)
    out_data = setNames(
      map(as.list(data[, embed_vars, with = FALSE]), function(x) {
        as.integer(x) - 1L
      }),
      embed_vars)
  if (length(embed_vars) < ncol(data))
    out_data$continuous = as.matrix(data[,setdiff(colnames(data), embed_vars), with = FALSE])

  list(data = out_data, fct_levels = fct_levels)
}
