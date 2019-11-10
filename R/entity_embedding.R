# Attention: This is WIP and will not yet work!
#' Create the embedding for a dataset.
#' Creates an input for each categorical var, concatenates those,
#' Adds batch-norm to continuous vars etc.
make_embedding = function(task, embed_size = NULL, embed_dropout = 0) {
  stop("This is not yet working")
  assert_task(task)
  assert_numeric(embed_size, null.ok = TRUE)
  assert_number(embed_dropout)

  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")
  n_cont = length(type[!embed_vars])

  # Embeddings for categorical variables
  embds = list()
  if (sum(embed_vars) > 0) {
    embds = Map(function(x, feat_name) {
      n_cat = length(levels(x))
      # Use heuristic from fast.ai https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py
      if (length(embed_size) >= 2) embed_size = embed_size[feat_name]
      if (length(embed_size) == 0) embed_size = min(600L, round(1.6 * n_cat^0.56))
      input = layer_input(shape = 1, dtype = "int32", name = feat_name)
      layers = input %>%
      layer_embedding(input_dim = as.numeric(n_cat), output_dim = as.numeric(embed_size),
        input_length = 1, name = paste0("embed_", feat_name),
        embeddings_initializer = initializer_he_uniform()) %>%
      layer_dropout(embed_dropout, input_shape = as.numeric(embed_size)) %>%
      layer_flatten()
      return(list(input = input, layers = layers))
    }, data[, embed_vars, drop = FALSE], names(type[embed_vars]))
  }
  # Layer for the continuous variables
  if (n_cont > 0) {
    input = layer_input(shape = n_cont, dtype = "float32", name = "continuous")
    layers = input %>% layer_batch_normalization(input_shape = n_cont, axis = 1)
    embds = c(embds, list(cont = list(input = input, layers = layers)))
  }

  # Concatenate in case
  if (length(embds) >= 2)
    layers = layer_concatenate(unname(lapply(embds, function(x) x$layers)))
  else
    layers = unname(embds[[1]]$layers)
   return(list(inputs = lapply(embds, function(x) x$input), layers = layers))
}