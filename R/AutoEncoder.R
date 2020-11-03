PipeOpAutoencoder = R6::R6Class(
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "autoencode", param_vals = list()) {
      ps = ParamSet$new(list(
        ParamInt$new("epochs", default = 100L, lower = 0L, tags = "train"),
        ParamDbl$new("validation_split", lower = 0, upper = 1, default = 1/3, tags = "train"),
        ParamInt$new("batch_size", default = 128L, lower = 1L, tags = c("train", "predict", "predict_fun")),
        ParamUty$new("callbacks", default = list(), tags = "train"),
        ParamInt$new("verbose", lower = 0L, upper = 1L, tags = c("train", "predict", "predict_fun")),
        ParamInt$new("n_max", default = 128L, tags = "train", lower = 1, upper = Inf),
        ParamInt$new("n_layers", default = 2L, tags = "train", lower = 1, upper = Inf),
        ParamInt$new("bottleneck_size", default = 10L, tags = "train", lower = 1, upper = Inf),
        ParamUty$new("initializer", default = "initializer_glorot_uniform()", tags = "train"),
        ParamUty$new("regularizer", default = "regularizer_l1_l2()", tags = "train"),
        ParamUty$new("optimizer", default = "optimizer_sgd()", tags = "train"),
        ParamFct$new("activation", default = "relu", tags = "train",
          levels = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu", "linear")),
        ParamLgl$new("use_batchnorm", default = TRUE, tags = "train"),
        ParamLgl$new("use_dropout", default = TRUE, tags = "train"),
        ParamDbl$new("dropout", lower = 0, upper = 1, tags = "train"),
        ParamFct$new("loss", default = "mean_squared_error", tags = "train",  levels = keras_reflections$loss$regr),
        ParamUty$new("metrics", tags = "train")
      ))
      ps$values = list(
        epochs = 100L,
        callbacks = list(),
        validation_split = 1/3,
        batch_size = 128L,
        activation = "relu",
        n_max = 128L,
        n_layers = 2L,
        bottleneck_size = 10L,
        initializer = initializer_glorot_uniform(),
        optimizer = optimizer_sgd(lr = 3*10^-4, momentum = 0.9),
        regularizer = regularizer_l1_l2(),
        use_batchnorm = FALSE,
        use_dropout = TRUE,
        dropout = 0,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        verbose = 0L
      )
      super$initialize(id = id, param_set = ps, param_vals = param_vals, feature_types = c("numeric", "integer"))
    }
  ),
  private = list(
    .train_task = function(task) {
      pars = self$param_set$values

      # Get columns from data
      dt_columns = private$.select_cols(task)
      cols = dt_columns
      if (!length(cols)) {
        self$state = list(dt_columns = dt_columns)
        return(task)
      }
      x = data.matrix(task$data(cols = task$feature_names))

      # Train model
      aenc = build_autoencoder(task, self$param_set$values)

      history = invoke(keras::fit,
        object = aenc$model,
        x = x,
        y = x,
        epochs = as.integer(pars$epochs),
        batch_size = as.integer(pars$batch_size),
        validation_split = pars$validation_split,
        verbose = as.integer(pars$verbose),
        callbacks = pars$callbacks
      )
      self$state = list(model = aenc$encoder, history = history)

      # Pass on encoded training data
      dt = data.table(aenc$encoder %>% predict(x))
      self$state$dt_columns = dt_columns
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },

    .predict_dt = function(dt, levels) {
      x = data.matrix(dt)
      self$state$model %>% predict(x)
    }
  )
)


# Feed-Forward Autoencoder
build_autoencoder = function(task, pars) {

  if ("factor" %in% task$feature_types$type && !pars$use_embedding)
    stop("Factor features are only available with use_embedding = TRUE!")

  # Get input and output shape for model
  input_shape = task$ncol - 1L
  bottleneck_size = pars$bottleneck_size

  model = keras_model_sequential()

  # Build hidden layers
  n_neurons_layer = integer(pars$n_layers)
  n_neurons_layer[1] = pars$n_max

  # Encoder
  enc_input = layer_input(shape = input_shape)
  enc_output = enc_input
  for (i in seq_len(pars$n_layers)) {
    enc_output = enc_output %>%
      layer_dense(
        units = n_neurons_layer[i],
        kernel_regularizer = pars$regularizer,
        kernel_initializer = pars$initializer,
        bias_regularizer = pars$regularizer,
        bias_initializer = pars$initializer
      ) %>%
      layer_activation(pars$activation)
    if (pars$use_batchnorm) enc_output = enc_output %>% layer_batch_normalization()
    if (pars$use_dropout) enc_output = enc_output %>% layer_dropout(pars$dropout)
    if(i < pars$n_layers)
      n_neurons_layer[i+1] = ceiling(n_neurons_layer[i] - (pars$n_max - bottleneck_size) / (pars$n_layers - 1L))
  }
  encoder = keras_model(enc_input, enc_output)

  # Decoder
  n_neurons_layer_rev = c(rev(n_neurons_layer)[-1], input_shape)
  dec_input = layer_input(shape = bottleneck_size)
  dec_output = dec_input
  for (i in seq_len(pars$n_layers)) {
    dec_output = dec_output %>%
      layer_dense(
        units = n_neurons_layer_rev[i],
        kernel_regularizer = pars$regularizer,
        kernel_initializer = pars$initializer,
        bias_regularizer = pars$regularizer,
        bias_initializer = pars$initializer
      ) %>%
      layer_activation(pars$activation)
    if (i != 1) {
      if (pars$use_batchnorm) dec_output = dec_output %>% layer_batch_normalization()
      if (pars$use_dropout) dec_output = dec_output %>% layer_dropout(pars$dropout)
    }
  }
  decoder = keras_model(dec_input, dec_output)

  # AutoEncoder
  a_input = layer_input(shape = input_shape)
  a_output = a_input %>%
    encoder() %>%
    decoder()
  model = keras_model(a_input, a_output)
  model %>% compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
  list(model = model, encoder = encoder, decoder = decoder)
}
