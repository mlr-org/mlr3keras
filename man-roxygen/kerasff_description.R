#' @description
#' Feed Forward Neural Network using Keras and Tensorflow.
#' This learner builds and compiles the keras model from the hyperparameters in `param_set`,
#' and does not require a supplied and compiled model.
#'
#' Calls [keras::fit()] from package \CRANpkg{keras}.
#' Layers are set up as follows:
#' * The inputs are connected to a `layer_dropout`, applying the `input_dropout`.
#'   Afterwards, each `layer_dense()` is followed by a `layer_activation`, and
#'   depending on hyperparameters by a `layer_batch_normalization` and or a
#'   `layer_dropout` depending on the architecture hyperparameters.
#'   This is repeated `length(layer_units)` times, i.e. one
#'   'dense->activation->batchnorm->dropout' block is appended for each `layer_unit`.
#'   The last layer is either 'softmax' or 'sigmoid' for classification or
#'   'linear' or 'sigmoid' for regression.
#'
#' Parameters:\cr
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `use_embedding`: A logical flag, should embeddings be used?
#'   Either uses `make_embedding` (if TRUE) or if set to FALSE `model.matrix(~. - 1, data)`
#'   to convert factor, logical and ordered factors into numeric features.
#' * `layer_units`: An integer vector storing the number of units in each
#'   consecutive layer. `layer_units = c(32L, 32L, 32L)` results in a 3 layer
#'   network with 32 neurons in each layer.
#'   Can be `integer(0)`, in which case we fit a (multinomial) logistic regression model.
#'
#' * `initializer`: Weight and bias initializer.
#'   ```
#'   "glorot_uniform"  : initializer_glorot_uniform(seed)
#'   "glorot_normal"   : initializer_glorot_normal(seed)
#'   "he_uniform"      : initializer_he_uniform(seed)
#'   "..."             : see `??keras::initializer`
#'   ```
#'
#' * `optimizer`: Some optimizers and their arguments can be found below.\cr
#'   Inherits from `tensorflow.python.keras.optimizer_v2`.
#'   ```
#'   "sgd"     : optimizer_sgd(lr, momentum, decay = decay),
#'   "rmsprop" : optimizer_rmsprop(lr, rho, decay = decay),
#'   "adagrad" : optimizer_adagrad(lr, decay = decay),
#'   "adam"    : optimizer_adam(lr, beta_1, beta_2, decay = decay),
#'   "nadam"   : optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
#'   ```
#'
#' * `regularizer`: Regularizer for keras layers:
#'   ```
#'   "l1"      : regularizer_l1(l = 0.01)
#'   "l2"      : regularizer_l2(l = 0.01)
#'   "l1_l2"   : regularizer_l1_l2(l1 = 0.01, l2 = 0.01)
#'   ```
#'
#' * `class_weights`: needs to be a named list of class-weights
#'   for the different classes numbered from 0 to c-1 (for c classes).
#'   ```
#'   Example:
#'   wts = c(0.5, 1)
#'   setNames(as.list(wts), seq_len(length(wts)) - 1)
#'   ```
#' * `callbacks`: A list of keras callbacks.
#'   See `?callbacks`.
