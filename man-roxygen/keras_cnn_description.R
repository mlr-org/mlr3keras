#' @description
#' Convolutional Neural Network (CNN) application from \CRANpkg{keras}.
#' This learner builds and compiles the keras model from the hyperparameters in `param_set`,
#' and does not require a supplied and compiled model.
#' The 'application' parameter refers to a 'keras::application_*' CNN architectures,
#' possibly with pre-trained weights.
#'
#' Calls [keras::fit_generator] together with [keras::flow_images_from_dataframe]  from package \CRANpkg{keras}.
#' Layers are set up as follows:
#' * The last layer (classification layer) is cut off the neural network.
#' * A classification layer with 'cl_layer_units' is added.
#' * The weights of all layers are frozen.
#' * The last 'unfreeze_n_last_layers' are unfrozen.
#'
#' Parameters:\cr
#' Most of the parameters can be obtained from the `keras` documentation.
#' Some exceptions are documented here.
#' * `application`: A (possibly pre-trained) CNN architecture.
#'   Default: [keras::application_resnet50].
#' * `cl_layer_units`: Number of units in the classification layer.
#'
#' * `unfreeze_n_last_layers`: Number of last layers to be unfrozen.
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
#' * `class_weights`: needs to be a named list of class-weights
#'   for the different classes numbered from 0 to c-1 (for c classes).
#'   ```
#'   Example:
#'   wts = c(0.5, 1)
#'   setNames(as.list(wts), seq_len(length(wts)) - 1)
#'   ```
#'
#' * `callbacks`: A list of keras callbacks.
#'   See `?callbacks`.
