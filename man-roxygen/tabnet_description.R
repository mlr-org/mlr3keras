#' @section  Hyper Parameter Tuning:
#'
#' Additional Arguments:
#' * `embed_size`: Size of embedding for categorical, character and ordered factors.
#'                Defaults to `min(600L, round(1.6 * length(levels)^0.56))`.
#' * `stacked`: Should a `StackedTabNetModel` be used instead of a normal `TabNetModel`? \cr
#'
#' @section Excerpt from paper:
#' We consider datasets ranging from 10K to 10M training points, with varying degrees of fitting
#' difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
#' selection: \cr 
#' * Most datasets yield the best results for Nsteps between 3 and 10. Typically, larger datasets and
#'     more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
#'     overfitting and yield poor generalization. \cr 
#' * Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
#'     between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
#'     very high value of Nd and Na may suffer from overfitting and yield poor generalization. \cr 
#' * An optimal choice of \eqn{\gamma} can have a major role on the overall performance. Typically a larger
#'     Nsteps value favors for a larger \eqn{\gamma}. \cr 
#' * A large batch size is beneficial for performance - if the memory constraints permit, as large
#'     as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
#'     much smaller than the batch size. \cr 
#' * Initially large learning rate is important, which should be gradually decayed until convergence. \cr
#' 
#' The R class wraps a python implementation found in \url{https://github.com/titu1994/tf-TabNet/tree/master/tabnet}.
#' 
#' @section Parameters:
#' * `feature_dim` (N_a): Dimensionality of the hidden representation in feature 
#'         transformation block. Each layer first maps the representation to a
#'         2*feature_dim-dimensional output and half of it is used to determine the
#'         nonlinearity of the GLU activation where the other half is used as an
#'         input to GLU, and eventually feature_dim-dimensional output is
#'         transferred to the next layer. \cr 
#' * `output_dim` (N_d): Dimensionality of the outputs of each decision step, which is
#'         later mapped to the final classification or regression output. \cr 
#' * `num_features`: The number of input features (i.e the number of columns for
#'         tabular data assuming each feature is represented with 1 dimension). \cr 
#' * `num_decision_steps` (N_steps): Number of sequential decision steps. \cr 
#' * `relaxation_factor` (gamma): Relaxation factor that promotes the reuse of each
#'         feature at different decision steps. When it is 1, a feature is enforced
#'         to be used only at one decision step and as it increases, more \cr 
#'         flexibility is provided to use a feature at multiple decision steps.
#' * `sparsity_coefficient` (lambda_sparse): Strength of the sparsity regularization.
#'         Sparsity may provide a favorable inductive bias for convergence to
#'         higher accuracy for some datasets where most of the input features are redundant. \cr 
#' * `norm_type`: Type of normalization to perform for the model. Can be either
#'         'batch' or 'group'. 'group' is the default. \cr 
#' * `batch_momentum`: Momentum in ghost batch normalization. \cr 
#' * `virtual_batch_size`: Virtual batch size in ghost batch normalization. The
#'         overall batch size should be an integer multiple of virtual_batch_size. \cr 
#' * `num_groups`: Number of groups used for group normalization. The number of groups
#'         should be a divisor of the number of input features (`num_features`) \cr 
#' * `epsilon`: A small number for numerical stability of the entropy calculations. \cr 
