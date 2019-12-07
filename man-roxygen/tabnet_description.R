#' @section  Hyper Parameter Tuning:
#' 
#' (Excerpt from paper)
#' We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
#' difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
#' selection:
#'     - Most datasets yield the best results for Nsteps between [3, 10]. Typically, larger datasets and
#'     more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
#'     overfitting and yield poor generalization.
#'     - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
#'     between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
#'     very high value of Nd and Na may suffer from overfitting and yield poor generalization.
#'     - An optimal choice of γ can have a major role on the overall performance. Typically a larger
#'     Nsteps value favors for a larger γ.
#'     - A large batch size is beneficial for performance - if the memory constraints permit, as large
#'     as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
#'     much smaller than the batch size.
#'     - Initially large learning rate is important, which should be gradually decayed until convergence.
#' Parameters:
#'     feature_dim (N_a): Dimensionality of the hidden representation in feature
#'         transformation block. Each layer first maps the representation to a
#'         2*feature_dim-dimensional output and half of it is used to determine the
#'         nonlinearity of the GLU activation where the other half is used as an
#'         input to GLU, and eventually feature_dim-dimensional output is
#'         transferred to the next layer.
#'     output_dim (N_d): Dimensionality of the outputs of each decision step, which is
#'         later mapped to the final classification or regression output.
#'     num_features: The number of input features (i.e the number of columns for
#'         tabular data assuming each feature is represented with 1 dimension).
#'     num_decision_steps(N_steps): Number of sequential decision steps.
#'     relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
#'         feature at different decision steps. When it is 1, a feature is enforced
#'         to be used only at one decision step and as it increases, more
#'         flexibility is provided to use a feature at multiple decision steps.
#'     sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
#'         Sparsity may provide a favorable inductive bias for convergence to
#'         higher accuracy for some datasets where most of the input features are redundant.
#'     norm_type: Type of normalization to perform for the model. Can be either
#'         'batch' or 'group'. 'group' is the default.
#'     batch_momentum: Momentum in ghost batch normalization.
#'     virtual_batch_size: Virtual batch size in ghost batch normalization. The
#'         overall batch size should be an integer multiple of virtual_batch_size.
#'     num_groups: Number of groups used for group normalization.
#'     epsilon: A small number for numerical stability of the entropy calculations.