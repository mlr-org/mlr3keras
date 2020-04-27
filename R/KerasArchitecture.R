#' @title Keras Neural Network architecture base class
#' @description
#'   A `KerasArchitecture` is a parametrized abstraction of a keras model.
#'   It can be used to more easily and flexibly add architectures.
#' @family KerasArchitectures
#' @rdname KerasArchitecture
#' @export
KerasArchitecture = R6::R6Class("KerasArchitecture",
  public = list(
    #' @field param_set [`ParamSet`] \cr A method's \ architecture's `ParamSet`.
    param_set = NULL,
    #' @field build_arch_fun [`function`] \cr Function that instantiates and compiles a model.
    build_arch_fn = NULL,
    #' @field transforms [`list`] \cr The corresponding x- and y_transform.
    transforms = list(),
    #' @description
    #'   Initialize architecture
    #' @param build_arch_fun [`function`] \cr Function that instantiates and compiles a model.
    #' @param x_transform [`function`] \cr Function used to transform the data for the model.
    #'   Defaults to `as.matrix(features)`.  See `x_transform` for more information.
    #' @param y_transform [`function`] \cr Function used to transform the targets for the model.
    #'   Default depends on the task type. See `y_transform` for more information.
    #' @param param_set [`ParamSet`] \cr A methods / architecture's `ParamSet`.
    initialize = function(build_arch_fn = NULL, x_transform = NULL, y_transform = NULL,
      param_set = ParamSet$new()) {
      if (!is.null(build_arch_fn)) self$build_arch_fn = assert_function(build_arch_fn)
      if (!is.null(x_transform)) self$set_transform("x", x_transform)
      else self$set_transform("x", private$.x_transform)
      if (!is.null(y_transform)) self$set_transform("y", y_transform)
      else self$set_transform("y", private$.y_transform)
      self$param_set = assert_param_set(param_set)
    },
    #' @description
    #'   Obtain the model. Called by Learner during `train_internal`.
    #' @param build_arch_fun [`Task`] \cr Function that instantiates and compiles a model.
    #' @param x_transform [`list`] \cr Function used to transform the data for the model.
    get_model = function(task, pars) {
      self$build_arch_fn(task, pars)
    },
    #' @description
    #'   Setter method for 'x_transform' and 'y_transform'.
    #' @param name [`character`] Either 'x' or 'y'.
    #' @param transform [`function`] \cr Function to set for the architecture
    set_transform = function(name, transform) {
      assert_choice(name, c("x", "y"))
      assert_function(transform)
      self$transforms[[name]] = transform
    },
    #' @description
    #'   Transform the features before passing them on to `keras::fit()`.
    #' @param features [`data.table`] \cr Function that convert the features to a
    #'   form that can be passed on to `keras::fit()`.
    #' @param pars [`list`] \cr Parameter values, i.e. self$param_set$get_values().
    #' @param ... [`any`] \cr Additional args passed on to x_transform.
    x_transform = function(features, pars, ...) {
      stop("Use .$set_transform() and `.$.x_transform;
            This method exists for documentation purposes only")
    },
    #' @description
    #'   Transform the target before passing it on to `keras::fit()`.
    #' @param target [`data.table`] \cr Function that convert the features to a
    #'   form that can be passed on to `keras::fit()`.
    #' @param pars [`list`] \cr Parameter values, i.e. self$param_set$get_values().
    #' @param ... [`any`] \cr Additional args passed on to x_transform.
    y_transform = function(target, pars, ...) {
      stop("Use .$set_transform() and `.$.y_transform;
            This method exists for documentation purposes only")
    }
  ),
  private = list(
    .x_transform = function(features, pars, ...) {
        as.matrix(features)
    },
    .y_transform = function(target, pars, model_loss) {
        stop("Abstract! Set y_transform according to the specific learner!")
    }
  )
)

#' @title Keras Neural Network custom architecture
#'
#' @description
#'   This is an architecture used for custom, user-supplied models.
#'   The `model`, i.e a compiled keras model is supplied to the learner as a hyperparameter.
#' @rdname KerasArchitecture
#' @family KerasArchitectures
#' @export
KerasArchitectureCustomModel = R6::R6Class("KerasArchitectureCustomModel",
  inherit = KerasArchitecture,
  public = list(
    initialize = function() {
      super$initialize(
        build_arch_fn = function(task, pars) stop("Abstract!")
      )
    },
    get_model = function(task, pars) {
      assert_class(pars$model, "keras.engine.training.Model")
      return(pars$model)
    }
  )
)

