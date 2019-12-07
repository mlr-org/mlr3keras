#' @title Keras Neural Network architecture base class
#' @description
#'   A `KerasArchitecture` is a parametrized abstraction of a keras model.
#'   It can be used to more easily and flexibly add architectures.
#' @rdname KerasArchitecture
#' @export
KerasArchitecture = R6::R6Class("KerasArchitecture",
  public = list(
    #' @field param_set [`ParamSet`] \cr A methods / architecure's `ParamSet`.
    param_set = NULL,
    #' @field build_arch_fun [`function`] \cr Function that instantiates and compiles a model.
    build_arch_fn = NULL,
    #' @field transforms [`list`] \cr The coresponding x- and y_transform.
    transforms = list(),
    #' @description
    #' Initialize architecture
    #' @param build_arch_fun [`function`] \cr Function that instantiates and compiles a model.
    #' @param x_transform [`function`] \cr Function used to transform the data for the model.
    #' @param y_transform [`function`] \cr Function used to transform the targets for the model.
    #' @param param_set [`ParamSet`] \cr A methods / architecure's `ParamSet`.
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
    #' Obtain the model. Called by Learner during `train_internal`. 
    #' @param build_arch_fun [`Task`] \cr Function that instantiates and compiles a model.
    #' @param x_transform [`list`] \cr Function used to transform the data for the model.
    get_model = function(task, pars) {
      self$build_arch_fn(task, pars)
    },
    #' @description 
    #' Setter method for 'x_transform' and 'y_transform'.
    #' @param name [`character`] Either 'x' or 'y'.
    #' @param transform [`function`] \cr Function to set for the architecture
    set_transform = function(name, transform) {
      assert_choice(name, c("x", "y"))
      assert_function(transform)
      self$transforms[[name]] = transform
    }
  ),
  private = list(
    .x_transform = function(features, pars, ...) {
        as.matrix(features)
    },
    .y_transform = function(task, pars, model, ...) {
        target = task$data(cols = task$target_names)
        if (inherits(task, "TaskRegr")) {
          y = as.numeric(target[[task$target_names]])
        } else if (inherits(task, "TaskClassif")) {
          y = to_categorical(as.integer(target[[task$target_names]]) - 1)
          if (model$loss == "binary_crossentropy") y = y[, 1, drop = FALSE]
        }
        return(y)
    }
  )
)

#' @title Keras Neural Network custom architecture
#' @rdname KerasArchitecture
#' @export
KerasArchitectureCustomModel = R6::R6Class("KerasArchitectureCustomModel",
  inherit = KerasArchitecture,
  public = list(
    initialize = function() {
      super$initialize(build_arch_fn = function(pars, input_shape, output_shape) {})
    },
    get_model = function(task, pars) { 
      assert_class(pars$model, "keras.engine.training.Model")
      return(pars$model)
    }
  )
)

#' @title Keras Neural Network Feed Forward architecture
#' @rdname KerasArchitecture
#' @export
KerasArchitectureFF = R6::R6Class("KerasArchitectureFF",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn, param_set) {
      super$initialize(build_arch_fn = build_arch_fn, param_set = param_set)
    }
  )
)

