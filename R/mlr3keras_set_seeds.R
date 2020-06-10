#' @title Set Seed for `mlr3keras`
#' @description `mlr3keras_set_seeds`: sets a seed in Random, Python, NumPy and Tensorflow.
#' Futhermore it disables hash seeds, and can disable GPU and CPU parallesim.
#' GPU and Cpu paralelissm can be a source of non deteministic executions.
#' For more information see \url{https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res}.
#' @param seed [`integer`]\cr
#' A seed to be set on different platforms
#' @param r_seed [`logical`]\cr
#' Should seed in R be set
#' @param random_seed [`logical`]\cr
#' Should seed in random be set
#' @param python_seed [`logical`]\cr
#' Should seed in python/NumPy be set
#' @param tensorflow_seed [`logical`]\cr
#' Should seed in tensorflow be set
#' @param disable_gpu [`logical`]\cr
#' Should GPU be disabled
#' @param disable_parallel_cpu [`logical`]\cr
#' Should CPU parallelism be disabled
#' @rdname mlr3keras_set_seeds
#' @export
mlr3keras_set_seeds = function(seed = 1L,
                     r_seed = TRUE,
                     random_seed = TRUE,
                     python_seed = TRUE,
                     tensorflow_seed = TRUE,
                     disable_gpu = FALSE,
                     disable_parallel_cpu = FALSE) {

  checkmate::assert_integerish(seed, len = 1L, lower = 1L, all.missing = FALSE)
  if (!is.integer(seed)) seed = as.integer(seed)

  # set seed in...
  if (r_seed) set.seed(seed) # R
  if (random_seed) {
    random <- reticulate::import("random")
    random$seed(seed) # Random
  }
  if (python_seed) reticulate::py_set_seed(seed, disable_hash_randomization = TRUE) # python and NumPy
  if (tensorflow_seed) {
    tensorflow = reticulate::import("tensorflow") # tensorflow, needs to be set after disbling hash!
    tensorflow$random$set_seed(seed)
  }

  if (tensorflow::tf_version() >= "2.0") {
    tf = tensorflow$compat$v1
  } else tf <- tensorflow

  # set up session and configurations for disabling gpu and cpu parallelism
  session <- NULL
  config <- configure_session(disable_gpu, disable_parallel_cpu)

  if (length(config) > 0L) {
    # call hook (returns TRUE if TF seed should be set, this allows users to
    # call this function even when using front-end packages like keras that
    # may not use TF as their backend)
    using_tf <- tensorflow:::call_hook("tensorflow.on_before_use_session", FALSE)
    if (using_tf) tf$reset_default_graph()
    session_conf <- do.call(tf$ConfigProto, config)
    session <- tf$Session(graph = tf$get_default_graph(), config = session_conf)
    # call after hook
    tf_call_hook("tensorflow.compat.v1.on_use_session", session, FALSE)
    tf$keras$backend$set_session(session)
  }
  invisible(session)
}

#' @describeIn mlr3keras_set_seeds configurations for [mlr3keras::mlr3keras_set_seeds]
configure_session <- function(disable_gpu, disable_parallel_cpu) {
  config <- list()
  if (disable_gpu) {
    Sys.setenv(CUDA_VISIBLE_DEVICES = "")
    config$device_count <-  list(gpu = 0L)
  }
  if (disable_parallel_cpu) {
    config$intra_op_parallelism_threads <- 1L
    config$inter_op_parallelism_threads <- 1L
  }
  config
}

# Re-export tensorflow:::call_hook
tf_call_hook = function (name, ...) {
    hooks <- getHook(name)
    if (!is.list(hooks))
        hooks <- list(hooks)
    response <- FALSE
    lapply(hooks, function(hook) {
        if (isTRUE(hook(...)))
            response <<- TRUE
    })
    response
}
