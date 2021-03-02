#' @title Keras CNN Architectures for Classification
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerascnn
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKeras].
#'
#' @section Construction:
#' ```
#' LearnerClassifKerasCNN$new()
#' mlr3::mlr_learners$get("classif.kerascnn")
#' mlr3::lrn("classif.kerascnn")
#' ```
#' @template keras_cnn_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name classif.kerascnn
#' @template example
#' @export
LearnerClassifKerasCNN = R6::R6Class("LearnerClassifKeras",
  inherit = LearnerClassifKeras,
  public = list(
    initialize = function() {
      ps = ParamSet$new(list(
        ParamUty$new("optimizer", default = "optimizer_adam(lr=3*10^-4)", tags = "train"),
        ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",  levels = keras_reflections$loss$classif),
        ParamFct$new("output_activation", levels = c("softmax", "linear", "sigmoid"), tags = "train"),
        ParamUty$new("application", tags = "train"),
        ParamInt$new("cl_layer_units", tags = "train", default = 1024L),
        ParamInt$new("unfreeze_n_last_layers", tags = "train", default = 5L),
        ParamUty$new("metrics", tags = "train"),
        ParamDbl$new("validation_fraction", tags = "train", default = 0.2, lower = 0, upper = 1)
      ))
      ps$values = list(
        application = keras::application_mobilenet,
        optimizer = optimizer_adam(lr = 3*10^-4),
        loss = "categorical_crossentropy",
        metrics = "accuracy",
        cl_layer_units = 1024L,
        output_activation = "softmax",
        unfreeze_n_last_layers = 5L,
        validation_fraction = 0.2
      )
      arch = KerasArchitectureFF$new(build_arch_fn = build_keras_pretrained_cnn_model, param_set = ps)
      super$initialize(
        feature_types = character(),
        man = "mlr3keras::mlr_learners_classif.keras",
        architecture = arch
      )
    }
  ),
  private = list(
    .train = function(task) {
      # CNN's as-is currently can only handle one image feature.
      assert_true(length(task$feature_types[type == "imagepath"][["id"]]) < 2L)
      pars = self$param_set$get_values(tags = "train")

      # Construct / Get the model depending on task and hyperparams.
      model = self$architecture$get_model(task, pars)
      df = cbind(task$data(), uri = task$uris$uri)

      # Data Augmentation Generator. FIXME:
      generator = image_data_generator(
        featurewise_center = FALSE,
        samplewise_center = FALSE,
        featurewise_std_normalization = FALSE,
        samplewise_std_normalization = FALSE,
        zca_whitening = FALSE,
        zca_epsilon = 1e-06,
        rotation_range = 0,
        width_shift_range = 0,
        height_shift_range = 0,
        brightness_range = NULL,
        shear_range = 0,
        zoom_range = 0,
        channel_shift_range = 0,
        fill_mode = "nearest",
        cval = 0,
        horizontal_flip = FALSE,
        vertical_flip = FALSE,
        rescale = NULL,
        preprocessing_function = NULL,
        data_format = NULL,
        validation_split = pars$validation_fraction
      )

      # Generators for train and validation data
      train_gen = keras::flow_images_from_dataframe(
        df, generator = generator, subset = "training",
        x_col = "uri", y_col = task$target_names,
        drop_duplicates = FALSE, batch_size = pars$batch_size, classes = task$class_names,
      )
      valid_gen = keras::flow_images_from_dataframe(
        df, generator = generator, subset = "validation",
        x_col= "uri", y_col = task$target_names,
        drop_duplicates = FALSE, batch_size = pars$batch_size, classes = task$class_names,
      )

      # And then fit
      history = invoke(keras::fit_generator,
        object = model,
        steps_per_epoch = train_gen$`__len__`(),
        generator = train_gen,
        epochs = as.integer(pars$epochs),
        class_weight = pars$class_weight,
        validation_data = valid_gen,
        validation_steps =  valid_gen$`__len__`(),
        verbose = pars$verbose,
        callbacks = pars$callbacks
      )
      return(list(model = model, history = history, class_names = task$class_names))
    },
    .predict = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      df = cbind(task$data(), uri = task$uris$uri)

      gen = keras::flow_images_from_dataframe(df,
        x_col="uri",
        y_col="class",
        drop_duplicates = FALSE,
        batch_size = pars$batch_size,
        classes = task$class_names,
        shuffle = FALSE
      )
      pf_pars = self$param_set$get_values(tags = "predict_fun")
      pf_pars = pf_pars[names(pf_pars) != "batch_size"]
      p = invoke(self$model$model$predict_generator, generator = gen, .args = pf_pars)
      fixup_target_levels_prediction_classif(p, task, self$predict_type)
    }
  )
)

#' @title Keras Neural Network Feed Forward architecture
#' @rdname KerasArchitecture
#' @family KerasArchitectures
#' @export
KerasArchitectureResNet = R6::R6Class("KerasArchitectureFF",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn, param_set) {
      super$initialize(build_arch_fn = build_arch_fn, param_set = param_set)
    }
  )
)



# Builds a Keras Feed Forward Neural Network
# @param task [`Task`] \cr
#   A mlr3 Task.
# @param pars [`list`] \cr
#   A list of parameter values from the Learner(Regr|Classif)KerasFF param_set.
# @template kerasff_description
# @return A compiled keras model
build_keras_pretrained_cnn_model = function(task, pars) {
  base = pars$application(
    include_top = FALSE,
    weights = "imagenet",
    input_tensor = NULL,
    input_shape = NULL,
    pooling = pars$pooling
  )
  model = keras_model(
    inputs = base$input,
    outputs = base$output %>%
      layer_global_average_pooling_2d() %>%
      layer_dense(units = pars$cl_layer_units, activation = "relu") %>%
      layer_dense(units = length(task$class_names), activation = pars$output_activation)
  )
  freeze_weights(base)
  unfreeze_weights(base, from = length(base$layers) - pars$unfreeze_n_last_layers)
  model$compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
  return(model)
}
