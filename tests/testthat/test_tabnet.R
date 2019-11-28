context("tabnet")

test_that("test tabnet", {
  library("reticulate")
  library("tensorflow")
  # keras::install_keras(extra_packages = c("tensorflow-hub", "tabnet==0.1.3"))
  skip_if_not(py_module_available("tabnet"))
  
  # Create a tf.data input from the task
  tsk = mlr_tasks$get("iris")
  make_feature_column = function(id, type) {
    if (type == "numeric") tf$feature_column$numeric_column(id)
    else tf$feature_column$categorical_column_with_vocabulary_list(id, tsk$levels(id)[[1]])
  }
  feature_columns = pmap(.f = make_feature_column, .x = tsk$feature_types)
 
  # Define and compile model
  tabnet = import("tabnet") 
  model = tabnet$TabNetClassification(feature_columns, num_classes=3,
    feature_dim=4, output_dim=4, num_decision_steps=2, relaxation_factor=1.0,
    sparsity_coefficient=1e-5, batch_momentum=0.98,
    virtual_batch_size=NULL, norm_type='group',
    num_groups=1)
  model %>% compile(
    loss='categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics=c('accuracy')
  )

  # Fit learner + adjust params
  lrn = LearnerClassifKeras$new()
  lrn$param_set$values$model = model
  lrn$param_set$values$epochs = 5L
  lrn$param_set$values$validation_split = NULL # Does not work with tf.data

  # We overwrite task -> dataset transformers here to stay flexible.
  lrn$set_transform("x", function(task, pars) {
    x = lapply(task$feature_names, function(x) { as.matrix(task$data(cols = x))})
    names(x) = task$feature_names
    return(x)
  })
  lrn$train(tsk)
  expect_class(lrn$model$model, "tabnet.tabnet.TabNetClassification")

  # Predict fails as TabNet comes with a predict method but no "predict_classes" etc.
  # This might  requie a more general aproach to prediction.
  # lrn$predict(tsk)

  k_clear_session()
})