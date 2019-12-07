#' Create a data generator for a task
#' 
#' Creates a data generator for a mlr3 task.
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param batch_size [`numeric`]\cr
#'   Batch size
#' @param x_transform
#'   Function used to transform data to a keras input format for features.
#' @param y_transform
#'   Function used to transform data to a keras input format for the response.
#' @examples
#' require(reticulate)
#' gen <- py_iterator(make_data_generator(tsk))
#' generator_next(gen) # Get next batch
#' @export
make_data_generator <- function(
  task, 
  training=TRUE,
  batch_size=128, 
  x_transform=function(x) {x},
  y_transform=function(y) {y}){
  
  # Data generator function
  # data_generator_fn <- function(task, batch_size, training) {
    if (training) {
      order_records <- sample(1:task$nrow)
    } else {
      order_records <- 1:task$nrow
    }

    start <- 1

    function() {
      end <- min(start + batch_size - 1, task$nrow) 
      
      # Get records before transform
      features <- task$data(rows=order_records[start:end], cols = task$feature_names)
      target <- task$data(rows=order_records[start:end], cols = task$target_names)[[task$target_names]]
      
      # Update slice
      if (start + batch_size < task$nrow) {
        start <<- start + batch_size
      } else {
        if (training) {
          # The generator is expected to loop over its data indefinitely during training.
          order_records <<- sample(1:task$nrow)
          start <<- 1 
        } 
      }
      
      # Output in right format for both generators
      if (training) {
        list(
          x_transform(features),
          y_transform(target)
        )
      } else {
        if (start < task$nrow) {
          print("OK")
          x_transform(features)
        } else {
          print("END")
          NULL
        }
      }      
    }
  # }
  
  # Return fn
  # data_generator_fn(task, batch_size, training)
}


