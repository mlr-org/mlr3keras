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
#' gen <- make_data_generator(tsk) # Make generator
#' generator_next(gen) # Get next batch
#' @export
make_data_generator <- function(
  task, 
  batch_size=128, 
  x_transform=function(x) {x},
  y_transform=function(y) {y}){
  
  # Data generator function
  data_generator_fn <-function(task, batch_size) {
    order_records <- rep(1:task$nrow)
    start <- 1
    
    function() {
      end <- min(start + batch_size - 1, task$nrow) 
      features <- task$data(rows=order_records[start:end], cols = task$feature_names)
      target <- task$data(rows=order_records[start:end], cols = task$target_names)
      start <<- start + batch_size
      
      list(
        x_transform(features),
        y_transform(target)
      )
    }
  }
  
  # Return iterator
  py_iterator(
    data_generator_fn(task, batch_size), 
    completed = NULL)
}


