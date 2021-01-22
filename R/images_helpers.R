#' Create a dataframe with imagepaths from dirs
#'
#' @param dirs [`character`]\cr
#'   List of dirs to create dataframes from.
#' @return [`data.frame`]\cr
#'   with columns "image" (an imagepath) and "class" (the class).
imagepathdf_from_imagenet_dir = function(dirs) {
  rbindlist(map(dirs, function(y) {
    dt = rbindlist(map(list.dirs(y, recursive = FALSE), function(x) {
      data.table(
        class = basename(x),
        image = list.files(x, full.names = TRUE)
      )
    }))
    set(dt, j = "class", value = as.factor(dt$class))
    set(dt, j = "image", value = as.imagepath(dt$image))
    return(dt)
  }))
}