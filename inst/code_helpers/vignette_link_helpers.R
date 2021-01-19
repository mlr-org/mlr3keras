#' @title Hyperlink to CRAN Package
#'
#' @description
#' Creates a markdown link to a CRAN package.
#'
#' @param pkg Name of the CRAN package.
#'
#' @return (`character(1)`) markdown link.
#' @export
cran_pkg = function(pkg) {
  checkmate::assert_string(pkg, pattern = "^[[:alnum:]._-]+$")
  sprintf("[%1$s](https://cran.r-project.org/package=%1$s)", trimws(pkg))
}

#' @title Hyperlink to mlr3 Package
#'
#' @description
#' Creates a markdown link to a mlr3 package with a "mlr-org.com" subdomain.
#'
#' @param pkg Name of the mlr3 package.
#'
#' @return (`character(1)`) markdown link.
#' @export
mlr_pkg = function(pkg) {
  checkmate::assert_string(pkg, pattern = "^[[:alnum:]._-]+$")
  sprintf("[%1$s](https://%1$s.mlr-org.com)", trimws(pkg))
}

#' @title Hyperlink to GitHub Repository
#'
#' @description
#' Creates a markdown link to GitHub repository.
#'
#' @param pkg Name of the repository specified as "{repo}/{name}".
#'
#' @return (`character(1)`) markdown link.
#' @export
gh_pkg = function(pkg) {
  checkmate::assert_string(pkg, pattern = "^[[:alnum:]_-]+/[[:alnum:]._-]+$")
  parts = strsplit(trimws(pkg), "/", fixed = TRUE)[[1L]]
  sprintf("[%s](https://github.com/%s)", parts[2L], pkg)
}
