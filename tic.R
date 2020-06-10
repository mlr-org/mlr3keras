# installs dependencies, runs R CMD check, runs covr::codecov()
do_package_checks(args = "--as-cran")

if (ci_on_ghactions()) {
  # creates pkgdown site and pushes to gh-pages branch
  do_pkgdown()
}
