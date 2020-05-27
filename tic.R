# installs dependencies, runs R CMD check, runs covr::codecov()
do_package_checks(args = "--as-cran")

get_stage("install") %>%
  add_code_step(
    keras::install_keras(extra_packages =
      c("IPython", "requests", "certifi", "urllib3", "tensorflow-hub", "tabnet==0.1.4.1")
  ))

if (ci_on_ghactions()) {
  # creates pkgdown site and pushes to gh-pages branch
  do_pkgdown()
}