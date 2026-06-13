## Package-level Python module reference
dnnmcshap <- NULL

.onLoad <- function(libname, pkgname) {
  dnnmcshap <<- reticulate::import("dnnmcshap", delay_load = TRUE)
}
