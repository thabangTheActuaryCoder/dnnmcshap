#' Estimate marginal distributions from training data
#'
#' @param df_raw A data.frame with original categorical columns.
#' @return A MarginalParams object.
#' @export
estimate_marginals <- function(df_raw) {
  df_py <- reticulate::r_to_py(df_raw)
  dnnmcshap$estimate_marginals(df_py)
}

#' Generate Monte Carlo profiles
#'
#' @param n Number of profiles to generate.
#' @param marginals MarginalParams object from estimate_marginals().
#' @param seed Random seed (default 42).
#' @return A data.frame of raw MC profiles.
#' @export
generate_mc_profiles <- function(n, marginals, seed = 42L) {
  np <- reticulate::import("numpy")
  rng <- np$random$default_rng(as.integer(seed))
  result <- dnnmcshap$generate_mc_profiles(as.integer(n), marginals, rng = rng)
  reticulate::py_to_r(result)
}
