#' Compute conditional SHAP at tail thresholds
#'
#' @param shap_values_11 Numeric matrix of aggregated SHAP values (n x 11).
#' @param y_mc Numeric vector of MC predicted costs.
#' @return A list of ConditionalResult objects.
#' @export
compute_conditional_shap <- function(shap_values_11, y_mc) {
  sv <- reticulate::np_array(shap_values_11, dtype = "float32")
  y <- reticulate::np_array(y_mc, dtype = "float64")
  result <- dnnmcshap$compute_conditional_shap(sv, y)
  reticulate::py_to_r(result)
}

#' Bootstrap confidence intervals for conditional SHAP
#'
#' @param shap_values_11 Numeric matrix of aggregated SHAP values.
#' @param y_mc Numeric vector of MC predicted costs.
#' @param tau Quantile level (default 0.99).
#' @param n_resamples Number of bootstrap resamples (default 10000).
#' @param alpha Significance level (default 0.05).
#' @param seed Random seed.
#' @return A BootstrapCIResult object.
#' @export
bootstrap_conditional_ci <- function(shap_values_11, y_mc,
                                     tau = 0.99, n_resamples = 10000L,
                                     alpha = 0.05, seed = 42L) {
  sv <- reticulate::np_array(shap_values_11, dtype = "float32")
  y <- reticulate::np_array(y_mc, dtype = "float64")
  dnnmcshap$bootstrap_conditional_ci(
    sv, y,
    tau = tau, n_resamples = as.integer(n_resamples),
    alpha = alpha, seed = as.integer(seed)
  )
}
