#' Compute SHAP values using GradientExplainer
#'
#' @param model Trained FunnelDNN model.
#' @param X_foreground_std Numeric matrix of standardised foreground data.
#' @param X_background Torch tensor of background data.
#' @param batch_size SHAP batch size (default 2000).
#' @param verbose Print progress.
#' @return A ShapResult object with shap_values_22, shap_values_11,
#'   phi_0, and efficiency_error.
#' @export
compute_shap_values <- function(model, X_foreground_std, X_background,
                                batch_size = 2000L, verbose = FALSE) {
  X_fg <- reticulate::np_array(X_foreground_std, dtype = "float32")
  result <- dnnmcshap$compute_shap_values(
    model, X_fg, X_background,
    batch_size = as.integer(batch_size),
    verbose = verbose
  )
  result
}
