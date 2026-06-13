#' Run the full DNN-MC-SHAP pipeline
#'
#' @param X_train Numeric matrix of raw encoded training features.
#' @param y_train Numeric vector of training targets.
#' @param X_val Numeric matrix of raw encoded validation features.
#' @param y_val Numeric vector of validation targets.
#' @param df_raw data.frame of raw training data with categorical columns.
#' @param mc_samples Number of MC profiles to generate.
#' @param verbose Print progress.
#' @return A PipelineResult object.
#' @export
run_pipeline <- function(X_train, y_train, X_val, y_val, df_raw,
                         mc_samples = 100000L, verbose = FALSE) {
  np <- reticulate::import("numpy")

  X_tr <- reticulate::np_array(X_train, dtype = "float32")
  y_tr <- reticulate::np_array(y_train, dtype = "float32")
  X_va <- reticulate::np_array(X_val, dtype = "float32")
  y_va <- reticulate::np_array(y_val, dtype = "float32")
  df_py <- reticulate::r_to_py(df_raw)

  config <- dnnmcshap$PipelineConfig(mc_samples = as.integer(mc_samples))

  dnnmcshap$run_pipeline(
    X_tr, y_tr, X_va, y_va, df_py,
    config = config,
    verbose = verbose
  )
}
