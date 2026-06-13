#' Encode raw profiles into model-ready format
#'
#' Apply reference-category dummy encoding to transform
#' 11 raw variables into 22 encoded features.
#'
#' @param df A data.frame with raw profile columns.
#' @return A numeric matrix with 22 columns.
#' @export
encode_profiles <- function(df) {
  df_py <- reticulate::r_to_py(df)
  result <- dnnmcshap$encode_profiles(df_py)
  return(reticulate::py_to_r(result))
}

#' Encode a raw DataFrame with charges
#'
#' @param df A data.frame including a 'charges' column.
#' @return A data.frame with 22 feature columns plus charges.
#' @export
encode_dataframe <- function(df) {
  df_py <- reticulate::r_to_py(df)
  result <- dnnmcshap$encode_dataframe(df_py)
  return(reticulate::py_to_r(result))
}
