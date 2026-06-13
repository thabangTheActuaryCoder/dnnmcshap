#' Load a trained DNN checkpoint
#'
#' @param path Path to the .pth checkpoint file.
#' @param device Device string ("cpu", "cuda", "mps"). Default "cpu".
#' @return A list with model, standardiser, and metadata.
#' @export
load_checkpoint <- function(path, device = "cpu") {
  result <- dnnmcshap$load_checkpoint(path, device = device)
  list(
    model = result[[1]],
    standardiser = result[[2]],
    metadata = reticulate::py_to_r(result[[3]])
  )
}

#' Save a model checkpoint
#'
#' @param path Output file path.
#' @param model Trained FunnelDNN model.
#' @param standardiser Fitted Standardiser.
#' @param architecture Architecture metadata list.
#' @param hyperparameters Hyperparameters list.
#' @param metrics Metrics list.
#' @export
save_checkpoint <- function(path, model, standardiser,
                            architecture = NULL,
                            hyperparameters = NULL,
                            metrics = NULL) {
  dnnmcshap$save_checkpoint(
    path, model, standardiser,
    architecture = architecture,
    hyperparameters = hyperparameters,
    metrics = metrics
  )
}

#' Score profiles through the DNN
#'
#' @param model Trained FunnelDNN model object.
#' @param X_encoded Numeric matrix of encoded features (n x 22).
#' @param standardiser Fitted Standardiser object.
#' @return Numeric vector of de-normalised predictions.
#' @export
score_dnn <- function(model, X_encoded, standardiser) {
  X_py <- reticulate::np_array(X_encoded, dtype = "float32")
  result <- dnnmcshap$score_dnn(model, X_py, standardiser)
  return(as.numeric(reticulate::py_to_r(result)))
}
