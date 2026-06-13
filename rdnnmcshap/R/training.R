#' Train a FunnelDNN model
#'
#' @param X_train Numeric matrix of standardised training features.
#' @param y_train Numeric vector of standardised training targets.
#' @param X_val Numeric matrix of standardised validation features.
#' @param y_val Numeric vector of standardised validation targets.
#' @param activation Activation function name ("CELU", "GELU", "Tanh").
#' @param lr Learning rate.
#' @param batch_size Mini-batch size.
#' @param max_epochs Maximum training epochs.
#' @param patience Early stopping patience.
#' @param verbose Print progress.
#' @return A list with training results.
#' @export
train_model <- function(X_train, y_train, X_val, y_val,
                        activation = "CELU", lr = 0.001,
                        batch_size = 256L, max_epochs = 200L,
                        patience = 10L, verbose = FALSE) {
  torch <- reticulate::import("torch")

  model <- dnnmcshap$FunnelDNN(
    input_dim = as.integer(ncol(X_train)),
    activation = activation
  )

  X_tr <- torch$tensor(X_train, dtype = torch$float32)
  y_tr <- torch$tensor(as.matrix(y_train), dtype = torch$float32)
  X_va <- torch$tensor(X_val, dtype = torch$float32)
  y_va <- torch$tensor(as.matrix(y_val), dtype = torch$float32)

  result <- dnnmcshap$train_model(
    model, X_tr, y_tr, X_va, y_va,
    lr = lr, batch_size = as.integer(batch_size),
    max_epochs = as.integer(max_epochs),
    patience = as.integer(patience),
    verbose = verbose
  )

  reticulate::py_to_r(result)
}
