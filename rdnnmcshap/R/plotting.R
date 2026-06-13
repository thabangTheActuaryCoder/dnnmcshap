#' Plot training and validation curves
#'
#' @param train_losses Numeric vector of training losses.
#' @param val_epochs Integer vector of validation epoch numbers.
#' @param val_losses Numeric vector of validation losses.
#' @param train_r2s Numeric vector of training R-squared values.
#' @param val_r2s Numeric vector of validation R-squared values.
#' @param best_epoch Integer, best epoch number.
#' @param save_path Optional file path to save the figure.
#' @return A matplotlib Figure object (invisible).
#' @export
plot_training_curves <- function(train_losses, val_epochs, val_losses,
                                train_r2s, val_r2s, best_epoch,
                                save_path = NULL) {
  plotting <- reticulate::import("dnnmcshap.plotting")
  fig <- plotting$plot_training_curves(
    train_losses, as.integer(val_epochs), val_losses,
    train_r2s, val_r2s, as.integer(best_epoch),
    save_path = save_path
  )
  invisible(fig)
}

#' Plot global SHAP importance
#'
#' @param df_global A data.frame with Variable, Mean Signed SHAP,
#'   and Mean |SHAP| columns.
#' @param save_path Optional file path to save the figure.
#' @return A matplotlib Figure object (invisible).
#' @export
plot_shap_global <- function(df_global, save_path = NULL) {
  plotting <- reticulate::import("dnnmcshap.plotting")
  df_py <- reticulate::r_to_py(df_global)
  fig <- plotting$plot_shap_global(df_py, save_path = save_path)
  invisible(fig)
}

#' Plot conditional SHAP importance
#'
#' @param conditional_results List of ConditionalResult objects.
#' @param save_path Optional file path to save the figure.
#' @return A matplotlib Figure object (invisible).
#' @export
plot_conditional_shap <- function(conditional_results, save_path = NULL) {
  plotting <- reticulate::import("dnnmcshap.plotting")
  fig <- plotting$plot_conditional_shap(conditional_results, save_path = save_path)
  invisible(fig)
}
