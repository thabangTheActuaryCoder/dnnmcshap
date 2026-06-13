# Quick start example for rdnnmcshap
#
# This script demonstrates loading a trained checkpoint, generating
# MC profiles, and computing SHAP values from R.

library(rdnnmcshap)

# Load a trained model checkpoint
# ckpt <- load_checkpoint("path/to/model.pth")
# model <- ckpt$model
# standardiser <- ckpt$standardiser

# For demonstration, create synthetic raw data
set.seed(42)
n <- 100
df_raw <- data.frame(
  age = sample(18:65, n, replace = TRUE),
  gender = sample(c("male", "female"), n, replace = TRUE),
  bmi = runif(n, 18, 50),
  children = sample(0:5, n, replace = TRUE),
  smoker = sample(c("yes", "no"), n, replace = TRUE, prob = c(0.2, 0.8)),
  region = sample(c("northeast", "southwest", "northwest", "southeast"),
                  n, replace = TRUE),
  medical_history = sample(c("None", "Heart disease",
                             "High blood pressure", "Diabetes"),
                           n, replace = TRUE),
  family_medical_history = sample(c("None", "Heart disease",
                                    "High blood pressure", "Diabetes"),
                                  n, replace = TRUE),
  exercise_frequency = sample(c("Rarely", "Occasionally",
                                "Frequently", "Never"),
                              n, replace = TRUE),
  occupation = sample(c("Unemployed", "Student",
                        "Blue collar", "White collar"),
                      n, replace = TRUE),
  coverage_level = sample(c("Basic", "Standard", "Premium"),
                          n, replace = TRUE),
  stringsAsFactors = FALSE
)

# Encode profiles
X_encoded <- encode_profiles(df_raw)
cat("Encoded shape:", dim(X_encoded), "\n")

# Estimate marginals
marginals <- estimate_marginals(df_raw)

# Generate MC profiles
df_mc <- generate_mc_profiles(50L, marginals, seed = 42L)
cat("MC profiles generated:", nrow(df_mc), "\n")

# Encode MC profiles
X_mc <- encode_profiles(df_mc)
cat("MC encoded shape:", dim(X_mc), "\n")

cat("\nQuickstart complete.\n")
