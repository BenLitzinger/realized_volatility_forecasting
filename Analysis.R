###############################################################################
# 1.1 Downloading and visualizing historical stock closing prices
###############################################################################

# Load required packages
library(quantmod)

# Specify the stock ticker symbol
ticker <- "ETN"

# Download data from Yahoo Finance
getSymbols(
  ticker,
  from = "2000-01-01",
  to   = "2024-11-12",
  src  = "yahoo"
)

# Convert into xts object if needed
if (!inherits(get(ticker), "xts")) {
  assign(ticker, as.xts(get(ticker)))
}

# Plot of the stock closing prices
plot(
  ETN$ETN.Close,
  main          = "ETN stock closing prices",
  ylab          = "stock price in USD",
  col           = "darkblue",
  main.timespan = FALSE,
  format.labels = "%Y",
  yaxis.right   = FALSE,
  type          = "h"
)


###############################################################################
# 1.2 Transforming stock closing prices into log return series
###############################################################################

# Compute log returns: R_t = ln(P_t) - ln(P_{t-1})
ETN$log_returns <- diff.xts(log(ETN$ETN.Close), lag = 1, differences = 1)

# Omit NAs
ETN <- na.omit(ETN)

# Plot the log returns
plot(
  ETN$log_returns,
  main          = "Daily percentage changes of ETN",
  main.timespan = FALSE,
  ylab          = "Log returns",
  col           = "darkblue",
  lwd           = 0.5,
  format.labels = "%Y",
  yaxis.right   = FALSE
)


###############################################################################
# 1.3 Analyzing normality in log returns visually
###############################################################################

mean_log_returns <- mean(ETN$log_returns, na.rm = TRUE)
sd_log_returns   <- sd(ETN$log_returns,   na.rm = TRUE)

x_vals <- seq(
  min(ETN$log_returns, na.rm = TRUE),
  max(ETN$log_returns, na.rm = TRUE),
  length.out = 1000
)

y_vals <- dnorm(
  x_vals,
  mean = mean_log_returns,
  sd   = sd_log_returns
)

hist(
  ETN$log_returns,
  freq    = FALSE,
  breaks  = 100,
  xlab    = "log returns",
  main    = "Density of log returns against normal distribution",
  col     = "darkblue",
  xlim    = c(-0.1, 0.1)
)

lines(
  x_vals,
  y_vals,
  col = "red",
  lwd = 2
)


###############################################################################
# 2.1 Assessing temporal dependencies and volatility clustering
###############################################################################

# Load forecast package for ACF plots
library(forecast)

par(mfrow = c(2, 2))

# Autocorrelogram of log returns
Acf(
  ETN$log_returns,
  lag.max = 10,
  type    = "correlation",
  plot    = TRUE,
  main    = "Autocorrelogram of returns",
  xlab    = "lag",
  ylab    = "correlation",
  ylim    = c(-0.3, 1)
)

# Square the log returns
ETN$sq_log_returns <- ETN$log_returns^2

# Autocorrelogram of squared log returns
Acf(
  ETN$sq_log_returns,
  lag.max = 10,
  type    = "correlation",
  plot    = TRUE,
  main    = "Autocorrelogram of squared returns",
  xlab    = "lag",
  ylab    = "correlation",
  ylim    = c(-0.3, 1)
)


###############################################################################
# 2.2 Testing for ARCH effects (Ljung-Box test on squared residuals)
###############################################################################

# Regress returns on a constant to get mean-adjusted residuals
ETN_meanadj <- lm(ETN$log_returns ~ 1)
ETN$meanadj_returns <- ETN_meanadj$residuals

# Square of mean-adjusted residuals
ETN$uhat_sq <- ETN$meanadj_returns^2

# Ljung-Box test
Ljung_Box_Test <- Box.test(
  ETN$uhat_sq,
  lag  = 10,
  type = "Ljung-Box"
)

print(Ljung_Box_Test)


###############################################################################
# 2.3 Testing for ARCH effects with the BIC (varying AR lags)
###############################################################################

library(stargazer)

bic_values <- numeric(10)
term1      <- numeric(10)
term2      <- numeric(10)
R2_values  <- numeric(10)
lags       <- 1:10
T_val      <- length(na.omit(ETN$log_returns))

# Compute total sum of squares (TSS)
log_returns_mean <- mean(ETN$log_returns, na.rm = TRUE)
TSS <- sum((ETN$log_returns - log_returns_mean)^2, na.rm = TRUE)

# Fit AR models with different lags and compute BIC
for (lag in lags) {
  ar_model      <- arima(ETN$log_returns, order = c(lag, 0, 0))
  bic_values[lag] <- BIC(ar_model)
  
  residuals <- ar_model$residuals
  RSS       <- sum(residuals^2)
  
  term1[lag] <- log(RSS / T_val)
  term2[lag] <- (lag + 1) * log(T_val) / T_val
  bic_values[lag] <- log(RSS / T_val) + (lag + 1) * (log(T_val) / T_val)
  
  # Compute R2
  R2_values[lag] <- 1 - (RSS / TSS)
}

bic_table <- data.frame(
  Lag  = lags,
  term1 = term1,
  term2 = term2,
  BIC   = bic_values,
  R2    = R2_values
)

stargazer(
  bic_table,
  title    = "BIC Values for Different Lags",
  summary  = FALSE,
  rownames = FALSE,
  header   = FALSE,
  digits   = 5,
  type     = "latex"
)


###############################################################################
# 3 Specification and Estimation of a GARCH(2,2) Model
###############################################################################

library(rugarch)

###############################################################################
# 3.1 Comparing optimisation routines for GARCH(2,2) estimation
###############################################################################

# Specify GARCH(2,2)
garch22_spec <- ugarchspec(
  variance.model = list(garchOrder = c(2, 2)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# Estimate the GARCH(2,2) model using solnp
garch22_fit_solnp <- ugarchfit(
  spec   = garch22_spec,
  data   = ETN$log_returns,
  solver = "solnp"
)

# Estimate the GARCH(2,2) model using lbfgs
garch22_fit_lbfgs <- ugarchfit(
  spec            = garch22_spec,
  data            = ETN$log_returns,
  solver          = "lbfgs",
  solver.control  = list(pgtol = 0.3, maxit = 10e5)
)


###############################################################################
# 3.2 Interpreting the significance of the estimators
###############################################################################

library(stargazer)

garch22_coef_lbfgs <- round(garch22_fit_lbfgs@fit[["matcoef"]], digits = 8)
garch22_coef_solnp <- round(garch22_fit_solnp@fit[["matcoef"]], digits = 8)

# Extracting the p-values
pvals_lbfgs <- garch22_fit_lbfgs@fit[["matcoef"]][, 4]
pvals_solnp <- garch22_fit_solnp@fit[["matcoef"]][, 4]

# Function to display very small p-values as 0
format_p_values <- function(pvals) {
  pvals[pvals < 0.0001] <- 0
  return(round(pvals, 3))
}

formatted_lbfgs_p_values <- format_p_values(pvals_lbfgs)
formatted_solnp_p_values <- format_p_values(pvals_solnp)

garch22_coef_df <- data.frame(
  Parameter       = rownames(garch22_coef_lbfgs),
  lbfgs_estimate  = garch22_coef_lbfgs[, 1],
  lbfgs_p_value   = formatted_lbfgs_p_values,
  solnp_estimate  = garch22_coef_solnp[, 1],
  solnp_p_value   = formatted_solnp_p_values
)

# Remove the mean parameter row for clarity
garch22_coef_df_filtered <- garch22_coef_df[-1, ]

stargazer(
  garch22_coef_df_filtered,
  summary  = FALSE,
  rownames = FALSE,
  header   = FALSE,
  title    = "Comparison of GARCH(2,2) Parameter Estimates from 'lbfgs' and 'solnp' Solvers"
)


###############################################################################
# 4 Plotting and forecasting the conditional standard deviation
###############################################################################

###############################################################################
# 4.1 Plotting mean adjusted returns with superimposed standard deviations
###############################################################################

# Extract conditional standard deviation from the GARCH(2,2) fit
ETN$sig_t_hat_garch22    <- sigma(garch22_fit_solnp)
ETN$neg_sig_t_hat_garch22 <- -ETN$sig_t_hat_garch22

# Plot mean-adjusted returns with +/- estimated sigma
plot(
  ETN[, c("sig_t_hat_garch22", "neg_sig_t_hat_garch22", "meanadj_returns")],
  ylab          = expression(epsilon[t] ~ "and" ~ "\u00B1" ~ widehat(sigma)[t]),
  lwd           = c(0.5, 0.5, 0.25),
  col           = c("red", "red", "blue"),
  format.labels = "%Y",
  main          = "Mean adjusted returns and estimated conditional standard deviation",
  main.timespan = FALSE,
  yaxis.right   = FALSE
)


###############################################################################
# 4.2 Forecasting volatility with the GARCH(2,2) model
###############################################################################

# One-step-ahead forecast of the conditional variance
garch22_fcast <- ugarchforecast(garch22_fit_solnp, n.ahead = 1)
fcast_val     <- as.numeric(garch22_fcast@forecast$sigmaFor)

# Insert the forecast value into the series
last_date <- tail(index(ETN), 1)
next_date <- last_date + 1

ETN$fcast_val <- NA

new_row <- xts(matrix(NA, ncol = ncol(ETN)), order.by = next_date)
colnames(new_row) <- colnames(ETN)
new_row[1, "fcast_val"] <- fcast_val

ETN <- rbind(ETN, new_row)


###############################################################################
# 5 Simulating and estimating a GARCH(1,1) model
###############################################################################

###############################################################################
# 5.1 Simulating Monte Carlo observations for analyzing GARCH(1,1) properties
###############################################################################

# Define parameters
n    <- 5000
M    <- 1000

mu    <- 0
omega <- 0.1
alpha1 <- 0.199999
beta1  <- 0.8

# Initialize a matrix to store estimates
estimates <- matrix(ncol = 4, nrow = M)
colnames(estimates) <- c("omega", "alpha1", "beta1", "uncon_var")

# Specify GARCH(1,1) with fixed parameters
garch11_fixed_pars_spec <- ugarchspec(
  variance.model = list(garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  fixed.pars     = list(mu = mu, omega = omega, alpha1 = alpha1, beta1 = beta1),
  distribution.model = "norm"
)

# Specify GARCH(1,1) without fixed parameters
garch11_spec <- ugarchspec(
  variance.model = list(garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)

# Monte Carlo loop
for (i in 1:M) {
  # Simulate from the fixed-parameter GARCH(1,1) model
  garch11_sim <- ugarchpath(garch11_fixed_pars_spec, n.sim = n)
  
  # Fit a GARCH(1,1) model to the simulated data
  garch11_fit <- ugarchfit(garch11_spec, garch11_sim@path[["seriesSim"]])
  
  # Extract parameter estimates
  estimates[i, "omega"]   <- garch11_fit@fit$robust.matcoef["omega", 1]
  estimates[i, "alpha1"]  <- garch11_fit@fit$robust.matcoef["alpha1", 1]
  estimates[i, "beta1"]   <- garch11_fit@fit$robust.matcoef["beta1", 1]
  
  # Calculate unconditional variance for each simulation
  estimates[i, "uncon_var"] <- estimates[i, "omega"] /
    (1 - estimates[i, "alpha1"] - estimates[i, "beta1"])
}


###############################################################################
# 5.2 Visualizing the parameters and the unconditional variance
###############################################################################

par(mfrow = c(2, 2))
hist(estimates[, 1], breaks = 50, freq = FALSE, main = "omega (0.1)",      col = "blue")
hist(estimates[, 2], breaks = 50, freq = FALSE, main = "alpha1 (0.199999)", col = "blue")
hist(estimates[, 3], breaks = 50, freq = FALSE, main = "beta1 (0.8)",       col = "blue")
hist(estimates[, 4], breaks = 50, freq = FALSE, main = "volatility",        col = "blue")
