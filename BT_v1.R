library(readr)
library(readxl)
library(quantmod)
library(torch)
library(dplyr)
library(xts)
library(caret)

# ^GSPC is the stock ticker of the S&P500 index
ticker <- "^GSPC"

# Download S&P500 data from Yahoo Finance
SPX <- getSymbols(ticker,
                  from = "1950-01-01",  # Start date of the data
                  to = "2017-12-31",    # End date of the data
                  auto.assign = FALSE)  # Do not save directly to the environment

# Plot the closing prices of the S&P500
plot(SPX$GSPC.Close)

# Calculate daily logarithmic returns
returns <- diff(log(SPX$GSPC.Close))  # Logarithmic returns: log(Close_t / Close_t-1)
returns <- na.omit(returns)           # Remove NA values (resulting from diff)

# Group returns by month
# Create a grouping factor based on the month and year of each return
monthly_groups <- format(index(returns), "%Y-%m")

# Calculate realized volatility for each month
# For each group (month), sum the squared daily returns
realized_volatility <- tapply(coredata(returns)^2, monthly_groups, function(x) {
  realized_variance <- sum(x)         # Realized variance: sum of squared daily returns
  log(sqrt(realized_variance))       # Realized volatility: log of the square root of the variance
})

# Convert realized volatility into an xts object
# Create an xts object where each value corresponds to the realized volatility of a month
dates <- as.Date(paste(names(realized_volatility), "01", sep = "-"))  # Generate dates for each month
realized_volatility_xts <- xts(realized_volatility, order.by = dates)

# Plot the realized volatility over time
plot(realized_volatility_xts, lwd = 1)

# Calculate and plot the autocorrelation function (ACF) of the realized volatility
acf(realized_volatility_xts)

#Realized Volatility ready
#-------------------------------------------------------------------------------------------------#
#Neural Network

set.seed(123)

data <- data.frame(Date = index(realized_volatility_xts), RV = coredata(realized_volatility_xts))
data <- data %>%
  mutate(Lag1 = lag(RV, 1),
         Lag2 = lag(RV, 2),
         Lag3 = lag(RV, 3)) %>%
  na.omit()

train_index <- createDataPartition(data$RV, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Normalisierung der Daten
preproc <- preProcess(train_data[, -1], method = c("center", "scale"))  # Exkludiere Datum
train_scaled <- predict(preproc, train_data[, -1])
test_scaled <- predict(preproc, test_data[, -1])

# Torch-Tensoren erstellen
x_train <- torch_tensor(as.matrix(train_scaled[, c("Lag1", "Lag2", "Lag3")]), dtype = torch_float())
y_train <- torch_tensor(as.matrix(train_scaled[, "RV", drop = FALSE]), dtype = torch_float())
x_test <- torch_tensor(as.matrix(test_scaled[, c("Lag1", "Lag2", "Lag3")]), dtype = torch_float())
y_test <- torch_tensor(as.matrix(test_scaled[, "RV", drop = FALSE]), dtype = torch_float())

# Modelldefinition
nn_model <- nn_module(
  initialize = function() {
    self$fc1 <- nn_linear(3, 2)  # 3 Inputs -> 5 Hidden Neuronen
    self$fc2 <- nn_linear(2, 1)  # 5 Hidden -> 1 Output
    self$activation <- nn_relu()  # Aktivierungsfunktion
  },
  forward = function(x) {
    x <- self$activation(self$fc1(x))
    x <- self$fc2(x)
    x
  }
)

model <- nn_model()

optimizer <- optim_adam(model$parameters, lr = 0.01)
loss_fn <- nn_mse_loss()

# Training
num_epochs <- 500
for (epoch in 1:num_epochs) {
  model$train()
  optimizer$zero_grad()
  
  # Vorhersage und Verlust berechnen
  y_pred <- model(x_train)
  loss <- loss_fn(y_pred, y_train)
  
  # Backpropagation und Optimierung
  loss$backward()
  optimizer$step()
  
  # Fortschritt anzeigen
  if (epoch %% 50 == 0) {
    cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
  }
}

# Model testing
model$eval()
y_pred_test <- model(x_test)
test_loss <- loss_fn(y_pred_test, y_test)
cat("Test Loss (MSE):", test_loss$item(), "\n")












#importing DP and EP data from Robert Shiller's website
Shiller_data <- read_xls("data/ie_data_cleaned.xls")
Shiller_data$DP <- Shiller_data$`D/P`
Shiller_data$EP <- Shiller_data$`E/P`

#importing  MKT, HML and SMB data from Fama French's website
FF_data <- read_csv("data/F-F_Research_Data_Factors_daily.CSV")

#importing STR data from Fama French's website
#missing are denoted by -99.99 or 99.99
FF_reversal_data <- read_csv("data/F-F_ST_Reversal_Factor_daily_cleaned.csv")

#importing TB, TS, DEF and INF data from datastream


#importing IP data from OECD database



