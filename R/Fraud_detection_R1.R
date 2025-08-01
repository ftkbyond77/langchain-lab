setwd("C:/Users/BM MONEY/OneDrive/Desktop/mycode/langchain-lab/langchain-lab/Ai Analyst/data")
data <- read.csv("PS_20174392719_1491204439457_log.csv")
print(data)

str(data)

head(data)

summary(data)

# library check
as.data.frame(installed.packages()[, c("Package", "Version")])

install.packages("tidyverse")
install.packages("GGally")
install.packages("corrplot")
install.packages("DataExplorer")

install.packages("randomForest")
install.packages("caret")
install.packages("xgboost")

library(tidyverse)
library(ggplot2)
library(GGally)
library(corrplot)
library(DataExplorer)
library(gridExtra)

# Distribution of Fraud and Non-Fraud
ggplot(data, aes(x = factor(isFraud), fill = factor(isFlaggedFraud))) +
  geom_bar(position = "dodge") +
  labs(title = "Fraud vs Flagged Fraud", 
       x = "isFraud", 
       fill = "isFlaggedFraud", 
       y = "Count") +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "firebrick")) +
  theme_minimal()

table(data$isFraud, data$isFlaggedFraud)


# Distribution of Amount
ggplot(data, aes(x = amount, fill = factor(isFraud))) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 50) +
  scale_x_log10() +
  labs(title = "Transaction Amount by Fraud", fill = "Fraud")

# Correlation
num_data <- data %>%
  select_if((is.numeric))

corrplot(cor(num_data), method = "color", type = "upper", addCoef.col = "black")

# Type Counting
ggplot(data, aes(x = type)) +
  geom_bar(fill = "cornflowerblue") +
  labs(title = "Transaction Types", x = "Type", y = "Count")

# Fraud Rate by type data
data %>%
  group_by(type) %>%
  summarise(total = n(), frauds = sum(isFraud)) %>%
  mutate(fraud_rate = frauds / total) %>%
  ggplot(aes(x = type, y = fraud_rate, fill = type)) +
  geom_col() +
  labs(title = "Fraud Rate by Transaction Type", y = "Fraud Rate") +
  theme_minimal()

# Destination and Balance Pattern
ggplot(data, aes(x = oldbalanceOrg, y = newbalanceOrig, color = factor(isFraud))) +
  geom_point(alpha = 0.5) +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "Old vs New Balance (Origin)", color = "isFraud")

# Automated Report (Optional)
create_report(data)

# Deeper Research 
data <- data %>%
  mutate(balanceDiff = oldbalanceOrg - newbalanceOrig,
         destBalanceDiff = newbalanceDest - oldbalanceDest)

ggplot(data, aes(x = balanceDiff, fill = factor(isFraud))) +
  geom_density(alpha = 0.5) +
  labs(title = "Balance Difference Distribution by Fraud")


# Machine Learning
install.packages(c("caret", "ROSE", "e1071"))
library(caret)
library(randomForest)
library(xgboost)
library(ROSE)     
library(dplyr)

# Prepare dataset
ml_data <- data %>%
  select(-nameOrig, -nameDest)

# Make sure type is factor
ml_data$type <- as.factor(ml_data$type)
ml_data$isFraud <- as.factor(ml_data$isFraud)

# Split dataset
set.seed(123)
trainIndex <- createDataPartition(ml_data$isFraud, p = 0.8, list = FALSE)
train <- ml_data[trainIndex, ]
test <- ml_data[-trainIndex, ]

# Handle Class Imbalance
train_balanced <- ROSE(isFraud ~ ., data = train, seed = 123)$data

table(train_balanced$isFraud)

# Random Forest Model
set.seed(42)
rf_model <- randomForest(isFraud ~ ., data = train_balanced, ntree = 100, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test)
confusionMatrix(rf_pred, test$isFraud)
varImpPlot(rf_model) # Feature Importance


# XGBoost Model
dummies <- dummyVars(isFraud ~ ., data = train_balanced)
train_matrix <- predict(dummies, newdata = train_balanced)
test_matrix <- predict(dummies, newdata = test)

train_label <- as.numeric(train_balanced$isFraud) - 1
test_label <- as.numeric(test$isFraud) - 1

# DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# Train XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)

xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)

xgb_pred_prob <- predict(xgb_model, newdata = dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)
confusionMatrix(factor(xgb_pred), factor(test_label))
