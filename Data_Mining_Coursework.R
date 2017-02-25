# Assignment #I-A Engineering the Inputs: Preparing the Datasets
# arff files pre-made in a TextEdit file
# No other pre-processing necessary
# Loading in the necessary packages and Weka packages with WPM
library(tidyr)
library(BBmisc)
library(ggplot2)
library(RWeka)
WPM("refresh-cache")
WPM("install-package", "classificationViaRegression")
WPM("install-package", "J48Consolidated")
WPM("install-package", "costSensitiveAttributeSelection")

# Sourcing functions
source("Cost_Sensitive_Meta_Learner.R")
source("Variable_Cost_Function.R")

# defining two training sets, one for regression (fit.faults) and one for binary
# classification (fit.class)
fit.faults <- read.arff("train_faults.arff")
fit.class <- fit.faults[, -ncol(fit.faults)]
fit.class$CLASS <- as.factor(ifelse(fit.faults$FAULTS < 2, "nfp", "fp"))

# similarly two test sets
test.faults <- read.arff("test_faults.arff")
test.class <- test.faults[, -ncol(test.faults)]
test.class$CLASS <- as.factor(ifelse(test.faults$FAULTS < 2, "nfp", "fp"))

# Assigntment #I-B Modeling Assignment: Prediction

# Loading the necessary Weka package for regression
# Linear Regression model using M5 method for attribute selection
mod.m5 <- LinearRegression(FAULTS ~ ., data = fit.faults, 
                           control = Weka_control(S = 0))

# 10-fold cross-validation on the original training(fit) data
m5.eval.fit <- evaluate_Weka_classifier(mod.m5, numFolds = 10, seed = 1)

# 10-fold cross-validation on the original test data
m5.eval.test <- evaluate_Weka_classifier(mod.m5, newdata = test.faults, 
                                         numFolds = 10, seed = 1) 

# Linear Regression model with no attribute selection
mod.none <- LinearRegression(FAULTS ~ ., data = fit.faults, 
                             control = Weka_control(S = 1))

# 10-fold cross-validation on the original training(fit) data
none.eval.fit <- evaluate_Weka_classifier(mod.none, numFolds = 10, seed = 1) 

# 10-fold cross-validation on the original test data
none.eval.test <- evaluate_Weka_classifier(mod.none, newdata = test.faults, 
                                           numFolds = 10, seed = 1) 

# Linear Regression model with greedy method for attribute selection
mod.greedy <- LinearRegression(FAULTS ~ ., data = fit.faults, 
                               control = Weka_control(S = 2))

# 10-fold cross-validation on the original training(fit) data
greedy.eval.fit <- evaluate_Weka_classifier(mod.greedy, numFolds = 10, 
                                            seed = 1) 
# Evaluation of model on test data set
greedy.eval.test <- evaluate_Weka_classifier(mod.greedy, newdata = test.faults, 
                                             numFolds = 10, seed = 1) 

# Decision Stump Model
mod.dstump <- DecisionStump(FAULTS ~ ., data = fit.faults)

# 10-fold cross-validation on the original training(fit) data
dstump.eval.fit <- evaluate_Weka_classifier(mod.dstump, numFolds = 10, 
                                            seed = 1) 
# 10-fold cross-validation on the original test data
dstump.eval.test <- evaluate_Weka_classifier(mod.dstump, newdata = test.faults, 
                                             numFolds = 10, seed = 1) 


# Assigntment #II Modeling Assignment: Classification Using Decision Trees
# Basic J48 (C4.5) decision tree with no adjustment to tuning parameters
tree <- J48(CLASS ~ ., data = fit.class)

# Evalution of base J48 tree on training data using 10-fold cross validation  
evaluate_Weka_classifier(tree, numFolds = 10, seed = 1, class = TRUE)

# Ploting the base J48 decision tree using the partykit package 
if(require("partykit", quietly = TRUE)) plot(tree)

# Evaluating the base J48 with the test data set
evaluate_Weka_classifier(tree, newdata = test.class, class = TRUE)

# A J48 decision tree with the unpruned tuning parameter set to true
tree.unpruned <- J48(CLASS ~ ., data = fit.class, 
                     control = Weka_control(U = TRUE))

# Evaluation of the unpruned J48 tree on training(fit) data using 10-fold
# cross validation
evaluate_Weka_classifier(tree.unpruned, numFolds = 10, seed = 1, class = TRUE)

# Ploting the unpruned J48 tree using the partykit package 
if(require("partykit", quietly = TRUE)) plot(tree.unpruned)

# Evaluating the unpruned J48 tree with the test data set
evaluate_Weka_classifier(tree.unpruned, newdata = test.class, class = TRUE)

# A J48 decision tree with the confidence factor set to 0.01 (default = 0.25)
tree.CF01 <- J48(CLASS ~ ., data = fit.class, control = Weka_control(C = 0.01))

# Evaluation of the J48 tree with confidence factor set to 0.01 on training data
# using 10-fold cross validation
evaluate_Weka_classifier(tree.CF01, numFolds = 10, seed = 1, class = TRUE)

# Ploting the J48 tree with confidence factor set to 0.01
if(require("partykit", quietly = TRUE)) plot(tree.CF01)

# Evaluating the J48 tree with confidence factor set to 0.01 with the 
# test data set
evaluate_Weka_classifier(tree.CF01, newdata = test.class, class = TRUE)

# Function definition for a function that takes x as its only argument and
# creates a model using x as the cost of Type II error in a Cost Sensitive
# Classifier as a meta-learner to a J48 tree. The functions returns the 10-fold
# cross validation on the original training data
GetModelErrors <- function(x){
  mod <- CostSensitiveClassifier(CLASS ~ ., fit.class, 
                                 control = Weka_control('cost-matrix' = matrix(c(0, x, 1, 0), 
                                                        nrow = 2, byrow = TRUE), W = 'J48'))
  evaluate_Weka_classifier(mod, class = TRUE, numFolds = 10, seed = 1)
}

# Defining the cost vector
cost <-  seq(0.1, 10, by = 0.1)

# Using ErrorsFromCostVec function to run each value in "cost" vector through
# GetModelErrors Function
output <- ErrorsFromCostVec(cost, GetModelErrors)

# Plot of error rates, training data
ggplot(output, aes(x = log10(cost), y = error, group = type, color = type))+
  geom_line() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(title = paste("Error Rates of Different Cost Ratios on \n a J48",
                     "Classifier (Training Data)", sep = ","),
       x = "Ratio of Type II error weight to Type I error weight (log10 scale)", 
       y = "Error (%)", colour = "Error type")

# Assignment #III Modeling Assignment: Using meta learning schemes with  
# a strong and a weak learner for classification

# Cost vector for all bagging classifiers
cost.bag <- c(1/10, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1:25)

# Cost sensitive classifier combined with bagging and J48 (10 iterations)
# training data
bag10.J48.fit <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, fit.class, 
                                   "Bagging", 10, 'J48')

# Plot of error rates
ggplot(bag10.J48.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier", 
                "with Bagging \n (10 iterations, Training Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call 
bag10.J48.fit2 <- spread(bag10.J48.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.158, Type II = 0.109)
table(bag10.J48.fit2$typeI, bag10.J48.fit2$typeII)

# Found which indices matched these error rates
ind.bag10.J48.fit2 <- which(bag10.J48.fit2$typeI == 0.158 & bag10.J48.fit2$typeII == 0.109)

# Subset data to find ideal cost 
ideal.bag10.J48.fit2 <- bag10.J48.fit2[ind.bag10.J48.fit2, "cost"]

# Ideal cost is when Type II errors are weighted 4 times Type I errors
ideal.bag10.J48.fit2

# Cost sensitive classifier combined with bagging and J48 (10 iterations)
# test data
bag10.J48.test <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, test.class, 
                                    "Bagging", 10, 'J48')

# Plot of error rates
ggplot(bag10.J48.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier",
                "with Bagging \n (10 iterations, Test Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag10.J48.test2 <- spread(bag10.J48.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.167, Type II = 0.179)
table(bag10.J48.test2$typeI, bag10.J48.test2$typeII)

# Found which indices matched these error rates
ind.bag10.J48.test2 <- which(bag10.J48.test2$typeI == 0.167 & bag10.J48.test2$typeII == 0.179)

# Subset data to find ideal cost 
ideal.bag10.J48.test2 <- bag10.J48.test2[ind.bag10.J48.test2, "cost"]

# Ideal cost is when Type II errors are weighted 4 times Type I errors
ideal.bag10.J48.test2 

# Cost sensitive classifier combined with bagging and Decision Stump (10 iterations)
# training data
bag10.DS.fit <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, fit.class, 
                                  "Bagging", 10, 'DecisionStump')

# Plot of error rates
ggplot(bag10.DS.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump",
                "Classifier with Bagging \n (10 iterations, Training Data)",
                sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag10.DS.fit2 <- spread(bag10.DS.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.165, Type II = 0.145)
table(bag10.DS.fit2$typeI, bag10.DS.fit2$typeII)

# Found which indices matched these error rates
ind.bag10.DS.fit2 <- which(bag10.DS.fit2$typeI == 0.165 & bag10.DS.fit2$typeII == 0.145)

# Subset data to find ideal cost
ideal.bag10.DS.fit2 <- bag10.DS.fit2[ind.bag10.DS.fit2, "cost"]

# Ideal cost is when Type I and Type II error weight are equal
ideal.bag10.DS.fit2 

# Cost sensitive classifier combined with bagging and Decision Stump (10 iterations)
# test data
bag10.DS.test <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, test.class, 
                                   "Bagging", 10, 'DecisionStump')

# Plot of error rates
ggplot(bag10.DS.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump",
                "Classifier with Bagging \n (10 iterations, Test Data)",
                sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag10.DS.test2 <- spread(bag10.DS.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.167, Type II = 0.143)
table(bag10.DS.test2$typeI, bag10.DS.test2$typeII)

# Found which indices matched these error rates
ind.bag10.DS.test2 <- which(bag10.DS.test2$typeI == 0.167 & bag10.DS.test2$typeII == 0.143)

# Subset data to find ideal cost
ideal.bag10.DS.test2 <- bag10.DS.test2[ind.bag10.DS.test2, "cost"]

# Ideal cost is when Type II errors are weighted 3 times Type I errors and when
# Type II errors are weighted 4 times Type I errors
ideal.bag10.DS.test2 

# Cost sensitive classifier combined with bagging and Decision Stump (25 iterations)
# training data
bag25.DS.fit <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, fit.class, 
                                  "Bagging", 25, 'DecisionStump')

# Plot of error rates
ggplot(bag25.DS.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump", 
          "Classifier with Bagging \n (25 iterations, Training Data)", 
          sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag25.DS.fit2 <- spread(bag25.DS.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.165, Type II = 0.164)
table(bag25.DS.fit2$typeI, bag25.DS.fit2$typeII)

# Found which indices matched these error rates
ind.bag25.DS.fit2 <- which(bag25.DS.fit2$typeI == 0.165 & bag25.DS.fit2$typeII == 0.164)

# Subset data to find ideal cost
ideal.bag25.DS.fit2 <- bag25.DS.fit2[ind.bag25.DS.fit2, "cost"]

# Ideal cost is when Type I and Type II error weight are equal
ideal.bag25.DS.fit2 

# Cost sensitive classifier combined with bagging and Decision Stump (25 iterations)
# test data
bag25.DS.test <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, test.class, 
                                   "Bagging", 25, 'DecisionStump')

# Plot of error rates
ggplot(bag25.DS.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump", 
          "Classifier with Bagging \n (25 iterations, Test Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag25.DS.test2 <- spread(bag25.DS.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.167, Type II = 0.143)
table(bag25.DS.test2$typeI, bag25.DS.test2$typeII)

# Found which indices matched these error rates
ind.bag25.DS.test2 <- which(bag25.DS.test2$typeI == 0.167 & bag25.DS.test2$typeII == 0.143)

# Subset data to find ideal cost
ideal.bag25.DS.test2 <- bag25.DS.test2[ind.bag25.DS.test2, "cost"]

# Ideal cost is when Type II errors are weighted 3 times Type I errors
ideal.bag25.DS.test2 

# Cost sensitive classifier combined with bagging and J48 (25 iterations)
# training data
bag25.J48.fit <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, fit.class, 
                                   "Bagging", 25, 'J48')

# Plot of error rates
ggplot(bag25.J48.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier with",
          "Bagging \n (25 iterations, Training Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag25.J48.fit2 <- spread(bag25.J48.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.135, Type II = 0.145)
table(bag25.J48.fit2$typeI, bag25.J48.fit2$typeII)

# Found which indices matched these error rates
ind.bag25.J48.fit2 <- which(bag25.J48.fit2$typeI == 0.135 & bag25.J48.fit2$typeII == 0.145)

# Subset data to find ideal cost
ideal.bag25.J48.fit2 <- bag25.J48.fit2[ind.bag25.J48.fit2, "cost"]

# Ideal cost is when Type II errors are weighted 2 times Type I errors
ideal.bag25.J48.fit2 

# Cost sensitive classifier combined with bagging and J48 (25 iterations)
# test data
bag25.J48.test <- ErrorsFromCostVec(cost.bag, CostSensMetaLearner, test.class,
                                    "Bagging", 25, 'J48')

# Plot of error rates
ggplot(bag25.J48.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier with", 
          "Bagging \n (25 iterations, Test Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
bag25.J48.test2 <- spread(bag25.J48.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.167, Type II = 0.179)
table(bag25.J48.test2$typeI, bag25.J48.test2$typeII)

# Found which indices matched these error rates
ind.bag25.J48.test2 <- which(bag25.J48.test2$typeI == 0.167 & bag25.J48.test2$typeII == 0.179)

# Subset data to find ideal cost
ideal.bag25.J48.test2 <- bag25.DS.test2[ind.bag25.J48.test2, "cost"]

# Ideal cost is when Type II errors are weighted 4 times Type I errors
ideal.bag25.J48.test2 

# Cost vector for AdaBoostM1 (10 iterations) J48 on the training data
cost.boost10.J48.fit <- seq(5, 75, by = 0.25)

# Cost sensitive classifier combined with boosting and J48 with 10 iterations
# training data
boost10.J48.fit <- ErrorsFromCostVec(cost.boost10.J48.fit, CostSensMetaLearner,
                                     fit.class, "AdaBoostM1", 10, 'J48')

# Plot of error rates
ggplot(boost10.J48.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier with",
          "AdaBoostM1 \n (10 iterations, Training Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type") 

# Spread data for table call
boost10.J48.fit2 <- spread(boost10.J48.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.105, Type II = 0.091)
table(boost10.J48.fit2$typeI, boost10.J48.fit2$typeII)

# Found which indices matched these error rates
ind.boost10.J48.fit2 <- which(boost10.J48.fit2$typeI == 0.105 & boost10.J48.fit2$typeII == 0.091)

# Subset data to find ideal cost
ideal.boost10.J48.fit2 <- boost10.J48.fit2[ind.boost10.J48.fit2, "cost"]

# Ideal cost is when Type II errors are weighted 25.5 times Type I errors
ideal.boost10.J48.fit2

# Cost vector for AdaBoostM1 (10 iterations) J48 on the test data
cost.boost10.J48.test <- c(seq(0, 75, by = 0.5))

# Cost sensitive classifier combined with boosting and J48 with 10 iterations
# test data
boost10.J48.test <- ErrorsFromCostVec(cost.boost10.J48.test, CostSensMetaLearner,
                                      test.class, "AdaBoostM1", 10, 'J48')

# Plot of error rates
ggplot(boost10.J48.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier", 
          " with AdaBoostM1 \n (10 iterations, Test Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type") 

# Spread data for table call
boost10.J48.test2 <- spread(boost10.J48.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.105, Type II = 0.091)
table(boost10.J48.test2$typeI, boost10.J48.test2$typeII)

# Found which indices matched these error rates
ind.boost10.J48.test2 <- which(boost10.J48.test2$typeI == 0.136 & boost10.J48.test2$typeII == 0.179)

# Subset data to find ideal cost
ideal.boost10.J48.test2 <- boost10.J48.test2[ind.boost10.J48.test2, "cost"]

# Ideal cost is when Type II errors are weighted 15.5 times Type I errors or
# when Type II errors are weight 58 times Type I
ideal.boost10.J48.test2

# Cost vector for AdaBoostM1 (25 iterations) J48 on the training data
cost.boost25.J48.fit <- c(seq(0, 125, by = 0.5))

# Cost sensitive classifier combined with boosting and J48 with 25 iterations
# training data
boost25.J48.fit <- ErrorsFromCostVec(cost.boost25.J48.fit, CostSensMetaLearner,
                                     fit.class, "AdaBoostM1", 25, 'J48')

# Plot of error rates
ggplot(boost25.J48.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier",
          "with AdaBoostM1 \n (10 iterations, Training Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type") 

# Spread data for table call
boost25.J48.fit2 <- spread(boost25.J48.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.083, Type II = 0.145)
table(boost25.J48.fit2$typeI, boost25.J48.fit2$typeII)

# Found which indices matched these error rates
ind.boost25.J48.fit2 <- which(boost25.J48.fit2$typeI == 0.083 & boost25.J48.fit2$typeII == 0.145)

# Subset data to find ideal cost
ideal.boost25.J48.fit2 <- min(boost25.J48.fit2[ind.boost25.J48.fit2, "cost"])

# Ideal cost is when Type II errors are weighted 90 times Type I errors
ideal.boost25.J48.fit2

# Cost vector for AdaBoostM1 (25 iterations) J48 on the test data
cost.boost25.J48.test <- c(seq(0, 85, by = 0.5))

# Cost sensitive classifier combined with boosting and J48 with 25 iterations
# test data
boost25.J48.test <- ErrorsFromCostVec(cost.boost25.J48.test, CostSensMetaLearner,
                                      test.class, "AdaBoostM1", 25, 'J48')

# Plot of error rates
ggplot(boost25.J48.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a J48 Classifier",
          "with AdaBoostM1 \n (25 iterations, Test Data)", sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
boost25.J48.test2 <- spread(boost25.J48.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.106, Type II = 0.214)
table(boost25.J48.test2$typeI, boost25.J48.test2$typeII)

# Found which indices matched these error rates
ind.boost25.J48.test2 <- which(boost25.J48.test2$typeI == 0.106 & boost25.J48.test2$typeII == 0.214)

# Subset data to find ideal cost
ideal.boost25.J48.test2 <- min(boost25.J48.test2[ind.boost25.J48.test2, "cost"])

# Ideal cost is when Type II errors are weighted 16.5 times Type I errors
ideal.boost25.J48.test2

# Cost vector for AdaBoostM1(10 iterations) Decision Stump on the training data
cost.boost10.DS.fit <- c(seq(0, 25, by = 0.05))

# Cost sensitive classifier combined with boosting and Decision Stump 
# with 10 iterations training data
boost10.DS.fit <- ErrorsFromCostVec(cost.boost10.DS.fit, CostSensMetaLearner, 
                                    fit.class, "AdaBoostM1", 10, 
                                    'DecisionStump')

# Plot of error rates
ggplot(boost10.DS.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump,
                Classifier with AdaBoostM1 \n (10 iterations, Training Data)",
                sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type") 

# Spread data for table call
boost10.DS.fit2 <- spread(boost10.DS.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.165, Type II = 0.164)
table(boost10.DS.fit2$typeI, boost10.DS.fit2$typeII)

# Found which indices matched these error rates
ind.boost10.DS.fit2 <- which(boost10.DS.fit2$typeI == 0.165 & boost10.DS.fit2$typeII == 0.164)

# Subset data to find ideal cost
ideal.boost10.DS.fit2 <- boost10.DS.fit2[ind.boost10.DS.fit2, "cost"]

# Ideal cost is when Type II errors are weighted 1.85 times Type I errors
ideal.boost10.DS.fit2

# Cost vector for AdaBoostM1(10 iterations) Decision Stump on the test data
cost.boost10.DS.test <- c(seq(0, 6, by = 0.05), 7:30)

# Cost sensitive classifier combined with boosting and Decision Stump 
# with 10 iterations test data
boost10.DS.test <- ErrorsFromCostVec(cost.boost10.DS.test, CostSensMetaLearner, 
                                     test.class, "AdaBoostM1", 10, 
                                     'DecisionStump')

# Plot of error rates
ggplot(boost10.DS.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump", 
          "Classifier with AdaBoostM1 \n (10 iterations, Test Data)",
          sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
boost10.DS.test2 <- spread(boost10.DS.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.167, Type II = 0.143)
table(boost10.DS.test2$typeI, boost10.DS.test2$typeII)

# Found which indices matched these error rates
ind.boost10.DS.test2 <- which(boost10.DS.test2$typeI == 0.167 & boost10.DS.test2$typeII == 0.143)

# Subset data to find ideal cost
ideal.boost10.DS.test2 <- boost10.DS.test2[ind.boost10.DS.test2, "cost"]

# Ideal cost is when Type II errors are weighted 4.7, 4.85, 4.9, or 4.95 times
# Type I errors
ideal.boost10.DS.test2

# Cost vector for AdaBoostM1(25 iterations) Decision Stump on the training data
cost.boost25.DS.fit <- c(seq(0, 10, by = 0.05), 11:30)

# Cost sensitive classifier combined with boosting and Decision Stump 
# with 25 iterations training data
boost25.DS.fit <- ErrorsFromCostVec(cost.boost25.DS.fit, CostSensMetaLearner, 
                                    fit.class, "AdaBoostM1", 25, 
                                    'DecisionStump')

# Plot of error rates
ggplot(boost25.DS.fit, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on \n a Decision Stump", 
                "Classifier with AdaBoostM1 \n (25 iterations, Training Data)", 
                sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type") 

# Spread data for table call
boost25.DS.fit2 <- spread(boost25.DS.fit, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.113, Type II = 0.164)
table(boost25.DS.fit2$typeI, boost25.DS.fit2$typeII)

# Found which indices matched these error rates
ind.boost25.DS.fit2 <- which(boost25.DS.fit2$typeI == 0.113 & boost25.DS.fit2$typeII == 0.164)

# Subset data to find ideal cost
ideal.boost25.DS.fit2 <- boost25.DS.fit2[ind.boost25.DS.fit2, "cost"]

# Ideal cost is when Type II errors are weighted 1.8 times Type I errors
ideal.boost25.DS.fit2

# Cost vector for AdaBoostM1(25 iterations) Decision Stump on the training data
cost.boost25.DS.test <- c(seq(0, 10, by = 0.05), 11:30)

# Cost sensitive classifier combined with boosting and Decision Stump 
# with 25 iterations test data
boost25.DS.test <- ErrorsFromCostVec(cost.boost25.DS.test, CostSensMetaLearner, 
                                     test.class, "AdaBoostM1", 25, 
                                     'DecisionStump')

# Plot of error rates
ggplot(boost25.DS.test, aes(x = cost, y = error, group = type, color = type)) +
  geom_line() +
  ggtitle(paste("Error Rates of Different Cost Ratios on a \n Decision Stump", 
                "Classifier with AdaBoostM1 \n (25 iterations, Test Data)", 
                sep = ",")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Ratio of Type II error weight to Type I error weight", 
       y = "Error (%)", colour = "Error type")

# Spread data for table call
boost25.DS.test2 <- spread(boost25.DS.test, type, error)

# Found from table lowest Type I and Type II errors with Type I approximately
# equal to Type II (Type I = 0.152, Type II = 0.143)
table(boost25.DS.test2$typeI, boost25.DS.test2$typeII)

# Found which indices matched these error rates
ind.boost25.DS.test2 <- which(boost25.DS.test2$typeI == 0.152 & boost25.DS.test2$typeII == 0.143)

# Subset data to find ideal cost
ideal.boost25.DS.test2 <- min(boost25.DS.test2[ind.boost25.DS.test2, "cost"])

# Ideal cost is when Type II errors are weighted 4.9 times Type I errors
ideal.boost25.DS.test2 

# Assignment #4- Module Ordering Models 
# Perfect ordering of the modules based on number of faults (training data)
ideal.faults <- sort(fit.faults$FAULTS, decreasing = TRUE)

# Using the predict function with the M5 Linear Regression model to predict the
# number of faults (training data)
fit.faults$predict.m5 <- predict(mod.m5, fit.faults)

# Correcting negative fault predictions as negative faults make no sense in the
# real world and will effect the Module Ordering Models (training data)
fit.faults$predict.m5 <- ifelse(fit.faults$predict.m5 < 0, 0, 
                                fit.faults$predict.m5)

# Ordered software modules based on M5 Linear Regression model's prediction
# (training data)
fit.faults.m5 <- fit.faults[order(fit.faults$predict.m5, decreasing = TRUE), ]

# New data frame with actual faults from predicted and ideal ordering (training data)
fit.faults.m5.ideal <- as.data.frame(cbind(fit.faults.m5$FAULTS, ideal.faults))

# Vector defining where each %5 cut-off increment should be based on the number
# of rows in fit.faults
cut.off.values <- round(seq(0.05, 0.5, by = 0.05)* nrow(fit.faults), 0)

# Vectors to store predicted ordering and ideal ordering fault sum (training data)
g.m5 <- rep(0, length(cut.off.values))
ghat.m5 <- rep(0, length(cut.off.values))

# For each cut-off point, g.m5 represents sum of the ideal ordering up to that
# cut-off point and ghat.m5 represents the sum of the predicted ordering up to
# the cut-off point (training data)
for (value in cut.off.values) {
  ind <- which(value == cut.off.values)
  g.m5[ind] <- sum(fit.faults.m5.ideal[1:value, 2])
  ghat.m5[ind] <- sum(fit.faults.m5.ideal[1:value, 1])
}

# Table of cutoffs, predicted faults, ideal gaults, performance (phi), % of
# predict / total (GhatF), and % of ideal / total (GF). (m5, training data)
alberg.table.m5 <- as.data.frame(cbind(seq(0.05, 0.5, by = 0.05), cut.off.values, 
                                 round(ghat.m5,0), g.m5, round(ghat.m5,0)/g.m5, 
                                 round(ghat.m5,0)/sum(fit.faults$FAULTS), 
                                 g.m5/sum(fit.faults$FAULTS)))
 
# Column names of alberg.table.m5 (training data)
colnames(alberg.table.m5) <- c("c", "cutoff index", "Ghat", 
                               "G", "phi", "GhatF", "GF")

# Ploting Alberg diagram for Training Data wit M5 Model (training data)
plot(x = alberg.table.m5$c, y = alberg.table.m5$GhatF, col = "red", type = "l", 
     ylim = c(0.3, 1.1), xlab = "Modules (%)", ylab = "Fault (%)", 
     main = "Alberg Diagram Training Data, M5 Model")
lines(x = alberg.table.m5$c, y = alberg.table.m5$GF, col = "blue")
legend(0.3, 0.6, c("ideal faults", "predicted faults"), lty = c(1,1),
       lwd=c(2.5,2.5), col=c("blue", "red"))

# Ploting performance for Training Data wit M5 Model (training data)
ggplot(alberg.table.m5, aes(x = c, y = phi)) +
  geom_line() +
  ggtitle("Performance of M5 Model on Training Data")

# Using the predict function with the greedy Linear Regression model to predict
# the number of faults (training data)
fit.faults$predict.greedy <- round(predict(mod.greedy, fit.faults), 0)

# Correcting negative fault predictions as negative faults make no sense in the
# real world and will effect the Module Ordering Models (training data)
fit.faults$predict.greedy <- ifelse(fit.faults$predict.greedy < 0, 0, 
                                    fit.faults$predict.greedy)

# Ordered software modules based on greedy Linear Regression model's prediction
# (training data)
fit.faults.greedy <- fit.faults[order(fit.faults$predict.greedy, decreasing = TRUE), ]

# New data frame with actual faults from predicted and ideal ordering (training data)
fit.faults.greedy.ideal <- as.data.frame(cbind(fit.faults.greedy$FAULTS, ideal.faults))

# Vectors to store predicted ordering and ideal ordering fault sum (training data)
g.greedy <- rep(0, length(cut.off.values))
ghat.greedy <- rep(0, length(cut.off.values))

# For each cut-off point, g.greedy represents sum of the ideal ordering up to that
# cut-off point and ghat.greedy represents the sum of the predicted ordering up to
# the cut-off point (training data)
for (value in cut.off.values) {
  ind <- which(value == cut.off.values)
  g.greedy[ind] <- sum(fit.faults.greedy.ideal[1:value, 2])
  ghat.greedy[ind] <- sum(fit.faults.greedy.ideal[1:value, 1])
}

# Table of cutoffs, predicted faults, ideal gaults, performance (phi), % of 
# predict / total (GhatF), and % of ideal / total (GF). (greedy, training data)
alberg.table.greedy <- as.data.frame(cbind(seq(0.05, 0.5, by = 0.05), 
                                           cut.off.values, ghat.greedy, 
                                           g.greedy, ghat.greedy/g.greedy, 
                                           ghat.greedy/sum(fit.faults$FAULTS), 
                                           g.greedy/sum(fit.faults$FAULTS)))

# Column names of alberg.table.greedy (training data)
colnames(alberg.table.greedy) <- c("c", "cutoff index", "Ghat", 
                                   "G", "phi", "GhatF", "GF")

# Ploting Alberg diagram for Training Data with M5 Model
plot(x = alberg.table.greedy$c, y = alberg.table.greedy$GhatF, col = "red", 
     type = "l", ylim = c(0.3, 1.1), xlab = "Modules (%)", ylab = "Fault (%)", 
     main = "Alberg Diagram Training Data, Greedy Model")
lines(x = alberg.table.greedy$c, y = alberg.table.greedy$GF, col = "blue")
legend(0.3, 0.6, c("ideal faults", "predicted faults"), lty = c(1,1),
       lwd=c(2.5,2.5), col=c("blue", "red"))

# Ploting performance for Training Data wit M5 Model (training data)
ggplot(alberg.table.greedy, aes(x = c, y = phi)) +
  geom_line() +
  labs(title = "Performance of Greedy Model on Training Data")

# Perfect ordering of the modules based on number of faults (test data)
ideal.faults.test <- sort(test.faults$FAULTS, decreasing = TRUE)

# Using the predict function with the M5 Linear Regression model to predict the
# number of faults (test data)
test.faults$predict.m5 <- predict(mod.m5, test.faults)

# Correcting negative fault predictions as negative faults make no sense in the
# real world and will effect the Module Ordering Models (test data)
test.faults$predict.m5 <- ifelse(test.faults$predict.m5 < 0, 0, 
                                 test.faults$predict.m5)
# Ordered software modules based on M5 Linear Regression model's prediction
# (test data)
test.faults.m5 <- test.faults[order(test.faults$predict.m5, decreasing = TRUE), ]

# New data frame with actual faults from predicted and ideal ordering (test data)
test.faults.m5.ideal <- as.data.frame(cbind(test.faults.m5$FAULTS, 
                                            ideal.faults.test))

# Vector defining where each %5 cut-off increment should be based on the number
# of rows in fit.faults
cut.off.values.test <- round(seq(0.05, 0.5, by = 0.05)* nrow(test.faults), 0)

# Vectors to store predicted ordering and ideal ordering fault sum (test data)
g.m5.test <- rep(0, length(cut.off.values.test))
ghat.m5.test <- rep(0, length(cut.off.values.test))

# For each cut-off point, g.m5.test represents sum of the ideal ordering up to that
# cut-off point and ghat.m5.test represents the sum of the predicted ordering up to
# the cut-off point (test data)
for (value in cut.off.values.test) {
  ind <- which(value == cut.off.values.test)
  g.m5.test[ind] <- sum(test.faults.m5.ideal[1:value, 2])
  ghat.m5.test[ind] <- sum(test.faults.m5.ideal[1:value, 1])
}

# Table of cutoffs, predicted faults, ideal gaults, performance (phi), % of 
# predict / total (GhatF), and % of ideal / total (GF). (m5, test data)
alberg.table.m5.test <- as.data.frame(cbind(seq(0.05, 0.5, by = 0.05), 
                                            cut.off.values.test, 
                                            round(ghat.m5.test,0), g.m5.test, 
                                            round(ghat.m5.test,0)/g.m5.test, 
                                            round(ghat.m5.test,0)/sum(test.faults$FAULTS), 
                                            g.m5.test/sum(test.faults$FAULTS)))


# Column names of alberg.table.m5.test (test data)
colnames(alberg.table.m5.test) <- c("c", "cutoff index", "Ghat", 
                                    "G", "phi", "GhatF", "GF")

# Ploting Alberg diagram for Test Data with M5 Model
plot(x = alberg.table.m5.test$c, y = alberg.table.m5.test$GhatF, col = "red",
     type = "l", ylim = c(0.3, 1.1), xlab = "Modules (%)", ylab = "Fault (%)", 
     main = "Alberg Diagram Test Data, M5 Model")
lines(x = alberg.table.m5.test$c, y = alberg.table.m5.test$GF, col = "blue")
legend(0.3, 0.6, c("ideal faults", "predicted faults"), lty = c(1,1),
       lwd=c(2.5,2.5), col=c("blue", "red"))

# Ploting performance for Training Data wit M5 Model (test data)
ggplot(alberg.table.m5.test, aes(x = c, y = phi)) +
  geom_line() +
  labs(title = "Performance of M5 Model on Test Data")

# Using the predict function with the greedy Linear Regression model to predict
# the number of faults (test data)
test.faults$predict.greedy <- round(predict(mod.greedy, test.faults), 0)

# Correcting negative fault predictions as negative faults make no sense in the
# real world and will effect the Module Ordering Models (test data)
test.faults$predict.greedy <- ifelse(test.faults$predict.greedy < 0, 0, 
                                     test.faults$predict.greedy)
# Ordered software modules based on greedy Linear Regression model's prediction 
# (test data)
test.faults.greedy <- test.faults[order(test.faults$predict.greedy, decreasing = TRUE), ]

# New data frame with actual faults from predicted and ideal ordering (test data)
test.faults.greedy.ideal <- as.data.frame(cbind(test.faults.greedy$FAULTS, 
                                                ideal.faults.test))

# Vectors to store predicted ordering and ideal ordering fault sum (test data)
g.greedy.test <- rep(0, length(cut.off.values.test))
ghat.greedy.test <- rep(0, length(cut.off.values.test))

# For each cut-off point, g.greedy.test represents sum of the ideal ordering up to that
# cut-off point and ghat.greedy.test represents the sum of the predicted ordering up to
# the cut-off point (training data)
for (value in cut.off.values.test) {
  ind <- which(value == cut.off.values.test)
  g.greedy.test[ind] <- sum(test.faults.greedy.ideal[1:value, 2])
  ghat.greedy.test[ind] <- sum(test.faults.greedy.ideal[1:value, 1])
}

# Table of cutoffs, predicted faults, ideal gaults, performance (phi), % of 
# predict / total (GhatF), and % of ideal / total (GF). (greedy, test data)
alberg.table.greedy.test <- as.data.frame(cbind(seq(0.05, 0.5, by = 0.05), 
                                                cut.off.values.test,
                                                ghat.greedy.test, g.greedy.test, 
                                                ghat.greedy.test/g.greedy.test, 
                                                ghat.greedy.test/sum(test.faults$FAULTS), 
                                                g.greedy.test/sum(test.faults$FAULTS)))

# Column names of alberg.table.greedy.test (test data)
colnames(alberg.table.greedy.test) <- c("c", "cutoff index", "Ghat", 
                                        "G", "phi", "GhatF", "GF")

# Ploting Alberg diagram for Test Data wit greedy Model
plot(x = alberg.table.greedy.test$c, y = alberg.table.greedy.test$GhatF, 
     col = "red", type = "l",  ylim = c(0.3, 1.1), xlab = "Modules (%)", 
     ylab = "Fault (%)", main = "Alberg Diagram for Test Data, Greedy Model")
lines(x = alberg.table.greedy.test$c, y = alberg.table.greedy.test$GF, 
      col = "blue") 
legend(0.3, 0.6, c("ideal faults", "predicted faults"), lty = c(1,1),
       lwd=c(2.5,2.5), col=c("blue", "red"))

# Ploting performance for Training Data wit greedy Model (test data)
ggplot(alberg.table.greedy.test, aes(x = c, y = phi)) +
  geom_line() +
  labs(title = "Performance of Greedy Model on Test Data")
