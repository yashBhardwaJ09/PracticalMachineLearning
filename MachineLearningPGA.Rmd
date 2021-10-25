---
title: "MachineLearningPGA1"
output:
  pdf_document: default
  html_document: default
---
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

## Data Preprocessing  
```{r, cache = T}
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
set.seed(1234)
```
```{r, cache = T}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```  
### Read the Data
After downloading the data from the data source, we can read the two csv files into two data frames.  
```{r, cache = T}
traincsv <- read.csv("./data/pml-training.csv")
testcsv <- read.csv("./data/pml-testing.csv")
dim(traincsv)
dim(testcsv)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict. 

### Clean the data
Removing unnecessary variables. Starting with N/A variables.
```{r, cache = T}
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .9] #removing mostly na columns
traincsv <- traincsv[,-c(1:7)]#removing metadata which is irrelevant to the outcome
```
Removing near zero variance variables.
```{r, cache = T}
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)
```
Now that we have finished removing the unnecessary variables, we can now split the training set into a validation and sub training set. The testing set “testcsv” will be left alone, and used for the final quiz test cases.
```{r, cache = T}
inTrain <- createDataPartition(y=traincsv$classe, p=0.7, list=F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]
```
###  Creating and Testing the Models
Here we will test a few popular models including: Decision Trees, Random Forest, Gradient Boosted Trees, and SVM. This is probably more than we will need to test, but just out of curiosity and good practice we will run them for comparison.

Set up control for training to use 3-fold cross validation.
```{r, cache = T}
control <- trainControl(method="cv", number=3, verboseIter=F)
```

## Decision Tree
 Model:
```{r, cache = T}
mod_trees <- train(classe~., data=train, method="rpart", trControl = control, tuneLength = 5)
fancyRpartPlot(mod_trees$finalModel)
```
 Prediction:
```{r, cache = T}
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
cmtrees
```
Random Forest
```{r, cache = T}
mod_rf <- train(classe~., data=train, method="rf", trControl = control, tuneLength = 5)
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))
cmrf
```

Gradient Boosted Trees
```{r, cache = T}
mod_gbm <- train(classe~., data=train, method="gbm", trControl = control, tuneLength = 5, verbose = F)
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))
cmgbm
```
Support Vector Machine
```{r, cache = T}
mod_svm <- train(classe~., data=train, method="svmLinear", trControl = control, tuneLength = 5, verbose = F)
pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))
cmsvm
```
## Predicting on Test Data Set
Running our test set to predict the classe (5 levels) outcome for 20 cases with the Random Forest model.
```{r, cache = T}
pred <- predict(mod_rf, testcsv)
print(pred)
```
## Appendix: Figures
1.correlation matrix of variables in training set  
```{r, cache = T}
corrPlot <- cor(train[, -length(names(train))])
corrplot(corrPlot, method="color")
```
2.Plotting the models
```{r, cache = T}
plot(mod_trees)
```
```{r, cache = T}
plot(mod_rf)
```
```{r, cache = T}
plot(mod_gbm)
```