

Predictive Model of Quality of Weight-Lifting Exercises
=======================================================

## Synopsis
A Random Forest model is trained to recognise different variations in quality of weight-lifting acvitity. The model achieves an estimated out-of-sample accuracy of 99%.
<br><br>


## I. Data Processing
First, we download a dataset containing sensor measurement output from a number of volunteers performing five variations of a barbell weight lift.

```r
# Download data and load 'caret' package for modeling
setInternet2(TRUE)
dat <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
library(caret)
```
The downloaded dataset contains a total of **19622** labeled observations.

To clean-up this data set for analysis and modeling, we remove columns that:
* Are irrelevant features, such as timestamps and subject names
* Contain mostly "NA" values
* Are predominantly blank

```r
# Remove unneccessary non-feature columns
nonFeatureColNames <-
      c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2',
      'cvtd_timestamp', 'new_window', 'num_window')
dat <- dat[, -match(nonFeatureColNames, names(dat))]
rm(nonFeatureColNames)

# Remove columns with over 30% of values being NAs
naPropUnacceptable <- 0.3
naProp <- colSums(is.na(dat)) / nrow(dat)
naColNums <- (1:ncol(dat))[naProp > naPropUnacceptable]
dat <- dat[, -naColNums]
rm(naPropUnacceptable, naProp, naColNums)

# Remove columns with many blanks (read as "factor" class)
colClasses <- sapply(dat, class)
factorColNums <- (1:ncol(dat))[colClasses == 'factor']
factorColNums <- factorColNums[1 : (length(factorColNums) - 1)]
dat <- dat[, -factorColNums]
rm(colClasses, factorColNums)
numFeats <- ncol(dat) - 1
```
The number of features used for modelling is **52**.

Lastly, we randomly split the dataset into a Training set (60%) and a Testing set (40%).

```r
# Split data set into Training & Testing set
set.seed(313)
indcsTrain <- createDataPartition(y = dat$classe, p = 0.6, list = FALSE)
datTrain <- dat[indcsTrain, ]
datTest <- dat[-indcsTrain, ]
rm(dat, indcsTrain)
```
The Training set contains **11776** observations, and the Test set contains **7846**.
<br><br>


## II. Training a Random Forest Classification Model 
A Random Forest model is fit to the Training set to predict the activity quality variable **classe** on the 52 selected features.

```r
# Train a Random Forest model
model <- train(classe ~ ., data = datTrain, method = 'rf',
      trControl = trainControl(method = "cv", number = 4),
      allowParallel = TRUE)
```
<br>

## III. Estimated Out-of-Sample Accuracy
The trained model achieves high predictive accuracy on the seperate Testing dataset, with Balanced Accuracy of **over 99%** for each of the five activity quality variations A-E.

```r
predTest <- predict(model, datTest)
confusionMatrix(predTest, datTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    5    0    0    0
##          B    1 1510   17    0    0
##          C    0    3 1349   22    7
##          D    0    0    2 1263    1
##          E    0    0    0    1 1434
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.995    0.986    0.982    0.994
## Specificity             0.999    0.997    0.995    1.000    1.000
## Pos Pred Value          0.998    0.988    0.977    0.998    0.999
## Neg Pred Value          1.000    0.999    0.997    0.997    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.161    0.183
## Detection Prevalence    0.285    0.195    0.176    0.161    0.183
## Balanced Accuracy       0.999    0.996    0.991    0.991    0.997
```
Finally, the model also performs well in the *Practical Machine Learning* course's separate testing dataset, classifying correctly 20 out of 20 examples.
