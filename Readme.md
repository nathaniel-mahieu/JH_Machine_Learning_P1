#Machine Learning Project 1: Predicting Excercize


This file is the entirety of the project.  I saved the generated models and included them in the repository (The code to generate the models is commented out below).  A clone of this repository should be executable.

```r
library(caret)
library(plyr)
library(doParallel)

cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

set.seed(987987)
```

## Data Import
Here we create two sets of data.  The training data will be used to develop the model and the testing data will be used to estimate the real-world error rate.

The columns of raw data are converted to numeric or factor as appropriate.  Columns 1:6 are names and times which should not be included as predictors and therefore are removed.

```r
rawdata = read.csv("pml-training.csv", stringsAsFactors=F)

rawdata[,160] = factor(rawdata[,160])
trainingdata = rawdata[,-(1:6)]
for (x in 1:(ncol(trainingdata)-1)) {
  trainingdata[,x] = as.numeric(trainingdata[,x])
}

zv = nearZeroVar(trainingdata, saveMetrics = T)
trainingdata = trainingdata[,!zv[,4]]


intrain = createDataPartition(y=trainingdata$classe, p=0.75, list=F)
training = trainingdata[intrain,]
testing = trainingdata[-intrain,]
```

## Training Data Model Building
Before fitting the model we impute and center and scale the data. We then train a Random Forests model as rfFit.

Our model is used to predict the known classes of the training data resulting in the in-sample error below. (~ 0.000)3al

```r
preProc = preProcess(training[,-ncol(training)], method=c("knnImpute", "center", "scale"))
training_proc = predict(preProc, training[,-ncol(training)])

#rfFit = train(training$classe ~ ., data=training_proc, method="rf", allowParallel=T)
#save("rfFit", file="rfFit.Rdata")
load("rfFit.Rdata")
print(rfFit)
```

```
## Random Forest 
## 
## 14718 samples
##   117 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##     2   0.9683687  0.9599498  0.002571220  0.003255631
##    60   0.9948133  0.9934363  0.001253299  0.001582857
##   118   0.9912962  0.9889850  0.002819840  0.003566945
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 60.
```

```r
modeloutput_train = predict(rfFit, training_proc)
confusionMatrix(modeloutput_train, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

## Testing Data Model Evaluation
We then use the model rfFit to predict the testing classes.  This results in the out-of-sample error below. (~ 0.0012)

```r
testing_proc = predict(preProc, testing[,-ncol(testing)])
                    
modeloutput_test = predict(rfFit, newdata = testing_proc)
confusionMatrix(modeloutput_test, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  945    0    0    0
##          C    0    1  855    5    0
##          D    0    1    0  798    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9978         
##                  95% CI : (0.996, 0.9989)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9972         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9958   1.0000   0.9925   0.9989
## Specificity            0.9994   1.0000   0.9985   0.9995   0.9998
## Pos Pred Value         0.9986   1.0000   0.9930   0.9975   0.9989
## Neg Pred Value         1.0000   0.9990   1.0000   0.9985   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1927   0.1743   0.1627   0.1835
## Detection Prevalence   0.2849   0.1927   0.1756   0.1631   0.1837
## Balanced Accuracy      0.9997   0.9979   0.9993   0.9960   0.9993
```

## Model comparisons
For fun we compare three different algorithms, Random Forests,  Partial Least Squares and weighted K Nearest Neighbors

```r
#
#plsFit = train(training$classe ~ ., data=training_proc, method="pls", allowParallel=T)
#save("plsFit", file="plsFit.Rdata")
load("plsFit.Rdata")

#kknnFit = train(training$classe ~ ., data=training_proc, method="kknn", allowParallel=T)
#save("kknnFit", file="kknnFit.Rdata")
load("kknnFit.Rdata")

resamps = resamples(list(rfFit = rfFit, plsFit = plsFit, kknnFit=kknnFit))
summary(resamps)
```

```
## 
## Call:
## summary.resamples(object = resamps)
## 
## Models: rfFit, plsFit, kknnFit 
## Number of resamples: 18 
## 
## Accuracy 
##           Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## rfFit   0.9910  0.9940 0.9949 0.9946  0.9952 0.9972    0
## plsFit  0.3488  0.3671 0.3826 0.3808  0.3883 0.4128    0
## kknnFit 0.9305  0.9333 0.9379 0.9370  0.9409 0.9421    0
## 
## Kappa 
##           Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## rfFit   0.9886  0.9924 0.9935 0.9932  0.9939 0.9965    0
## plsFit  0.1548  0.1826 0.2038 0.2004  0.2121 0.2448    0
## kknnFit 0.9122  0.9157 0.9215 0.9204  0.9253 0.9268    0
```


##Predicting on Validation Set
The predictions are made here on the 20 samples for which the classe is unknown.


```r
verify = read.csv("pml-testing.csv", stringsAsFactors=F)
verify[,160] = factor(verify[,160])
verifydata = verify[,-c(1:6)]
verifydata = verifydata[,!zv[,4]]

for (x in 1:(ncol(verifydata)-1)) {
  verifydata[,x] = as.numeric(verifydata[,x])
}


verify_proc = predict(preProc, verifydata[,-ncol(verifydata)])
modeloutput_verify = predict(rfFit, newdata = verify_proc)
modeloutput_verify
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
