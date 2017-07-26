
library(pec)
library(rms)
library(Hmisc)
library(xgboost)
library(data.table)
library(cplm)
library(readr)
library(plyr)
library(DiagrammeR)
library(dplyr)
library(caret)

setwd("C:/YZhang1/kagg/Aws")
dt=read.csv('device_failure.csv')
str(dt)

############################# section 1: data exploration #################
#1, data distributions
table(dt$failure)

summary(dt$attribute1)
summary(dt$attribute2)
summary(dt$attribute3)
summary(dt$attribute4)
summary(dt$attribute5)
summary(dt$attribute6)
summary(dt$attribute7)
summary(dt$attribute8)
summary(dt$attribute9)

#2,correlations. 
rcorr(as.matrix(dt[,c(4:12)]),type='spearman')
# The two columsn attribite7 and attribite8 are identicial. 
dt$attribute8=NULL


#3, time  differences by days
max(dt$date)
min(dt$date)
x=data.frame(as.Date(dt$date, format = "%Y-%m-%d") - as.Date('2015-01-01',format = "%Y-%m-%d"))
colnames(x)='days'

dt=cbind(dt,x)

dt$days = as.numeric(dt$days)
hist(dt$days)

#how many devices?
length(unique(dt$device))


#4,xgboost train/test
set.seed(1234)
dt$random <- runif(nrow(dt))
mean(dt$random)    
train <- dt[dt$random <= 0.7,]  #train data set
test <- dt[dt$random > 0.7,]  


############################# section 2: train data #################
# 9+ predictors
include <-  c('attribute1','attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute9','days')

# retains the missing values
options(na.action = 'na.pass')
x <- sparse.model.matrix(~ . - 1, data = dt[, include])
options(na.action = 'na.omit')

# response
y <- dt[, "failure"]

d_train <- xgb.DMatrix(data = x, label = y, missing = NA)

# 5-fold cross validation
set.seed(1234)
nrounds=1000
cv.res <- xgb.cv(   data=d_train,
                    label=y,
                    nfold = 5,
                    nround =nrounds,
                    objective           = "binary:logistic", 
                    booster             = "gbtree",
                    eval_metric         = "error",
                    eta                 = 0.1,  #0.02
                    max_depth           = 6,    #5
                    subsample=0.7,
                    colsample_bytree=0.7,
                    #max_delta_step     = 3,  #3. default =0 (0,10) for imbalanced data
                    verbose             = 2,
                    maximize            = FALSE,
                    prediction = TRUE,   #last iterations
                    missing             =NaN
)

#cv.res
str(cv.res)

plot(1:nrounds,cv.res$evaluation_log$train_error_mean,type="l",col="red")
lines(1:nrounds,cv.res$evaluation_log$test_error_mean,col="blue")
legend('topright', c('Train','Test'),lty=c(1,1),lwd=c(2.5,2.5), col=c('red','blue'))

#bestround<-which.min(abs(as.matrix(cv.res)[,1] - as.matrix(cv.res)[,3]))
bestround<-which.min(abs(as.matrix(cv.res$evaluation_log$train_rmse_mean)-as.matrix(cv.res$evaluation_log$test_rmse_mean)))
bestround




#train
params <- list(
  objective = 'binary:logistic',
  eval_metric = 'error', 
  max_depth = 6,
  subsample=0.7,
  colsample_bytree=0.7,
  eta = 0.1)

bst <- xgb.train(
  data = d_train, 
  params = params, 
  nthreads =2,
  maximize = FALSE,
  watchlist = list(train = d_train), 
  nrounds = 800)   # after xgb.cv, the bestround is around 450


# ROC curve
library(pROC) 
train.pred<-as.integer(round(predict(bst, data.matrix(train[,include]),missing=NaN)))
g1 <- roc(train$failure,train.pred,C(0,1))
plot(g1,legacy.axes = T,col='blue')
cutoff <- coords(g1, x = "best", best.method = "closest.topleft")
cutoff


# performance on test dataset
test.pred<-as.integer(round(predict(bst, data.matrix(test[,include]),missing=NaN)))
table(test.pred)

#kappa
library("Metrics")
ScoreQuadraticWeightedKappa(as.numeric(test$failure),as.numeric(test.pred)) 

#confusion matrix
confusionMatrix(test.pred,test$failure,positive = '1')


