library(data.table)

# Load Data ---------------------------------------------------------------

rm(list=ls())
setwd("C:\\")

train <- fread("device_failure.csv")

table(train$failure)

table(train$device)

# survival analysis ------------------------------------------------------

