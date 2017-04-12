library(dplyr)
setwd("~/JQXXXX/kaggle/modelers/yc/")

train = read.csv("train.csv", header = T, stringsAsFactors = F)
test = read.csv("test.csv", header = T, stringsAsFactors = F)

summary(train)
table(train$TripType)
plot(density(train[which(train$TripType!=999),]$TripType))
table(train$VisitNumber)
plot(density(train$VisitNumber))

length(unique(train$TripType))
length(unique(train$DepartmentDescription))

train[which(is.na(train$Upc)),]$Upc = -1
train[which(is.na(train$FinelineNumber)),]$FinelineNumber = -1
test[which(is.na(test$Upc)),]$Upc = -1
test[which(is.na(test$FinelineNumber)),]$FinelineNumber = -1

# create a return column
ind = which(train$ScanCount < 0)
train$Return = 0
train$Return[ind] = abs(train$ScanCount)

# create a purchase column
ind = which(train$ScanCount > 0)
train$Purchase = 0
train$Purchase[ind] = train$ScanCount

