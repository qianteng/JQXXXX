library(dplyr)
setwd("~/JQXXXX/kaggle/modelers/yc/")
source("new_func_def.R")

library(xgboost)
library(Matrix)
library(data.table)
library(entropy)
library(stringr)

train = read.csv("train.csv", header = T, stringsAsFactors = F)
test = read.csv("test.csv", header = T, stringsAsFactors = F)

train$Upc = as.character(train$Upc)
test$Upc = as.character(test$Upc)

summary(train)
table(train$TripType)
plot(density(train[which(train$TripType!=999),]$TripType))
table(train$VisitNumber)
plot(density(train$VisitNumber))

length(unique(train$TripType))
length(unique(train$DepartmentDescription))

# remove HEALTH AND BEAUTY AIDS from train data
train = train[which(train$DepartmentDescription != "HEALTH AND BEAUTY AIDS"),]

# fill na values on column FinelineNumber and Upc
train[which(is.na(train$Upc)),]$Upc = "0000"
train[which(is.na(train$FinelineNumber)),]$FinelineNumber = -1
test[which(is.na(test$Upc)),]$Upc = "0000"
test[which(is.na(test$FinelineNumber)),]$FinelineNumber = -1

# convert weekday to numbers
lists = as.character(unique(train$Weekday))
weekdays = c(5,6,7,1,2,3,4)
train$Weekday_convert = 0
test$Weekday_convert = 0
for (i in 1:length(lists)) {
    ind = which(train$Weekday == lists[i])
    train$Weekday_convert[ind] = weekdays[i]
    
    ind = which(test$Weekday == lists[i])
    test$Weekday_convert[ind] = weekdays[i]
    
}

# create a return column
ind = which(train$ScanCount < 0)
train$Return = 0
train$Return[ind] = abs(train$ScanCount[ind])

ind = which(test$ScanCount < 0)
test$Return = 0
test$Return[ind] = abs(test$ScanCount[ind])

# create a purchase column
ind = which(train$ScanCount > 0)
train$Purchase = 0
train$Purchase[ind] = train$ScanCount[ind]

ind = which(test$ScanCount < 0)
test$Return = 0
test$Return[ind] = abs(test$ScanCount[ind])

deptlist = sort(unique(train$DepartmentDescription))

uniqueVisitNumber = sort(unique(train$VisitNumber))
splits = split(uniqueVisitNumber, ceiling(1:length(uniqueVisitNumber)/10000))
str(splits)

for (i in 1:length(splits)) {
    dat = train %>% filter(TripType %in% splits[[i]]) %>% 
        group_by(TripType, VisitNumber) %>% 
        summarise(day = Weekday,
                  scanLength = length(ScanCount),
                  scanMax = max(ScanCount),
                  scanMin = min(ScanCount),
                  scanSum = sum(abs(ScanCount)),
                  scanMean = mean(abs(ScanCount)),
                  scanSD = sd(ScanCount),
                  scanSD_abs = sd(abs(ScanCout)),
                  purchaseSum = sum(Purchase),
                  purchaseCount = length(which(Purchase != 0)),
                  returnSum = sum(Return),
                  returnCount = length(which(Return != 0)),
                  unique_dept_count = length(unique(DepartmentDescription)),
                  unique_fineLine_count = length(unique(FinelineNumber)),
                  bins = generateBins_Fineline(Upc),
                  )
    
}


generateBins_Upc<-function(x)
{
    upcode<-x[which(nchar(x)>=4)]
    upc_temp<-strtoi(substr(upcode,start=1,stop=4), base = 0L)
    #estimation 50 per bin
    v<-discretize(upc_temp,numBins=500,r=range(0,10000)) 
    names(v)<-paste("upc_bins",c(1:500),sep="_")
    v<-v/length(upc_temp)
    v
}

#to return the number of 4 digit upc codes
generatefourLengthCounts<-function(x)
{
    indexes_length<-length(which(str_length(x$Upc)<=4))
    indexes_length
}




# convert day of the week to number


train$DepartmentDescription = as.factor(train$DepartmentDescription)
test$DepartmentDescription = as.factor(test$DepartmentDescription)

train_grouped = train %>% group_by(VisitNumber, Weekday_convert, DepartmentDescription) %>% 
    summarise(total_visit = sum(VisitNumber),
              max_weekday = max(Weekday_convert),
              min_weekday = min(Weekday_convert),
              )
)