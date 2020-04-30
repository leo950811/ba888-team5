library(ggplot2)
library(tidyverse)
library(plotly)
library(corrplot)
library(skimr)
library(randomForest)
library(xgboost)
library(caret)
library(gbm)
library(ggstatsplot)
library(data.table)
library(pROC)
library(naivebayes)
library(DMwR)
library(ROSE)
library(rpart)

#read data
application = read.csv('application_train.csv')

#current work period(deal with invaild data)
application$DAYS_EMPLOYED0 = -application$DAYS_EMPLOYED / 365
application$DAYS_EMPLOYED[application$DAYS_EMPLOYED0 < 0] = 0

#data cleaning
## delete variables according three rules
## 1. have more than 50% missing values
## 2. have 0 standard deviation
## 3. highly correlated variables (correlation > 80% and only drop one from each pair)
#select numeric data & fill missing values with 0
applicants0 <- application %>% 
  select(CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH, 
         DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, FLAG_EMP_PHONE, 
         FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, REGION_RATING_CLIENT,
         HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION,
         REG_CITY_NOT_LIVE_CITY, LIVE_CITY_NOT_WORK_CITY,  
         EXT_SOURCE_2, EXT_SOURCE_3, FLOORSMAX_AVG, FLOORSMAX_MODE, 
         FLOORSMAX_MEDI, TOTALAREA_MODE,
         OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, 
         DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_3, 
         AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON,
         AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR)  
applicants0[is.na(applicants0)] = 0

#take a look of correlation
cor(cbind(applicants0, application$TARGET))

#select categorical data
applicants1 = application %>% select(CODE_GENDER, 
                                FLAG_OWN_CAR, FLAG_OWN_REALTY, 
                                NAME_CONTRACT_TYPE, NAME_EDUCATION_TYPE, 
                                NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, 
                                NAME_INCOME_TYPE, 
                                OCCUPATION_TYPE, ORGANIZATION_TYPE, 
                                WEEKDAY_APPR_PROCESS_START)


#sum all document variables to avoid losing info as TOTAL_FLAG_DOCUMENT
application$TOTAL_FLAG_DOCUMENT = application %>% 
  select(FLAG_DOCUMENT_2:FLAG_DOCUMENT_21) %>% rowSums()

#after pre-processed data - applicants
applicants = cbind(application$TARGET, applicants0, applicants1)

###########################random forest########################
set.seed(7)
SAMP = sample(1:nrow(train), 260000)
trainData = train[SAMP, ] %>% select(-ORGANIZATION_TYPE)
testData = train[-SAMP, ] %>% select(-ORGANIZATION_TYPE)  

#manually balance 
trainT1 = trainData %>% filter(TARGET == 1)
trainT0 = trainData %>% filter(TARGET == 0)
partTrain = trainT0[sample(1:nrow(trainT0), nrow(trainT1)), ]
train_final = rbind(trainT1, partTrain) 

#build model
mod_r <- randomForest(TARGET %>% as.factor() ~ ., train_final)
print(mod_r)

#apply test data
preds = predict(mod_r, newdata = testData)
tab = table(preds, testData$TARGET)
yardstick::accuracy(tab)
varImpPlot(mod_r)
