
# Prudential Life Insurance Assessment
# XGBoost learning
require(xgboost)
require(data.table)
require(Metrics)
require(stringr)
require(Matrix)
require(SparseM)
require(DiagrammeR)
require(Ckmeans.1d.dp)
require(kernlab)
require(randomForest)
require(caret)
setwd("./R/kaggle/Prudential Life Insurance Assessment/")
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")
sampleSubmission <- read.csv("data/sample_submission.csv")

str(train)
str(test)

# All features shared, making feature transformations simultaneously. 
response <- train$Response
train$training <- 1
test$training  <- 0

data <- rbind(train[-c(1,128)], test[-1])
colnames(data)

prop.table(table(response))
plot(prop.table(table(response)))

feature.names <- names(data[-127])
for( f in feature.names ){
  if(class(data[[f]]) == "factor"){
    levels <- unique(c(train[[f]],test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]]), levels = levels)
    test[[f]] <- as.integer(factor(test[[f]]), levels = levels)
    data[[f]] <- as.integer(factor(data[[f]]), levels = levels)
    
  }
}

data.roughfix <- na.roughfix(data)

# Using training data to identify most important features with xgboost.
model_xgboost <- xgboost(data = data.matrix(data.roughfix[data.roughfix$training==1,]), 
                         label  = response, 
                        nround  = 55, 
                      objective = "reg:linear",
                    eval_metric = "rmse",
                        missing = NaN)
test_xgb_cv <- xgb.cv(params = list("objective" = "reg:linear", "eval_metric" = "rmse"), 
       data = data.matrix(data.roughfix[data.roughfix$training==1,]),
       missing = NaN, nfold=10, label = response,  nrounds = 50)
plot(test_xgb_cv$test.rmse.mean)
plot(test_xgb_cv$test.rmse.mean) # 55 looks like minimum. 1.852797

# [99]	train-rmse:1.587900

model_dump <- xgb.dump(model_xgboost, with.stats = T)
importance.matrix <- xgb.importance(names(data.roughfix), model_xgboost, filename_dump = NULL)
xgb.plot.importance(importance.matrix[1:30])


# Creating a feature counting the medical keywords for each instance (medical keywords is column 80:127)
medkeywords <- apply(data.roughfix[,79:126], 1, sum)
data.roughfix$medkeywords <- as.integer(medkeywords)
partition <- createDataPartition(response, times = 1, p = 0.75)
training <- data.roughfix[data.roughfix$training==1,]
training_train <- training[partition$Resample1,-127]
training_test <- training[-partition$Resample1,-127]

model_xgboost <- xgboost(data = data.matrix(training_train), 
                         label  = response[partition$Resample1], 
                         nround  = 55, 
                         objective = "reg:linear",
                         eval_metric = "rmse",
                         missing = NaN)
# Small improvement. [99]	train-rmse:1.585579

pred <- predict(model_xgboost, data.matrix(training_test), missing=NaN)
actual <- response[-partition$Resample1]
offset <- lm(actual~pred-1)$coef

data <- data.frame(pred)
data$actual <- actual
data[1:10,]
data$diff <- data$actual-round(data$pred)


SQWKfun = function(x = seq(1.5, 7.5, by = 1), pred) {
  preds = data$pred
  true = data$actual
  cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
  preds = as.numeric(Hmisc::cut2(preds, cuts))
  err = Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8)
  return(-err)
}

optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, pred = data)
preds = as.numeric(Hmisc::cut2(data$pred, c(-Inf, optCuts$par, Inf)))

data$preds <- preds
head(data)
data$diff2 <- data$actual - data$preds
data[1:20,]

continous <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1",
                "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", 
                "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")
# Inspecting continous variables.
xyplot(Response~-exp(Wt), data = train)

# Categorical variables. One-hot encoding the most important.
categorical_string <- as.character("Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41")
categorical_names <- unlist(strsplit(categorical_string, split = ", "))
top30features <- importance.matrix$Feature[1:30]
which(top30features %in% categorical_names)
top30categorical_names <- top30features[which(top30features %in% categorical_names)]
# One-hot encoding top 15 categorical variables
top30categorical_factor <- as.data.frame(apply(data.roughfix[,top30categorical_names],2,as.factor))
categorical_one_hot <- as.data.frame(model.matrix(~.-1, top30categorical_factor[-8])) # Except Medical_History_2 which has too many levels.
categorical_one_hot2 <- as.data.frame(sapply(categorical_one_hot,as.factor))
str(categorical_one_hot2)

data.roughfix2 <- cbind(data.roughfix, categorical_one_hot2)
one_hot_cv.xboost <- test_xgb_cv <- xgb.cv(params = list("objective" = "reg:linear", "eval_metric" = "rmse"), 
                                           data = data.matrix(data.roughfix2[data.roughfix2$training==1,]),
                                           missing = NaN, nfold=10, label = (response-1),  nrounds = 100, nthread = 2)
plot(one_hot_cv.xboost$test.rmse.mean[20:70], col="blue", lty=1)
lines(test_xgb_cv$test.rmse.mean[20:70], col="red")

model2 <- xgboost(data = data.matrix(data.roughfix2[data.roughfix2$training==1,]), 
                         label  = (response-1), 
                         nround  = 100, 
                         objective = "reg:linear",
                         eval_metric = "rmse",
                         missing = NaN)
# Benchmark: train-rmse:1.585579
# Crossvalidating with ScoreQuadraticWeightedKappa

folds <- createFolds(response, 10)
training <- data.roughfix[data.roughfix$training == 1,]
cv_results <- lapply(folds, function(x){
  train <- data.matrix(training[-x,])
  test <- data.matrix(training[x,])
  model <- xgboost(data = train,
                   label = response[-x],
                   nrounds = 55,
                   objective = "reg:linear",
                   eval_metric = "rmse",
                   missing = NaN)
  model_pred <- round(predict(model, test, missing=NaN))
  model_pred[model_pred>8] <- 8
  model_pred[model_pred<1] <- 1
  actual <- response[x]
  qwkappa <- ScoreQuadraticWeightedKappa(actual, model_pred)
  print(qwkappa)
  return(qwkappa)
})
cv_results



model <- xgboost(data = data.matrix(data.roughfix[data.roughfix$training == 1,]),
                 label = response,
                 nrounds = 55,
                 objective = "reg:linear",
                 eval_metric = "rmse",
                 missing = NaN)

model_pred <- predict(model_xgboost, data.matrix(data.roughfix[data.roughfix$training == 0,]), missing=NaN)
model_pred_offset <- model_pred*offset
model_pred_offset <- round(model_pred_offset)
model_pred_offset[model_pred_offset>8] <- 8
model_pred_offset[model_pred_offset<1] <- 1
plot(prop.table(table(model_pred_offset)))

results_df <- data.frame(test$Id)
colnames(results_df) <- "Id"
results_df$Response <- model_pred_offset
head(results_df)
write.table(results_df, "results.csv", row.names=F, sep=",")


preds = as.numeric(Hmisc::cut2(model_pred, c(-Inf, optCuts$par, Inf)))

results2 <- cbind(test$Id, preds)
colnames(results2) <- c("Id","Response")
write.table(results2, "results2.csv", row.names=F, sep=",")
