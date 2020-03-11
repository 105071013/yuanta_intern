##data set
df <- read.csv("./data/3227set.csv", header = TRUE)
df <- df[, -1]
df <- df[-c(1, dim(df)[1]), ]
##some package
library(randomForest)
library(dplyr)
library(caret)
##random forest
pred_rf <- c()
#change to dim(df)[1] - 15 (last 5 samples are NaN)
#start at 9th sample cause top 8 samples are NaN
for(i in 1:(dim(df)[1] - 10)){
  train <- df[i:(i+9), ]
  test  <- df[i+10, ]
  rf <- randomForest(next_mid1 ~., data = train)
  rf_pred <- predict(rf, newdata = test)
  pred_rf <- append(pred_rf, rf_pred)
}
par(mfrow=c(1,2))
##group
plot(pred_rf[1:500], col = 'green', type = 'l', lwd = 2)
lines(df$next_mid1[11: 510], col = 'red', lwd = 2)
legend("topright", legend = c("predicted", "reality"),
       lty = c(1, 1), col = c('green', 'red'), cex = 1)
plot(pred_rf, df$next_mid1[11:253058])
##split
plot(pred_rf[1:500], col = 'green', type = 'l', lwd = 2)
plot(df$next_mid1[11: 510], col = 'red', type= 'l', lwd = 2)
#test mse
t_mse <- sum((pred_rf - df$next_mid1[11: 253058])^2)
sum(pred_rf == df$next_mid1[11:253058])
#accruacy
#acc_1 <- sum(pred_rf == slice(df_1['pred'], 19:(dim(df_1)[1]-5))) / length(pred_rf)
#confusion matrix
#test_y <- as.matrix(slice(df_1['pred'], 19:(dim(df_1)[1]-5)))
#table(pred_rf, test_y)

##xgb
library('xgboost')
#data
df_1matrix <- as.matrix(df_1)
#setting parameters
params <- list(booster='gbtree',
               eval_metric='auc', # aucpr, ndcg, cox-nloglik
               eta=0.1,           #learning rate
               max_depth=12,       #tree depth
               min_child_weight=1.,
               #max_delta_step=0,
               subsample=1,       #percentage data use in every iteration
               colsample_bytree=0.6,  #percentage covariate use in every iteration
               lambda=0.7, 
               alpha=1)
#rolling window
pred_xgb <- c()
for(i in 9:(dim(df_1)[1]-15)){
  df_1train <- df_1matrix[i:(i+9), 1:4]
  df_1trainy <- df_1matrix[i:(i+9), 5]
  df_1test <- df_1matrix[i+10, 1:4]
  df_1testy <- df_1matrix[i+10, 5]
  dtrain <- xgb.DMatrix(data=df_1train, label = df_1trainy)
  dtest  <- xgb.DMatrix(data=df_1test , label = df_1testy )
  xgb_model <- xgb.train(params=params,
                         data=dtrain,
                         nrounds=200,
                         print_every_n=10,
                         objective='multi:softmax', 
                         num_class=3)
  xgb_pred <- predict(xgb_model, newdata=dtest)
  pred_xgb <- append(pred_xgb, xgb_pred) 
}