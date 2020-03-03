######## Ridge and LASSO in R ########

#### package : glmnet ####
#install.packages("glmnet")
library(glmnet)

#-------------------------------------

#### data : Boston ####
library(MASS)
data("Boston")
head(Boston)
data = Boston[ ,-c(4,9)]  # 去掉類別型
Y = data[,12]             # 房價中位數
X = data[,-12]  

#-------------------------------------

#### X : 標準化 , Y : 中心化 ####
# 記得做Ridge或Lasso前，皆須將資料標準化
X = scale(X)      # 標準化
Y = Y - mean(Y)   # 中心化
# Y = scale(Y,scale = F)
# apply(X,2,mean)
# apply(X,2,var)
# mean(Y)

#-------------------------------------

#### LASSO ####
LASSO = glmnet(X,Y,family = "gaussian",alpha = 1,nlambda = 100)    

# X 要放矩陣
class(X)

# family : Y是連續型  : gaussian
#          Y是二元    : binomial
#          Y是多元類別: multinomial

# alpha : 調整L1 norm and L2 norm的比例 
#         alpha = 0 : Ridge
#         alpha = 1 : LASSO

# nlambda : 跑幾個不同的lambda

# lambda : 直接設定lambda值

?glmnet

#-------------------------------------

#### Result ####
LASSO        #不同lambda下選到的變數個數
LASSO$beta   #不同lambda下的參數估計值

# 看參數估計值如何隨著lambda增加而改變
plot(LASSO,xvar = "lambda",main = "Solution path",label = TRUE)   # 畫solution path

#-------------------------------------

#### 找最佳lambda值 -> cross validation ####
set.seed(10)
CVLASSO = cv.glmnet(X,Y,family = "gaussian",nfold = 5,alpha = 1)
CVLASSO
# lambda.min -> 最小cv error對應到的lambda   
# (value of lambda that gives minimum cvm)
# lambda.1se -> 找跟lambda.min差不多但更簡單的model   
# (largest value of lambda such that error is within 1 standard error of the minimum)

plot(CVLASSO)
26.508 + 4.951  # lambda.min : cvm + cvsd = 31.459
# lambda.1se -> 找cvm < 31.459 且lambda最大者(第18個lambda : 1.393833)
CVLASSO$lambda.min
CVLASSO$lambda.1se

# Use lambda.min
lasso1 = glmnet(X,Y,family = "gaussian",alpha = 1,lambda = CVLASSO$lambda.min)  #lambda代lambda.min
lasso1
lasso1$beta               # 係數
which(lasso1$beta != 0)   # 選到的變數

# Use lambda.1se
lasso2 = glmnet(X,Y,family = "gaussian",alpha = 1,lambda = CVLASSO$lambda.1se)   #lambda代lambda.1se
lasso2
lasso2$beta               # 係數
which(lasso2$beta != 0)   # 選到的變數

#-------------------------------------

#### Ridge ####
RIDGE = glmnet(X,Y,family = "gaussian",alpha = 0,nlambda = 100)  

#-------------------------------------

#### Result ####
RIDGE  #不同lambda下選到的變數個數
plot(RIDGE,xvar = "lambda",main = "Solution path")   #畫solution path

#-------------------------------------

#### 找最佳lambda值 -> cross validation ####
CVRIDGE = cv.glmnet(X,Y,family = "gaussian",nfold = 5,alpha = 0)
CVRIDGE

plot(CVRIDGE)
25.83049 + 3.499500  # lambda.min : cvm + cvsd = 29.32999
# lambda.1se -> 找cvm < 29.32999 且lambda最大者(第78個lambda : 5.2476911)
CVRIDGE$lambda.min
CVRIDGE$lambda.1se

# Use lambda.min
ridge1 = glmnet(X,Y,family = "gaussian",alpha = 0,lambda = CVRIDGE$lambda.min)   # lambda代lambda.min
ridge1$beta   #係數

# Use lambda.1se
ridge2 = glmnet(X,Y,family = "gaussian",alpha = 0,lambda = CVRIDGE$lambda.1se)   # lambda代lambda.1se
ridge2$beta   #係數

