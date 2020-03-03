# Statitical learning 2019/12/09 demo

###########################################
#### Tree and Bagging and Randomforest ####
###########################################

# ¦w¸Ë©Ò»Ý®M¥ó
# install.packages("rpart")
library(rpart)
# install.packages("rpart.plot")
library(rpart.plot)
# install.packages("tree")
library(tree)
# install.packages("randomForest")
library(randomForest)


# ¨Ï¥Î«e¦C¸¢Àù¯g¸ê®Æ
# install.packages("lasso2")
data(Prostate, package="lasso2")
str(Prostate)
head(Prostate)
# response variable : lpsa(Log Prostate Specific Antigen)(ÄáÅ@¸¢¯S²§§Ü­ì)
# variable explanation : https://rafalab.github.io/pages/649/prostate.html

#### ±N¸ê®Æ¤Á¦¨†ç‚ºtrain 80% ,test 20% ####
set.seed(5)
index = sample(1:nrow(Prostate),ceiling(0.8*nrow(Prostate)))
train = Prostate[index,]
test = Prostate[-index,]


##############
#### Tree ####
##############

# rpart() and tree()

# use rpart function (·|¦Û°Ê­×°Å)
tree = rpart(lpsa~.,data=train)  
tree
# the deviance is simply the sum of squared errors for the tree
summary(tree)
rpart.plot(tree)  #µøÄ±¤Æ

pred = predict(tree,newdata = test)  #¹w´útest data
mean((pred - test$lpsa)^2)  # ­pºâMSE = 0.838

# MSE = 0.7189

# ¥i½Õ¾ã°Ñ¼Æ
# minsplit¡G¨C¤@­Ónode³Ì¤Ö­n´X­Ódata
# minbucket¡G¦b¥½ºÝªºnode¤W³Ì¤Ö­n´X­Ódata
# cp¡Gcomplexity parameter
# maxdepth¡GTreeªº²`«×

# cp : alpha , ¹w³]=0.01
# alpha¶V¤j , ¿ï¨ìªº¤l¾ð¶V¤p
# cp = 0.05
tree = rpart(lpsa~.,data=train,cp=0.05)  
rpart.plot(tree)  #µøÄ±¤Æ
# cp = 0.08
tree = rpart(lpsa~.,data=train,cp=0.08)  
rpart.plot(tree)  #µøÄ±¤Æ


# use tree function (­n¦Û¤v°µ­×°Å)
tree2 = tree(lpsa~.,data=train)
tree2
plot(tree2) ;  text(tree2,pretty=0)
#If pretty = 0 then the level names of a factor split attributes are used unchanged. 
#If pretty = NULL, the levels are presented by a, b, ¡K z, 0 ¡K 5. 
#If pretty is a positive integer, abbreviate is applied to the labels with that value for its argument minlength.

# §Q¥Î¥¼­×°Åªºtree¹ïtest data°µ¹w´ú
pred1 = predict(tree2,newdata = test)
mean((pred1 - test$lpsa)^2)  # ­pºâMSE = 1.01 -> «Ü¤j(¦]¬°¨S¦³­×°Å)

# MSE = 0.9603 => overfitting

# §Q¥Îcross-validation§ä³Ì¨Î¾ðªº¤j¤p
set.seed(98)
cv_tree = cv.tree(tree2,K=5)  # K-folds CV
cv_tree  
plot(cv_tree$size ,cv_tree$dev ,type="b")
# choose best size = 5
#dev -> sum of square error for diff models

prune_tree = prune.tree(tree2,best = 5)  #­×°Å
plot(prune_tree)
text(prune_tree,pretty=0)

pred2 = predict(prune_tree,newdata = test)  #§Q¥Î­×°Å§¹ªºtree¹ïtest data°µ¹w´ú
mean((pred2 - test$lpsa)^2)  # ­pºâMSE = 0.8488447

# MSE = 0.7058


#################
#### Bagging ####
#################
B = 100  # ©â100¦¸boostrap sample
B_pred = matrix(ncol=B,nrow=nrow(test))
for (i in 1:B){
  B_sample = train[sample(1:nrow(train),replace = T),]   # bootstrap sample
  B_tree = rpart(lpsa~.,data=B_sample)
  B_pred[,i] = predict(B_tree,newdata=test)
}
mean((apply(B_pred,1,mean) - test$lpsa)^2)  # ¨ú100¦¸¥­§¡ ­pºâMSE = 0.7978106

# MSE = 0.5884


######################
#### Randomforest ####
######################
set.seed(75)
rf = randomForest(lpsa~.,data=train)
rf

pred3 = predict(rf,newdata = test)
mean((pred3 - test$lpsa)^2) # ­pºâMSE

# MSE = 0.6943

# ¥i½Õ¾ã°Ñ¼Æ
# ntree : ¥Í¦¨´X´Ê¾ð (¹w³]500)
# mtry : ¨C¦¸¤Á³Î¦Ò¼{ªºÅÜ¼Æ­Ó¼Æ (¹w³]class => sqrt(p) ; regression => p/3)

# ¨M©w­n¥Í¦¨´X´Ê¾ð
plot(rf)  #Æ[¹î¦b¤£¦Ptreeªº­Ó¼Æ¤UªºMSE
# ¨úntree=150

# ¨M©w¨C¦¸¤Á³Î¦Ò¼{ªºÅÜ¼Æ­Ó¼Æ (mtry)
tuneRF(x=train[,-9],y=train[,9],stepFactor=0.5) 
# ¨úmtry = 4

set.seed(75)
rf_model = randomForest(lpsa~. , data = train ,
                        ntree = 150 , 
                        mtry = 4 , 
                        importance=T)  
rf_model

pred4 = predict(rf_model,newdata = test)
mean((pred4 - test$lpsa)^2) # MSE

# MSE = 0.6301

# ÅÜ¼Æ­«­n©Ê
importance(rf_model)
varImpPlot(rf_model,main="variable importance plot")
# ­È¶V¤jªº¥Nªí¸ÓÅÜ¼Æ¹ï¹w´ú¶V­«­n
# The former is based upon the mean decrease of accuracy in predictions on the out of bag samples
# when a given variable is excluded from the model
# The latter is a measure of the total decrease in node impurity that results from splits over that
# variable, averaged over all trees
# https://www.quora.com/How-do-we-calculate-variable-importance-for-a-regression-tree-in-random-forests



