############ packages #############
#install.packages("e1071")
library(e1071)
# install.packages("ISLR")
library(ISLR)
# install.packages("MASS")
library(MASS)
# install.packages("class")
library(class)
# install.packages("tidyverse")
library(tidyverse)

#---- Smarket data from ISLR -----#
data(Smarket) # Daily percentage returns for the stock between 2001 and 2005.
head(Smarket,10)
dim(Smarket)

attach(Smarket)
train <- (Year < 2005)
Smarket_2005 <- Smarket[!train, ]
Direction_2005 <- Direction[!train]
length(Direction_2005)

######### Naive Bayes 0.591 ########
fit_nb <- naiveBayes(Direction~Lag1+Lag2, data=Smarket, subset=train)
fit_nb

# prior probabilities
prior <- Smarket %>%
  filter(Year < 2005) %>%
  select(Direction) %>%
  table() %>%
  prop.table()
prior

# group means and sds
group <- Smarket %>%
  filter(Year < 2005) %>%
  group_by(Direction) %>%
  summarize(Lag1_mean=mean(Lag1),
            Lag1_sd=sd(Lag1),
            Lag2_mean=mean(Lag2),
            Lag2_sd=sd(Lag2)) %>%
  column_to_rownames(var = 'Direction')
group

# predict Direction, result : class
pred_nb <- predict(fit_nb, Smarket_2005)
table(pred_nb)
table(pred_nb, Direction_2005)

# accuracy = 0.591
mean(pred_nb == Direction_2005)

# predict Direction, result : posterior probabilities
pred_nb_prob <- predict(fit_nb, Smarket_2005, type="raw")
head(pred_nb_prob)

# check the predicted probability
posterior <- function(x,prior,group){
  x <- as.numeric(x)
  # prior[1] : P(Y=Down) ; prior[2] : P(Y=Up)
  # x[1] : Lag1 ; x[2] : Lag2
  # group[1,1] : mean(Lag1|Y=Down)
  # group[1,2] : sd(Lag1|Y=Down)
  p.do <- as.numeric(prior[1] * dnorm(x[1],group[1,1],group[1,2]) * dnorm(x[2],group[1,3],group[1,4]))
  p.up <- as.numeric(prior[2] * dnorm(x[1],group[2,1],group[2,2]) * dnorm(x[2],group[2,3],group[2,4]))
  sav  <- c(p.do, p.up)/(p.do + p.up)
  return(sav)
}
head(t(apply(Smarket_2005[,c(2,3)], 1, posterior, prior, group)))

############# LDA 0.560 ############
fit_lda <- lda(Direction~Lag1+Lag2, data=Smarket, subset=train)
fit_lda

# predict Direction
pred_lda <- predict(fit_lda, Smarket_2005)
table(pred_lda$class)
table(pred_lda$class, Direction_2005)
head(pred_lda$posterior)

# accuracy = 0.560
mean(pred_lda$class == Direction_2005)

############# QDA 0.599 #############
fit_qda <- qda(Direction~Lag1+Lag2, data=Smarket, subset=train)
fit_qda

# predict Direction
pred_qda <- predict(fit_qda, Smarket_2005)
table(pred_qda$class)
table(pred_qda$class, Direction_2005)
head(pred_qda$posterior)

# accuracy = 0.599
mean(pred_qda$class == Direction_2005)

############# KNN 0.536 #############
train_X <- cbind(Lag1, Lag2)[train, ]
test_X <- cbind(Lag1, Lag2)[!train, ]
train_Y <- Direction[train]

set.seed(1)
# predict when k=1
pred_knn_1 <- knn(train_X, test_X, train_Y, k=1)
table(pred_knn_1, Direction_2005)

# accuracy = 0.5
mean(pred_knn_1 == Direction_2005)

# change k
pred_knn_3 <- knn(train_X, test_X, train_Y, k=3)
table(pred_knn_3, Direction_2005)

# accuracy = 0.536
mean(pred_knn_3 == Direction_2005)

###### Logistic Regression 0.560 ######
fit_glm <- glm(Direction ~ Lag1+Lag2, data=Smarket, family=binomial, subset=train)
summary(fit_glm)

# predict Direction (link)
pred_glm <- predict(fit_glm, Smarket_2005)
head(pred_glm)
head(exp(pred_glm) / (1 + exp(pred_glm)))

# predict Direction (response)
pred_glm <- predict(fit_glm, Smarket_2005, type="response")
head(pred_glm)
prob_glm <- ifelse(pred_glm > 0.5, "Up", "Down")
table(prob_glm, Direction_2005)

# accuracy = 0.559
mean(prob_glm == Direction_2005)

#####
detach(Smarket)
