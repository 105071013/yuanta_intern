###Linear Model and plot and table
###2019/09/23

#data
library(MASS)
Boston
head(Boston)
View(Boston) #to see the data
help(Boston)#F1

#feature
head(Boston$crim,10) # to see the first 10 samples of boston crime rate
###########plot##############
#scatter plot 
plot(Boston$crim,Boston$medv)
# pch:指定資料點的圖案，cex:圖形中資料點的大小,col:資料點的顏色
plot(Boston$crim,Boston$medv,pch = 12, cex = 0.5, col = 3,
     xlab = "per capita crime rate by town(crim)", ylab = "price(medv)", main = "Scatter Plot") 
#more about plot
# “p”: Points.
# “l”: Lines.
# “b”: Both.
# “c”: The lines part alone of “b”
# “o”: Both “overplotted”
# “h”: Histogram like (or high-density) vertical lines.
# “n”: No plotting.
plot(1:5, c(2,3,1,5,4)) #預設:points
plot(1:5, c(2,3,1,5,4), type = "p")
plot(1:5, c(2,3,1,5,4), type = "l")
plot(1:5, c(2,3,1,5,4), type = "b")
# lwd:指定線條寬度，為數值型，表示相對默認大小的倍數
plot(1:5, c(2,3,1,5,4), type = "b", lwd = 5)
# lines() : 在已有圖上加線，lty = 2 :虛線
lines(1:5, c(3,4,5,1,2), lty = 2)
# 在已有圖上加上點
points(3,3,col=4,pch=15,cex=4)
# abline -> 在已有圖上畫線
# （1）a 要繪製的直線截距
# （2）b 直線的斜率
# （3）h 繪製水平線時的縱軸值
# （4）v 繪製垂直線時的橫軸值
abline(a = 1, b = 2, col = "red")
abline(h = 3, col = 4, lty = 3)
abline(v = 3, col = 3, lty = 4)
###########Linear model###########

#one covariate
lm(medv~crim,data=Boston)
#y=Bx <-> medv = B0 + B1 * crim

#two covariates 
lm(medv~crim+zn,data=Boston)
# medv = B0 + B1 * crim + B2 * zn

#covariates and interaction
lm(medv~crim*zn,data=Boston)
#medv~crim*zn 等於 medv=B0+B1*crim+b2*zn+B3*(crim*zn)
#medv~crim:zn 等於 medv=B0 + B1*(crim*zn)

#only interaction
lm(medv~I(crim*zn),data=Boston)
lm(medv~crim:zn,data=Boston)
# medv = B0 + B1*(crim*zn)

#All
lm(medv~.,data=Boston)

#remove one
lm(medv~.-crim,data=Boston)

#remove Intercept
lm(medv~crim+zn-1,data=Boston)
# medv = B1*crim+B2*zn

#square
lm(medv~I(crim^2),data=Boston)

#more detail
fit=lm(medv~crim,data=Boston)
summary(fit)

#fit.value
head(fit$fitted.values)
# 下方皆表示residual
head(fit$residuals)
head(Boston$medv-fit$fitted.values)
#risidual的平方和
sum(fit$residuals^2)

# add regression line
plot(Boston$crim, Boston$medv, pch = 12, cex = 0.5, col = 2,
     xlab = "per capita crime rate by town(crim)", ylab = "price(medv)", main = "Scatter Plot")
abline(fit, col = 4, lwd = 2)

#prediction
predict(fit, newdata = data.frame(crim=c(10,20,30,40,50)))
# give the confidence interval
predict(fit, newdata = data.frame(crim=c(10,20,30,40,50)), interval = "confidence")


############table#########
# 需要加上replace = TRUE 才能抽樣出大於個數3的樣本
V1=sample(c("a","b","c"),50,replace = TRUE)
V2=sample(c("d","e","f"),50,replace = TRUE)
V3=sample(c("g","h"),50,replace = TRUE)
# cbind : 水平concat
test_data=as.data.frame(cbind(V1,V2,V3))
colnames(test_data)=c("abc","def","gh")
#one covariate
#各欄位原素的個數
table(test_data$abc)
table(test_data$def)
table(test_data$gh)
#各個元素對應的次數
#two covariate
table(test_data$abc,test_data$def)
table(test_data$abc,test_data$gh)
#three covariate
table(test_data$abc,test_data$def,test_data$gh)
