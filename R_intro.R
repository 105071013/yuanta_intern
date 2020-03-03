##### R introduction #####

##### 安裝套件 #####
install.packages("glmnet")
library(glmnet)

##### Help #####
?glmnet
??glmnet

##### 匯入資料 #####
# 1.使用內建之dataset
library(datasets)
data(iris)    
iris[,1]
iris[,4:5]
iris$Species
names(iris)   #看資料內有哪些變數
dim(iris)     #看資料維度
head(iris,10) #看前10筆資料
tail(iris,10) #看後10筆資料
summary(iris) #得到敘述統計量

# 2.從外部匯入資料
data = read.table("final.txt",header=T)
data_csv = read.csv("nbadata.csv")

##### 資料型態 #####
# 1.資料屬性
d1 = 1L     #整數
d2 = 0.1    #實數
d3 = "0.1"  #字串
d4 = TRUE   #布林值
class(d1)
class(d2)
class(d3)
class(d4)

# 2.資料結構
# (1)vector
v1 = c(1,3,5,7)
v2 = c("1","2")
v3 = c(TRUE,FALSE)
v4 = c(1,"2",TRUE)
class(v1)
class(v2)
class(v3)
class(v4)   #會自動轉成相同屬性

length(v1)  #元素個數
v1[2]
v1[3]

# (2)factor 用來表示類別變數
summary(v1)
f1 = as.factor(v1)
f1
class(f1)
summary(f1)

# (3)list
l1 = list(variable1 = c(2,3,4,5) , variable2 = factor(c("M","F","M")))
l1
l1[1]   #回傳包含變數名稱
l1[[1]] #使用雙括號則回傳不含變數名稱
l1$variable1
class(l1)
class(l1[[1]])
l1[[1]][2] #取出第一個變數內第2筆數值

l2 = list(1,"M",TRUE) #比向量更有彈性,list可以存放「任何型態」的變數
l2
class(l2)
class(l2[[3]])

# (4)matrix
m1 = matrix(1:4,nrow=2,ncol=2)
m1
class(m1)
m1[1,2]
m1[1,]
m1[,2]
t(m1)    #轉置
dim(m1)  #維度
det(m1)  #計算行列式值
eigen(m1)#計算eigenvalue
solve(m1)#計算反矩陣
#跟向量一樣只能儲存相同屬性
m2 = matrix(5:8,nrow=2,ncol=2)
m1%*%m2  #矩陣相乘

# (5)data.frame
iris
class(iris)
rownames(iris)
colnames(iris)
names(iris)
dim(iris)
summary(iris)
class(iris$Species)

##### 基本運算 #####
# 1.數學基本運算
a = 5
b = 2
a+b
a-b
a*b
a/b
a%%b   #餘數
a^b    #次方

c = 3.75
round(c)   #四捨五入
floor(c)   #無條件進位
ceiling(c) #無條件捨去

# 2.邏輯運算
a > b
a < b
a >= b
a <= b
a == b   #為了不與變數設定混淆，判斷兩變數是否相等，要用雙等號
a | b >= 5  # or
a & b >= 5  # and

# 3.常用統計函數
x = c(4,8,12,13,20,21)
mean(x)    #平均值
var(x)     #變異數
sd(x)      #標準差
median(x)  #中位數
max(x)     #最大值
min(x)     #最小值
sum(x)     #總和
quantile(x)#分位數
summary(x) 


##### 簡單繪圖 #####
# 1.長條圖
barplot(table(iris$Species),main="長條圖")

# 2.直方圖
hist(iris$Sepal.Length,main="直方圖",xlab="Sepal.Length")

# 3.盒型圖
boxplot(iris$Sepal.Length,main="Boxplot")

# 4.散布圖
plot(iris[,1],iris[,2],main="scatter plot")
plot(iris[,1],iris[,2],main="scatter plot",xlab=colnames(iris)[1],ylab=colnames(iris)[2],col=4,pch=15)
pairs(iris)

# 5.設定一頁顯示多張圖
par(mfrow=c(2,2))
barplot(table(iris$Species),main="長條圖")
hist(iris$Sepal.Length,main="直方圖",xlab="Sepal.Length")
boxplot(iris$Sepal.Length,main="Boxplot")
plot(iris[,1],iris[,2],main="scatter plot",xlab=colnames(iris)[1],ylab=colnames(iris)[2],col=4,pch=15)