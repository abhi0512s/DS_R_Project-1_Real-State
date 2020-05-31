setwd("D:/AI & Data Science/R/Project 1 Real State")

housing_test=read.csv("housing_test.csv",stringsAsFactors = F)
housing_train=read.csv("housing_train.csv", stringsAsFactors = F)

library(dplyr)
library(car)

housing_test$Price=NA

housing_test$Data="Test"
housing_train$Data="Train"

housing=rbind(housing_train,housing_test)

glimpse(housing)

n=table(housing$Suburb)
dim(n) #can't consider holding too many values
n=round(tapply(housing$Price,housing$Suburb,mean,na.rm=T))
sort(n) # can be consider if taking mean and grouping
#same goes with SellerG,Postcode
n=round(tapply(housing$Price,housing$Postcode,mean,na.rm=T))

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

n=table(housing$SellerG)
dim(n)
n
lapply(housing,function(x) sum(is.na(x)))

for(col in names(housing)){
  if(sum(is.na(housing[,col]))>0 & (col %in% c("Bedroom2","Bathroom",
                              "Car","Landsize","BuildingArea","YearBuilt"))){
    housing[is.na(housing[,col]),col]=round(mean(housing[,col],na.rm=T))
  }
}

housing=CreateDummies(housing ,"Type",1000)
housing=CreateDummies(housing ,"Method",50)

glimpse(housing)

housing_train=housing %>% filter(Data=='Train') %>% select(-Data)
housing_test=housing %>% filter(Data=='Test') %>% select(-Data,-Price)

glimpse(housing_train)

set.seed(2)
s=sample(1:nrow(housing_train),.75*nrow(housing_train))
housing_train1=housing_train[s,]
housing_train2=housing_train[-s,]

#Linear Rgression Model
fit=lm(Price~.-Suburb-Address-SellerG-CouncilArea,data=housing_train1)
sort(vif(fit),decreasing = T)
fit=step(fit)
summary(fit)
formula(fit)
fit=lm(Price ~ Rooms + Distance + Postcode + Bedroom2 + Bathroom + Car + 
         Landsize + BuildingArea + YearBuilt + Type_u + Type_h + Method_PI + 
         Method_S,data=housing_train1)

val.pred=predict(fit,newdata=housing_train2)
plot(val.pred,housing_train2$Price)
errors=housing_train2$Price-val.pred
RMSE_lr=errors**2 %>% mean() %>% sqrt()
RMSE_lr
Score_lr =212467/RMSE_lr
Score_lr

#DTree Model
library(rpart)
library(rpart.plot)
library(tidyr)
library(randomForest)
require(rpart)
library(tree)

dtModel = tree(Price~.-Suburb-Address-SellerG-CouncilArea,data=housing_train1)
plot(dtModel)
dtModel

val.score=predict(dtModel,newdata = housing_train2)
plot(val.score,housing_train2$Price)

errors=housing_train2$Price-val.score
RMSE_dt=errors**2 %>% mean() %>% sqrt()
RMSE_dt
Score_dt =212467/RMSE_dt
Score_dt

#Random Forest Model
randomForestModel = randomForest(Price~.-Suburb-Address-SellerG-CouncilArea,data=housing_train1)
d=importance(randomForestModel)
d
names(d)
d=as.data.frame(d)
d$IncNodePurity=rownames(d)
d %>% arrange(desc(IncNodePurity))

val.score=predict(randomForestModel,newdata = housing_train2)
plot(val.score,housing_train2$Price)

errors=housing_train2$Price-val.score
RMSE_rf=errors**2 %>% mean() %>% sqrt()
RMSE_rf
Score_rf =212467/RMSE_rf
Score_rf
#greater than 0.51


#GBM Model
library(gbm)
library(cvTools)

param=list(interaction.depth=c(1:7),
           n.trees=c(50,100,200,500,700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=30
my_params=subset_paras(param,num_trials)

myerror=9999999

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))

    params=my_params[i,]
  
  k=cvTuning(gbm,Price~.-Suburb-Address-SellerG-CouncilArea,
             data =housing_train1,
             tuning =params,
             args = list(distribution="gaussian"),
             folds = cvFolds(nrow(housing_train1), K=10, type = "random"),
             seed =2,
             predictArgs = list(n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this<myerror){
    print(params)

    myerror=score.this
    print(myerror)

    best_params=params
  }
  
  print('DONE')

}

Score_rf =212467/myerror
Score_rf

myerror

best_params

best_params=data.frame(interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.1,
                       n.minobsnode=2)

bs.gbm.final=gbm(Price~.-Suburb-Address-SellerG-CouncilArea,data=housing_train1,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "gaussian")

bs.gbm.final

test.pred=predict(bs.gbm.final,newdata=housing_train2,
                  n.trees = best_params$n.trees)

plot(test.pred,housing_train2$Price)
errors=housing_train2$Price-test.pred
RMSE_gbm=errors**2 %>% mean() %>% sqrt()
RMSE_gbm
Score_gbm =212467/RMSE_gbm
Score_gbm

##########Final Submission#############
for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  params=my_params[i,]
  k=cvTuning(gbm,Price~.-Suburb-Address-SellerG-CouncilArea,
             data =housing_train,
             tuning =params,
             args = list(distribution="gaussian"),
             folds = cvFolds(nrow(housing_train), K=10, type = "random"),
             seed =2,
             predictArgs = list(n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this<myerror){
    print(params)
    myerror=score.this
    print(myerror)
    best_params=params
  }
  
  print('DONE')
}

Score_rf =212467/myerror
Score_rf

myerror
best_params

best_params=data.frame(interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.1,
                       n.minobsnode=2)

bs.gbm.final=gbm(Price~.-Suburb-Address-SellerG-CouncilArea,data=housing_train,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "gaussian")
bs.gbm.final

test.pred=predict(bs.gbm.final,newdata=housing_test,
                  n.trees = best_params$n.trees)

write.table(test.pred,file ="Abhilash_Singh_P1_part2.csv",
          row.names = F,col.names="Price")


####################Quiz##############################
#1
var(housing$Price, na.rm = T)

#2
lapply(housing_train,function(x) sum(is.na(x)))
lapply(housing,function(x) sum(is.na(x)))

#3
s=housing %>% 
  select(Type,Price) %>% 
  group_by(Type) %>% 
  summarise(avg_price=mean(Price,na.rm = T))
View(s)
t=s$avg_price[1]-s$avg_price[2]
t

#4
table(housing$Postcode)
table(housing_train$Postcode)

r=housing %>% 
  group_by(Postcode) %>% 
  summarise(count=n())
dim(r)


#5
glimpse(housing)

#6
plot(housing_train$Distance,housing_train$Price)
library(ggplot2)

p=ggplot(housing_train,aes(x=Distance,y=Price))
p
p+geom_line()
p+geom_point()
p+geom_point()+geom_line()+geom_smooth()

ggplot(housing_train, aes(x=Distance)) + 
  geom_histogram(binwidth=.25, colour="black", fill="white")

ggplot(housing_train,aes(x=Distance))+geom_density(color="red")+
  geom_histogram(aes(y=..density..,alpha=0.5))+
  stat_function(fun=dnorm,aes(x=Distance),color="green")

ggplot(housing_train, aes(x=Distance))+
  geom_histogram(aes(y=..density..),binwidth=.25,colour="black",fill="white")+
  stat_function(fun=dnorm,lwd=2,col='red',
                args=list(mean=mean(housing_train$Distance),
                          sd=sd(housing_train$Distance)))

shapiro.test(housing_train$Distance[1:5000])
shapiro.test(sample(1:nrow(housing_train),0.65*nrow(housing_train)))

#7
table(housing$SellerG)

s=housing_train %>% 
  select(SellerG,Price) %>% 
  group_by(SellerG) %>%
  summarise(val=sum(Price)) %>% 
  filter(val==max(val))
View(s)


#8
table(housing$CouncilArea)

s=housing_train %>% 
  select(CouncilArea,Price) %>% 
  group_by(CouncilArea) %>%
  summarise(val=mean(Price)) %>% 
  filter(val==max(val))
View(s)

#9
table(housing$CouncilArea)

s=housing_train %>% 
  select(CouncilArea,Price) %>% 
  group_by(CouncilArea) %>%
  summarise(val=var(Price)) %>% 
  filter(val==max(val))
View(s)


#10
glimpse(housing)
################################################################