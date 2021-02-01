#Classification model to see which custpomers are more likely to subscribe the term deposit

bank_data = read.csv(file = "/Users/jyoticaa/Desktop/blank1_030.csv")

#exploring the data
dim(bank_data)
names(bank_data)

head(bank_data)
tail(bank_data)

#checking for missing values
sum(is.na(bank_data))
#it says we have 0 missing values

#Visualizing the data with respect to the variables avaiable with us.
library(ggplot2)
#checking how age is distributed in our data
ggplot(data = bank_data, aes(x=bank_data$age))+geom_density()+
  labs(x="age", y= " Distribution")
# here we can see that majority of people present here are aged between 25 to 60 years

#for education
ggplot(data = bank_data, aes(x=bank_data$education))+
  geom_bar()+
  labs(x="Education", y="no of people")

#now here we can see that major education is secondary and least being unknown
#here unknown can be treated as our missing or null values

summary(bank_data$education)
#from the summary we can see that 187 values are unknown in the education variable


# we have 4521 rows from which 187 are unknown values for education 
#this maybe or may not be affecting to our final results  
#lets see


#for marital status
ggplot(data = bank_data, aes(x=bank_data$marital))+
  geom_bar()+
  labs(x="Marital Status", y="Count")
#no unknown values are present here


#for job
ggplot(data = bank_data, aes(x=bank_data$job))+
  geom_bar()+
  labs(x="Job Category", y="Count") + coord_flip()
#here too we have unknown values  
summary(bank_data$job)
#38 are unknown
#from 4521 38 are unknown  
#i think this is very minimum and would not affect our final result

#for credit 
ggplot(data = bank_data, aes(x= bank_data$default))+
  geom_bar()+
  labs(x="Credit Default", y="no. of people")

#majorly people dont have credit as default that means the customers do not have any due payments to make
#from which we can say that we can target these people who can pay their dues on time  

#for balance
ggplot(data = bank_data, aes(x=bank_data$balance))+geom_histogram(bins=30)+
  labs(x="Bank Balance", y= " Distribution")
#mojority of the population is seen to be of economical class.

#for housing
ggplot(data = bank_data, aes(x= bank_data$housing.loan))+
  geom_bar()+
  labs(x="housing_loan", y="no. of people")
summary(bank_data$housing.loan)
#more than 50 percent of the population has taken a housing loan




#for personal loan
ggplot(data = bank_data, aes(x= bank_data$personal.loan))+
  geom_bar()+
  labs(x="personal loan", y="no. of people")
summary(bank_data$personal.loan)
#less than 20 percent of the population has taken personal loan 

#from the above two we can say that here people prefer taking the housing loan more than that of personal loans




#previous campaign
ggplot(data = bank_data, aes(x=bank_data$previous.campaign))+
  geom_histogram(bins=30)+
  labs(x="Calls Made", y="no. of people")
# contacts with zero are the highest so we can say that most of the customers were first timers 
#in this campaign



ggplot(data = bank_data, aes(x=bank_data$current.campaign))+
  geom_bar()+
  labs(x="Number of Campaings")+
  scale_x_continuous(breaks=seq(0,30,2))
#here majority is seen in 1 and 2 which means most of the customers are/were either engaged
#with the company once or twice 



#Since we have explored all the variables and we have good knowlege of them we can start with the modelling
#MODELLING


#first, splitting the data into training and testing(80% for training and remaining 20% for testing)
library(caTools)
set.seed(190) 
sample = sample.split(bank_data$subscribes_030, SplitRatio = .75)
train = subset(bank_data, sample == TRUE)
test  = subset(bank_data, sample == FALSE)

#verifying the splitting of our data
prop.table(table(train$subscribes_030))
prop.table(table(test$subscribes_030))
#from the result we can say that they have been fairly splitted



#Now, we can start building our model on top of these splitted datasets


bank_data$job_unk <- ifelse(bank_data$job == "unknown", 1, 0)
bank_data$edu_unk <- ifelse(bank_data$education == "unknown", 1, 0)

#the above code will create extra columns for our unknown values which are present in the data

model<-svm(subscribes_030 ~ .,data = train)
summary(model)
pred1<-predict(model,test)
confusionMatrix(as.factor(test$subscribes_030),as.factor(pred1))

bank_data$job <- as.numeric(as.factor(bank_data$job))
bank_data$marital <- as.numeric(as.factor(bank_data$marital))
bank_data$education <- as.numeric(as.factor(bank_data$education))
bank_data$default<- ifelse(bank_data$default == "yes", 1, 0)
bank_data$housing.loan <- ifelse(bank_data$housing.loan== "yes", 1, 0)
bank_data$personal.loan<- ifelse(bank_data$personal.loan== "yes", 1, 0)
#the above will make our character data into data frame format so we can move furthur


#now our target variable is also in character format 
#so we will need to chnange it to factor format
bank_data$subscribes_030 <- as.factor(bank_data$subscribes_030)
bank_data$subscribes_030


dim(train)
dim(test)
#just checking the proportion of our train test set again

percentt <- nrow(test[test$subscribes_030 == "yes",]) / nrow(test)
percentt
#in our test set only 11% of the total people have subscribes to the term deposit 
#now lets see how our algorithm predicts

library(dplyr)
library(caret)
library(e1071)
model_naive <- naiveBayes(train[, !names(train) %in% c("subscribes_030")],
                    train$subscribes_030, na.action = na.pass)
# Type classifier to examine the function call, a-priori probability, and conditional
#probability
model_naive
#from the above output we observed the A-priori probabilities #A-priori is desired_outcomes/totalnumber of outcomes
#they are prior probabilities in bayes theorm,  That is, how frequently each level of class occurs in the training dataset.
#and we already know that conditional prob. is nothing but the occurence of one event knowing the occurence
#of second event
#For example in the above output if we take marital in consideration it will mean that
#prob of person not subscribing to the term deposit knowing that he/she is divorced P(divorced/not subscribing term deposit) is 0.113


#preparing data for test prediction
x<- test[, !names(test) %in% c("subscirbes_030")]
y <- test$subscribes_030

#predicting the bayes table
bayes.table <- table(predict(model_naive, x), y)
bayes.table

#here from the bayes contingency table we can see that,
#970 were predicted negative and they were negative i.e those people did not subscribe
##30 were false positives. prediction was that they subscribes but actually they did not
#114 people were predicted that they did not subscribe but actually they did subscribe the term deposit
#this is what is to worry about
#16 were were predicted that they subscribes and they actually did subscribe to the term

#checking the misclassification rate
1-sum(bayes.table[row(bayes.table)==col(bayes.table)])/sum(bayes.table)
#about 12% of the test data was misclassified during the prediction

confusionMatrix(bayes.table)
#The confusion matrix above shows the 87% of acuracy and 95% confidence interval of the
#perdicted acuracy.
#The cross table from the confusion matrix shows that model predicted more 
#closely for the customer who did not suscribe term deposit but for those custome who had subscribed 
#term deposit has been predicted badly.



