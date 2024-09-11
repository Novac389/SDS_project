################################################################################
#### Precision and accuracy analysis with different classes probability ########
#### authors: Andriani P. Manai E. Velardita M. ################################
################################################################################

library(glmnet)
library(ROSE)


setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("02_performance_measures.R")


data = read.csv("df_8pred_normal.csv")

#dynamic selection of predictors
num_var=8 #CHANGE {3, 6, 8}
predictor_vars = c("age", "sysBP", "diaBP", "glucose", "totChol", "BMI", "cigsPerDay", "heartRate")
selected_vars = predictor_vars[1:num_var]

formula_str = paste("TenYearCHD ~", paste(selected_vars, collapse = " + "))
data = cbind(data["TenYearCHD"], data[,selected_vars])

# Sample indexes to select data for train set
train_index = sample(x = nrow(data), 
                     size = round(0.8 * length(data$TenYearCHD)), 
                     replace = FALSE)

# Split data into train and test set
train_set = data[train_index,]
test_set = data[-train_index,]

x_train = as.matrix(train_set[,2:ncol(train_set)])
y_train = as.factor(train_set[,1]) 

x_test = as.matrix(test_set[,2:ncol(train_set)])
y_test = as.factor(test_set[,1]) 

#train a linear model with ridge penalization
cv_out = glmnet::cv.glmnet(x = x_train,
                           y= y_train, 
                           nfolds=10,
                           family = 'binomial')

rid_model = glmnet(x = x_train, 
                   y = y_train, alpha = 0, 
                   family = 'binomial',
                   lambda = cv_out$lambda.min)

#predict values of test set
pred = predict(rid_model, x_test, type = 'class') # get class prediction

#calculate model preformance
sensitivity_model = sensitivity(pred,y_test)
specificity_model = specificity(pred,y_test)
accuracy_model = accuracy(pred,y_test)
precision_model = precision(pred,y_test)

#calculate number of negative and positive class records 
positives = sum(test_set$TenYearCHD==1)
negatives = sum(test_set$TenYearCHD==0)
#crate rebalance factor from 0.001 to 1
rebalance_factor= seq(0.001, 1, 0.001)

#calculate the new sample size with positive rebalance
new_positives=as.integer(positives/rebalance_factor)
sample_size=negatives+as.integer(new_positives)
sample_size = unique(sample_size)

probabilities = list()
precisions = list()

#test the model in different test set sample with increasing number of positive
#instances each time

for(i in 1:length(sample_size)){
  #create test set with oversampled minority class
  data_balanced = ovun.sample(TenYearCHD~., 
                              data=test_set, 
                              method="over", 
                              N=sample_size[i])$data
  
  #P(C)
  probability=sum(data_balanced$TenYearCHD==1)/nrow(data_balanced)
  
  #split test set in x and y
  x_test = as.matrix(data_balanced[,2:length(data_balanced)])
  y_test = as.factor(data_balanced[,1])
  
  #predict class
  pred = predict(rid_model, x_test, type = 'class')
  
  #calculate the precision of the model
  precision_model = precision(pred,y_test)
  
  #save P(C) and precision of the model
  probabilities=append(probabilities,probability)
  precisions=append(precisions,precision_model)
  
  #print(sprintf("%f %f",probability,precision_model))
}

plot(probabilities,precisions,xlab='P(C)', ylab='P(C|+)',pch = 20,cex=0.9)

#Theoretical precision whit different probability of begin positive
f_precision = function(x) (sensitivity_model*x)/(sensitivity_model*x + (1-specificity_model)*(1-x))
curve(f_precision(x), xlim=c(0, 1), ylim=c(0,1),  font.lab=2, lwd=4, col="red",add=TRUE)

#calculate accuracies over probabaility
probabilities = list()
accuracies = list()

for(i in 1:length(sample_size)){
  #create test set with oversampled minority class
  data_balanced = ovun.sample(TenYearCHD~., 
                              data=test_set, 
                              method="over", 
                              N=sample_size[i])$data
  
  #P(C)
  probability=sum(data_balanced$TenYearCHD==1)/nrow(data_balanced)
  
  x_test = as.matrix(data_balanced[,2:length(data_balanced)])
  y_test = as.factor(data_balanced[,1])
  
  pred = predict(rid_model, x_test, type = 'class')
  precision_model = accuracy(pred,y_test)
  
  probabilities=append(probabilities,probability)
  accuracies=append(accuracies,precision_model)
  #print(sprintf("%f %f",probability,precision_model))
}



plot(probabilities,accuracies,xlab='P(C)', ylab='P(Ŷ=Y)',pch = 20,cex=0.9)

#accuracy over P(C)
f_accuracy = function(x) (sensitivity_model * x) + (specificity_model * (1-x))
curve(f_accuracy(x), xlim=c(0, 1), ylim=c(0,1), xlab='P(C)', ylab='accuracy', font.lab=2, lwd=2, col=2,add=TRUE)


