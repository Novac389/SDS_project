###############################################################################
############ Class imbalance: case study ######################################
#### original code from https://github.com/benvancalster/classimb_calibration #
#### modified by Andriani P. Manai E Velardita M ##############################
###############################################################################
#data cleaning
library(tidyverse) 
library(DescTools)

# models
library(glmnet)
library(glmnetUtils)

#imbalance
library(caret) 
library(smotefamily) 

# Performance
library(xtable) 
library(CalibrationCurves)
library(rmda)

# Plot in grid
library(gridExtra)

# set current working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# load necessary code for performance measures
source("02_performance_measures.R")


#####################
#### Data import ####
#####################

# Import normalized dataset with only the 8 selected predictors
data = read.csv("df_8pred_normal.csv")
data

#dynamic selection of predictors
# can choose how many predictor to use
num_var=3 
predictor_vars = c("age", "sysBP", "diaBP", "glucose", "totChol", "BMI", "cigsPerDay", "heartRate")

#selected vars to employ to train models
selected_vars = predictor_vars[1:num_var]

# join target variable with spline of predictors with nk knots. Use nk = 4 if num_var=8
formula_str = paste("TenYearCHD ~", paste("rcs(", selected_vars, ", nk = 2)", collapse = " + "))

# merge target with predictors
data = cbind(data["TenYearCHD"], data[,selected_vars])
data

#########################
#### Check imbalance ####
#########################

# Check and plot imbalance of original data
data %>% ggplot()+geom_bar(mapping = aes(TenYearCHD))

n_minor = data %>% 
  filter(TenYearCHD == 1) %>% 
  nrow()

n_major = data %>% 
  filter(TenYearCHD == 0) %>% 
  nrow()

n_minor/n_major # 0.1812247

#########################################################
####   DATA SPLITTING into train and test 0.8:0.2    ####
#########################################################

#seed for reproducibility
set.seed(1333)

# Sample indexes to select data for train set
train_index = sample(x = nrow(data), 
                      size = round(0.8 * length(data$TenYearCHD)), 
                      replace = FALSE)

# Split data into train and test set
train_set = data[train_index,]
test_set = data[-train_index,]

##################################
#### CREATE BALANCED DATASETS ####
##################################
x = train_set[,2:ncol(train_set)] 
y = as.factor(train_set[,1]) # 

#downSample and upSample from caret library

## Random undersampling ##
##########################

train_down = downSample(x = x, y = y, yname = 'TenYearCHD') 
train_down %>% 
  ggplot()+
  geom_bar(mapping = aes(TenYearCHD))

# Class imbalance resulted 0: 454, 1: 454

## Random oversampling ##
#########################

train_up = upSample(x = x, y = y, yname = 'TenYearCHD')  #5172 obj
train_up %>% 
  ggplot()+
  geom_bar(mapping = aes(TenYearCHD))

#Class imbalance resulted 0: 2586, 1: 2586

## SMOTE ##
###########
#dup_size: number of synthetic sample to generate for each minority point
#default dup_size=0 -> nearly equal to the number of negative instances
train_smote = SMOTE(X= x, target = y, dup_size = 0)
train_smote = train_smote$data

train_smote = train_smote %>% 
  rename(TenYearCHD = class) 

train_smote$TenYearCHD  = as.factor(train_smote$TenYearCHD)

train_smote %>% 
  ggplot()+
  geom_bar(mapping = aes(TenYearCHD))

class_counts = table(train_smote$TenYearCHD)
print(class_counts) 

#Class imbalance resulted 0: 0: 2586, 1: 2270

## ADASYN added by Velardita, Manai, Andriani##

# Perform ADASYN oversampling using ADAS function
train_adasyn = ADAS(X= x , target = y, K = 3)
train_adasyn = train_adasyn$data

train_adasyn = train_adasyn %>% 
  rename(TenYearCHD = class)

train_adasyn$TenYearCHD  = as.factor(train_adasyn$TenYearCHD)

train_adasyn %>% 
  ggplot()+
  geom_bar(mapping = aes(TenYearCHD))

class_counts = table(train_adasyn$TenYearCHD)
print(class_counts) 

# Class imbalance resulted 0: 0: 2586, 1: 2660

######################################
#### Standard logistic regression ####
######################################

## uncorrected dataset ##
#########################

# rcs(): restricted cubic spline
# Utilizing rcs allows modeling nonlinear relationships between variables.
# It employs restricted cubic splines to capture data complexities, enhancing 
# regression model accuracy. The choice of the number of nodes reflects 
# the desired complexity of the relationship. (too much nodes lead to overfitting!)

# define logistic regression model
# formula takes dynamically selected predictors
# family binomial due to binary outcome
log_model = glm(formula = formula_str, family = 'binomial', data = train_set)

log_probs = predict(log_model, test_set, type = 'response') # get probabilities: log_probs contains probabilities in [0,1]

# 0.5 threshold
log_pred = rep(0, length(log_probs)) # empty vector for class prediction
log_pred[log_probs > .5] = 1  # predict class with 0.5 threshold
# imbalance ratio threshold
log_pred2 = rep(0, length(log_probs)) # empty vector for class prediction 
log_pred2[log_probs > .1812] = 1  # predict class with threshold set to imbalance ratio 

###################
## Undersampling ##
###################

log_model_down = glm(formula = formula_str, family = 'binomial', data = train_down)

log_probs_down = predict(log_model_down, test_set, type = 'response') # get probabilities

log_pred_down = rep(0, length(log_probs_down)) # empty vector for class prediction
log_pred_down[log_probs_down > .5] = 1 # predict class with 0.5 threshold

log_pred2_down = rep(0, length(log_probs_down)) # empty vector for class prediction
log_pred2_down[log_probs_down > .1812] = 1 # predict class with threshold set to imbalance ratio.

##################
## Oversampling ##
##################

log_model_up =glm(formula = formula_str, family = 'binomial', data = train_up)

log_probs_up = predict(log_model_up, test_set, type = 'response') # get probabilities

log_pred_up = rep(0, length(log_probs_up)) # empty vector for class prediction
log_pred_up[log_probs_up > .5] = 1 # predict class with 0.5 threshold

log_pred2_up = rep(0, length(log_probs_up)) # empty vector for class prediction
log_pred2_up[log_probs_up > .1812] = 1 # predict class with threshold set to imbalance ratio.

###########
## SMOTE ##
###########

log_model_smote =glm(formula = formula_str, family = 'binomial', data = train_smote)

log_probs_smote = predict(log_model_smote, test_set, type = 'response') # get probabilities.

log_pred_smote = rep(0, length(log_probs_smote)) # empty vector for class prediction
log_pred_smote[log_probs_smote > .5] = 1 # predict class with 0.5 threshold

log_pred2_smote = rep(0, length(log_probs_smote)) # empty vector for class prediction
log_pred2_smote[log_probs_smote > .1812] = 1 # predict class with threshold set to imbalance ratio.

            ############
            ## ADASYN ##
            ############
#####Velardita Manai Andriani######

log_model_adasyn =glm(formula = formula_str, family = 'binomial', data = train_adasyn)

log_probs_adasyn = predict(log_model_adasyn, test_set, type = 'response') # get probabilities

log_pred_adasyn = rep(0, length(log_probs_adasyn)) # empty vector for class prediction
log_pred_adasyn[log_probs_adasyn > .5] = 1 # predict class with 0.5 threshold

log_pred2_adasyn = rep(0, length(log_probs_adasyn)) # empty vector for class prediction
log_pred2_adasyn[log_probs_adasyn > .1812] = 1 # predict class with threshold set to imbalance ratio.

######################################################################################
######################## Ridge logistic regression ###################################
######################################################################################

# split also train into x, y
x_train = as.matrix(train_set[,2:ncol(train_set)])
y_train = as.factor(train_set[,1]) 

x_test = as.matrix(test_set[,2:ncol(train_set)])
y_test = as.factor(test_set[,1]) 

## uncorrected dataset ##
#########################

# Get hyperparameter
# get lambdas (ridge coefficients) by using 10-fold cross validation
cv_out = glmnet::cv.glmnet(x = x_train,
                           y= y_train, 
                           nfolds=10,
                           family = 'binomial')
# Fit model and use min lambda found
rid_model = glmnet(x = x_train, 
                   y = y_train, alpha = 0, 
                   family = 'binomial',
                   lambda = cv_out$lambda.min)

rid_probs = predict(rid_model, x_test, type = 'response') # get probabilities

rid_pred = rep(0, length(rid_probs)) # vector for class prediction
rid_pred[rid_probs > .5] = 1  # predict classes with 0.5 threshold
rid_pred2 = rep(0, length(rid_probs)) # vector for class prediction
rid_pred2[rid_probs > .1812] = 1  # predict classes with unbalanced ratio


## random undersampling ##
##########################
x_train_down = as.matrix(train_down[,1:ncol(train_down)-1])
y_train_down = as.factor(train_down[,ncol(train_down)]) 


# Tune hyperparameter
cv_out_down = glmnet::cv.glmnet(x = x_train_down,
                           y = y_train_down, 
                           nfolds=10,
                           family = 'binomial')
# Fit model use min lambda found
rid_model_down = glmnet(x = x_train_down, 
                        y = y_train_down, alpha = 0, 
                        family = 'binomial',
                        lambda = cv_out_down$lambda.min)

rid_probs_down = predict(rid_model_down, x_test, type = 'response') # get probabilities

rid_pred_down = rep(0, length(rid_probs_down)) # Create vector for class predictions
rid_pred_down[rid_probs_down > .5] = 1 # predict classes with threshold 0.5
rid_pred2_down = rep(0, length(rid_probs_down)) # Create vector for class predictions
rid_pred2_down[rid_probs_down > .1812] = 1 # predict classes with  with unbalanced ratio


## Random oversampling ##
#########################
x_train_up = as.matrix(train_up[,1:ncol(train_up)-1])
y_train_up = as.factor(train_up[,ncol(train_up)]) 

# Tune hyperparameter
cv_out_up = glmnet::cv.glmnet(x = x_train_up,
                              y= y_train_up, 
                              nfolds=10,
                              family = 'binomial')
# Fit model use min lambda found
rid_model_up = glmnet(x = x_train_up, 
                      y = y_train_up, alpha = 0, 
                      family = 'binomial',
                      lambda = cv_out_up$lambda.min)

rid_probs_up = predict(rid_model_up, x_test, type = 'response') # get probabilities

#calculate class from probabilities
rid_pred_up = rep(0, length(rid_probs_up)) # vector for class predictions
rid_pred_up[rid_probs_up > .5] = 1 # predict class with threshold 0.5
rid_pred2_up = rep(0, length(rid_probs_up)) # vector for class predictions
rid_pred2_up[rid_probs_up > .1812] = 1 # predict class with unbalanced ratio


## SMOTE ##
###########

x_train_smote = as.matrix(train_smote[,1:ncol(train_smote)-1])
y_train_smote = as.factor(train_smote[,ncol(train_smote)]) 

# Hyperparameter tuning
cv_out_smote = glmnet::cv.glmnet(x = x_train_smote,
                                 y= y_train_smote,
                                 nfolds=10,
                                 family = 'binomial')
# Fit model use min lambda found
rid_model_smote = glmnet(x = x_train_smote, 
                         y = y_train_smote, 
                         alpha = 0,
                         family = 'binomial',
                         lambda = cv_out_smote$lambda.min)

rid_probs_smote = predict(rid_model_smote, x_test, type = 'response')

rid_pred_smote = rep(0, length(rid_probs_smote)) # Create vector for class prediction
rid_pred_smote[rid_probs_smote > .5] = 1 # predict classes with threshold 0.5
rid_pred2_smote = rep(0, length(rid_probs_smote)) # Create vector for class prediction
rid_pred2_smote[rid_probs_smote > .1812] = 1 # predict classes with unbalanced ratio


## ADASYN ##
############
#### added by Velardita Manai Andriani #######

x_train_adasyn = as.matrix(train_adasyn[,1:ncol(train_adasyn)-1])
y_train_adasyn = as.factor(train_adasyn[,ncol(train_adasyn)]) 

# Hyperparameter tuning
cv_out_adasyn = glmnet::cv.glmnet(x = x_train_adasyn, 
                                  y = y_train_adasyn, 
                                  nfolds=10, 
                                  family = 'binomial')

# Fit model use min lambda found
rid_model_adasyn = glmnet(x = x_train_adasyn, 
                          y = y_train_adasyn, 
                          alpha = 0, 
                          family = 'binomial',
                          lambda = cv_out_adasyn$lambda.min)


rid_probs_adasyn = predict(rid_model_adasyn, x_test, type = 'response')

rid_pred_adasyn = rep(0, length(rid_probs_adasyn)) # Create vector for class prediction
rid_pred_adasyn[rid_probs_adasyn > .5] = 1 # predict classes with threshold 0.5
rid_pred2_adasyn = rep(0, length(rid_probs_adasyn)) # Create vector for class prediction
rid_pred2_adasyn[rid_probs_adasyn > .1812] = 1 # predict classes with  with unbalanced ratio



##############################
#### Test set performance ####
##############################

# In this part of the code, all performance measures are computed and placed in a 
# table. Also, calibration plots are generated to visualize the performance.

# Creating table with all estimated probabilities
probs_table = cbind(log_probs, log_probs_down, log_probs_up, log_probs_smote, log_probs_adasyn,
                     rid_probs, rid_probs_down, rid_probs_up, rid_probs_smote, rid_probs_adasyn)

# Creating table with all predicted classes
pred_table = cbind(log_pred, log_pred_down, log_pred_up, log_pred_smote, log_pred_adasyn,
                    rid_pred, rid_pred_down, rid_pred_up, rid_pred_smote, rid_pred_adasyn)
pred2_table = cbind(log_pred2, log_pred2_down, log_pred2_up, log_pred2_smote,log_pred2_adasyn,
                     rid_pred2, rid_pred2_down, rid_pred2_up, rid_pred2_smote, rid_pred2_adasyn)

# Creating empty vectors for performance measures
accuracy_vector = rep(NA, ncol(pred_table))
sensitivity_vector = rep(NA, ncol(pred_table))
specificity_vector = rep(NA, ncol(pred_table))
accuracy2_vector = rep(NA, ncol(pred2_table))
sensitivity2_vector = rep(NA, ncol(pred2_table))
specificity2_vector = rep(NA, ncol(pred2_table))

# Loop over all models and imbalance approaches to get performance measures
for (i in 1:ncol(pred_table)){
  accuracy_vector[i] = accuracy(pred_table[,i], y_test)
  sensitivity_vector[i] = sensitivity(pred_table[,i], y_test)
  specificity_vector[i] = specificity(pred_table[,i], y_test)
}
for (i in 1:ncol(pred2_table)){
  accuracy2_vector[i] = accuracy(pred2_table[,i], y_test)
  sensitivity2_vector[i] = sensitivity(pred2_table[,i], y_test)
  specificity2_vector[i] = specificity(pred2_table[,i], y_test)
}

# Bind all results together
results = cbind(accuracy_vector, sensitivity_vector, specificity_vector,
                 accuracy2_vector, sensitivity2_vector, specificity2_vector)

# Round results to 2 digits
results = format(round(results, digits = 2), nsmall = 2)

# Name rows and columns of the results object
colnames(results) = c("Accuracy (0.5)", "Sensitivity (0.5)", "Specificity (0.5)",
                       "Accuracy (EF)", "Sensitivity (EF)", "Specificity (EF)")
rownames(results) = c("LR", "LR down", "LR up", "LR smote", "LR adasyn",
                       "Ridge", "Ridge down", "Ridge up", "Ridge smote", "Ridge adasyn")

results = data.frame(results) %>% 
  apply(2, as.character) %>% 
  apply(2, as.numeric)

results = data.frame(results)



# Save results
saveRDS(results, "cases_study_results.RDS")


## Calibration plots ##
#######################

plot1 = valProbggplot(p = probs_table[,1],
                    y = test_set$TenYearCHD, 
                    smooth="loess",
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot1 = plot1 + labs(title = "Logistic: Uncorrected")

plot2=valProbggplot(p = probs_table[,2],
                    y = test_set$TenYearCHD, 
                    smooth="loess",
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot2 = plot2 + labs(title = "Logistic: RUS")

plot3=valProbggplot(p = probs_table[,3], 
                    y = test_set$TenYearCHD,
                    smooth="loess",
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot3 = plot3 + labs(title = "Logistic: ROS")

plot4=valProbggplot(p = probs_table[,4], 
                    y = test_set$TenYearCHD, 
                    smooth="loess", 
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot4 = plot4 + labs(title = "Logistic: SMOTE")

plot5=valProbggplot(p = probs_table[,5], 
                    y = test_set$TenYearCHD, 
                    smooth="loess",
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot5 = plot5 + labs(title = "Logistic: ADASYN")

plot6=valProbggplot(p = probs_table[,6],
                    y = test_set$TenYearCHD, 
                    smooth="loess", 
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot6 = plot6 + labs(title = "Logistic+Ridge: Uncorrected")

plot7=valProbggplot(p = probs_table[,7],
                    y = test_set$TenYearCHD, 
                    smooth="loess", 
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot7 = plot7 + labs(title = "Logistic+Ridge: RUS")

plot8=valProbggplot(p = probs_table[,8],
                    y = test_set$TenYearCHD, 
                    smooth="loess", 
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot8 = plot8 + labs(title = "Logistic+Ridge: ROS")

plot9=valProbggplot(p = probs_table[,9],
                    y = test_set$TenYearCHD, 
                    smooth="loess", 
                    dostats = F,
                    lwd.ideal = 0.8,
                    lwd.smooth = 0.8,
                    CL.smooth=F,
                    size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                 axis.title = element_blank(),
                                                 plot.title = element_text(size = 8))
plot9 = plot9 + labs(title = "Logistic+Ridge: SMOTE")

plot10=valProbggplot(p = probs_table[,10],
                     y = test_set$TenYearCHD, 
                     smooth="loess", 
                     dostats = F,
                     lwd.ideal = 0.8,
                     lwd.smooth = 0.8,
                     CL.smooth=F,
                     size.d01=2.5)$ggPlot + theme(legend.position = "none",
                                                  axis.title = element_blank(),
                                                  plot.title = element_text(size = 8))
plot10 = plot10 + labs(title = "Logistic+Ridge: ADASYN")

#Logistic in the first column,  Ridge in the second column
grid.arrange(plot1,
             plot6,
             plot2,
             plot7, 
             plot3,
             plot8,
             plot4,
             plot9,
             plot5,
             plot10,
             ncol=2,nrow=5)

# prepare data to plot decision curves (decision risk threshold VS net benefit )
dcdataLR = as.data.frame(cbind(test_set$TenYearCHD,probs_table[,1:5]))
dcdataL2 = as.data.frame(cbind(test_set$TenYearCHD,probs_table[,6:10]))
colnames(dcdataLR) = c("Y", "No", "RUS", "ROS", "SMOTE", "ADASYN")
colnames(dcdataL2) = c("Y", "No", "RUS", "ROS", "SMOTE", "ADASYN")

LRund = decision_curve(Y~No, data = dcdataLR, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
LRrusd = decision_curve(Y~RUS, data = dcdataLR, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
LRrosd = decision_curve(Y~ROS, data = dcdataLR, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
LRsmoted = decision_curve(Y~SMOTE, data = dcdataLR, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
LRadasyned = decision_curve(Y~ADASYN, data = dcdataLR, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 

L2und = decision_curve(Y~No, data = dcdataL2, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
L2rusd = decision_curve(Y~RUS, data = dcdataL2, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
L2rosd = decision_curve(Y~ROS, data = dcdataL2, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
L2smoted = decision_curve(Y~SMOTE, data = dcdataL2, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 
L2adasyned = decision_curve(Y~ADASYN, data = dcdataL2, fitted.risk = TRUE, thresholds = seq(0, 1, by = .05), bootstraps = 200) 

#Log Reg
plot_decision_curve( list(LRund, LRrusd, LRrosd, LRsmoted, LRadasyned), 
                     curve.names = c("Uncorrected", "RUS", "ROS", "SMOTE", "ADASYN"),
                     col = c("black", "red", "green", "blue", "violet"), 
                     ylim = c(-0.25, 0.2), #set ylim
                     xlim = c(0,1),
                     lty = c(1,1,1,1,1), lwd = c(2,2,2,2,2), confidence.intervals = FALSE,
                     standardize = FALSE, #plot Net benefit instead of standardized net benefit
                     legend.position = "topright",xlab="Risk threshold",ylab = "Net Benefit", cost.benefit.axis=F) 
#Ridge
plot_decision_curve( list(L2und, L2rusd, L2rosd, L2smoted, L2adasyned), 
                     curve.names = c("Uncorrected", "RUS", "ROS", "SMOTE", "ADASYN"),
                     col = c("black", "red", "green", "blue", "violet"), 
                     ylim = c(-0.25,0.2), #set ylim
                     xlim = c(0,1),
                     lty = c(1,1,1,1,1), lwd = c(2,2,2,2,2), confidence.intervals = FALSE,
                     standardize = FALSE, #plot Net benefit instead of standardized net benefit
                     legend.position = "topright",xlab="Risk threshold",ylab = "Net Benefit", cost.benefit.axis=F) 

