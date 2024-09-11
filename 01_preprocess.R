####################################################
#### Data preprocessing and visualization ##########
#### authors: Andriani P. Manai E. Velardita M. ####
####################################################

library(caret)
library(reshape2)
# set current working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))


#######################################
#### data cleaning and preparation ####
#######################################

#read Framingham original dataset
df=read.csv("framingham_heart disease orig.csv")
categoricalCols= c("male","education","currentSmoker","BPMeds","prevalentStroke","prevalentHyp","diabetes")
df = df[, -which(names(df) %in% categoricalCols)]

colSums(is.na(df))#count the null values for each column
df = na.omit(df)#remove all rows that contain na in at least one feature
y = df[9]


df_8predictors=cbind(y,df[1:8])
write.csv(df_8predictors, "df_8predictors.csv", row.names=FALSE)


#####################
# Scaling data with min-max of caret library 
# Show boxplots of variables value and outliers with Turkey's fences (1.5 from IQR)
#####################



# load necessary code for performance measures
data <- read.csv("df_8predictors.csv")
data

####### MIN-MAX scaling of data in values between 0 and 1
#
scaling_transformer <- preProcess(data, method = "range")

# Apply transformation to data
scaled_data <- predict(scaling_transformer, newdata = data)

print(scaled_data)

# save normalized data
write.csv(scaled_data, file = "df_8pred_normal.csv", row.names = FALSE)

#####Explore data######
#report data explorer
#library(DataExplorer)
#create_report(scaled_data)
###########

# Take columns except the target
data <- data[, -1]

#############
# Grid of boxplot of min-max scaled data to highlight outliers & check distribution of values

scaled_data <- scaled_data[, -1]
scaled_data

ggplot(melt(scaled_data), aes(x = variable, y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) +
  geom_point(data = subset(melt(scaled_data), value < quantile(value, 0.25) - 1.5 * IQR(value) | value > quantile(value, 0.75) + 1.5 * IQR(value)), 
             aes(color = "outliers"), size = 1)+
  scale_color_manual(values = c("outliers" = "red"))+ 
labs(color = "Legend")

