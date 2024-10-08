####################################################
### Comparison among models precision ##############
#### authors: Andriani P. Manai E. Velardita M. ####
####################################################


#load results table generated by script_case_study_general_v2.R
results <- readRDS("cases_study_results.RDS")

classifiers_names <- c("SLR", "Down", "Up", "Smote", "Ada",
                       "Ridge", "R-Down", "R-Up", "R-SMote", "R-Ada")

#extract the 4th column (Precision column)
precision <- results[,4]

#Vector of bars heights
bar_heights <- precision

#create barplot Precision
bp <- barplot(bar_heights, names.arg = classifiers_names, col = "lightblue",
              main = "Classifiers Precision", xlab = "Classifiers trained on different rebalanced dataset", ylab = "Observed Precision",
              cex.names=1, ylim=c(0,0.9))

#width of bars
bar_width <- bp[2]-bp[1]

bar_colors <- c(rep("#a78d84", 5), rep("#ead6c6", 5))

#overlap colored resized rectangle on existing barplot
for(i in 1:10){
  rect(bp[i]-bar_width/2, 0, bp[i]+bar_width/2, bar_heights[i], col=bar_colors[i])
}

#add precision values 
text(x=bp, y=bar_heights, labels=round(bar_heights, 2), pos=3, cex=1.5, col="black")



#########################################
### Comparison among models accuracy ###
#########################################

#extract the 1st column (Accuracy column)
accuracy <- results[,1]

#Vector of bars heights
bar_heights <- accuracy

#create accuracy barplot
bp <- barplot(bar_heights, names.arg = classifiers_names, col = "lightblue",
              main = "Classifiers Accuracy", xlab = "Classifiers trained on different rebalanced dataset", ylab = "Observed Accuracy",
              cex.names=1, ylim=c(0,1))

#width of bars
bar_width <- bp[2]-bp[1]

bar_colors <- c(rep("#a78d84", 5), rep("#ead6c6", 5))

#overlap colored resized rectangle on existing barplot
for(i in 1:10){
  rect(bp[i]-bar_width/2, 0, bp[i]+bar_width/2, bar_heights[i], col=bar_colors[i])
}

#add accuracy values 
text(x=bp, y=bar_heights, labels=round(bar_heights, 2), pos=3, cex=1.5, col="black")



#########################################
### Comparison among models sensitivity ###
#########################################

#extract the 2nd column (Sensitivity column)
sensitivity <- results[,2]

#Vector of bars heights
bar_heights <- sensitivity

#create sensitivity barplot
bp <- barplot(bar_heights, names.arg = classifiers_names, col = "lightblue",
              main = "Classifiers Sensitivity", xlab = "Classifiers trained on different rebalanced dataset", ylab = "Observed Sensitivity",
              cex.names=1, ylim=c(0,1))

#width of bars
bar_width <- bp[2]-bp[1]

bar_colors <- c(rep("#a78d84", 5), rep("#ead6c6", 5))

#overlap colored resized rectangle on existing barplot
for(i in 1:10){
  rect(bp[i]-bar_width/2, 0, bp[i]+bar_width/2, bar_heights[i], col=bar_colors[i])
}

#add sensitivity values 
text(x=bp, y=bar_heights, labels=round(bar_heights, 2), pos=3, cex=1.5, col="black")

