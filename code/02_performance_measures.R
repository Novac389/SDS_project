######################################################################################
############ Performance measures ####################################################
#### original code from: https://github.com/benvancalster/classimb_calibration #######
#### modified by Andriani P. Manai E Velardita M #####################################
######################################################################################

# In this script, the functions to compute the performance measures used in
# the case study are defined. 

# accuracy
accuracy <- function(pred, outcome){
  correct_predictions <- sum(pred == outcome)
  total_predictions <- length(pred)
  correct_predictions / total_predictions
}

# sensitivity 
sensitivity <- function(pred, outcome) {
  true_positives <- sum(pred == 1 & outcome == 1)
  false_negatives <-  sum(pred == 0 & outcome == 1)
  true_positives/(true_positives+false_negatives)
}

# specificity
specificity <- function(pred, outcome) {
  true_negatives <- sum(pred == 0 & outcome == 0)
  false_positives <-  sum(pred == 1 & outcome == 0)
  true_negatives/(true_negatives+false_positives)
}

# precision
precision <- function(pred, outcome){
  true_positives <- sum(pred == 1 & outcome == 1)
  false_positives <-  sum(pred == 1 & outcome == 0)
  true_positives/(true_positives+false_positives)
}





