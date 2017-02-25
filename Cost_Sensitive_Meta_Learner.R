
#' CostSensMetaLearner
#' 
#' Function builds an RWeka model with a CostSensitive classifier, with a given 
#' data set and options of either Boosting or Bagging (with option for number of
#' iterations) as meta-learners to two options Decision Stump or J48
#' 
#' @param x a non-negative numeric value used to represent the scale factor type
#'   II errors cost relative to type I errors, e.g. if x = 3 this would mean we 
#'   want type II errors cost 3 times that of type I
#' @param type a character string which is the name of a RWeka meta-learner, 
#'   either Bagging or AdaBoostM1 for this Assignment
#' @param I a numeric value representing the number of iterations for variable 
#'   'type'
#' @param W a character string whic is the name of a RWeka binary classifier, 
#'   either 'J48' or 'DecisionStump'for this Assignment
#'   
#' @return  A Weka classifier evaluation using 10-fold cross validation on the 
#'   model using the original training data.
#' @export
#' 
#' @examples
CostSensMetaLearner <- function(x, data, type, I, W){
  checkArg(x, "numeric", min.len = 1L, lower = 0)
  checkArg(type, "character", choices = c("Bagging", "AdaBoostM1"))
  checkArg(I, "numeric", min.len = 1L, lower = 1)
  checkArg(W, "character", choices = c("J48", "DecisionStump"))
  mat <- matrix(c(0, x, 1, 0), nrow = 2, byrow = TRUE)
  mod4 <- CostSensitiveClassifier(CLASS ~ ., data = data, 
                                  control = Weka_control('cost-matrix' = mat,
                                                         W = list(type, 
                                                                  I = I, 
                                                                  S = 1,
                                                                  W = W)))
  evaluate_Weka_classifier(mod4, class = TRUE, numFolds = 10, seed = 1) 
}
