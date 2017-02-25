#' ErrorsFromCostVec
#' 
#' Function takes a cost vector to be iterated through a second argument, a
#' model with an argument option for a cost value
#' 
#' @param cost.vec a vector of non-negative numeric costs for RWeka's 
#'   CostSensitiveClassifier
#' @param model.function a classification model's 10-fold cross validated 
#'   evaluation built in RWeka with one of 8 predefine choices: Bag10J48, 
#'   Bag25J48, Bag10DS, Bag25DS, Boost10J48, Boost25J48, Boost10DS, or Boost25DS
#' @param ... optional arguements to "model.function"
#'   
#' @return a dataframe with 3 columns: cost, type, and error; a long dataframe 
#'   which makes ggplot with geom_line call easier
#' @export
#' 
#' @examples 
ErrorsFromCostVec <- function(cost.vec, model.function, ...) {
  checkArg(cost.vec, "numeric", min.len = 1L, lower = 0)
  checkArg(model.function, "model", choices = c(CostSensMetaLearner, GetModelErrors))
  typeI <- rep(0, length(cost.vec))
  typeII <- rep(0, length(cost.vec))
  for (i in cost.vec) {
    model <- model.function(i, ...)
    typeI[which(cost.vec == i)] <- round(model$detailsClass[1,1], 3)
    typeII[which(cost.vec == i)] <- round(model$detailsClass[2,1], 3)
    }
  
    output <- cbind(cost = round(cost.vec, 3), typeI, typeII)
    output <- gather(as.data.frame(output), type, error, - cost)
    return(output)
}
