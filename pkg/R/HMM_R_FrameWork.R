
generate_initial_parameters <- function(index_matrix) {
  transition <- (index_matrix>0) * runif(length(index_matrix))

  transition <- t(apply(transition,1,function(x) if (sum(x)==0) x else x/sum(x)))
  result <- tapply(transition, index_matrix, mean)
  if (min(index_matrix) == 0) result = result[-1]

  return(result);
}

duplicate_transition_matrix <- function(transition_matrix)
{
  result = 
    rbind(
      cbind(transition_matrix, diag(rep(0,nrow(transition_matrix)))), 
      cbind(diag(rep(0,nrow(transition_matrix))), transition_matrix)
    )
    
  result[nrow(transition_matrix), ncol(transition_matrix)+1] = 1
  return(result)
}


get_parameters <- function(index_matrix, transition_matrix)
{
  result <- tapply(transition_matrix, index_matrix, mean)[-1]
  return(result)
}


generate_transition_matrix <- function(index_matrix, parameters)
{
  transition = index_matrix*0
  transition[index_matrix>0] = parameters[index_matrix[index_matrix>0]]
  return(transition)
}


transition_to_indexmatrix <- function(transition_matrix)   # includes a duplication mechanism for haplotype HMMs
{
  n_par = sum(transition_matrix > 0)
  index_matrix = matrix(0, nrow=nrow(transition_matrix), ncol=ncol(transition_matrix))
  index_matrix[transition_matrix>0] = 1:n_par
  index_matrix = rbind(cbind(index_matrix,diag(rep(0,nrow(index_matrix)))), cbind(diag(rep(0,nrow(index_matrix))), index_matrix))
  index_matrix[nrow(transition_matrix),nrow(transition_matrix)+1] = n_par+1
  return(index_matrix)
}

simulation <- function(index_matrix, parameters, emissions, Nsamp=10)
{
  transitions = generate_transition_matrix(index_matrix, parameters);
  result <- matrix(0,ncol=ncol(emissions), nrow=Nsamp)
  for (i in 1:Nsamp)
  {
    state = 1
    max = 1e4
    count = 0
    while (count < max && state != nrow(transitions)) {
      state = which(1==rmultinom(n=1,size=1,prob=transitions[state,]))
      result[i, ] = result[i,] + emissions[state,]
      count=count+1;
    }
  }
  
  return(result)
}