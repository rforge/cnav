 
#source("HMM_R_FrameWork.R")

testSkript <- function() 
{

  simTransition = rbind(
    c(0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0.1),
    c(0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0),
    c(0, 0, 0, 0, 0, 0, 0.8, 0.2, 0, 0),
    c(0, 0, 0, 0, 0, 0, 0.2, 0.8, 0, 0),
    c(0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0),
    c(0, 0, 0, 0, 0, 0, 0.01, 0.99, 0, 0),
    c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    c(0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.8),
    c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  )
  
  simEmission = rbind(
    # A  B  C  D  E  F
    c(0, 0, 0, 0, 0, 0),
    c(0, 0, 0, 0, 0, 0),
    c(1, 0, 1, 0, 0, 0),
    c(1, 0, 0, 1, 0, 0),
    c(0, 1, 1, 0, 0, 0),
    c(0, 1, 0, 1, 0, 0),
    c(0, 0, 0, 0, 1, 0),
    c(0, 0, 0, 0, 0, 1),
    c(0, 0, 0, 0, 0, 0),
    c(0, 0, 0, 0, 0, 0)
  )
  
  simIndex = transition_to_indexmatrix(simTransition)
  simParameter = get_parameters(simIndex, duplicate_transition_matrix(simTransition))
  
  simGenotypes = simulation(simIndex, simParameter, rbind(simEmission,simEmission), Nsamp=300)
  
  stufen=10
  simMindist = 0.19
  simMaxdist = 10

  test_transition = (simTransition>0)*runif(length(simTransition))
  test_transition = test_transition/rowSums(test_transition)
  test_transition[nrow(test_transition),] = 0
  
  
  #***
  #  
  #  print(unique(simGenotypes))
  #  cat("Anzahl = ", nrow(unique(simGenotypes)),"\n")
  #  
  #*******************************************************************************
  # prepare some data
  
  individuals = sort(round(runif(nrow(simGenotypes),min=1, max=300)))
  weights = runif(nrow(simGenotypes))
  weights = sapply(1:length(weights), function(i) weights[i]/sum(weights[individuals==individuals[i]]))
    
  
  #*******************************************************************************
  
  #  if (!is.loaded("CNAV")) dyn.load("CNAV.so")
  cat("Number of individuals N = ", length(unique(individuals)),"\n")
  # cat(":-)\n")


    temperaturen = seq(1,0,len=10)^3

    unitGibbs = .C("unitHMMGibbs", genotypes=simGenotypes, individuals=individuals, weights=weights, transition_matrix = test_transition,
		  emission_matrix = simEmission, temperatures = temperaturen,
		  percentage = 0.95, burnin = 0, mc = 100, seed=42 , PACKAGE="CNAV")
		  
   print(colMeans(unitGibbs))		  

}