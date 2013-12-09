 
# cnav.unormdensity
# This function wraps the functions to calculate an unnormalized density 

# input parameters:
# genotypes : an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene
# transition_matrix : the indicator matrix for valid graph edges (numeric matrix)
# emission_matrix : each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)
# n_sim: number of simulations
# random_seed: just to set the data
# max_seq_len: restricts length

# output:
# a List, containing
# Unnormalized.Density : the unnormalized and approximated density given the transition_matrix and observed data
# Likelihood : just the Likelihood alone, i.e. P(observation | theta)
# Prior.Density : the prior density alone, i.e. P(theta)

cnav.unormdensity = function(genotypes,
                           transition_matrix,
                           emission_matrix,
			   n_sim = 1e6,
			   random_seed = trunc(runif(1, min=0, max = 1e6)),
                           max_seq_len = 1000)

                          
{
  # first: some checks
  if (nrow(genotypes) == 0) stop("No genotype data!\n")

    # all row sums of the transition matrix must be 1, except the last
  transition_matrix <- transition_matrix / rowSums(transition_matrix)
  transition_matrix[nrow(transition_matrix),] = rep(0, ncol(transition_matrix))

  # check compatibility of emission matrix with transition matrix
  if (nrow(emission_matrix) != nrow(transition_matrix)) stop("transition matrix does not match emission matrix!\n")

  result = .Call("HMMunnormalizedDensity", genotypes = genotypes, 
                 transition_matrix = transition_matrix, emission_matrix = emission_matrix, count = n_sim,
                 random_seed = random_seed, max_seq_len = max_seq_len,
                 PACKAGE="CNAV")

  return(result);
}