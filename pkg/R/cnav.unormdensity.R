 
# cnav.unormdensity
# This function wraps the functions to calculate an unnormalized density 

# input parameters:
# genotypes : an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene
# individuals : a factor with the individuals belonging to the rows in genotypes
# weights : a numeric vector the probability weight for each genotype (multiple pseudoindividuals with a probability summing up to 1.0)
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
                           individuals,
                           weights,
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

  # genotype matrix, individuals and weights length identical
  if (nrow(genotypes) != length(individuals) || nrow(genotypes) != length(weights)) stop("Genotype matrix does not match individuals or weights!\n");

  # check compatibility of emission matrix with transition matrix
  if (nrow(emission_matrix) != nrow(transition_matrix)) stop("transition matrix does not match emission matrix!\n")

  # correct order of individuals
  neworder <- order(individuals)
  individuals = individuals[neworder]
  weights = weights[neworder]
  genotypes = genotypes[neworder,]
  # correct sums of weights
  for (ina in unique(individuals)) {
    if (sum(individuals == ina) == 1) {
      weights[ina == individuals] = 1
    } else {
      weights[ina==individuals][1] = 1 - sum(weights[ina==individuals][-1])
    }
  }

  result = .Call("HMMunnormalizedDensity", genotypes = genotypes, individuals = individuals, weights=weights,
                 transition_matrix = transition_matrix, emission_matrix = emission_matrix, count = n_sim,
                 random_seed = random_seed, max_seq_len = max_seq_len,
                 PACKAGE="CNAV")

  return(result);
}