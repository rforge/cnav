
# cnav.regression
# This function wraps the actual C++ function for HMM inference

# input parameters:
# genotypes : an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene
# individuals : a factor with the individuals belonging to the rows in genotypes 
# weights : a numeric vector the probability weight for each genotype (multiple pseudoindividuals with a probability summing up to 1.0)
# transition_matrix : the indicator matrix for valid graph edges (numeric matrix)
# emission_matrix : each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)
# temperatures : a temperature set, always starting with 1.0 as the highest temperature and ending with 0.0 as the lowest one, numeric vector
# percentage : a tuning parameter, how many genotypes should be approximated, double
# preparation : another tuning parameter - how many sequences should be generated for approximation
# max_unbiased_sequence_generation_repeats - the maximum number of trials to generate the genotypes unbiasedly
# max_sequence_length: just a restriction - the maximum length of a Markov path
# burnin : how many samples should be dropped beforehand, integer
# mc : number of MC samples, integer
# seed: a random number to make the MCMC (partly) reproducible, integer
 
# output:
# temperature.indices : temperature level - for each MCMC sample
# number.of.sequence.generation.repeats : number of resamples necessary to generate a preset amount of unbiased sequences - for each MCMC sample
# amount.of.unbiasedly.simulated.sequences : amount of sequences which are unbiased - for each MCMC sample
# mc.samples.transition.matrix : the random samples for the transition matrix, order is column-wise
# mean.temperature.jumping.probabilities : the probability to jump from lower to higher temperature ... useful to determine temperature steps
# kullback.leibler.divergences : the information amount taken up by the posterior distribution, compared to the prior distribution
# 

cnav.regression <- function(genotypes,
                            individuals,
                            weights,
                            transition_matrix,
                            emission_matrix,
                            temperatures,
                            burnin = 100,
                            mc = 1000,
			    percentage = 0.95,
			    preparation = 100,
                            max_unbiased_sequence_generation_repeats = 30000,
                            max_sequence_length = 1000,
                            seed = 42)
{
  # first: some checks

  # genotype matrix, individuals and weights length identical
  if (nrow(genotypes) != length(individuals) || nrow(genotypes) != length(weights)) stop("Genotype matrix does not match individuals or weights!\n");

  # check temperatures
  if (any(order(temperatures, decreasing=T) != 1:length(temperatures))) {
    cat("Somethings wrong with the temperature set! Try to correct \n")
    cat("Old: " , temperature , "\n");
    temperatures = sort(temperatures, decreasing=T)
    temperatures[1] = 1.0
    temperatures[length(temperatures)] = 0.0     
    cat("New (corrected): " , temperature , "\n");
  }

  # check compatibility of emission matrix with transition matrix
  if (nrow(emission_matrix) != nrow(transition_matrix)) stop("Transition matrix does not match emission matrix!\n")

  # check compatibility of emission_matrix with genotypes
  if (ncol(emission_matrix) != ncol(genotypes)) stop("Emission matrix does not match genotypes!\n")

  # check other settings
  if (any(c(burnin, mc, preparation, max_sequence_length, seed, max_unbiased_sequence_generation_repeats) <= 0)) stop("Please correct control settings!\n");

  # Everythings okay? ... then start
                        
  result = .Call("HMMinterface",
              genotypes=genotypes,
              individuals=as.integer(individuals),
              weights=as.double(weights),
              transition_matrix = transition_matrix,
              emission_matrix = emission_matrix,
              temperatures = as.double(temperatures),
              percentage =  as.double(percentage),
              r_how_many_sequence_tries = as.integer(max_unbiased_sequence_generation_repeats),
              r_preparation = as.integer(preparation),
              r_maxsequence_length = as.integer(max_sequence_length),
              burnin = as.integer(burnin),
              mc = as.integer(mc),
              seed = as.integer(seed),
              PACKAGE="CNAV")

  class(result) = "cnav.result"
  return(result);
}