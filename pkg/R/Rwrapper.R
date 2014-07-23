
# cnav.regression
# This function wraps the actual C++ function for HMM inference

cnav.regression <- function(genotypes,
                            transition_matrix,
                            emission_matrix,
                            temperatures = 1,
	                    burnin = 100,
                            mc = 1000,
			    incomplete_samplings = 1000,
			    squirrel_kernel_parameter = 100,
			    n_swappings = 0,
                            max_sequence_length = 1000,
                            seed = 42)
{
  # first: some checks
  if (nrow(genotypes) == 0) stop("No genotype data!\n")

  # check temperatures
  if (any(order(temperatures, decreasing=T) != 1:length(temperatures))) {
    cat("Somethings wrong with the temperature set! Try to correct \n")
    cat("Old: " , temperatures , "\n");
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
  if (n_swappings<0 ||burnin < 0 || any(c(mc, max_sequence_length, seed, incomplete_samplings, squirrel_kernel_parameter) <= 0)) stop("Please correct control settings!\n");

  # Everythings okay? ... then start
                        
  result = .Call("HMMinterface",
              genotypes=genotypes,
              transition_matrix = transition_matrix,
              emission_matrix = emission_matrix,
              temperatures = as.double(temperatures),
              r_how_many_sequence_tries = as.integer(incomplete_samplings),
              r_maxsequence_length = as.integer(max_sequence_length),
	      internal_sampling = as.integer(squirrel_kernel_parameter), 
	      n_swappings = as.integer(n_swappings),
	      collapsed_sampling = FALSE, 
              burnin = as.integer(burnin),
              mc = as.integer(mc),
              chib_samples = as.integer(mc/30),
              seed = as.integer(seed),
              PACKAGE="CNAV")

  class(result) = "cnav.result"
  return(result);
}