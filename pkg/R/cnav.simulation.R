 
# cnav.simulation
# This function wraps the functions to simulate haplotypes or diplotypes

# input parameters:
# target: what type of information should be resampled - haplotypes = half, diplotypes = genotypes = full
# transition_matrix: the transition matrix containing transition probabilities
# emission_matrix: emissions for each state
# n_sim: number of simulations
# random_seed: just to set the data
# max_seq_len: restricts length

# output:
# a List, containing
# Haplotypes or Genotypes : a matrix containing row vectors of genotypes or half genotypes, i.e. haplotypes
# Counts : the number of simulations that produced the above haplotypes or genotypes
# Frequency.MLE : approximate frequency for each haplotype 

cnav.simulation = function(target = c("haplotypes","diplotypes", "half", "full", "genotypes"),
                           transition_matrix,
                           emission_matrix,
                           n_sim,
                           random_seed = trunc(random(1, min=0, max = 1e6)),
                           max_seq_len = 1000)
{
  target = match.arg(target)

  # check compatibility of emission matrix with transition matrix
  if (nrow(emission_matrix) != nrow(transition_matrix)) stop("transition matrix does not match emission matrix!\n")

  if (target %in% c("haplotypes", "half")) 
  {
    result = .Call("HMMgenerateHaplotype", transition_matrix=transition_matrix, emission_matrix = emission_matrix, count=n_sim,
                                           random_seed = random_seed, max_seq_len = max_seq_len, PACKAGE="CNAV")
  } else {
    result = .Call("HMMgenerateDiplotype", transition_matrix=transition_matrix, emission_matrix = emission_matrix, count=n_sim,
                                           random_seed = random_seed, max_seq_len = max_seq_len, PACKAGE="CNAV")
  }
  return(result);
}