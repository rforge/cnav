 \name{cnav.unormdensity}
\alias{print.cnav.unormdensity}
\alias{summary.cnav.unormdensity}
\alias{print.summary.cnav.unormdensity}
\title{GENOME ANALYSIS OF JOINT COPY-NUMBER AND ALLELIC VARIATION BY BAYESIAN HIDDEN MARKOV MODEL INFERENCE}
\description{Determines the unnormalized density for a transition probability matrix, given the individuals, genotypes and weights.}
\usage{
cnav.unormdensity = function(genotypes, individuals, weights,
                           transition_matrix, emission_matrix,
			   n_sim = 1e6, 
			   random_seed = trunc(runif(1, min=0, max = 1e6)),
                           max_seq_len = 1000)
}
\arguments{
\item{genotypes}{ an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene }
\item{individuals}{ a factor with the individuals belonging to the rows in genotypes }
\item{weights}{ a numeric vector the probability weight for each genotype (multiple pseudoindividuals with a probability summing up to 1.0)}
\item{transition_matrix}{the transition probabilities as matrix (numeric matrix)}
\item{emission_matrix}{ each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)}
\item{n_sim}{ number of resampled genotypes to approximate the likelihood function, integer}
\item{random_seed}{ a random number to make the result reproducible, integer}
\item{max_seq_len}{ maximum length of a Markov path, integer}
}
\value{
An List including elements
\item{Unnormalized.Density}{The unnormalized log density}
\item{Likelihood}{The approximated log likelihood value}
\item{Prior.Density}{The prior density, given Jeffrey's prior for the transition probabilities (applies to each row vector of the matrix)}
}
\author{Andreas Recke}
\examples{
  # None yet
  
}
\keyword{Markov model} 