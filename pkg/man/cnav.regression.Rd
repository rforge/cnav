\name{cnav.regression}
\alias{cnav.regression}
\alias{print.cnav.regression}
\alias{summary.cnav.regression}
\alias{print.summary.cnav.regression}
\docType{package}
\title{GENOME ANALYSIS OF JOINT COPY-NUMBER AND ALLELIC VARIATION BY BAYESIAN HIDDEN MARKOV MODEL INFERENCE}
\description{Determine posterior distribution of HMM transition matrix by serially tempered Gibbs sampling.}
\usage{
cnav.regression <- function(genotypes, transition_matrix, emission_matrix,
                            temperatures = 1,
                            burnin = 100, mc = 1000,
			    incomplete_samplings = 1000, squirrel_kernel_parameter = 100, n_swappings = 0,
                            max_sequence_length = 1000, seed = 42)


}
\arguments{
\item{genotypes}{ an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene }
\item{transition_matrix}{the indicator matrix for valid graph edges (numeric matrix, values may only be 0 or 1)}
\item{emission_matrix}{ each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)}
\item{burnin}{ how many samples should be dropped beforehand, integer}
\item{mc}{ number of MC samples, integer}
\item{temperatures}{ for experimental purposes. The temperature levels for parallel tempering.}
\item{incomplete_samplings}{ the number of simulations for the incomplete Gibbs sampler}
\item{squirrel_kernel_parameter}{ a tuning parameter for the squirrel kernel - improves mixing}
\item{n_swappings}{ number of temperature exchanges between particels for parallel tempering. Exchanges are chosen randomly and only between neighboured
temperature levels. }
\item{max_sequence_length}{ the length a possible Markov path may have. Needs limitation for practical reasons. }
\item{seed}{ a random number to make the MCMC (partly) reproducible, integer}
}
\details{
This function is the interface to the MCMC sampler. As input, the genotypes (allele counts) are given, together with the corresponding emission matrix and
the transition graph encoding valid transitions between states (1=valid connection, 0=no connection).

The output is the inferred posterior distribution of the transition probability matrix.

In the current version, parallel tempering is implemented but needs further development to be reliable. Interested users may refer to the author (see below).
}
\value{
An object of class \code{cnav.result}, basically a list including elements
\item{temperature.indices}{temperature levels trace recorded together with sampling}
\item{mc.samples.transition.matrix}{A 3-dimensional array (row = vectorized transition matrix A sample, column = a single element \eqn{a_ij} of A, slice =
samples of a single temperature level) of transition matrix MCMC samples. A row contains the n*n elements of a transition matrix, ordered column-wise as
typical for Gnu R.}
\item{mc.samples.hash.trace}{A 3-dimensional array (row = a sample collected each sampling cycle, column = hash value of Markov path of a single
individual, slice = samples of a single temperature level) of MCMC samples of Markov path hash values. This is for diagnostic purposes, to see the
diversity of
Markov paths for each individual genotype.}
\item{mc.samples.likelihood.trace}{A 3-dimensional array (row = a sample collected each sampling cycle, column = likelihood value of a Markov path of a
single individual, slice = samples of a single temperature level) of MCMC samples of Markov path likelihood valuess. This is also for diagnostic
purposes, to evaluate the likelihood traces for individual genotype.}
\item{sorted.genotyping.data}{The input genotypes that correspond to the \code{mc.samples.hash.trace} and the \code{mc.samples.likelihood.trace}.}
\item{chib.marginal.likelihoods}{estimator for the log marginal likelihood of the model, calculated after the Chib method.
Provided as a matrix.
 \tabular{rlll}{
    [,1] \tab Chib.Marginal.Likelihood   \tab numeric \tab the actual Chib estimator, calculated as log likeliood + log prior density - log posterior density \cr
    [,2] \tab Point.Likelihood \tab numeric \tab log likelihood for given theta \cr
    [,3] \tab Point.Prior.Density    \tab numeric \tab log prior density for given theta \cr
    [,4] \tab Point.Posterior.Density   \tab numeric \tab log posterior density for given theta \cr
  } 
}
\item{chib.estimation.points}{sample points at which the chib marginal likelihood is calculated.}
\item{naive.dirichlet.marginal.likelihoods}{log marginal likelihood when a conjugate Dirichlet prior is used for naive estimation of genotype frequencies. For
comparison with the marginal likelihood of the HMM model.}
\item{transition.graph}{The transition graph - just for completeness.}
\item{emission.matrix}{The emission matrix - the same, just for completeness.}
\item{n.samples}{Another feedback. How many samples have actually been produced.}
\item{jumping.probabilities}{A diagnostic output to judge the likelihood at each temperature levels. Useful for convergence diagnostics.}
}
\author{Andreas Recke <andreas.recke@uksh.de>}
\examples{

# load library
library(CNAV)

# Activate data
data(fcgr_genotype_controls)

# Extract data of FCGR3B
newdata = aggregate(Probability ~ ID + FCGR3B.Na1 + FCGR3B.Na2 + FCGR3B.SH, data=fcgr_genotype_controls, FUN=sum)
newdata$Probability = newdata$Probability / with(newdata, tapply(Probability, ID, sum)[ID])
newdata$Probability = sapply(1:nrow(newdata), function(i) 1-sum(newdata$Probability[setdiff(which(newdata$ID == newdata$ID[i]),i)]))

# build transition matrix
transitions = matrix(c(
          #   S  2  Na1  Na2  SH  3  E
             0,  1,  0,  0,   0,  0,  1,
             0,  0,  1,  1,   1,  0,  0,
             0,  0,  0,  0,   0,  1,  0,
             0,  0,  0,  0,   0,  1,  0,
             0,  0,  0,  0,   0,  1,  0,
             0,  1,  0,  0,   0,  0,  1,
             0,  0,  0,  0,   0,  0,  0), nrow=7, byrow=T)

# build emission matrix
emissions = matrix(c(
          #  Na1  Na2  SH
             0,  0,  0,
             0,  0,  0,
             1,  0,  0,
             0,  1,  0,
             0,  0,  1,
             0,  0,  0,
             0,  0,  0 ), nrow=7, byrow=T)

colnames(emissions) = c("FCGR3B.Na1", "FCGR3B.Na2", "FCGR3B.SH")

result <- cnav.regression( genotypes= as.matrix(newdata[,colnames(emissions)]),
			      transition_matrix=transitions,
			      emission_matrix=emissions,
			      seed = trunc(runif(1,min=0,max=1e5)))

# print estimator for the transition matrix
print(matrix(colMeans(result$mc.samples.transition.matrix[,,1]),ncol=ncol(transitions))))

# print log marginal likelihood
print(median(result$chib.marginal.likelihoods))

# compare naive log marginal likelihood
print(median(result$naive.dirichlet.marginal.likelihoods))

}
\keyword{Markov model} 
