\name{cnav.regression}
\alias{print.cnav.regression}
\alias{summary.cnav.regression}
\alias{print.summary.cnav.regression}
\title{Genome analysis of joint copy-number and allelic variation by Bayesian Hidden Markov Model inference.}
\description{Determine posterior distribution of HMM transition matrix by serially tempered Gibbs sampling.}
\usage{
cnav.regression(genotypes, individuals, weights, transition_matrix, emission_matrix, temperatures,
                            burnin = 100, mc = 1000, percentage = 0.95, preparation = 100,
                            max_unbiased_sequence_generation_repeats = 30000, max_sequence_length = 1000,
                            seed = 42)
}
\arguments{
\item{genotypes}{ an integer matrix containing individual genotypes in a row, i.e. counts of alleles for each gene }
\item{individuals}{ a factor with the individuals belonging to the rows in genotypes }
\item{weights}{ a numeric vector the probability weight for each genotype (multiple pseudoindividuals with a probability summing up to 1.0)}
\item{transition_matrix}{the indicator matrix for valid graph edges (numeric matrix)}
\item{emission_matrix}{ each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)}
\item{temperatures}{ a temperature set, always starting with 1.0 as the highest temperature and ending with 0.0 as the lowest one, numeric vector}
\item{percentage}{ a tuning parameter, how many genotypes should be approximated, double}
\item{preparation}{ another tuning parameter - how many sequences should be generated for approximation}
\item{max_unbiased_sequence_generation_repeats}{ the maximum number of trials to generate the genotypes unbiasedly}
\item{max_sequence_length}{ just a restriction - the maximum length of a Markov path}
\item{burnin}{ how many samples should be dropped beforehand, integer}
\item{mc}{ number of MC samples, integer}
\item{seed}{ a random number to make the MCMC (partly) reproducible, integer}

# output:
# temperature.indices : temperature level - for each MCMC sample
# number.of.sequence.generation.repeats : number of resamples necessary to generate a preset amount of unbiased sequences - for each MCMC sample
# amount.of.unbiasedly.simulated.sequences : amount of sequences which are unbiased - for each MCMC sample
# mc.samples.transition.matrix : the random samples for the transition matrix, order is column-wise
# mean.temperature.jumping.probabilities : the probability to jump from lower to higher temperature ... useful to determine temperature steps
# kullback.leibler.divergences : the information amount taken up by the posterior distribution, compared to the prior distribution
# 


\value{
An object of class \code{cnav.result}, basically a list including elements
\item{temperature.indices}{temperature levels trace recorded together with Gibbs sampling}
\item{number.of.sequence.generation.repeats}{number of resamplings necessary to generate a preset amount of unbiased Markov paths recorded together with Gibbs sampling. For control purposes.}
\item{amount.of.unbiasedly.simulated.sequences}{amount of sequences which are unbiased, recorded during Gibbs sampling. For control purposes.}
\item{mc.samples.transition.matrix}{Gibbs output. The random samples for the transition matrix, order is column-wise.}
\item{mean.temperature.jumping.probabilities}{the probability to jump from lower to higher temperature for each temperature index ... useful to determine temperature steps}
\item{kullback.leibler.divergences}{the information amount taken up by the posterior distribution, compared to the prior distribution.}
}
\author{Andreas Recke}
\examples{
  cat("This will be an example!\n")
}
\keyword{Markov model} 
