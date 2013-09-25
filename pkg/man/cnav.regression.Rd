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
}
\value{
An object of class \code{cnav.result}, basically a list including elements
\item{temperature.indices}{temperature levels trace recorded together with Gibbs sampling}
\item{number.of.sequence.generation.repeats}{number of resamplings necessary to generate a preset amount of unbiased Markov paths recorded together with Gibbs sampling. For control purposes.}
\item{amount.of.unbiasedly.simulated.sequences}{amount of sequences which are unbiased, recorded during Gibbs sampling. For control purposes.}
\item{mc.samples.transition.matrix}{Gibbs output. The random samples for the transition matrix, order is column-wise.}
\item{mean.temperature.jumping.probabilities}{the probability to jump from lower to higher temperature for each temperature index ... useful to determine temperature steps}
\item{kullback.leibler.divergences}{the information amount taken up by the posterior distribution, compared to the prior distribution.}
\item{chib.marginal.likelihoods}{estimator for the log marginal likelihood of the model, calculated after the Chib method.}
\item{naive.dirichlet.marginal.likelihoods}{log marginal likelihood when a conjugate Dirichlet prior is used for naive estimation of genotype frequencies. For
comparison with the marginal likelihood of the HMM model.}
}
\author{Andreas Recke}
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

temperaturen <- seq(1,0,len=5)^4

result <- cnav.regression( genotypes= as.matrix(newdata[,colnames(emissions)]),
			      individuals = newdata$ID,
			      weights=newdata$Probability,
			      transition_matrix=transitions,
			      emission_matrix=emissions,
			      temperatures = temperaturen,
			      burnin = 10, mc = 300, percentage = 1.0,
			      preparation = 100, max_unbiased_sequence_generation_repeats = 30000,
			      max_sequence_length = 1000, seed = trunc(runif(1,min=0,max=1e5)))

# print estimator for the transition matrix
print(matrix(colMeans(result$mc.samples.transition.matrix[result$temperature.indices == 0,]),ncol=ncol(transitions)))

# print log marginal likelihood
print(median(result$chib.marginal.likelihoods)))

# compare naive log marginal likelihood
print(median(result$naive.dirichlet.marginal.likelihoods)))

}
\keyword{Markov model} 
