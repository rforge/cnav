\name{cnav.simulation}
\alias{cnav.simulation}
\alias{print.cnav.simulation}
\alias{summary.cnav.simulation}
\alias{print.summary.cnav.simulation}
\docType{package}
\title{GENOME ANALYSIS OF JOINT COPY-NUMBER AND ALLELIC VARIATION BY BAYESIAN HIDDEN MARKOV MODEL INFERENCE}
\description{Generate a set of genotypes given a Markov model with transition probabilities and emission matrix}
\usage{
cnav.simulation = function(target = c("haplotypes","diplotypes", "half", "full", "genotypes"),
                           transition_matrix,
                           emission_matrix,
                           n_sim,
                           random_seed = trunc(runif(1, min=0, max = 1e6)),
                           max_seq_len = 1000)
}
\arguments{
\item{target}{Determines whether complete \code{diplotypes} (syn. \code{full} and \code{genotypes}) or \code{haplotypes} (syn. \code{half}) should be
simulated.}
\item{transition_matrix}{the transition matrix for the HMM (numeric matrix)}
\item{emission_matrix}{ each row reflects the allele/gene counting events for a certain state of the HMM (integer matrix)}
\item{n_sim}{ number of simulated samples, integer}
\item{random_seed}{ a random number to make the MCMC (partly) reproducible, integer}
\item{max_seq_len}{ just a restriction - the maximum length of a Markov path}
}
\value{
An object of class \code{cnav.sim.result.diplotypes} 
\item{Genotypes}{a list of unique genotypes}
\item{Counts}{for each row of genotypes, the number of occurences}
\item{Frequency.MLE}{for each row of genotypes, the frequency of occurences}

or an object of class \code{cnav.sim.result.haplotypes}
\item{Haplotypes}{a list of unique genotypes}
\item{Counts}{for each row of genotypes, the number of occurences}
\item{Frequency.MLE}{for each row of genotypes, the frequency of occurences}

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

# calculate model
cnav.inferred.transitions = matrix(colMeans(result$mc.samples.transition.matrix[result$temperature.indices == 0,]),ncol=ncol(transitions))

cnav.sim.hap = cnav.simulation(target = "haplotypes",
                            transition_matrix = cnav.inferred.transitions,
                            emission_matrix = emissions,
                            n_sim = 1e4)


simulation = cbind(cnav.sim.hap$Haplotypes, cnav.sim.hap$Frequency.MLE)
colnames(simulation) = c(colnames(emissions), "Frequency")

print(simulation)

}
\keyword{Markov model} 
