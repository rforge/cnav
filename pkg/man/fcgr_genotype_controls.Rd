\name{fcgr_genotype_controls}
\alias{fcgr_genotype_controls}

\docType{data}      

\title{Fc gamma receptor allele copy numbers}

\description{
  Fc gamma receptor genotypes from MLPA analysis of a control cohort
} 

\usage{
  data(fcgr_genotype_controls)       
} 
                             
\format{
  An object of class \code{data frame}.
}

\details{
Fc gamma receptor genotypes determined from 387 individuals of a (northern) German control population by multiplex ligation-dependent probe amplification (MLPA).
The data set consist of allele counts for individual Fc gamma receptor genes (<gene>.<allele>). To reflect ambiguities in the MLPA interpretation, single
individuals can have multiple genotypes (pseudo-individuals), together with a probability weight.

\tabular{rl}{
ID   \tab ID of a each individual. Individuals might have multiple associated genotypes.   \cr
Age   \tab Age of individuals.   \cr
Gender   \tab Gender of individuals.   \cr
Probability   \tab Probability weight for pseudo-individuals. For each real individual, probability weights of pseudo-individual must sum up to 1.0.   \cr
FCGR2A.H   \tab Fc gamma receptor IIa, 131H variant, number of alleles.   \cr
FCGR2A.R   \tab Fc gamma receptor IIa, 131R variant, number of alleles.   \cr
HSPA6   \tab Heat-shock protein A6 gene copy number without allelic variation.   \cr
FCGR3A.F   \tab Fc gamma receptor IIIa, 158F variant, number of alleles.   \cr
FCGR3A.V   \tab Fc gamma receptor IIIa, 158V variant, number of alleles.   \cr
FCGR2BC.T   \tab Fc gamma receptor IIb/c, -120T promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.   \cr
FCGR2BC.A   \tab Fc gamma receptor IIb/c, -120A promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.   \cr
FCGR2BC.G   \tab Fc gamma receptor IIb/c, -386G promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.   \cr
FCGR2BC.C   \tab Fc gamma receptor IIb/c, -386C promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.   \cr
HSPA7   \tab Heat-shock protein A7 (pseudo-)gene copy number without allelic variation.   \cr
FCGR3B.Na1   \tab Fc gamma receptor IIIb, Na1 or HNA1a variant, number of alleles.   \cr
FCGR3B.Na2   \tab Fc gamma receptor IIIb, Na2 or HNA1b variant, number of alleles.   \cr
FCGR3B.SH   \tab Fc gamma receptor IIIb, SH or HNA1c variant, number of alleles.   \cr
FCGR2B.I   \tab Fc gamma receptor IIb, 232I variant, number of alleles.   \cr
FCGR2B.T   \tab Fc gamma receptor IIb, 232T variant, number of alleles.   \cr
}
}
\references{
  N.N.       
}

\source{German control cohort}            

\keyword{datasets}         
