\name{fcgr_genotype_controls}

\docType{data}      

\title{Fc gamma receptor allele copy numbers}

\description{
  Fc gamma receptor genotypes from MLPA analysis of a control cohort
} 

\usage{data{fcgr_genotype_controls}}        # hier wird beschrieben, wie die Daten aufzurufen sind 
                             (dies ist meist der Name des Datensatzes).
\format{
  An object of class \code{data frame}.
}

\details{
Fc gamma receptor genotypes determined from 387 individuals of a (northern) German control population by multiplex ligation-dependent probe amplification (MLPA).
The data set consist of allele counts for individual Fc gamma receptor genes (<gene>.<allele>). To reflect ambiguities in the MLPA interpretation, single individuals
can have multiple genotypes (pseudo-individuals), together with a probability weight.

\item{ID}{ID of a each individual. Individuals might have multiple associated genotypes.}
\item{Age}{Age of individuals.}
\item{Gender}{Gender of individuals.}
\item{Probability}{Probability weight for pseudo-individuals. For each real individual, probability weights of pseudo-individual must sum up to 1.0.}
\item{FCGR2A.H}{Fc gamma receptor IIa, 131H variant, number of alleles.}
\item{FCGR2A.R}{Fc gamma receptor IIa, 131R variant, number of alleles.}
\item{HSPA6}{Heat-shock protein A6 gene copy number without allelic variation.}
\item{FCGR3A.F}{Fc gamma receptor IIIa, 158F variant, number of alleles.}
\item{FCGR3A.V}{Fc gamma receptor IIIa, 158V variant, number of alleles.}
\item{FCGR2BC.T}{Fc gamma receptor IIb/c, -120T promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.}
\item{FCGR2BC.A}{Fc gamma receptor IIb/c, -120A promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.}
\item{FCGR2BC.G}{Fc gamma receptor IIb/c, -386G promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.}
\item{FCGR2BC.C}{Fc gamma receptor IIb/c, -386C promotor polymorphism, sum of alleles of both Fc gamma receptor IIb and c genes.}
\item{HSPA7}{Heat-shock protein A7 (pseudo-)gene copy number without allelic variation.}
\item{FCGR3B.Na1}{Fc gamma receptor IIIb, Na1 or HNA1a variant, number of alleles.}
\item{FCGR3B.Na2}{Fc gamma receptor IIIb, Na2 or HNA1b variant, number of alleles.}
\item{FCGR3B.SH}{Fc gamma receptor IIIb, SH or HNA1c variant, number of alleles.}
\item{FCGR2B.I}{Fc gamma receptor IIb, 232I variant, number of alleles.}
\item{FCGR2B.T}{Fc gamma receptor IIb, 232T variant, number of alleles.}
}

# \references{QUELLEN}       # hiermit können Literaturangaben angegeben werden.

\source{German control cohort}            

\keyword{datasets}         
