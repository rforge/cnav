// unittests

#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMgibbs.hpp"

std::ostream& operator<< (std::ostream& os, arma::urowvec val)
{
	arma::urowvec::const_iterator iter = val.begin();
	for (; iter != val.end(); iter++) os << " " << (*iter);
	return os;
}


//~ RcppExport SEXP unitHMMdataSet(SEXP genotypes, SEXP individuals, SEXP weights, SEXP transition_matrix, SEXP emission_matrix, 
  //~ SEXP percentage, SEXP seed)
//~ {
	//~ BEGIN_RCPP
	//~ 
    //~ using namespace Rcpp; 
//~ 
    //~ IntegerMatrix iGenotypes(genotypes);
    //~ arma::imat ia_genotypes(iGenotypes.begin(), iGenotypes.rows(), iGenotypes.cols(), true);
    //~ arma::umat ua_genotypes = arma::conv_to<arma::umat>::from(ia_genotypes);
    //~ 
    //~ IntegerVector iIDs(individuals);
    //~ arma::ivec ia_ids(iIDs.begin(), iIDs.size(), true);
    //~ arma::uvec ua_ids = arma::conv_to<arma::uvec>::from(ia_ids);
    //~ 
    //~ NumericVector iWeights(weights);
    //~ arma::vec na_weights(iWeights.begin(), iWeights.size(), true);
    	//~ 
    //~ arma::uword randseed = as<arma::uword>(seed);
    //~ BasicTypes::base_generator_type	rgen(randseed);
     //~ 
    //~ double a_percentage = as<double>(percentage);
       //~ 
    //~ NumericMatrix iTransitionMatrix(transition_matrix);
    //~ arma::mat na_transition(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    //~ arma::umat ua_graph = na_transition > 0.0;
    //~ 
    //~ IntegerMatrix iEmission(emission_matrix);
    //~ arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    //~ arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    	//~ 
	//~ HMMdataSet dataTest(ua_genotypes, ua_ids, na_weights);
	//~ 
	//~ Rcpp::Rcout << "\n\n";
	//~ 
	//~ arma::uword N = 10;
	//~ arma::umat counts(dataTest.get_ref_count(),N);
	//~ for (arma::uword i = 0; i < N; i++) counts(arma::span::all, i) = dataTest.random_draw(rgen);
	//~ 
	//~ Rcpp::Rcout << "\nFirst Run: N = " << arma::conv_to<arma::urowvec>::from(arma::sum(counts)) << "\n";
	//~ for (unsigned i = 0; i < dataTest.get_ref_count(); i++) {
		//~ Rcpp::Rcout << "Ref " << std::setw(2) << i << " " << dataTest.get_genotype(i) << " counts: ";
		//~ for (unsigned j = 0; j < N; j++) Rcpp::Rcout << std::setw(4) << counts(i,j);
		//~ Rcpp::Rcout << "\n";
	//~ }
	//~ 
	//~ Rcpp::Rcout << "Now we test the sequence production ... \n"; Rcpp::Rcout.flush();
	//~ 
	//~ 
	//~ 
	//~ HMMsequenceProducer productionTest(dataTest, ua_graph, ua_Emission, randseed, a_percentage, 100000, 1000);
	//~ 
	//~ Rcpp::Rcout << "! ... \n"; Rcpp::Rcout.flush();
	//~ 
	//~ for (unsigned i = 0; i < 10; i++)
	//~ {
		//~ arma::umat transition_counts = arma::zeros<arma::umat>(ua_graph.n_rows, ua_graph.n_cols);
		//~ double amount;
		//~ arma::uword repeats = productionTest.simulate_transition_counts(na_transition, transition_counts, amount); 
		//~ Rcpp::Rcout << "Test: " << i << " with " <<  repeats << " repetitions for " << round(amount*100.0) << "% non-approximated sequences \n";
		//~ transition_counts.print();
	//~ }
	//~ 
	//~ END_RCPP
//~ }


//************************************************

RcppExport SEXP unitHMMGibbs(SEXP genotypes, SEXP individuals, SEXP weights, SEXP transition_matrix, SEXP emission_matrix, 
                             SEXP temperatures, SEXP percentage, SEXP burnin, SEXP mc, SEXP seed)
{
	BEGIN_RCPP
	
    using namespace Rcpp; 

    IntegerMatrix iGenotypes(genotypes);
    arma::imat ia_genotypes(iGenotypes.begin(), iGenotypes.rows(), iGenotypes.cols(), true);
    arma::umat ua_genotypes = arma::conv_to<arma::umat>::from(ia_genotypes);
    
    IntegerVector iIDs(individuals);
    arma::ivec ia_ids(iIDs.begin(), iIDs.size(), true);
    arma::uvec ua_ids = arma::conv_to<arma::uvec>::from(ia_ids);
    
    NumericVector iWeights(weights);
    arma::vec na_weights(iWeights.begin(), iWeights.size(), true);
    
    NumericVector iTemps(temperatures);
    arma::vec na_temps(iTemps.begin(), iTemps.size(), true);
    	
    arma::uword randseed = as<arma::uword>(seed);
    BasicTypes::base_generator_type	rgen(randseed);
    
    arma::uword imc = as<arma::uword>(mc),
                iburn = as<arma::uword>(burnin);
    
    double a_percentage = as<double>(percentage);
    
    NumericMatrix iTransitionMatrix(transition_matrix);
    arma::mat na_transition(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    arma::umat ua_graph = na_transition > 0.0;
    
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    	    	
	HMMdataSet dataTest(ua_genotypes, ua_ids, na_weights);
	
	Gibbs_Sampling Runner(dataTest, na_temps, ua_graph, ua_Emission, a_percentage, 100, 1000, randseed);
	std::cout << "\nRun!!!\n"; std::cout.flush();
	arma::mat runResult = Runner.run(iburn, imc);
	
	return wrap(runResult);
	END_RCPP
}
