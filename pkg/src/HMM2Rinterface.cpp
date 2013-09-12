#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMgibbs.hpp"

RcppExport SEXP HMMinterface(SEXP genotypes, SEXP individuals, SEXP weights, SEXP transition_matrix, SEXP emission_matrix, 
                             SEXP temperatures, SEXP percentage, SEXP r_how_many_sequence_tries, 
                             SEXP r_preparation, SEXP r_maxsequence_length,
                             SEXP burnin, SEXP mc, SEXP seed)
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
                iburn = as<arma::uword>(burnin),
                how_many_sequence_tries = as<arma::uword>(r_how_many_sequence_tries),
                preparation = as<arma::uword>(r_preparation),
                maxseqlength = as<arma::uword>(r_maxsequence_length);
    
    double a_percentage = as<double>(percentage);
    
    IntegerMatrix iTransitionMatrix(transition_matrix);
    arma::imat ia_graph(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    arma::umat ua_graph = arma::conv_to<arma::umat>::from(ia_graph);
    
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    	    	
	HMMdataSet dataTest(ua_genotypes, ua_ids, na_weights);
	
	Gibbs_Sampling Runner(dataTest, na_temps, ua_graph, ua_Emission, a_percentage, preparation, maxseqlength, how_many_sequence_tries, randseed);
	arma::mat runResult = Runner.run(iburn, imc);
	
	arma::uvec res_temperature_indices = arma::conv_to<arma::uvec>::from(runResult(arma::span::all,2));
	arma::uvec number_of_sequence_generation_repeats = arma::conv_to<arma::uvec>::from(runResult(arma::span::all,0));
	arma::vec amount_of_unbiasedly_simulated_sequences = runResult(arma::span::all,1);
	arma::mat mc_samples_transition_matrix = runResult(arma::span::all, arma::span(3,runResult.n_cols-1));
	
	List HMMresult = List::create( _("temperature.indices") = wrap(res_temperature_indices),
	                               _("number.of.sequence.generation.repeats") = wrap(number_of_sequence_generation_repeats),
	                               _("amount.of.unbiasedly.simulated.sequences") = wrap(amount_of_unbiasedly_simulated_sequences ),
	                               _("mc.samples.transition.matrix") = wrap(mc_samples_transition_matrix),
	                               _("mean.temperature.jumping.probabilities") = wrap(Runner.get_temperature_probabilities()),
	                               _("kullback.leibler.divergences") = wrap(Runner.get_kullback_divergence()) );
	                            
	
	return HMMresult;
	END_RCPP
}
 
