#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMgibbs.hpp"

#include <boost/lexical_cast.hpp>

RcppExport SEXP HMMinterface(SEXP genotypes, SEXP individuals, SEXP weights, SEXP transition_matrix, SEXP emission_matrix, 
                             SEXP temperatures, SEXP percentage, SEXP r_how_many_sequence_tries, 
                             SEXP r_preparation, SEXP r_maxsequence_length,
                             SEXP exact, SEXP collect, SEXP betterSamplingOrder,
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
    bool exact_sampling = as<bool>(exact);
    bool collect_during_sampling = as<bool>(collect);
    bool improvedSampling = as<bool>(betterSamplingOrder);
    
    IntegerMatrix iTransitionMatrix(transition_matrix);
    arma::imat ia_graph(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    arma::umat ua_graph = arma::conv_to<arma::umat>::from(ia_graph);
    
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    	    	
	HMMdataSet dataTest(ua_genotypes, ua_ids, na_weights);
	
	Gibbs_Sampling Runner(dataTest, na_temps, ua_graph, ua_Emission, a_percentage, preparation, maxseqlength, 
	                        how_many_sequence_tries, exact_sampling, collect_during_sampling, improvedSampling, randseed);
	arma::mat runResult = Runner.run(iburn, imc);
	
	arma::uvec res_temperature_indices = arma::conv_to<arma::uvec>::from(runResult(arma::span::all,3));
	arma::uvec number_of_sequence_generation_repeats = arma::conv_to<arma::uvec>::from(runResult(arma::span::all,0));
	arma::vec amount_of_unbiasedly_simulated_sequences = runResult(arma::span::all,1);
	arma::vec jumpingProbs = runResult(arma::span::all,2);
	arma::mat mc_samples_transition_matrix = runResult(arma::span::all, arma::span(4,runResult.n_cols-1));
	
	// just calculate some marginal likelihoods
	arma::uword number_of_samples = 1 + imc / na_temps.n_elem / 10;
	arma::mat marlik = arma::zeros<arma::mat>(number_of_samples, 4);
	arma::uvec indexlist = arma::shuffle(arma::linspace<arma::uvec>(0,imc-1,imc));
	arma::uword i = 0, j = 0;
	bool finito = false;
	Rcpp::Rcout << "\nCalculating marginal likelihood\n>";
	arma::mat marginal_calculation_points = arma::zeros<arma::mat>(number_of_samples, mc_samples_transition_matrix.n_cols);
	while (!finito) 
	{
		while (i < imc && res_temperature_indices[i] != 0) i++;
		if (i < imc) {
			marlik(j, arma::span::all) = Runner.get_Chib_marginal_likelihood(mc_samples_transition_matrix(i, arma::span::all));
			marginal_calculation_points(j,arma::span::all) = mc_samples_transition_matrix(i, arma::span::all);
		}
		i++; j++;
		finito = i >= imc || j >= number_of_samples;
		Rcpp::Rcout << "."; Rcpp::Rcout.flush();
	}
	Rcpp::Rcout << "<\n";
	
	// Don't know how to do it better
	NumericMatrix chibResult(marlik.n_rows, marlik.n_cols);
	for (unsigned a = 0; a < marlik.n_rows; a++) for (unsigned b = 0; b < marlik.n_cols; b++) chibResult(a,b) = marlik(a,b);
	
	CharacterVector cvec(number_of_samples);
	for (unsigned icv = 0; icv < number_of_samples; icv++) {
		cvec[icv] = boost::lexical_cast<std::string>(icv);
	}
	
	List dimnms = List::create(
	  cvec, CharacterVector::create("Chib.Marginal.Likelihood", "Point.Likelihood", "Point.Prior.Density","Point.Posterior.Density"));
	chibResult.attr("dimnames") = dimnms;
	
	Rcpp::Rcout << "\nCalculation naive likelihood for comparison\n";
	
	arma::vec naiveMarlik = Runner.get_naive_marginal_likelihood(1000);
	arma::uword approximate_realizations = Runner.get_number_of_prepared_realizations();
		
	List HMMresult = List::create( _("temperature.indices") = wrap(res_temperature_indices),
	                               _("number.of.sequence.generation.repeats") = wrap(number_of_sequence_generation_repeats),
	                               _("amount.of.unbiasedly.simulated.sequences") = wrap(amount_of_unbiasedly_simulated_sequences ),
	                               _("mc.samples.transition.matrix") = wrap(mc_samples_transition_matrix),
	                               _("mean.temperature.jumping.probabilities") = wrap(Runner.get_temperature_probabilities()),
	                               _("kullback.leibler.divergences") = wrap(Runner.get_kullback_divergence()),
	                               _("chib.marginal.likelihoods") = chibResult,
	                               _("chib.estimation.points") = wrap(marginal_calculation_points),
	                               _("naive.dirichlet.marginal.likelihoods") = wrap(naiveMarlik),
	                               _("setsize.of.approximation.sequences") = approximate_realizations,
	                               _("transition.graph") = wrap(ua_graph),
	                               _("emission.matrix") = wrap(ua_Emission),
	                               _("n.samples") = imc,
	                               _("jumping.probabilities") = wrap(jumpingProbs));
	                            
	
	return HMMresult;
	END_RCPP
}
 
