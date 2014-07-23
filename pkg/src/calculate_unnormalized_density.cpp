#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/special_functions/gamma.hpp>


#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"


//********************

double transitionDensity(const arma::mat& transition_probs, const arma::mat& transition_counts)
{
	using namespace arma;
	double result = 0.0;
	
	for (uword zeile = 0; zeile < transition_counts.n_rows-1; zeile++) 
	{
		double zeilensumme = 0.0;
		
		for (uword spalte = 0; spalte < transition_counts.n_cols; spalte++)
			if (transition_probs(zeile,spalte) > 0.0)
		{
			zeilensumme += transition_counts(zeile,spalte);
			result += log(transition_probs(zeile,spalte)) * (transition_counts(zeile,spalte) - 1.0) 
			        - boost::math::lgamma( transition_counts(zeile,spalte) );
		}
		result += boost::math::lgamma(zeilensumme);
	}
	
	return result;
}



// outsourced functions to generate the Markov sequences

arma::uword select_state(const arma::rowvec& row_parameters, BasicTypes::base_generator_type& rand_gen)
{
	boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_real_distribution<> > test_randoms(rand_gen, test_dist);
	
	double rnum = test_randoms(), summe = row_parameters[0];
	arma::uword index = 0;
	
	while (index < row_parameters.n_elem && rnum >= summe) 
	{
		index++;
		summe += row_parameters[index];
	}
	
	return index;
}

// This is as simple copy of that one in HMMsequenceProducer. I did not want to initialize the whole thing 

BasicTypes::IDRefSequenceCountTuple produce_random_sequence(
  const arma::mat& transition_probabilities, 
  const arma::umat& emission_matrix,
  HMMdataSet& observed_data,  
  BasicTypes::base_generator_type& rand_gen,
  const arma::uword MAXIMUM_SEQUENCE_LENGTH
)
{
	using namespace arma;
	urowvec sequence(1);
	sequence[0] = 0;
	
	urowvec sim_genotype = zeros<urowvec>(emission_matrix.n_cols);
	umat transitSave = zeros<umat>(transition_probabilities.n_rows, transition_probabilities.n_cols);
	uword refGenotype = 0;
	bool validity = false;
	
	uword j = 0, state = 0;
	bool ran_twice = false;
	while (state != transition_probabilities.n_rows-1 && j < MAXIMUM_SEQUENCE_LENGTH) 
	{
		// calculate new state
		uword newstate = select_state(transition_probabilities(state, span::all),rand_gen);
		state = newstate;
		// add emission to resulting genotype
	    sim_genotype = sim_genotype + emission_matrix(state,span::all);
	    
	    if (!ran_twice && state==transition_probabilities.n_rows-1)   // the HMM is simulated twice!
	    {
			state = 0;
			ran_twice = true;
		}
	    j++;
	}
    
    	
	if (j == MAXIMUM_SEQUENCE_LENGTH-1 && state != transition_probabilities.n_rows-1) validity = false;
		
	BasicTypes::IDRefSequenceCountTuple result(0, 0, sequence.subvec(0, j), transitSave, validity);
	if (validity) observed_data.get_ref(result, sim_genotype);
		
	return result;    
}

 
RcppExport SEXP HMMunnormalizedDensity(
              SEXP genotypes, 
              SEXP transition_matrix, SEXP emission_matrix, 
              SEXP count, SEXP random_seed, SEXP max_seq_len)
{
	BEGIN_RCPP
	using namespace Rcpp;
	
	// Take up values
	IntegerMatrix iGenotypes(genotypes);
    arma::imat ia_genotypes(iGenotypes.begin(), iGenotypes.rows(), iGenotypes.cols(), true);
    arma::umat ua_genotypes = arma::conv_to<arma::umat>::from(ia_genotypes);

	// And initialize a data object
	HMMdataSet dataTest(ua_genotypes);
		
	// take up the rest of values
	arma::uword randseed = as<arma::uword>(random_seed);
	arma::uword n_sim_count = as<arma::uword>(count);
	arma::uword MAXLEN = as<arma::uword>(max_seq_len);
    BasicTypes::base_generator_type	rgen(randseed);
	
	NumericMatrix ntransition_matrix(transition_matrix); 
	arma::mat na_transitions(ntransition_matrix.begin(), ntransition_matrix.rows(), ntransition_matrix.cols(), true);
	
	arma::mat prior_transitions = arma::conv_to<arma::mat>::from(na_transitions>0)*0.5;
		
	IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
	
	// now ... build a counting system to approximate the probabilities
	arma::vec probabilities = 0.5 + arma::zeros<arma::vec>(dataTest.get_ref_count());
	
	for (arma::uword i = 0; i < n_sim_count; i++)
	{
		BasicTypes::IDRefSequenceCountTuple seq;
	    seq = produce_random_sequence(na_transitions, ua_Emission, dataTest, rgen, MAXLEN);	
		
		if (seq.get<4>()) probabilities[seq.get<1>()] += 1.0;	    
	}
	
	probabilities = probabilities / (double(n_sim_count) + 0.5 * double(1 + dataTest.get_ref_count()));
	
	// and calculate the likelihood
	double likelihood = dataTest.calculate_likelihood(probabilities);
	
	// as well as the prior density
	double priordensity = transitionDensity(na_transitions, prior_transitions);
	
	// include controls +++++
	arma::umat genos(dataTest.get_ref_count(), ua_Emission.n_cols);
	for (arma::uword i = 0; i < dataTest.get_ref_count(); i++) genos(i,arma::span::all) = dataTest.get_genotype(i);
		
	// Now wrap the result nicely
	List result = List::create ( _("Unnormalized.Density") = likelihood + priordensity, _("Likelihood") = likelihood, 
	                             _("Prior.Density") = priordensity,
	                             _("Unique.genotypes") = wrap(genos), _("Probabilities") = probabilities);
	
	return result;
	
	END_RCPP
} 
