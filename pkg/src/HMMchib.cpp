/*
 * HMMchib.cpp
 * 
 * Copyright 2013 Andreas Recke <andreas@Dianeira>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */


#include "HMMchib.hpp"

/*
 * Constructor - initializes the collection of augmented data samples
 * name: HMMchib
 * @param: randseed - random seed
 * @return: nothing
 */

HMMchib::HMMchib(arma::uword randseed) : common_rgen(randseed) {}


/*
 * This function save intermediate sampling results for the counting matrix
 * name: save_transition_counts
 * @param: generator ... just to get the counting matrix
 * @return: nothing
 */
void HMMchib::save_transition_counts(HMMsequenceProducer& generator)
{
	using namespace arma;
	
	umat original = generator.get_transition_instance().get_transition_counts();
	mat pb = arma::conv_to<mat>::from(original);
	
	transition_counts_vector.push_back(pb);
}


/*
 * 
 * name: calculate_marginal_likelihood
 * @param
 * @return
 * 
 */
arma::rowvec  HMMchib::calculate_marginal_likelihood(HMMsequenceProducer& generator, const arma::mat& transition_matrix)
{
	using namespace arma;
	
	// Chib principle for two blocks Gibbs samplers
	
	// 1.: calculate likelihood function
	double log_likelihood = generator.get_observations_instance().calculate_likelihood(simulate_frequencies(generator, transition_matrix));
	
	// 2.: calculate density of transition matrix 
	vec transitionDensities = zeros<vec>(transition_counts_vector.size());
	double posterior_density = 0.0;
	
	transition_counts_vector_type::const_iterator viter = transition_counts_vector.begin();
	vec::iterator tditer = transitionDensities.begin();
	for (; viter != transition_counts_vector.end(); tditer++, viter++) *tditer = generator.get_transition_instance().transition_matrix_density(transition_matrix, *viter);

    // sum up log values	
	double maxval = transitionDensities.max();
	posterior_density = log(accu(exp(transitionDensities - maxval))) - log(double(transitionDensities.n_elem)) + maxval;
	
	// 3.: calculate prior density
	double priordensity = 
	  generator.get_transition_instance().transition_matrix_density(transition_matrix, zeros<mat>(transition_matrix.n_rows, transition_matrix.n_cols));
	
	//~ transition_matrix.print("TM=");
	
	
	// 4.: with all components, calculate the Chib marginal likelihood
	//~ Rcpp::Rcout << "\nComponents: " <<
	//~ " log_likelihood = " << log_likelihood << "\t" <<
	//~ " priordensity = " << priordensity << "\t" <<
	//~ " posterior_density = " << posterior_density << "\n";
	//~ Rcpp::Rcout << "Chib = " << log_likelihood + priordensity - posterior_density << "\n";
	
	
	rowvec result;
	result << (log_likelihood + priordensity - posterior_density) << log_likelihood << priordensity << posterior_density;
	
	return result;
}


/*
 * 
 * name: simulate_frequencies
 * @param
 * @return
 * 
 */
arma::vec HMMchib::simulate_frequencies(HMMsequenceProducer& generator, const arma::mat& transition_matrix) 
{
	using namespace arma;
	// using a 1000 times overestimation
	
	vec probabilities = 0.5 + zeros<vec>(generator.get_observations_instance().get_ref_count());
	arma::uword event_sim = 1000 * generator.get_observations_instance().n_individuals();
	
	for (uword i = 0; i < event_sim; i++)
	{
		BasicTypes::SequenceReferenceTuple seq;
	    seq = generator.produce_random_sequence(transition_matrix, common_rgen);	
		
		if (seq.get<3>()) probabilities[seq.get<1>()] += 1.0;	    
	}
	
	probabilities = probabilities / (double(event_sim) + 0.5 * double(1 + generator.get_observations_instance().get_ref_count()));
		
	return probabilities;	
}
