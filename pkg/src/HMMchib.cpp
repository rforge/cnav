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
void HMMchib::save_transition_counts(HMMtransitionMatrix& matrix_instance)
{
	using namespace arma;
	
	uword i = 0;
	while (matrix_instance.get_current_particle_temperatures()[i] != 0) ++i;
	mat pb = arma::conv_to<mat>::from(matrix_instance.get_transition_counts(i));
	
	transition_counts_vector.push_back(pb);
}


/*
 * 
 * name: transitionDensity
 * @param: transition_probs - transition probabilities
 * @param: transition_counts - transition counts = Dirichlet parameters
 * @param: prior - Dirichlet prior
 * @return: the density - product of a Dirichlet distribution for each state
 * 
 */
double HMMchib::transitionDensity(const arma::mat& transition_probs, const arma::mat& transition_counts, double prior)
{
	using namespace arma;
	double result = 0.0;
	
	for (uword zeile = 0; zeile < transition_counts.n_rows-1; zeile++) 
	{
		double zeilensumme = 0.0;
		
		for (uword spalte = 0; spalte < transition_counts.n_cols; spalte++)
			if (transition_probs(zeile,spalte) > 0.0)
		{
			zeilensumme += transition_counts(zeile,spalte) + prior;
			result += log(transition_probs(zeile,spalte)) * (transition_counts(zeile,spalte) + prior - 1.0) 
			        - boost::math::lgamma( transition_counts(zeile,spalte) + prior );
		}
		result += boost::math::lgamma(zeilensumme);
	}
	
	return result;
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
	for (; viter != transition_counts_vector.end(); tditer++, viter++) 
		*tditer = transitionDensity(transition_matrix, *viter, generator.get_transition_instance().get_prior());

    // sum up log values	
	double maxval = transitionDensities.max();
	posterior_density = log(accu(exp(transitionDensities - maxval))) - log(double(transitionDensities.n_elem)) + maxval;
	
	// 3.: calculate prior density
	double priordensity = 
	  transitionDensity(transition_matrix, zeros<mat>(transition_matrix.n_rows, transition_matrix.n_cols), generator.get_transition_instance().get_prior());
	
		
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
