/*
 * HMMchib.hpp
 * 
 * Copyright 2014 Andreas Recke <andreas@Persephone>
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
 * 
 * 
 */
 

#pragma once
#include <iostream>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>

#include "BasicTypes.hpp"

class HMMchib 
{	
    BasicTypes::base_generator_type common_rgen;
    typedef boost::container::vector<arma::mat> transition_counts_vector_type;  
    transition_counts_vector_type transition_counts_vector;
    
    typedef HMMparticle::IDRefSequenceCountTuple IDRefSequenceCountTuple;
    
    arma::umat emission_matrix;
    arma::mat transition_matrix_prior;
    arma::uword MAXDEPTH;
        
    
    IDRefSequenceCountTuple produce_random_sequence(const arma::mat& transition_matrix)
	{
		using namespace arma;
		
		urowvec sequence(MAXDEPTH);
		sequence[0] = 0;
		
		urowvec sim_genotype = zeros<urowvec>(emission_matrix.n_cols);
		umat transitSave = zeros<umat>(transition_matrix.n_rows, transition_matrix.n_cols);
		uword refGenotype = 0;
		uword endstate = transition_matrix.n_rows-1;
		bool validity = true;
		
		uword j = 0, state = 0;
		bool ran_twice = false;
		while (state != endstate && j < MAXDEPTH - 2) 
		{
			// calculate new state
			rowvec transition_probabilities = transition_matrix(state, span::all);
			boost::random::discrete_distribution<> diskrete(transition_probabilities.begin(), transition_probabilities.end());
			boost::variate_generator<boost::random::mt19937&, boost::random::discrete_distribution<> > diskrete_randoms(common_rgen, diskrete); 	
			uword newstate = diskrete_randoms();
			
			++j;
			sequence[j] = newstate;
			transitSave(state, newstate) = transitSave(state, newstate) + 1;
		    state = newstate;
			// add emission to resulting genotype
		    sim_genotype = sim_genotype + emission_matrix(state,span::all);
		    
		    if (!ran_twice && state == endstate)   // the HMM is simulated twice!
		    {
				++j;
				sequence[j] = 0;
				state = 0;
				ran_twice = true;
			}
			
		}
		
		if (j == MAXDEPTH-1 && state != endstate) validity = false;
		
		IDRefSequenceCountTuple result(0, 0, sequence.subvec(0, j), transitSave, validity);
		if (validity) data_set.get_ref(result, sim_genotype);
		
		return result;
	}	
	
    //***************************************************************************************************************************    
        
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
				zeilensumme += transition_counts(zeile,spalte) + transition_matrix_prior(zeile,spalte);
				result += log(transition_probs(zeile,spalte)) * (transition_counts(zeile,spalte) +  transition_matrix_prior(zeile,spalte) - 1.0) 
				        - boost::math::lgamma( transition_counts(zeile,spalte) +  transition_matrix_prior(zeile,spalte) );
			}
			result += boost::math::lgamma(zeilensumme);
		}
		
		return result;
	}
    
    //***************************************************************************************************************************    
    /*
	 * 
	 * name: simulate_frequencies
	 * @param
	 * @return
	 * 
	*/
	arma::vec simulate_frequencies(const arma::mat& transition_matrix) 
	{
		using namespace arma;
		// using a 1000 times overestimation
		
		vec probabilities = 0.5 + zeros<vec>(data_set.get_ref_count());
		arma::uword event_sim = 1000 * data_set.n_individuals();
		
		for (uword i = 0; i < event_sim; i++)
		{
			IDRefSequenceCountTuple seq;
		    seq = produce_random_sequence(transition_matrix);	
			
			if (seq.get<4>()) probabilities[seq.get<1>()] += 1.0;	    
		}
		
		probabilities = probabilities / (double(event_sim) + 0.5 * double(1 + data_set.get_ref_count()));
			
		return probabilities;	
	}
    
    //***************************************************************************************************************************    
    HMMdataSet& data_set;    

    //***************************************************************************************************************************    
    public:
	
	HMMchib(arma::uword randseed, const arma::umat& my_emission_matrix,
	         const arma::mat& my_transition_prior, arma::uword myMAXDEPTH, HMMdataSet& my_data_set ) :
	  data_set(my_data_set),
	  common_rgen(randseed),
	  emission_matrix(my_emission_matrix),
	  transition_matrix_prior(my_transition_prior),
	  MAXDEPTH(myMAXDEPTH)
	{
	}
	
	//***************************************************************************************************************************    
	void save_transition_counts(const arma::umat& count_matrix)
	{
		transition_counts_vector.push_back(arma::conv_to<arma::mat>::from(count_matrix));
	}
	
	//***************************************************************************************************************************    		
	arma::rowvec calculate_marginal_likelihood(const arma::mat& transition_matrix)
	{
		using namespace arma;
		
		// Chib principle for two blocks Gibbs samplers
		// 1.: calculate likelihood function
		double log_likelihood = data_set.calculate_likelihood(simulate_frequencies(transition_matrix));
		
		// 2.: calculate density of transition matrix 
		vec transitionDensities = zeros<vec>(transition_counts_vector.size());
		double posterior_density = 0.0;
		
		transition_counts_vector_type::const_iterator viter = transition_counts_vector.begin();
		vec::iterator tditer = transitionDensities.begin();
		for (; viter != transition_counts_vector.end(); tditer++, viter++) 
			*tditer = transitionDensity(transition_matrix, *viter);
	
	    // sum up log values	
		double maxval = transitionDensities.max();
		posterior_density = log(accu(exp(transitionDensities - maxval))) - log(double(transitionDensities.n_elem)) + maxval;
		
		// 3.: calculate prior density
		double priordensity = 
		  transitionDensity(transition_matrix, zeros<mat>(transition_matrix.n_rows, transition_matrix.n_cols));
					
		rowvec result;
		result << (log_likelihood + priordensity - posterior_density) << log_likelihood << priordensity << posterior_density;
		
		return result;
	}
};
