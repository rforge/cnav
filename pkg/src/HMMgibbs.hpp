/*
 * HMMgibbs.hpp
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

#pragma once
#include <iostream>

#include <boost/container/vector.hpp>
#include <boost/scoped_array.hpp> 
#include <boost/range/algorithm.hpp>

#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"
#include "HMMparticle.hpp"
#include "HMMkernelwrapper.hpp"
#include "HMMchib.hpp"

//~ #include "HMMchib.hpp"

class Gibbs_Sampling
{
	boost::container::vector<HMMparticle> sampling_particles;	
	arma::uword parameter_count;
	arma::vec lambda_levels;
	
	arma::mat level_likelihood_trace;
	arma::umat temperature_trace;
	arma::cube hash_trace;
	arma::cube sequences_likelihood_trace;
	
	arma::uword gibbs_sequence_tries;
	arma::uword saved_rand_seed;
	arma::uword n_swapping_tries;
	arma::uword multiple_tries;
	
	HMMchib chib_ML_estimation;
	
	public:
	
	Gibbs_Sampling(
	    HMMdataSet& observed_data,	
	    const arma::vec& init_lambda, 
		const arma::mat& transition_matrix_prior, 
		const arma::umat& init_emission_matrix,
		arma::uword max_sequence_length = 1000,  
		arma::uword how_many_sequence_tries = 50, 
		arma::uword internal_sampling = 1 ,
		arma::uword n_swappings = 1,
		arma::uword rand_seed = 42); 
	
	arma::cube run(arma::uword burnin, arma::uword mc);
	
	arma::mat get_temperature_likelihoods();
	arma::imat get_temperature_trace();
	arma::cube get_hash_trace();
	arma::cube get_sequences_likelihood_trace();
	
	arma::rowvec get_Chib_marginal_likelihood(const arma::rowvec& transition_matrix_sample);
	
	double get_naive_marginal_likelihood();
	
};
