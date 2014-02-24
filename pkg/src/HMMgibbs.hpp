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

#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMtransitionMatrix.hpp"
#include "HMMchib.hpp"

class Gibbs_Sampling
{
	
	HMMtransitionMatrix TransitionInstance;
	HMMsequenceProducer SequencerInstance;
	HMMchib chib_ML_estimation;
	bool samplingOrderImproved;	
	
	arma::mat jumping_probs_statistics;
	
	public:
	
	Gibbs_Sampling(const HMMdataSet& observed_data,	const arma::vec& init_lambda, 
	const arma::umat& init_transition_graph, const arma::umat& init_emission_matrix,
	arma::uword max_sequence_length,  
	arma::uword how_many_sequence_tries = 50, 
	arma::uword path_sampling_repetitions = 1,
	arma::uword internal_sampling = 1,
	arma::uword n_swappings = 1, 
	bool use_collapsed_sampler = false,
	arma::uword rand_seed = 42);
	
	arma::cube run(arma::uword burnin, arma::uword mc);
	
	arma::mat get_temperature_likelihoods();
	
	arma::rowvec get_Chib_marginal_likelihood(const arma::rowvec& transition_matrix_sample);
	
	double get_naive_marginal_likelihood();
	
};
