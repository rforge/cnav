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

class Gibbs_Sampling
{
	HMMsequenceProducer SequencerInstance;
	
	public:
	
	Gibbs_Sampling(const HMMdataSet& observed_data,	const arma::vec& init_lambda, 
	const arma::umat& init_transition_graph, const arma::umat& init_emission_matrix,
	double amount, arma::uword preparation, arma::uword max_sequence_length,  
	arma::uword how_many_sequence_tries = 30000, 
	arma::uword rand_seed = 42);
	
	arma::mat run(arma::uword burnin, arma::uword mc);
	
	arma::vec get_temperature_probabilities();
	arma::vec get_kullback_divergence();

};
