/*
 * HMMtransitionMatrix.hpp
 * 
 * Copyright 2013 Andreas Recke <andreas@Persephone>
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

#include "BasicTypes.hpp"
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/scoped_array.hpp>

class HMMtransitionMatrix 
{
	arma::uword temperature, endstate;
	arma::vec lambda_set;
	double prior;
	
	boost::scoped_array< arma::running_stat<double> > information_statistics;	
	boost::scoped_array< arma::running_stat<double> > jump_statistics;	
	
	arma::umat transition_graph;
	arma::umat emission_matrix;
	arma::umat transition_counts;
	arma::mat transition_matrix;
	BasicTypes::base_generator_type rng_engine;
	
	double log_dirichlet_density(const arma::rowvec& x, const arma::rowvec& alpha);	
	double likelihood(double r_temp);
		
	public:
		
	HMMtransitionMatrix(arma::vec init_lambda, const arma::umat& init_transition_graph, 
	                    const arma::umat& init_emission_matrix, arma::uword rand_seed, 
	                    double i_prior = 0.5);  // setting the alpha prior
	
	HMMtransitionMatrix(const HMMtransitionMatrix& obj);
	
	void set_transition_counts(const arma::umat& new_transition_counts);
	
	void random_matrix();       // draws new matrix
	void random_temperature();  // changes temperature
	
	arma::mat get_transition_matrix();
	const arma::umat& get_emission_matrix() const;
	arma::uword get_temperature();
	
	arma::uword n_states();

	arma::rowvec get_parameters();
	arma::uword get_endstate() const;
	
	const arma::umat& get_transition_graph() const;
	
	void printDKL();
	arma::vec get_temperature_probabilities();
	arma::vec get_kullback_divergence();
};
