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
 
#include <iostream>
#define ARMA_DONT_USE_BLAS
#include <Rcpp.h>
#include <RcppArmadillo.h>

#include "BasicTypes.hpp"
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/scoped_array.hpp>

class HMMtransitionMatrix 
{
	arma::uword temperature, endstate;
	arma::vec lambda_set;
	
	
	double prior;
	
	boost::scoped_array< arma::running_stat<double> > information_statistics;	
	boost::scoped_array< arma::running_stat<double> > jump_statistics;	
	
	arma::vec constants_updated_sum;
	arma::uword constants_updated_count;
	
	arma::umat transition_graph;
	arma::umat emission_matrix;
	arma::umat transition_counts;
	arma::mat transition_matrix;
	BasicTypes::base_generator_type rng_engine;
	
	double log_dirichlet_density(const arma::rowvec& x, const arma::rowvec& alpha);	
	double likelihood(double r_temp);
	
	arma::mat denominator, numerator; // the two matrices that reflect the values for constants
	arma::umat w_counter;
	arma::vec mean_normalization_constants;
	
	// internal class for normalization constants, includes ability for bootstrapping
	
	class normalization_class
	{
		arma::uword collected;
		arma::uvec weight_ref_columns;
		arma::mat collection;
		BasicTypes::base_generator_type boot_rng_engine;
		bool reoptimized;
		arma::rowvec normalized_constants;
				
		double accu_log_values(arma::vec logs);
		arma::uvec bootstrap_draw();
		
		void exp_max_constants(const arma::uvec& indices, arma::uword depth=5);
				
		public:
		
		normalization_class(arma::uword n_lambda_levels, arma::uword rand_seed);
		normalization_class(const normalization_class& obj);
		
		void save_weights(arma::rowvec weights_each_level, arma::uword from_level);
		
		double get_weight_normalization(arma::uword level);
		arma::mat export_data();
	};
	
	normalization_class NormConstantsWrapper;
	double normalization_constant(double r_temp);
	void add_realization_to_constants(arma::uword drawn_at_temp);	
	
	public:
		
	HMMtransitionMatrix(arma::vec init_lambda, 
	                    const arma::umat& init_transition_graph, 
	                    const arma::umat& init_emission_matrix, arma::uword rand_seed, 
	                    double i_prior = 0.5);  // setting the alpha prior
	
	HMMtransitionMatrix(const HMMtransitionMatrix& obj);
	
	void set_transition_counts(const arma::umat& new_transition_counts);
	
	void random_matrix();       // draws new matrix
	double random_temperature();  // changes temperature
	
	arma::mat get_transition_matrix();
	const arma::umat& get_emission_matrix() const;
	arma::uword get_temperature();
	void set_temperature(arma::uword new_temp);
	arma::uword n_temperatures();
	
	
	double get_sequence_likelihood(const arma::umat& sequence_as_matrix, bool full_posterior);
	
	arma::uword n_states();

	arma::rowvec get_parameters();
	arma::uword get_endstate() const;
	
	const arma::umat& get_transition_graph() const;
	
	const arma::umat& get_transition_counts() const;
	
	void printDKL();
	arma::vec get_temperature_probabilities();
	arma::vec get_kullback_divergence();
	arma::vec get_normalization_constants();
	
	double transition_matrix_density(const arma::mat& c_transition_matrix, const arma::mat& c_count_matrix);
	
	void save_realization_for_constants();
	
	arma::mat export_constant_data();
};
