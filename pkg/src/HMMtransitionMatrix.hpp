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
	arma::vec lambda_set;
	double multinomial_coef;
	
	double prior;
	
	arma::umat transition_graph;
	arma::umat emission_matrix;
	arma::ucube transition_counts_cube;
	arma::cube transition_matrix_cube;
	
	arma::urowvec current_particle_temperature;
	// row is the particle, col is the likelihood at that temperature 
	arma::mat current_particle_likelihoods;
	
	arma::uword n_swapping_tries;
	boost::scoped_array< arma::running_stat<double> > jump_statistics;	
	
	BasicTypes::base_generator_type rng_engine;
	
	//~ double log_dirichlet_density(const arma::rowvec& x, const arma::rowvec& alpha);	
	//~ double likelihood(double r_temp);
	//~ double new_likelihood(double r_temp);  // calculates the likelihood with the new formula
	arma::rowvec likelihood_vector(arma::uword particle); // for the latest revision 01/11/2014 
	
	class lgamma_functor_class
	{
		double density;
		arma::vec lgamma_preserved;
		public:
		
		lgamma_functor_class();
		
		double operator()(double value);
		
	};
	
	lgamma_functor_class loggamma;
	
	public:
		
	HMMtransitionMatrix(arma::vec init_lambda, 
	                    const arma::umat& init_transition_graph, 
	                    const arma::umat& init_emission_matrix, 
	                    arma::uword rand_seed, 
	                    arma::uword n_swaps = 10,
	                    double i_prior = 0.5,
	                    arma::uword open_cycles = 20);  // setting the alpha prior
	
	HMMtransitionMatrix(const HMMtransitionMatrix& obj);
	
	void set_transition_counts(const arma::ucube& new_transition_counts);  // revised 12.1.2014
	void set_transition_counts(arma::uword particle, const arma::umat& new_counts); // revised 18.1.2014
	
	void set_multinomial_coefficient(double value);    // revised 12.1.2014
	
	void random_matrices();       // draws new matrix    // revised 12.1.2014
	
	// double random_temperature();  // changes temperature
	
	const arma::urowvec& get_current_particle_temperatures() const; // revised 18.1.2014
	arma::vec particle_swapping();// revised 18.1.2014
	
	const arma::umat& get_transition_counts(arma::uword particle) const; // revised 21.1.2014
	const arma::mat& get_transition_matrix(arma::uword particle) const; // revised 18.1.2014
	const arma::umat& get_emission_matrix() const;
	const double get_prior() const;
	//~  arma::uword get_temperature(arma::uword particle);
	//~ void set_temperature(arma::uword new_temp);
	arma::uword n_temperatures();
	
	const double get_particle_temperature(arma::uword particle) const;
	
	bool is_particle_at_zero_lambda(arma::uword particle);
	arma::uword get_particle_at_next_temperature(arma::uword particle);
	//~ 
	//~ void provide_marginal_likelihood(double marlik);
	//~ double collapsed_likelihood_difference(const arma::umat& old_counts, const arma::umat& new_counts);
	//~ 
	//~ double get_sequence_likelihood(const arma::umat& sequence_as_matrix, bool full_posterior);
	
	arma::uword n_states();

	arma::mat get_all_transition_matrices(); // revised 18.1.2014
	arma::uword get_endstate() const; // revised 18.1.2014
	
	double collapsed_likelihood_difference(const arma::umat& old_counts, const arma::umat& new_counts, arma::uword particle_number); // revised 18.1.2014
	double likelihood(const arma::umat& counts, arma::uword particle);
	
	const arma::umat& get_transition_graph() const; // revised 18.1.2014
	
	//~ const arma::umat& get_transition_counts() const;
	
	//~ void printDKL();
	//~ arma::vec get_temperature_probabilities();
	//~ arma::vec get_kullback_divergence();
	
	//~ arma::vec get_normalization_constants();
	
	double transition_matrix_density(const arma::mat& c_transition_matrix, const arma::mat& c_count_matrix);
	
	//~ void save_realization_for_constants();
	
	//~ void set_open_cycle(arma::uword Ncycles);
	
	//~ arma::mat export_constant_data();
};
