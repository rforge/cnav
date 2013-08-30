/*
 * HMMtransitionMatrix.cpp
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
	
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>	
#include <boost/math/special_functions/gamma.hpp>

#include <iostream>
#include <algorithm>

#include "BasicTypes.hpp"
#include "HMMtransitionMatrix.hpp"
	
HMMtransitionMatrix::HMMtransitionMatrix(arma::vec init_lambda, const arma::umat& init_transition_graph, const arma::umat& init_emission_matrix, arma::uword rand_seed, double i_prior) :
  rng_engine(rand_seed),
  lambda_set(init_lambda),
  transition_graph(init_transition_graph),
  emission_matrix(init_emission_matrix),
  prior(i_prior)
{
	information_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem]);
	jump_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem-1]);
	
	for (unsigned i=0; i < lambda_set.n_elem; i++) information_statistics[i].reset();
	for (unsigned i=0; i < lambda_set.n_elem-1; i++) jump_statistics[i].reset();
	
	endstate = init_transition_graph.n_rows - 1;
	temperature = 0;
	transition_counts = arma::zeros<arma::umat>(init_transition_graph.n_rows, init_transition_graph.n_cols);
	random_matrix();
}


HMMtransitionMatrix::HMMtransitionMatrix(const HMMtransitionMatrix& obj) :
  rng_engine(obj.rng_engine),
  lambda_set(obj.lambda_set),
  transition_graph(obj.transition_graph),
  emission_matrix(obj.emission_matrix),
  prior(obj.prior),
  transition_counts(obj.transition_counts),
  transition_matrix(obj.transition_matrix)
{
	information_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem]);
	jump_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem-1]);
	
	for (unsigned i=0; i < lambda_set.n_elem; i++) information_statistics[i].reset();
	for (unsigned i=0; i < lambda_set.n_elem-1; i++) jump_statistics[i].reset();
	
	endstate = transition_graph.n_rows - 1;
	temperature = 0;	
}


void HMMtransitionMatrix::printDKL()
{
	Rcpp::Rcout << "\nInformation statistics:" <<
	               "\nTemperatures:                     ";
	for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3)  << lambda_set[i] << " ";
	                                           
	Rcpp::Rcout << "\nKullback-Leibler Divergence:      ";
	for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << information_statistics[i].mean() << " ";
	
	Rcpp::Rcout << "\nTemperatures jumping probability: " << "   ";
	for (unsigned i = 0; i < lambda_set.n_elem - 1; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << jump_statistics[i].mean() << " ";
	
	Rcpp::Rcout << "\n";
}

arma::uword HMMtransitionMatrix::n_states() 
{
	return transition_graph.n_rows;
}


double HMMtransitionMatrix::log_dirichlet_density(const arma::rowvec& x, const arma::rowvec& alpha)
{
	double summe = boost::math::lgamma(arma::accu(alpha));
	for (arma::uword i=0; i < x.n_elem; i++) if (alpha[i] > 0.0) 
	{
		summe = summe + log(x[i])*(alpha[i]-1.0) - boost::math::lgamma(alpha[i]);
	}
	return summe;
}

double HMMtransitionMatrix::likelihood(double r_temp)
{
	double logsumme = 0;
	
	arma::mat alphamatrix = arma::conv_to<arma::mat>::from(transition_graph > 0.0) * prior + 
		                        arma::conv_to<arma::mat>::from(transition_counts) * r_temp;
		                        
	for (arma::uword i=0; i < transition_matrix.n_rows - 1; i++) 
	{
		logsumme += log_dirichlet_density(transition_matrix(i, arma::span::all), alphamatrix(i, arma::span::all));
	}
	
	return logsumme;
}

void HMMtransitionMatrix::set_transition_counts(const arma::umat& new_transition_counts)
{
	transition_counts = new_transition_counts;
}
	
const arma::umat& HMMtransitionMatrix::get_transition_graph() const
{
	return transition_graph;
}
	
const arma::umat& HMMtransitionMatrix::get_emission_matrix() const
{
	return emission_matrix;
}
	
	
void HMMtransitionMatrix::random_matrix()
{
	using namespace arma;
    
    double mytemp = lambda_set[temperature];
    
	transition_matrix = zeros<mat>(transition_graph.n_rows, transition_graph.n_cols);
	for (uword i = 0; i < transition_graph.n_rows-1; i++)
	{
		for (uword j = 0; j < transition_graph.n_cols; j++) 
			if (transition_graph(i,j) > 0.0)
			{
				boost::gamma_distribution<> mygamma(prior + mytemp*double(transition_counts(i,j)));
		        boost::variate_generator<BasicTypes::base_generator_type&, boost::gamma_distribution<> > random_gamma(rng_engine, mygamma);
				transition_matrix(i,j) = random_gamma();
		    }	
			
			transition_matrix(i, span::all) = transition_matrix(i, span::all) / accu(transition_matrix(i,span::all));
	}
}

void HMMtransitionMatrix::random_temperature()
{
	// assumes that the temperature can go up or down linearly
	using namespace arma;
	
	if (lambda_set.n_elem > 1)  // to exclude the normal case
	{
		uword proposed_temp, old_neighbors, new_neighbors;
		double log_proposal;
		
		boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
		boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_real_distribution<> > test_randoms(rng_engine, test_dist);
		
		// first: select a neighbour
		if (temperature == 0) proposed_temp = 1; 
		else 
		if (temperature == lambda_set.n_elem - 1) proposed_temp = lambda_set.n_elem - 2;
		else 
		{
			if (test_randoms() <= 0.5) 
				proposed_temp = temperature - 1; 
			else 
				proposed_temp = temperature + 1; 
		}
		
		if (temperature == 0 || temperature == lambda_set.n_elem - 1) old_neighbors = 1; else old_neighbors = 2;
		if (proposed_temp == 0 || proposed_temp == lambda_set.n_elem - 1) new_neighbors = 1; else new_neighbors = 2;
		
		// calculate likelihood
		double log_forward = likelihood(lambda_set[proposed_temp]),
		       log_backward = likelihood(lambda_set[temperature]),
		       log_zero = likelihood(0.0);
		
		log_proposal = log_forward - log(double(new_neighbors)) - log_backward + log(double(old_neighbors));
		
		information_statistics[temperature](log_backward - log_zero); 
		
		double jump_probability = (log_proposal>0)? 1: exp(log_proposal); 
		if (temperature < proposed_temp) jump_statistics[temperature](jump_probability); else jump_statistics[proposed_temp](jump_probability);
		
		if (log(test_randoms()) < log_proposal) temperature = proposed_temp;
	}
}
		
arma::mat HMMtransitionMatrix::get_transition_matrix()
{
	return transition_matrix;
}
	
arma::uword HMMtransitionMatrix::get_temperature()
{
	return temperature;
}

arma::rowvec HMMtransitionMatrix::get_parameters()
{
	arma::rowvec retval(transition_matrix.n_elem+1);  // return transition matrix and current temperature (at first position)
	retval[0] = temperature;
	std::copy(transition_matrix.begin(), transition_matrix.end(), retval.begin()+1);
	return retval;
}

arma::uword HMMtransitionMatrix::get_endstate() const
{
	return endstate;
}
