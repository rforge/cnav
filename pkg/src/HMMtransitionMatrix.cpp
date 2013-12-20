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
	

#include <boost/random/uniform_int.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>	
#include <boost/math/special_functions/gamma.hpp>

#include <iostream>
#include <algorithm>

#include "BasicTypes.hpp"
#include "HMMtransitionMatrix.hpp"
	
HMMtransitionMatrix::HMMtransitionMatrix(arma::vec init_lambda, 
  const arma::umat& init_transition_graph, const arma::umat& init_emission_matrix, 
  arma::uword rand_seed, double i_prior) :
  rng_engine(rand_seed),
  lambda_set(init_lambda),
  transition_graph(init_transition_graph),
  emission_matrix(init_emission_matrix),
  prior(i_prior),
  NormConstantsWrapper(init_lambda.n_elem, rand_seed)
{
	information_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem]);
	jump_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem-1]);
	
	for (unsigned i=0; i < lambda_set.n_elem; i++) information_statistics[i].reset();
	for (unsigned i=0; i < lambda_set.n_elem-1; i++) jump_statistics[i].reset();
	
	endstate = init_transition_graph.n_rows - 1;
	temperature = lambda_set.n_elem - 1;
	transition_counts = 1.0/double(lambda_set.n_elem * 100) + arma::zeros<arma::umat>(init_transition_graph.n_rows, init_transition_graph.n_cols);
	random_matrix();
	
	denominator = arma::zeros<arma::mat>(lambda_set.n_elem, lambda_set.n_elem);
	numerator = arma::zeros<arma::mat>(lambda_set.n_elem, lambda_set.n_elem);
	mean_normalization_constants = arma::ones<arma::vec>(lambda_set.n_elem);
	w_counter = arma::zeros<arma::umat>(lambda_set.n_elem, lambda_set.n_elem);
	
	constants_updated_sum = arma::zeros<arma::vec>(lambda_set.n_elem);
	constants_updated_count = 0;
}


HMMtransitionMatrix::HMMtransitionMatrix(const HMMtransitionMatrix& obj) :
  rng_engine(obj.rng_engine),
  lambda_set(obj.lambda_set),
  transition_graph(obj.transition_graph),
  emission_matrix(obj.emission_matrix),
  prior(obj.prior),
  transition_counts(obj.transition_counts),
  transition_matrix(obj.transition_matrix),
  denominator(obj.denominator),
  numerator(obj.numerator),
  mean_normalization_constants(obj.mean_normalization_constants),
  w_counter(obj.w_counter),
  constants_updated_count(obj.constants_updated_count),
  constants_updated_sum(obj.constants_updated_sum),
  NormConstantsWrapper(obj.NormConstantsWrapper)
{
	information_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem]);
	jump_statistics.reset(new arma::running_stat<double>[lambda_set.n_elem-1]);
	
	for (unsigned i=0; i < lambda_set.n_elem; i++) information_statistics[i].reset();
	for (unsigned i=0; i < lambda_set.n_elem-1; i++) jump_statistics[i].reset();
	
	endstate = transition_graph.n_rows - 1;
	temperature = lambda_set.n_elem - 1;	
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
	
	Rcpp::Rcout << "\nNormalization constants:" << "   ";
	for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << NormConstantsWrapper.get_weight_normalization(i) << " ";
		
	Rcpp::Rcout << "\n";
	
}

arma::vec HMMtransitionMatrix::get_temperature_probabilities()
{
	using namespace arma;
	vec result = zeros<vec>(lambda_set.n_elem);
	
	for (uword i = 0; i < lambda_set.n_elem-1; i++) result[i] = jump_statistics[i].mean();
	
	return result;
}


arma::vec HMMtransitionMatrix::get_kullback_divergence()
{
	using namespace arma;
	vec result = zeros<vec>(lambda_set.n_elem);
	
	for (uword i = 0; i < lambda_set.n_elem; i++) result[i] = information_statistics[i].mean();
	
	return result;
}

arma::vec HMMtransitionMatrix::get_normalization_constants()
{
	arma::vec nc(lambda_set.n_elem);
	for (arma::uword i=0; i < lambda_set.n_elem; i++) nc[i] = NormConstantsWrapper.get_weight_normalization(i);
	return nc;
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

double HMMtransitionMatrix::new_likelihood(double r_temp)
{
	double logsumme = multinomial_coef;
	
	arma::umat::const_iterator trans_iter = transition_counts.begin();
	arma::mat::const_iterator prob_iter = transition_matrix.begin();
	
	for (; trans_iter != transition_counts.end(); trans_iter++, prob_iter++) 
		if (*prob_iter > 0) logsumme += log(*prob_iter) * double(*trans_iter);

    logsumme *= r_temp;
	
	return logsumme;
}


double HMMtransitionMatrix::transition_matrix_density(const arma::mat& c_transition_matrix, const arma::mat& c_count_matrix) 
{
	double logsumme = 0;
	
	arma::mat with_prior_matrix =  arma::conv_to<arma::mat>::from(transition_graph > 0) * prior + c_count_matrix;
	
	for (arma::uword i=0; i < c_transition_matrix.n_rows - 1; i++) 
	{
		logsumme += log_dirichlet_density(c_transition_matrix(i, arma::span::all), with_prior_matrix(i, arma::span::all));
	}
	
	return logsumme;
	
}

void HMMtransitionMatrix::set_multinomial_coefficient(double value)
{
	multinomial_coef = value;
}


void HMMtransitionMatrix::set_transition_counts(const arma::umat& new_transition_counts)
{
	transition_counts = new_transition_counts;
}
	
const arma::umat& HMMtransitionMatrix::get_transition_graph() const
{
	return transition_graph;
}

const arma::umat& HMMtransitionMatrix::get_transition_counts() const
{
	return transition_counts;
}
	
const arma::umat& HMMtransitionMatrix::get_emission_matrix() const
{
	return emission_matrix;
}
	

double HMMtransitionMatrix::normalization_constant(double r_temp)
{
	double logsumme = 0.0;
	
	arma::mat alphamatrix = arma::conv_to<arma::mat>::from(transition_graph > 0.0) * prior + 
		                        arma::conv_to<arma::mat>::from(transition_counts) * r_temp;
	
	for (arma::uword i=0; i < transition_matrix.n_rows - 1; i++) 
	{
		logsumme -= boost::math::lgamma(arma::accu(alphamatrix(i,arma::span::all)));
		
		for (arma::uword j=0; j < transition_matrix.n_cols; j++) if (alphamatrix(i,j) > 0.0) 
		{
			logsumme += boost::math::lgamma(alphamatrix(i,j));
		}
	}
	
	return logsumme;
}


void HMMtransitionMatrix::add_realization_to_constants(arma::uword drawn_at_temp)
{
	arma::rowvec sample(lambda_set.n_elem);
	
	for (arma::uword i_temp = 0; i_temp < lambda_set.n_elem; i_temp++) 
	{
		sample[i_temp] = normalization_constant(lambda_set[i_temp]);
	}
	
	// NormConstantsWrapper.save_weights(sample, drawn_at_temp);
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

double HMMtransitionMatrix::random_temperature()
{
	// assumes that the temperature can go up or down linearly
	using namespace arma;
	
	if (lambda_set.n_elem > 1)  // to exclude the normal case
	{
		uword proposed_temp, old_neighbors, new_neighbors;
		double log_proposal = 0;
		
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
		if (temperature > 0) 
			NormConstantsWrapper.save_likelihood(new_likelihood(lambda_set[temperature-1] - lambda_set[temperature]), temperature-1);
				
		if (test_randoms() < 0.03)	
		{
			double log_forward = new_likelihood(lambda_set[proposed_temp]) - NormConstantsWrapper.get_weight_normalization(proposed_temp),
			       log_backward = new_likelihood(lambda_set[temperature]) - NormConstantsWrapper.get_weight_normalization(temperature),
			       log_zero = 0 - NormConstantsWrapper.get_weight_normalization(lambda_set.n_elem-1);
					
			log_proposal = log_forward - log(double(new_neighbors)) - log_backward + log(double(old_neighbors));
			
			information_statistics[temperature](log_backward - log_zero); 
		
		    double jump_probability = (log_proposal>0)? 1: exp(log_proposal); 
		    if (temperature < proposed_temp) jump_statistics[temperature](jump_probability); else jump_statistics[proposed_temp](jump_probability);
			
			if (log(test_randoms()) < log_proposal) temperature = proposed_temp;
		}
		return (log_proposal>0)?1:exp(log_proposal);
	} 
	else return 0;
}
		
void HMMtransitionMatrix::save_realization_for_constants()
{
	add_realization_to_constants(temperature);
}
	
		
arma::mat HMMtransitionMatrix::get_transition_matrix()
{
	return transition_matrix;
}
	
arma::uword HMMtransitionMatrix::get_temperature()
{
	return temperature;
}

arma::uword HMMtransitionMatrix::n_temperatures()
{
	return lambda_set.n_elem;
}

void HMMtransitionMatrix::set_temperature(arma::uword new_temp)
{
	temperature = new_temp;
}

//*************************************************************************************

double HMMtransitionMatrix::get_sequence_likelihood(const arma::umat& sequence_as_matrix, bool full_posterior)
{
	double logsumme = 0;
	double thistemperature = full_posterior? 1.0 : lambda_set[temperature];
	
	arma::mat alphamatrix = arma::conv_to<arma::mat>::from(transition_graph > 0.0) * prior + 
		                        arma::conv_to<arma::mat>::from(sequence_as_matrix) * thistemperature;
		                        
	for (arma::uword i=0; i < transition_matrix.n_rows - 1; i++) 
	{
		logsumme += log_dirichlet_density(transition_matrix(i, arma::span::all), alphamatrix(i, arma::span::all));
	}
	
	return logsumme;
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

//**************** internal class normalization ********************


HMMtransitionMatrix::normalization_class::normalization_class(arma::uword n_lambda_levels, arma::uword rand_seed) :
  boot_rng_engine(rand_seed)
{
	 using namespace arma;
	 
	 collect_count = zeros<uvec>(n_lambda_levels-1);
	 collection = zeros<mat>(1000, n_lambda_levels-1);
	 normalized_constants = zeros<rowvec>(n_lambda_levels-1);
 }

HMMtransitionMatrix::normalization_class::normalization_class(const HMMtransitionMatrix::normalization_class& obj) :
	collect_count(obj.collect_count),
	collection(obj.collection),
	boot_rng_engine(obj.boot_rng_engine),
	normalized_constants(obj.normalized_constants)
{	
}


double HMMtransitionMatrix::normalization_class::accu_log_values(arma::vec logs)
{
	// summarizes log values from a vector. Sorts the data before
	// the idea is that there are rounding errors with log values, which can be overcome by 
	// subsequent addition of increasing values
	
	using namespace arma;
	logs = sort(logs);
	double summe = logs[0]; // start with the smallest one
	
	arma::vec::iterator iter = logs.begin() + 1;
	for (; iter != logs.end(); iter++) {
		
		if (summe > (*iter) ) 
			summe = log(1 + exp( (*iter) - summe)) + summe;
		else
		    summe = log( exp(summe - (*iter)) + 1) + (*iter);
	}
	return summe; // returns the log value of the sum
}


arma::uvec HMMtransitionMatrix::normalization_class::bootstrap_indices(arma::uword n)
{
	arma::uvec result(n);
	
	if (n <= 1) 
		result[0] = 0; 
    else
    {
		boost::uniform_int<arma::uword> dist(0, n-1);
		boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_int<arma::uword>  > random_int(boot_rng_engine, dist);
		for (arma::uvec::iterator it = result.begin(); it != result.end(); it++) *it = random_int();
	}
	return result;
}


void HMMtransitionMatrix::normalization_class::save_likelihood(double likelihood_ratio, arma::uword for_temperature)
{
    collection(collect_count[for_temperature], for_temperature)	= likelihood_ratio;
    arma::vec view(collect_count[for_temperature]+1); 
    arma::uvec indices = bootstrap_indices(collect_count[for_temperature]+1); 
    arma::uvec::const_iterator it2 = indices.begin();
    
    for (arma::vec::iterator it = view.begin() ;  it != view.end(); it++, it2++) 
	{
		*it = collection(*it2, for_temperature);
	}
            
    normalized_constants[for_temperature] = accu_log_values(view) - log(double(collect_count[for_temperature]+1));
    collect_count[for_temperature] += 1;
}


double HMMtransitionMatrix::normalization_class::get_weight_normalization(arma::uword level)
{
	double summe = 0;
	for (arma::uword i = level; i < normalized_constants.n_elem; i++) summe += normalized_constants[i];
	
	return summe;
}

arma::mat HMMtransitionMatrix::normalization_class::export_data()
{
	return collection(arma::span(0,arma::max(collect_count)-1), arma::span::all);
}


arma::mat HMMtransitionMatrix::export_constant_data()
{
	return NormConstantsWrapper.export_data();
}


