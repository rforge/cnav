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
#include <boost/thread.hpp> 

#include <iostream>
#include <algorithm>

#include "BasicTypes.hpp"
#include "HMMtransitionMatrix.hpp"
	
HMMtransitionMatrix::HMMtransitionMatrix(arma::vec init_lambda,                // revised 12.1.2014
  const arma::umat& init_transition_graph, const arma::umat& init_emission_matrix, 
  arma::uword rand_seed, arma::uword n_swaps,
  double i_prior, arma::uword open_cycles ) :
  rng_engine(rand_seed),
  lambda_set(init_lambda),
  transition_graph(init_transition_graph),
  emission_matrix(init_emission_matrix),
  n_swapping_tries(n_swaps),
  prior(i_prior),
  loggamma()
 {
	transition_counts_cube = arma::zeros<arma::ucube>(init_transition_graph.n_rows, init_transition_graph.n_cols, init_lambda.n_elem);
	transition_matrix_cube = arma::zeros<arma::cube>(init_transition_graph.n_rows, init_transition_graph.n_cols, init_lambda.n_elem);
	current_particle_temperature = arma::linspace<arma::urowvec>(0,init_lambda.n_elem-1,init_lambda.n_elem);
	current_particle_likelihoods = arma::zeros<arma::mat>(init_lambda.n_elem, init_lambda.n_elem);
	
	random_matrices();
	
}


HMMtransitionMatrix::HMMtransitionMatrix(const HMMtransitionMatrix& obj) :
  rng_engine(obj.rng_engine),
  lambda_set(obj.lambda_set),
  transition_graph(obj.transition_graph),
  emission_matrix(obj.emission_matrix),
  n_swapping_tries(obj.n_swapping_tries),
  prior(obj.prior),
  transition_counts_cube(obj.transition_counts_cube),
  transition_matrix_cube(obj.transition_matrix_cube),
  current_particle_temperature(obj.current_particle_temperature),
  current_particle_likelihoods(obj.current_particle_likelihoods),
  loggamma(obj.loggamma)
{
}


//~ void HMMtransitionMatrix::printDKL()
//~ {
	//~ 
	//~ Rcpp::Rcout << "\nInformation statistics:" <<
	               //~ "\nTemperatures:                     ";
	//~ for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3)  << lambda_set[i] << " ";
	                                           //~ 
	//~ Rcpp::Rcout << "\nKullback-Leibler Divergence:      ";
	//~ for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << information_statistics[i].mean() << " ";
	//~ 
	//~ Rcpp::Rcout << "\nTemperatures jumping probability: " << "   ";
	//~ for (unsigned i = 0; i < lambda_set.n_elem - 1; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << jump_statistics[i].mean() << " ";
	//~ 
	//~ Rcpp::Rcout << "\nNormalization constants:" << "   ";
	//~ for (unsigned i = 0; i < lambda_set.n_elem; i++) Rcpp::Rcout << std::setw(7) << std::setprecision(3) << NormConstantsWrapper.get_weight_normalization(i) << " ";
		//~ 
	//~ Rcpp::Rcout << "\n";
	//~ 
//~ }



arma::uword HMMtransitionMatrix::n_states() 
{
	return transition_graph.n_rows;
}


const arma::urowvec& HMMtransitionMatrix::get_current_particle_temperatures() const
{
	return current_particle_temperature;
}

const double HMMtransitionMatrix::get_particle_temperature(arma::uword particle) const
{
	return current_particle_temperature[particle];
}


//~ double HMMtransitionMatrix::log_dirichlet_density(const arma::rowvec& x, const arma::rowvec& alpha)
//~ {
	//~ double summe = boost::math::lgamma(arma::accu(alpha));
	//~ for (arma::uword i=0; i < x.n_elem; i++) if (alpha[i] > 0.0) 
	//~ {
		//~ summe = summe + log(x[i])*(alpha[i]-1.0) - boost::math::lgamma(alpha[i]);
	//~ }
	//~ return summe;
//~ }
//~ 
//~ double HMMtransitionMatrix::likelihood(double r_temp)
//~ {
	//~ double logsumme = multinomial_coef;
	//~ 
	//~ arma::mat alphamatrix = arma::conv_to<arma::mat>::from(transition_graph > 0.0) * prior + 
		                    //~ arma::conv_to<arma::mat>::from(transition_counts) * r_temp;
		                        //~ 
	//~ for (arma::uword i=0; i < transition_matrix.n_rows - 1; i++) 
	//~ {
		//~ logsumme += log_dirichlet_density(transition_matrix(i, arma::span::all), alphamatrix(i, arma::span::all));
	//~ }
	//~ 
	//~ return logsumme;
//~ }
//~ 
double HMMtransitionMatrix::likelihood(const arma::umat& counts, arma::uword particle)
{
	double summe =  0.0;
	arma::umat::const_iterator citer = counts.begin();
	arma::mat::const_iterator piter = transition_matrix_cube.slice(particle).begin();
	for (; citer != counts.end(); ++citer, ++piter)
		if (*piter > 0) summe+= log(*piter) * double(*citer);
		
	return summe * lambda_set[current_particle_temperature[particle]];
}


bool HMMtransitionMatrix::is_particle_at_zero_lambda(arma::uword particle)
{
	return current_particle_temperature[particle] == lambda_set.n_elem-1;
}
	
arma::uword HMMtransitionMatrix::get_particle_at_next_temperature(arma::uword particle)
{
	arma::uword next_one = 0;
	bool found = false;
	while (next_one < current_particle_temperature.n_elem && !found) 
	{
		found = current_particle_temperature[next_one] == current_particle_temperature[particle] + 1;
		if (!found) ++next_one;
	}
	
	return next_one;
}



const double HMMtransitionMatrix::get_prior() const
{
	return prior;
}

//~ double HMMtransitionMatrix::new_likelihood(double r_temp)
//~ {
	//~ double logsumme = multinomial_coef;
	//~ 
	//~ arma::umat::const_iterator trans_iter = transition_counts.begin();
	//~ arma::mat::const_iterator prob_iter = transition_matrix.begin();
	//~ 
	//~ for (; trans_iter != transition_counts.end(); trans_iter++, prob_iter++) 
		//~ if (*prob_iter > 0) logsumme += log(*prob_iter) * double(*trans_iter);
//~ 
    //~ logsumme *= r_temp;
	//~ 
	//~ return logsumme;
//~ }

arma::rowvec HMMtransitionMatrix::likelihood_vector(arma::uword particle)
{
	// generates a complete vector of likelihoods
	double logsumme = 0.0;
	
	arma::umat::const_iterator trans_iter = transition_counts_cube.slice(particle).begin();
	arma::mat::const_iterator prob_iter = transition_matrix_cube.slice(particle).begin();
	
	for (; trans_iter != transition_counts_cube.slice(particle).end(); trans_iter++, prob_iter++) 
		if (*prob_iter > 0) logsumme += log(*prob_iter) * double(*trans_iter);

	return logsumme * arma::trans(lambda_set);
	
}

boost::mutex pstop;

double HMMtransitionMatrix::collapsed_likelihood_difference(const arma::umat& old_counts, const arma::umat& new_counts, arma::uword particle_number)
{
	using namespace arma;
	
	double summe = 0.0, intermediate_sum = 0.0;
	double mylambda = lambda_set[current_particle_temperature[particle_number]];
	for (uword zeile = 0; zeile < transition_graph.n_rows-1; zeile++)
	{
		double csum_old = 0.0, csum_new = 0.0;
		
		for (uword spalte = 0; spalte < transition_graph.n_cols; spalte++)
		
		if (transition_graph(zeile, spalte) > 0)
	    {
			double alt_counts = double(transition_counts_cube(zeile,spalte,particle_number)) * mylambda + prior;
			
			double neu_counts = alt_counts - double(old_counts(zeile,spalte)) * mylambda 
			                               + double(new_counts(zeile, spalte)) * mylambda;
		
			intermediate_sum = loggamma(neu_counts+2.0) - (neu_counts+1.0) - neu_counts  
			                 -(loggamma(alt_counts + 2.0) - (alt_counts+1.0) - alt_counts );
		
			summe += intermediate_sum;
			csum_old += alt_counts;
			csum_new += neu_counts;
			
		} 
		intermediate_sum = (loggamma(csum_old + 2.0) - (csum_old + 1.0) - csum_old)
		                 - (loggamma(csum_new + 2.0) - (csum_new + 1.0) - csum_new);
		summe += intermediate_sum;
	}
	//~ pstop.lock();
	//~ std::cout << " " << std::setprecision(10) << summe << " "; std::cout.flush();
	//~ pstop.unlock();
	
	return summe;
}


double HMMtransitionMatrix::transition_matrix_density(const arma::mat& c_transition_matrix, const arma::mat& c_count_matrix) 
{
	double logsumme = 0;
	
	arma::mat with_prior_matrix =  arma::conv_to<arma::mat>::from(transition_graph > 0) * prior + c_count_matrix;
	
	for (arma::uword i=0; i < c_transition_matrix.n_rows - 1; i++) 
	{
		logsumme += boost::math::lgamma(arma::accu(with_prior_matrix.row(i)));
	
		for (arma::uword j = 0; j < c_transition_matrix.n_cols; j++) 
			if (with_prior_matrix(i,j) > 0)
		{
			logsumme += log(c_transition_matrix(i,j)) * (with_prior_matrix(i,j) - 1.0) - boost::math::lgamma(with_prior_matrix(i,j));
		}
	}
	
	return logsumme;
	
}

void HMMtransitionMatrix::set_multinomial_coefficient(double value)    // revised 12.1.2014
{
	multinomial_coef = value;
}


void HMMtransitionMatrix::set_transition_counts(const arma::ucube& new_transition_counts)   // revised 12.1.2014
{
	transition_counts_cube = new_transition_counts;
}
	
const arma::umat& HMMtransitionMatrix::get_transition_graph() const
{
	return transition_graph;
}

//~ const arma::umat& HMMtransitionMatrix::get_transition_counts() const
//~ {
	//~ return transition_counts;
//~ }
	
const arma::umat& HMMtransitionMatrix::get_emission_matrix() const
{
	return emission_matrix;
}
	

	
void HMMtransitionMatrix::random_matrices()    // revised 18.1.2014
{
	using namespace arma;
    transition_matrix_cube.fill(0.0);
    for (uword particle = 0; particle < current_particle_temperature.n_elem; ++particle)
    {
	    double mytemp = lambda_set[current_particle_temperature[particle]];
	    
		for (uword i = 0; i < transition_graph.n_rows-1; i++)
		{
			for (uword j = 0; j < transition_graph.n_cols; j++) 
				if (transition_graph(i,j) > 0.0)
				{
					boost::gamma_distribution<> mygamma(prior + mytemp*double(transition_counts_cube(i,j,particle)));
			        boost::variate_generator<BasicTypes::base_generator_type&, boost::gamma_distribution<> > random_gamma(rng_engine, mygamma);
					transition_matrix_cube(i,j,particle) = random_gamma();
			    }	
			
			transition_matrix_cube(span(i), span::all, span(particle)) = 
				transition_matrix_cube(span(i), span::all, span(particle)) / accu(transition_matrix_cube(span(i), span::all, span(particle)));
		}
	}
	
}


const arma::mat& HMMtransitionMatrix::get_transition_matrix(arma::uword particle) const
{
	return transition_matrix_cube.slice(particle);
}


const arma::umat& HMMtransitionMatrix::get_transition_counts(arma::uword particle) const
{
	return transition_counts_cube.slice(particle);
}

void HMMtransitionMatrix::set_transition_counts(arma::uword particle, const arma::umat& new_counts)
{
	transition_counts_cube.slice(particle) = new_counts;
}

	
arma::uword HMMtransitionMatrix::n_temperatures()
{
	return lambda_set.n_elem;
}

//~ 
//~ //*************************************************************************************
//~ 
//~ double HMMtransitionMatrix::get_sequence_likelihood(const arma::umat& sequence_as_matrix, bool full_posterior)
//~ {
	//~ double logsumme = 0;
	//~ double thistemperature = full_posterior? 1.0 : lambda_set[temperature];
	//~ 
	//~ arma::mat alphamatrix = arma::conv_to<arma::mat>::from(transition_graph > 0.0) * prior + 
		                        //~ arma::conv_to<arma::mat>::from(sequence_as_matrix) * thistemperature;
		                        //~ 
	//~ for (arma::uword i=0; i < transition_matrix.n_rows - 1; i++) 
	//~ {
		//~ logsumme += log_dirichlet_density(transition_matrix(i, arma::span::all), alphamatrix(i, arma::span::all));
	//~ }
	//~ 
	//~ return logsumme;
//~ }
//~ 


arma::mat HMMtransitionMatrix::get_all_transition_matrices()
{
	using namespace arma;
	mat retval(transition_matrix_cube.n_rows * transition_matrix_cube.n_cols, transition_matrix_cube.n_slices);  
	
	std::copy(transition_matrix_cube.begin(), transition_matrix_cube.end(), retval.begin());  // this should work
	
	return join_cols(conv_to<rowvec>::from(trans(current_particle_temperature)), retval);
}

arma::uword HMMtransitionMatrix::get_endstate() const
{
	return transition_graph.n_rows - 1;
}

	
arma::vec HMMtransitionMatrix::particle_swapping()
{
	// just run through all temperatures and swap 
	boost::uniform_int<arma::uword> dist(1, lambda_set.n_elem-1);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_int<arma::uword>  > swap_random(rng_engine, dist);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_real<> > runif(rng_engine, boost::uniform_real<>(0.0, 1.0));

    // before, calculate the current likelihoods	
	for (arma::uword zeile = 0; zeile < current_particle_likelihoods.n_rows; ++zeile)
	{
		current_particle_likelihoods(zeile, arma::span::all) = likelihood_vector(zeile);
	}
	
	// determine result
	arma::vec retval(current_particle_temperature.n_elem);
	
	// reference from the temperatures to the particles
	arma::uvec temp_to_particle(lambda_set.n_elem);
	for (arma::uword particle = 0; particle < current_particle_temperature.n_elem; ++particle)
	{
		temp_to_particle[current_particle_temperature[particle]] = particle;
		
		retval[current_particle_temperature[particle]] = current_particle_likelihoods(particle, current_particle_temperature[particle]);
	}
	
	
	
	// then start swapping	
	for (arma::uword i=0; i < n_swapping_tries; ++i) 
	{
		// first: select the temperature level for an exchange
		arma::uword temp_auswahl1 = swap_random();
		arma::uword temp_auswahl2 = temp_auswahl1 - 1;
	
		// then: grab the corresponding particles
		arma::uword particle_auswahl1 = temp_to_particle[temp_auswahl1], 
		            particle_auswahl2 = temp_to_particle[temp_auswahl2];
		
		// and test them
		double log_decision = current_particle_likelihoods(particle_auswahl1, temp_auswahl2) + 
							   current_particle_likelihoods(particle_auswahl2, temp_auswahl1) -
							   current_particle_likelihoods(particle_auswahl1, temp_auswahl1) -
							   current_particle_likelihoods(particle_auswahl2, temp_auswahl2);
								   
		if (log(runif()) < log_decision) 
		{
			// if accepted, then exchange temperatures
			current_particle_temperature[particle_auswahl1] = temp_auswahl2;
			current_particle_temperature[particle_auswahl2] = temp_auswahl1;
			
			// and the back-references
			temp_to_particle[temp_auswahl1] = particle_auswahl2;
			temp_to_particle[temp_auswahl2] = particle_auswahl1;
		}
	}
	
	return retval; 		
}

//*****************************************



HMMtransitionMatrix::lgamma_functor_class::lgamma_functor_class()
{
	density = 0.01;
	arma::uword anzahl = trunc(10000.0 / density);
	lgamma_preserved = arma::ones<arma::vec>(anzahl);
	for (arma::uword i = 0; i < lgamma_preserved.n_elem; i++) lgamma_preserved[i] = boost::math::lgamma(double(i+1.0) * density);
}
		

double HMMtransitionMatrix::lgamma_functor_class::operator()(double value)
{
	if (trunc(value/density) > 1.0) 
	{
		arma::uword unten_index = arma::uword(trunc(value/density))-1, 
		            oben_index = unten_index + 1;
		
		if (oben_index >= lgamma_preserved.n_elem) 
			return boost::math::lgamma(value);
		else 
		{	
			// simple linear interpolation
			double offset = (value/density - double(unten_index));
			double oben = lgamma_preserved[oben_index], unten = lgamma_preserved[unten_index];
			return unten + (oben-unten)*offset;
		}
	} 
	else 
	{
		return boost::math::lgamma(value);
	}
}
		
	
	
