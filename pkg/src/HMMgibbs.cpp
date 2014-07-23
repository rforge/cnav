/*
 * HMMgibbs.cpp
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

#include <boost/foreach.hpp>
#include "HMMgibbs.hpp"
#include "HMMparticle.hpp"
#include "HMMkernelwrapper.hpp"
#include "SquirrelKernel.hpp"
#include "MultipleSquirrel.hpp"
#include "ImprovedMultipleSquirrel.hpp"
#include "LongjumpSquirrelKernel.hpp"
	
Gibbs_Sampling::Gibbs_Sampling(
    HMMdataSet& observed_data,	
    const arma::vec& init_lambda, 
	const arma::mat& transition_matrix_prior, 
	const arma::umat& init_emission_matrix,
	arma::uword max_sequence_length,  
	arma::uword how_many_sequence_tries,
	arma::uword internal_sampling ,
	arma::uword n_swappings,
	arma::uword rand_seed) 
	:
	lambda_levels(init_lambda),
	gibbs_sequence_tries(how_many_sequence_tries),
	saved_rand_seed(rand_seed),
	n_swapping_tries(n_swappings),
	multiple_tries(internal_sampling),
	chib_ML_estimation(rand_seed, init_emission_matrix, transition_matrix_prior*0.5, max_sequence_length, observed_data ) 
{
	//~ Rcpp::Rcout << "\nGibbs start ...!\n"; 
	arma::uword nid = observed_data.n_individuals();
	//~ Rcpp::Rcout << "\nNumber of individuals = " << nid << "\n";
	
	for (arma::uword i = 0; i < lambda_levels.n_elem; ++i) 
	{
		//~ Rcpp::Rcout << i << "\t"; Rcpp::Rcout.flush();
		
		HMMparticle next(observed_data, transition_matrix_prior*0.5, init_emission_matrix, i, lambda_levels[i], i, rand_seed + i, max_sequence_length);
		sampling_particles.push_back(next);
	}
	
	parameter_count = transition_matrix_prior.n_elem;
	//~ Rcpp::Rcout << "\n:-\n";Rcpp::Rcout.flush();
}


arma::mat Gibbs_Sampling::get_temperature_likelihoods()
{
	return level_likelihood_trace;
}
	
arma::imat Gibbs_Sampling::get_temperature_trace()
{
	return arma::conv_to<arma::imat>::from(temperature_trace);
}

arma::cube Gibbs_Sampling::get_hash_trace()
{
	return hash_trace;
}

arma::cube Gibbs_Sampling::get_sequences_likelihood_trace()
{
	return sequences_likelihood_trace;
}


arma::cube Gibbs_Sampling::run(arma::uword burnin, arma::uword mc)
{
	// init containers for traces
	arma::cube samples(mc, parameter_count, lambda_levels.n_elem); 
	level_likelihood_trace = arma::zeros<arma::mat>(mc, lambda_levels.n_elem);
	temperature_trace = arma::zeros<arma::umat>(mc, lambda_levels.n_elem);
	hash_trace = arma::zeros<arma::cube>(sampling_particles.begin()->size_of_path_vector(), lambda_levels.n_elem, mc);
	sequences_likelihood_trace = arma::zeros<arma::cube>(sampling_particles.begin()->size_of_path_vector(), lambda_levels.n_elem, mc);
	
	// init transition kernels
	// HMMkernelwrapper<HMMsquirrelKernel> first_kernel(saved_rand_seed);
	// HMMkernelwrapper<HMMMultipleSquirrel> first_kernel(saved_rand_seed, multiple_tries);
	// HMMkernelwrapper<HMMImprovedMultipleSquirrel> first_kernel(saved_rand_seed, multiple_tries);
	HMMkernelwrapper<HMMlongjump_squirrelKernel, true> third_kernel(saved_rand_seed, lambda_levels.n_elem, multiple_tries);
	HMMkernelwrapper<IncompleteGibbsKernel, false> second_kernel(saved_rand_seed, lambda_levels.n_elem, gibbs_sequence_tries);
	//~ HMMkernelwrapper<SquirrelSwarmKernel, true> third_kernel(saved_rand_seed, lambda_levels.n_elem, multiple_tries);
	
	// and init and random generator
	boost::random::mt19937 random_generator(saved_rand_seed);
	boost::random::uniform_int_distribution<arma::uword> dist(1, (lambda_levels.n_elem > 1)? lambda_levels.n_elem-1 : 2);
    boost::variate_generator<boost::random::mt19937&, boost::random::uniform_int_distribution<arma::uword> > select_random(random_generator, dist); 	
	
	// initialize timer
    arma::uword counter = 0;
    arma::wall_clock zeit;
    zeit.tic();
   
    // give a message that we're starting
   
    Rcpp::Rcout << "\nStarting Gibbs sampling with " << burnin << " burn-in samples and " << mc << " normal samples \n>";
    Rcpp::Rcout.flush();
  
	arma::uword perc_tick = mc/4; if (perc_tick == 0) perc_tick=1;
    arma::uword stroke_tick= mc/100; if (stroke_tick == 0) stroke_tick=1;
  
    // sampling
   	while (counter < mc)
	{
		try {
			
			//~ Rcpp::Rcout << "\n-->";Rcpp::Rcout.flush();
			if (lambda_levels.n_elem > 1) 
			{
				for (arma::uword swapcounter = 0; swapcounter < n_swapping_tries; ++swapcounter)
				{
					arma::uword sel = select_random();
					exchange_temperatures(sampling_particles[sel],sampling_particles[sel-1], false);
				}
			}
			 //~ Rcpp::Rcout << "exchanged-->";Rcpp::Rcout.flush();
			
			BOOST_FOREACH(HMMparticle &ref, sampling_particles) 
			{
				// first: make a random draw to the matrix
				//~ Rcpp::Rcout << "Random Matrix";Rcpp::Rcout.flush();
				ref.random_matrix();
				
				//~ std::cout << "\n3rd"; std::cout.flush();
				third_kernel.apply_kernel(&ref);
				//~ std::cout << " 2rd"; std::cout.flush();
				second_kernel.apply_kernel(&ref);
				//~ std::cout << " 1rd"; std::cout.flush();
				// first_kernel.apply_kernel(&ref);				
				//~ std::cout << " ready\n"; std::cout.flush();
				//~ Rcpp::Rcout << "kernel applied ...";Rcpp::Rcout.flush();
				
				if (burnin == 0) 
				{
					//~ Rcpp::Rcout << ".. a";Rcpp::Rcout.flush();
					level_likelihood_trace(counter,ref.get_lambda_ref()) = ref.relative_countmatrix_likelihood();
					//~ level_likelihood_trace(counter,ref.get_lambda_ref()) = ref.full_untempered_collapsed_likelihood();
					//~ Rcpp::Rcout << ".. b";Rcpp::Rcout.flush();
					temperature_trace(counter,ref.get_my_running_number()) = ref.get_lambda_ref();
					//~ Rcpp::Rcout << ".. c";Rcpp::Rcout.flush();
					samples.slice(ref.get_lambda_ref()).row(counter) = ref.get_transition_matrix_as_rowvector();
					//~ Rcpp::Rcout << ".. d";Rcpp::Rcout.flush();
					hash_trace.slice(counter).col(ref.get_lambda_ref()) = ref.current_sequences_hash_value_vector();
					//** 
					sequences_likelihood_trace.slice(counter).col(ref.get_lambda_ref()) = ref.current_sequences_likelihood_vector();
					//**
					if (ref.get_lambda_ref() == 0) chib_ML_estimation.save_transition_counts(ref.get_transition_counts());
					
				}
			}
			
			if (burnin == 0) ++counter; else --burnin;
			
		    if (counter > 0) {
			    if (counter % perc_tick == 0) {
						
					 Rcpp::Rcout <<  counter*100/mc << "%"; Rcpp::Rcout.flush(); 
				} else {
				    if (counter % stroke_tick == 0)  Rcpp::Rcout <<  "-"; Rcpp::Rcout.flush(); 
				}
		    }
		    
		    
		} catch (std::runtime_error& ex)
		{
			Rcpp::Rcout << "Some error occured during sampling: " << ex.what() << "\n"; Rcpp::Rcout.flush();
			throw ex; 
		}
	}
	
	double n_secs = zeit.toc();
	Rcpp::Rcout << "<\nRunning time: " << n_secs << " seconds\n\n";
	
	// first_kernel.print_monitor();
	second_kernel.print_monitor();
	third_kernel.print_monitor();
		
	return samples;	
}


arma::rowvec Gibbs_Sampling::get_Chib_marginal_likelihood(const arma::rowvec& transition_matrix_sample)
{
	using namespace arma;

	rowvec result = zeros<rowvec>(4);	
	uword n_states = sampling_particles[0].get_transition_matrix().n_rows;
	
	mat transmatrix(n_states,n_states);
	std::copy(transition_matrix_sample.begin(), transition_matrix_sample.end(), transmatrix.begin());
	
	return chib_ML_estimation.calculate_marginal_likelihood(transmatrix);
}

