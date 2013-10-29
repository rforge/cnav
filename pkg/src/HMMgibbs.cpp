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

#include "HMMgibbs.hpp"

	
Gibbs_Sampling::Gibbs_Sampling(const HMMdataSet& observed_data,	
    const arma::vec& init_lambda, 
	const arma::umat& init_transition_graph, 
	const arma::umat& init_emission_matrix,
	double amount, 
	arma::uword preparation, 
	arma::uword max_sequence_length,  
	arma::uword how_many_sequence_tries,
	bool exact, bool collect, bool improvedSampling, 
	arma::uword rand_seed) : 
	SequencerInstance(observed_data, HMMtransitionMatrix(init_lambda, init_transition_graph, init_emission_matrix, rand_seed, 0.5), 
	                  rand_seed, exact, collect, amount, preparation, max_sequence_length, how_many_sequence_tries),
	chib_ML_estimation(rand_seed)
{
	Rcpp::Rcout << "\nGibbs start ...!\n"; 
	arma::uword nid = observed_data.n_individuals();
	Rcpp::Rcout << "\nNumber of individuals = " << nid << "\n";
	samplingOrderImproved = improvedSampling;
}


arma::vec Gibbs_Sampling::get_temperature_probabilities()
{
	return SequencerInstance.get_transition_instance().get_temperature_probabilities();
}
	
arma::vec Gibbs_Sampling::get_kullback_divergence()
{
	return SequencerInstance.get_transition_instance().get_kullback_divergence();
}


arma::mat Gibbs_Sampling::run(arma::uword burnin, arma::uword mc)
{
	arma::mat samples(mc, SequencerInstance.get_transition_instance().get_parameters().n_elem + 3); // first position is the amount of approximated data
    arma::uword counter = 0;
    arma::wall_clock zeit;
    zeit.tic();
    bool has_been_interrupted = false;
    
    Rcpp::Rcout << "\nStarting tempered Gibbs sampling with " << burnin << " burn-in and " << mc << " normal samples \n>";
    Rcpp::Rcout.flush();
   
    arma::uword perc_tick = mc/4; if (perc_tick == 0) perc_tick=1;
    arma::uword stroke_tick= mc/100; if (stroke_tick == 0) stroke_tick=1;
   
	while (counter < mc)
	{
		try {
			// First step: simulate a hidden sequence set - matching the observed data
			arma::umat transition_counts;
			double amounts;
			arma::uword repeats = SequencerInstance.simulate_transition_counts(amounts);

            // I suspect that the sampling order is critical
            if (!samplingOrderImproved) SequencerInstance.get_transition_instance().random_matrix(); // Second step: simulate the transition matrix
						
			// Third step: change temperature
			double jumpingProb = SequencerInstance.get_transition_instance().random_temperature(); 
			
			// I suspect that the sampling order is critical. See above
            if (samplingOrderImproved) SequencerInstance.get_transition_instance().random_matrix(); // Second step: simulate the transition matrix

			// Now save data
			if (burnin == 0) 
			{
				
				samples(counter,0) = repeats;
				samples(counter,1) = amounts;
				samples(counter,2) = jumpingProb;
				samples(counter, arma::span(3,samples.n_cols-1)) = SequencerInstance.get_transition_instance().get_parameters();
			    counter++;
		    
			} else {
				burnin--;
			}
			
		    if (counter > 0) {
			    if (counter % perc_tick == 0) {
					 Rcpp::Rcout <<  counter*100/mc << "%"; Rcpp::Rcout.flush(); 
				} else {
				    if (counter % stroke_tick == 0)  Rcpp::Rcout <<  "-"; Rcpp::Rcout.flush(); 
				}
		    }
		    
		    if (SequencerInstance.get_transition_instance().get_temperature() == 0) 
		    {
				chib_ML_estimation.save_transition_counts(SequencerInstance);
			}
		    
		} catch (std::runtime_error& ex)
		{
			if (!SequencerInstance.system_interrupted()) 
			  throw ex; 
			else 
			  has_been_interrupted = true;
		}
	}
	
	double n_secs = zeit.toc();
	Rcpp::Rcout << "<\nRunning time: " << n_secs << " seconds\n\n";
	SequencerInstance.get_transition_instance().printDKL();
	
	if (has_been_interrupted) Rcpp::Rcout << "\nSampling interrupted!\n";
	
	return samples;	
}


arma::rowvec Gibbs_Sampling::get_Chib_marginal_likelihood(const arma::rowvec& transition_matrix_sample)
{
	using namespace arma;
	
	uword n_states = SequencerInstance.get_transition_instance().get_endstate() + 1;
	
	mat transmatrix(n_states,n_states);
	std::copy(transition_matrix_sample.begin(), transition_matrix_sample.end(), transmatrix.begin());
	
	return chib_ML_estimation.calculate_marginal_likelihood(SequencerInstance, transmatrix);
}


arma::vec Gibbs_Sampling::get_naive_marginal_likelihood(arma::uword n_samp)
{
	return SequencerInstance.get_naive_marginal_likelihood(n_samp);
}

arma::uword Gibbs_Sampling::get_number_of_prepared_realizations()
{
	return SequencerInstance.get_number_of_prepared_realizations();
}
