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
	arma::mat samples(mc, SequencerInstance.get_transition_instance().get_parameters().n_elem + 1); // first position is the jumping_prob, second the temperature
    arma::uword counter = 0;
    arma::wall_clock zeit;
    zeit.tic();
    bool has_been_interrupted = false;
    
    exchange_saver = arma::zeros<arma::umat>(mc, 2);
    
    constants_merker = arma::zeros<arma::mat>(mc, SequencerInstance.get_transition_instance().get_normalization_constants().n_elem);
    
    Rcpp::Rcout << "\nStarting tempered Gibbs sampling with " << burnin << " burn-in and " << mc << " normal samples \n>";
    Rcpp::Rcout.flush();
   
    arma::uword perc_tick = mc/4; if (perc_tick == 0) perc_tick=1;
    arma::uword stroke_tick= mc/100; if (stroke_tick == 0) stroke_tick=1;
   
    // pre-sampling to calculate the normalization constants for each temperature
    //~ SequencerInstance.get_transition_instance().set_temperature(SequencerInstance.get_transition_instance().n_temperatures() - 1);
	//~ while (burnin > 0)
	//~ {
		//~ double amounts;
		//~ arma::uword repeats = SequencerInstance.simulate_transition_counts(amounts);
		//~ SequencerInstance.get_transition_instance().save_realization_for_constants();
		//~ SequencerInstance.get_transition_instance().random_matrix();
		//~ Rcpp::Rcout <<  "*"; Rcpp::Rcout.flush(); 
		//~ burnin--;
	//~ }
//~ 
    //~ SequencerInstance.get_transition_instance().set_temperature(SequencerInstance.get_transition_instance().n_temperatures() - 1);
    //~ 
	
    // sampling
   	while (counter < mc)
	{
		try {
			// First step: simulate a hidden sequence set - matching the observed data
			arma::umat transition_counts;
			
			arma::uword ex1, ex2;
			SequencerInstance.simulate_transition_counts(ex1, ex2);

		    // I suspect that the sampling order is critical
            if (!samplingOrderImproved) SequencerInstance.get_transition_instance().random_matrix(); // Second step: simulate the transition matrix
						
			// Third step: change temperature
			double jumpingProb = 0;
			jumpingProb = SequencerInstance.get_transition_instance().random_temperature(); 
			
			// I suspect that the sampling order is critical. See above
            if (samplingOrderImproved) SequencerInstance.get_transition_instance().random_matrix(); // Second step: simulate the transition matrix

			// Now save data
			if (burnin == 0) 
			{
				constants_merker(counter,arma::span::all) = arma::trans(SequencerInstance.get_transition_instance().get_normalization_constants());
				samples(counter,0) = jumpingProb;
				samples(counter, arma::span(1,samples.n_cols-1)) = SequencerInstance.get_transition_instance().get_parameters();
				exchange_saver(counter,0) = ex1;
				exchange_saver(counter,1) = ex2;
				
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


double Gibbs_Sampling::get_naive_marginal_likelihood()
{
	return SequencerInstance.get_naive_marginal_likelihood();
}


arma::vec Gibbs_Sampling::get_constants()
{
	return SequencerInstance.get_transition_instance().get_normalization_constants();
}

arma::mat Gibbs_Sampling::get_normalizer_data()
{
	return SequencerInstance.get_transition_instance().export_constant_data();
}

arma::umat Gibbs_Sampling::get_exchanger()
{
	return exchange_saver;
}
