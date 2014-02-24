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
	arma::uword max_sequence_length,  
	arma::uword how_many_sequence_tries,
	arma::uword path_sampling_repetitions,
	arma::uword internal_sampling ,
	arma::uword n_swappings,
	bool use_collapsed_sampler,
	arma::uword rand_seed) :
	
	TransitionInstance(init_lambda, init_transition_graph, init_emission_matrix, rand_seed, n_swappings, 0.5),
	SequencerInstance(observed_data, TransitionInstance, 
	                  rand_seed, max_sequence_length, how_many_sequence_tries, internal_sampling, path_sampling_repetitions, use_collapsed_sampler),
	chib_ML_estimation(rand_seed)
{
	Rcpp::Rcout << "\nGibbs start ...!\n"; 
	arma::uword nid = observed_data.n_individuals();
	Rcpp::Rcout << "\nNumber of individuals = " << nid << "\n";
	
}


arma::mat Gibbs_Sampling::get_temperature_likelihoods()
{
	return jumping_probs_statistics;
}
	

arma::cube Gibbs_Sampling::run(arma::uword burnin, arma::uword mc)
{
	arma::mat parameters = SequencerInstance.get_transition_instance().get_all_transition_matrices();
	
	arma::cube samples(parameters.n_rows, parameters.n_cols, mc); 
	jumping_probs_statistics = arma::zeros<arma::mat>(mc, SequencerInstance.get_transition_instance().n_temperatures());
	
	// first position is the jumping_prob, second the temperature
    arma::uword counter = 0;
    arma::wall_clock zeit;
    zeit.tic();
    bool has_been_interrupted = false;
   
    Rcpp::Rcout << "\nStarting Gibbs sampling with " << burnin << " burn-in samples and " << mc << " normal samples \n>";
    Rcpp::Rcout.flush();
  
	arma::uword perc_tick = mc/4; if (perc_tick == 0) perc_tick=1;
    arma::uword stroke_tick= mc/100; if (stroke_tick == 0) stroke_tick=1;
  
    // sampling
   	while (counter < mc)
	{
		try {
			//~ Rcpp::Rcout << "\tTransitions\t"; Rcpp::Rcout.flush();	       
			SequencerInstance.simulate_transition_counts();
			//~ Rcpp::Rcout << "Matrices\t"; Rcpp::Rcout.flush();	       
            SequencerInstance.get_transition_instance().random_matrices(); 
						
			//~ Rcpp::Rcout << "Swapping\t"; Rcpp::Rcout.flush();	       			
			arma::rowvec jumping_likelihoods = trans(SequencerInstance.get_transition_instance().particle_swapping());
			
			//~ Rcpp::Rcout << "Saving\t"; Rcpp::Rcout.flush();	       			
			// Now save data
			if (burnin == 0) 
			{
				samples.slice(counter) = SequencerInstance.get_transition_instance().get_all_transition_matrices();
				jumping_probs_statistics(counter, arma::span::all) = jumping_likelihoods;
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
		    //~ Rcpp::Rcout << "Chib Saving\t"; Rcpp::Rcout.flush();	       			
		    chib_ML_estimation.save_transition_counts(SequencerInstance.get_transition_instance());
		    //~ Rcpp::Rcout << "next\n"; Rcpp::Rcout.flush();	       			
			
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

