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
	arma::uword rand_seed) : 
	SequencerInstance(observed_data, HMMtransitionMatrix(init_lambda, init_transition_graph, init_emission_matrix, rand_seed, 0.5), 
	                  rand_seed, amount, preparation, max_sequence_length)
{
	std::cout << "\nGibbs start ...!\n"; std::cout.flush();
	arma::uword nid = observed_data.n_individuals();
	Rcpp::Rcout << "\nNumber of individuals Gibbs = " << nid << "\n";
}


arma::mat Gibbs_Sampling::run(arma::uword burnin, arma::uword mc)
{
	arma::mat samples(mc, SequencerInstance.get_transition_instance().get_parameters().n_elem + 2); // first position is the amount of approximated data
    arma::uword counter = 0;
    arma::wall_clock zeit;
    zeit.tic();
    bool has_been_interrupted = false;
    
    Rcpp::Rcout << "\nStarting tempered Gibbs sampling with " << burnin << " burn-in and " << mc << " normal samples \n>";
    Rcpp::Rcout.flush();
   
	while (counter < mc)
	{
		try {
			// First step: simulate a hidden sequence set - matching the observed data
			arma::umat transition_counts;
			double amounts;
			arma::uword repeats = SequencerInstance.simulate_transition_counts(amounts);
			
			// Second step: simulate the transition matrix
			SequencerInstance.get_transition_instance().random_matrix();
			
			// Third step: change temperature
			SequencerInstance.get_transition_instance().random_temperature();
			
			// Now save data
			if (burnin == 0) 
			{
				samples(counter,0) = repeats;
				samples(counter,1) = amounts;
				samples(counter, arma::span(2,samples.n_cols-1)) = SequencerInstance.get_transition_instance().get_parameters();
			    counter++;
			} else {
				burnin--;
			}
			
		    if (counter > 0) {
			    if (counter % (mc/4) == 0) {
					 Rcpp::Rcout <<  counter*100/mc << "%"; Rcpp::Rcout.flush(); 
				} else {
				    if (counter % (mc/100) == 0)  Rcpp::Rcout <<  "-"; Rcpp::Rcout.flush(); 
				}
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
