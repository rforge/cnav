/*
 * MultipleSquirrel.hpp
 * 
 * Copyright 2014 Andreas Recke <andreas@Dianeira>
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

// Multiple Try

#include <string>
#include <boost/utility.hpp> 
#include <boost/tuple/tuple.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/container/vector.hpp>

#include "SquirrelKernel.hpp"


class HMMMultipleSquirrel : public HMMsquirrelKernel
{
	
	
	
	//*************************************************************************************************************************************************
	
	class log_discrete {
		
		// this class uses a CFTP sampler to sample from logarithmic unnormalized likelihoods
		
		arma::vec likelihoods;
			
		public:
		
		log_discrete(arma::vec unweighted_densities) : likelihoods(unweighted_densities) {}
		
		arma::uword operator()(BasicTypes::base_generator_type& random_generator)
		{
			arma::uword state1 = 1, state2 = likelihoods.n_elem-1;
		    
			boost::random::uniform_real_distribution<> pseudo_random_dist(0.0, 1.0);
		    boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > 
				pseudo_randoms(random_generator, pseudo_random_dist);
			
			boost::uniform_int<arma::uword> unidist(0, likelihoods.n_elem-1);
	   	    boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_int<arma::uword> > unirandoms(random_generator, unidist);
	
			while (state1 != state2) {
				
				arma::uword abb1 = unirandoms(), abb2 = unirandoms();
				if (abb1 <= abb2) {
					double test = log(pseudo_randoms());
					if (test < likelihoods[abb1] - likelihoods[state1]) state1 = abb1;
					if (test < likelihoods[abb2] - likelihoods[state2]) state2 = abb2;
				}
			}
			
			return state1;
		}
		
    };
	
	
	//*************************************************************************************************************************************************
	
	
	double logsum(const arma::vec& logdens)
	{
		using namespace arma;
		vec sortierung = sort(logdens);
		vec::const_iterator it = sortierung.begin();
		
		double summe = *it; ++it;
		while (it != sortierung.end())
		{
			if ((*it) > summe) 
				summe = log( 1.0 + exp( summe - (*it) ) ) + (*it);
			else
				summe = log( 1.0 + exp( (*it) - summe ) ) + summe;
			++it;
		}
		
		return summe;
	}
		
	
	//****************************************************************************************************

    double square(double x)
    {
		return x*x;
	}

	//****************************************************************************************************
	
	public:
	
	static std::string KernelName() {
		return "Standard Multiple Proposal Squirrel Kernel";
	}
	
	HMMMultipleSquirrel(arma::uword param) : HMMsquirrelKernel(param) {}
	
	//****************************************************************************************************
	
	
	virtual bool step(HMMparticle::sequence_vector_iterator_type current_path_ref, boost::random::mt19937& random_generator, bool CollapsedVersion)
	{		
		bool accepted = false;
		//~ std::cout << "\nhier geht es los"; std::cout.flush();
		// define random number generators
		boost::uniform_int<arma::uword> unidist(0, std::numeric_limits<arma::uword>::max());
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_int<arma::uword> > unirandoms(random_generator, unidist);
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_real<> > central_vargen(random_generator, boost::uniform_real<>(0.0, 1.0));
	
		//~ std::cout << "\nAlt: "; std::cout.flush();
		//~ for (arma::uword i=0; i < current_path_ref->get<2>().n_elem; ++i ) std::cout << current_path_ref->get<2>()[i] << " ";
		//~ std::cout << "\nNeu: "; std::cout.flush();
		
		// forward
		
		boost::container::vector<IDRefSequenceCountTuple> proposals(parameter);
		arma::vec likelihoods = arma::zeros<arma::vec>(parameter);
		
		arma::urowvec current_genotype;
		arma::urowvec new_sequence;
		bool clockwise;
		bool old_path_found;
		
		//~ double ref_likelihood = associatedParticle->single_path_likelihood(current_path_ref->get<3>(), 1.0);
		
		//~ std::cout << "\nStep " << parameter << " >"; std::cout.flush();
				
		for (arma::uword i = 0; i < parameter; ++i)
		{	
			current_genotype = arma::zeros<arma::urowvec>(associatedParticle->get_emission_matrix().n_cols);
		    new_sequence = arma::zeros<arma::uvec>(associatedParticle->MAXDEPTH);
			base_pseudo_generator_type new_generator(unirandoms());
			clockwise = central_vargen() < 0.5;
			old_path_found = false;
			
			IDRefSequenceCountTuple next = *current_path_ref;
			bool success = recursive_tree_search( next, 0, new_sequence, false, current_genotype, 1, clockwise, old_path_found, new_generator);
			if (success) 
			{
				double star_likelihood;
				
				if (CollapsedVersion)
				{
					star_likelihood = associatedParticle->collapsed_likelihood_absolute(current_path_ref->get<3>(), next.get<3>());
				}
				else {	
				    star_likelihood = associatedParticle->single_path_likelihood(next.get<3>());
				}
				likelihoods[i] = star_likelihood; // * associatedParticle->get_lambda() +  square(star_likelihood - ref_likelihood);
				//~ std::cout << likelihoods[i]  << " "; std::cout.flush();
				// proposals.push_back(next);
				proposals[i] = next;
			}
			else
				throw(std::runtime_error("Some weird error occured!"));
		}
		
		//~ std::cout << "< Finalized\n >"; std::cout.flush();
		//~ throw "Bye";
		double forward_sum = logsum(likelihoods);
		
		//~ std::cout << " -> forward_sum = " << forward_sum << "\n>"; std::cout.flush();
		
		// likelihoods = arma::exp(likelihoods - forward_sum);  // normalization
		// boost::random::discrete_distribution<> diskrete(likelihoods.begin(), likelihoods.end());
		// boost::variate_generator<boost::random::mt19937&, boost::random::discrete_distribution<> > diskrete_randoms(random_generator, diskrete); 	
		
		log_discrete diskrete_randoms(likelihoods);
		arma::uword selektion = diskrete_randoms(random_generator);
		IDRefSequenceCountTuple selected_proposal = proposals[selektion];
		// backward
		if (CollapsedVersion) associatedParticle->update_countmatrix(current_path_ref->get<3>(), selected_proposal.get<3>());
		
		proposals[0] = *current_path_ref;
		likelihoods[0] = likelihoods[selektion];
		// ref_likelihood * associatedParticle->get_lambda() + square( associatedParticle->single_path_likelihood(selected_proposal.get<3>(), 1.0) - ref_likelihood);
				
		for (arma::uword i = 1; i < parameter; ++i)
		{
			base_pseudo_generator_type new_generator(unirandoms());
			current_genotype = arma::zeros<arma::urowvec>(associatedParticle->get_emission_matrix().n_cols);
		    new_sequence = arma::zeros<arma::uvec>(associatedParticle->MAXDEPTH);
			clockwise = central_vargen() < 0.5;
			old_path_found = false;
			
			IDRefSequenceCountTuple next = selected_proposal;
			bool success = recursive_tree_search( next, 0, new_sequence, false, current_genotype, 1, clockwise, old_path_found, new_generator);
			
			if (success) 
			{
				double star_likelihood;
				
				if (CollapsedVersion)
				{
					star_likelihood = associatedParticle->collapsed_likelihood_absolute(selected_proposal.get<3>(), next.get<3>());
				}
				else {	
				    star_likelihood = associatedParticle->single_path_likelihood(next.get<3>());
				}
				likelihoods[i] = star_likelihood; 
				
				//~ 
				//~ double star_likelihood = associatedParticle->single_path_likelihood(next.get<3>(), 1.0);
				//~ likelihoods[i] = star_likelihood * associatedParticle->get_lambda() +
				                 //~ square(star_likelihood - ref_likelihood); 
				//~ likelihoods[i] = associatedParticle->single_path_likelihood(next.get<3>(), associatedParticle->get_lambda());
				//~ std::cout << likelihoods[i]  << " "; std::cout.flush();
				proposals[i] = next;
			}
			else
				throw(std::runtime_error("Some weird error occured!"));
		}
		double backward_sum = logsum(likelihoods);
		//~ std::cout << " -> backward_sum = " << backward_sum << " -> selected " << selektion << "\n>"; std::cout.flush();
		
		if (log(central_vargen()) < forward_sum - backward_sum) 
		{	
			accepted = true;
			(*current_path_ref) = selected_proposal;
		} else {
			if (CollapsedVersion) associatedParticle->update_countmatrix(selected_proposal.get<3>(), current_path_ref->get<3>());
		}
		return accepted;
	}
	
};
