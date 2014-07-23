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


class HMMImprovedMultipleSquirrel
{
	
	protected:
	
	typedef boost::random::taus88 base_pseudo_generator_type;
	typedef BasicTypes::IDRefSequenceCountTuple IDRefSequenceCountTuple;
	
	typedef boost::container::vector<IDRefSequenceCountTuple> proposal_vector_type;
 
	HMMparticle *associatedParticle;
	arma::uword parameter;
	
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
	
	double symmetric_function(arma::umat m1, arma::umat m2)
	{
		using namespace arma;
		return accu( ( conv_to<mat>::from(m1) - conv_to<mat>::from(m2) ) % 
		              ( conv_to<mat>::from(m1) - conv_to<mat>::from(m2) ) 		              
		              );
		
		//~ double lik1 = associatedParticle->single_path_likelihood(m1);
		//~ double lik2 = associatedParticle->single_path_likelihood(m2);
		//~ return square(lik1-lik2);
	}
	
	
	
	//*************************************************************************************************************************************************
	
	bool recursive_tree_search(
	    const IDRefSequenceCountTuple& target,
		arma::uword current_state,              	 	// state of the system before recursion
		arma::urowvec& current_sequence,			    // the sequence that will be transferred into the target
		bool ran_twice,                					// the Markov chain must run twice to simulate two chromosomes
		arma::urowvec& current_genotype,             	// genotype at the time before recusion
		arma::uword depth,                    			// depth of tree
		bool clockwise, 								// direction of tree-search
		bool& old_path_found,							// a switch for the first one
		base_pseudo_generator_type new_generator,	// pseudo_random generator, we use boost::random::taus88 because of its small size
		arma::uword& current_container_index,		    // to save the data
		proposal_vector_type& proposals 
	)
	{
		arma::uword width = associatedParticle->get_transition_matrix().n_cols;
		
		if (depth > associatedParticle->MAXDEPTH || arma::accu(current_genotype > associatedParticle->get_observed_data().get_genotype(target.get<1>())) > 0) 
		{
			// no solution found or possible to find
			return false;
		} 
		else if (current_state == width-1)
		{
			// an end has been reached
			if (!ran_twice) 
			{   // switch second chromosome
				current_sequence[depth] = 0;
				return recursive_tree_search(target, 0, current_sequence, true, current_genotype, depth+1, clockwise, 
				                              old_path_found, new_generator, current_container_index, proposals);
			} 
			else {
				
			    // the end is reached, we can finalize
				if (arma::accu(current_genotype != associatedParticle->get_observed_data().get_genotype(target.get<1>())) == 0) 
				{
					if (!old_path_found) {
						old_path_found = true;
						return false;
					}
					else {
						// everything is okay, we have a sequence
						
						IDRefSequenceCountTuple new_entry = target;
							
						new_entry.get<2>()  = arma::zeros<arma::urowvec>(depth);
						for (arma::uword i=0; i < depth; ++i) new_entry.get<2>()[i] = current_sequence[i];
						
						new_entry.get<3>() = arma::zeros<arma::umat>(width, width); // generate countmatrix
					
						for (arma::urowvec::const_iterator it = new_entry.get<2>().begin(); (it+1) != new_entry.get<2>().end(); ++it)
							if (*it != width-1) 
								new_entry.get<3>()(*it, *(it+1)) += 1;
						
						proposals[current_container_index] = new_entry;
						++current_container_index;
						return current_container_index >= parameter;
					}
				} 
				else {
					// the end is reached, but it is not okay
					return false;
				}				
			}	
		}
		 
		// third alternative, the interesting part
		
		// define a random number generator
	    boost::random::uniform_real_distribution<> pseudo_random_dist(0.0, 1.0);
	    typedef boost::random::variate_generator<base_pseudo_generator_type&, boost::random::uniform_real_distribution<> > taus_gentype;
	    taus_gentype pseudo_randoms(new_generator, pseudo_random_dist);
		
		// fill a vector
		arma::vec preorder(width);
		for (arma::vec::iterator oit = preorder.begin(); oit != preorder.end(); oit++) *oit = pseudo_randoms();
		// and find the order to traverse
		arma::uvec order;
		if (clockwise) 
			order = arma::sort_index(preorder, 0);
		else
			order = arma::sort_index(preorder, 1);
			
		arma::uword walker = 0;
		bool finished = false;
		
		// 
		if (!old_path_found) {
			
			while (order[walker] != target.get<2>()[depth] && walker < width) ++walker;
			
			//~ std::cout << " [" << target.get<2>()[depth] << "=" << order[walker] << "] "; std::cout.flush();
			//~ 
			if (walker == width) throw(std::runtime_error("This does not work!")); // emergency, to prevent infinite loops
			
		}
		
		do {
			if (associatedParticle->get_transition_matrix()(current_state, order[walker]) > 0) 
			{
				current_sequence[depth] = order[walker];
				current_genotype = current_genotype + associatedParticle->get_emission_matrix()(order[walker], arma::span::all);
				finished = recursive_tree_search(target, order[walker], current_sequence, ran_twice, current_genotype, depth+1, 
				                                 clockwise, old_path_found, new_generator, current_container_index, proposals);
				current_genotype = current_genotype - associatedParticle->get_emission_matrix()(order[walker], arma::span::all);
			}
			walker++;	
			if (depth == 1 && walker >= width) walker = 0;  // only with the first one re-do again and again until finished becomes true!
			
		} while (!finished && walker < width);
	
		return finished;
	}	
	
	//****************************************************************************************************
	
	
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
		return "Improved Multiple Proposal Squirrel Kernel";
	}
		
	HMMImprovedMultipleSquirrel(arma::uword param) : parameter(param) {}
	
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
		
		proposal_vector_type proposals(parameter);
		arma::uword prop_counter = 0;
		arma::vec likelihoods = arma::zeros<arma::vec>(parameter);
		
		arma::urowvec current_genotype;
		arma::urowvec new_sequence;
		bool clockwise;
		bool old_path_found;
		
		current_genotype = arma::zeros<arma::urowvec>(associatedParticle->get_emission_matrix().n_cols);
		new_sequence = arma::zeros<arma::urowvec>(associatedParticle->MAXDEPTH);
		base_pseudo_generator_type new_generator(unirandoms());
		clockwise = central_vargen() < 0.5;
		old_path_found = false;
		
		IDRefSequenceCountTuple old_state = *current_path_ref;
		
		bool success = recursive_tree_search( old_state, 0, new_sequence, false, 
											   current_genotype, 1, clockwise, old_path_found, new_generator, prop_counter, proposals);
		
		if (!success) throw(std::runtime_error("Some weird error occured!"));
		
		arma::vec::iterator lit = likelihoods.begin();
		for (proposal_vector_type::iterator it = proposals.begin(); it != proposals.end(); ++it, ++lit)
		{
			//~ (*lit) = associatedParticle->single_path_likelihood(it->get<3>()) + symmetric_function(old_state.get<3>(), it->get<3>());
			if (CollapsedVersion)
			{
				(*lit) = associatedParticle->collapsed_likelihood_absolute(old_state.get<3>(), it->get<3>());
			}
			else {	
			    (*lit) = associatedParticle->single_path_likelihood(it->get<3>());
			}
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
		prop_counter = 1;
		clockwise = central_vargen() < 0.5;			
		old_path_found = false;	
		base_pseudo_generator_type new_generator2(unirandoms());
		
		success = recursive_tree_search( selected_proposal, 0, new_sequence, false, 
									   current_genotype, 1, clockwise, old_path_found, new_generator2, prop_counter, proposals);
		
		if (!success) throw(std::runtime_error("Some weird error occured!"));
						
		lit = likelihoods.begin();
		for (proposal_vector_type::iterator it = proposals.begin(); it != proposals.end(); ++it, ++lit)
		{
			
			if (CollapsedVersion)
			{
				(*lit) = associatedParticle->collapsed_likelihood_absolute(selected_proposal.get<3>(), it->get<3>());
			}
			else {	
			    (*lit) = associatedParticle->single_path_likelihood(it->get<3>());
			}
			//~ 
			//~ 
			//~ (*lit) = associatedParticle->single_path_likelihood(it->get<3>()) + symmetric_function(selected_proposal.get<3>(), it->get<3>());;
		}	
				
		double backward_sum = logsum(likelihoods);
		
		if (log(central_vargen()) < forward_sum - backward_sum) 		
		{	
			accepted = true;
			(*current_path_ref) = selected_proposal;
		} else {
			accepted = true;
			if (CollapsedVersion) associatedParticle->update_countmatrix(selected_proposal.get<3>(), current_path_ref->get<3>());
		}
		
		return accepted;
	}
	
	
	//****************************************************************************************************	
	
	void associateParticle(HMMparticle *Particle)
	{
		associatedParticle = Particle;
	}
	
};
