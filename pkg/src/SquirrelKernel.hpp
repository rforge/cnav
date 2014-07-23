/*
 * SquirrelKernel.hpp
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

#pragma once
#include <iostream>
#include <string>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>
 
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>

#include "BasicTypes.hpp"
#include "HMMparticle.hpp"

class HMMsquirrelKernel
{	
	
	protected:
	
	typedef boost::random::taus88 base_pseudo_generator_type;
	typedef BasicTypes::IDRefSequenceCountTuple IDRefSequenceCountTuple;
 
	HMMparticle *associatedParticle;
	arma::uword parameter;
	
	bool recursive_tree_search(IDRefSequenceCountTuple& target,
		arma::uword current_state,              	 	// state of the system before recursion
		arma::urowvec& current_sequence,			    // the sequence that will be transferred into the target
		bool ran_twice,                					// the Markov chain must run twice to simulate two chromosomes
		arma::urowvec& current_genotype,             	// genotype at the time before recusion
		arma::uword depth,                    			// depth of tree
		bool clockwise, 								// direction of tree-search
		bool& old_path_found,							// a switch for the first one
		base_pseudo_generator_type new_generator	// pseudo_random generator, we use boost::random::taus88 because of its small size
    )
	{
		arma::uword width = associatedParticle->get_transition_matrix().n_cols;
		//~ std::cout << " " << depth << ":" current_state << " "; std::cout.flush();
		
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
				return recursive_tree_search(target, 0, current_sequence, true, current_genotype, depth+1, clockwise, old_path_found, new_generator);
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
						
						// Die Funktion subvec ist irgendwie kaputt
						target.get<2>()  = arma::zeros<arma::urowvec>(depth);
						for (arma::uword i=0; i < depth; ++i) target.get<2>()[i] = current_sequence[i];
						
						target.get<3>() = arma::zeros<arma::umat>(width, width); // generate countmatrix
					
						for (arma::urowvec::const_iterator it = target.get<2>().begin(); (it+1) != target.get<2>().end(); ++it)
							if (*it != width-1) 
								target.get<3>()(*it, *(it+1)) += 1;
						
						return true;
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
				finished = recursive_tree_search(target, order[walker], current_sequence, ran_twice, current_genotype, depth+1, clockwise, old_path_found, new_generator);
				current_genotype = current_genotype - associatedParticle->get_emission_matrix()(order[walker], arma::span::all);
			}
			walker++;	
			//~ std::cout << ".";; std::cout.flush();
			if (depth == 1 && walker >= width) walker = 0;  // only with the first one re-do again and again until finished becomes true!
			
		} while (!finished && walker < width);
	
		return finished;
	}	


	//**********************************************************************************************************************************************************
		
	public:
	
	static std::string KernelName() {
		return "Standard Squirrel Kernel";
	}
	
	HMMsquirrelKernel(arma::uword param) : parameter(param) {}
	
	
	virtual bool step(HMMparticle::sequence_vector_iterator_type current_path_ref, boost::random::mt19937& random_generator, bool CollapsedVersion)
	{		
		bool accepted = false;
		// define random number generators
		boost::uniform_int<arma::uword> unidist(0, std::numeric_limits<arma::uword>::max());
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_int<arma::uword> > unirandoms(random_generator, unidist);
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_real<> > central_vargen(random_generator, boost::uniform_real<>(0.0, 1.0));
	
		arma::urowvec current_genotype = arma::zeros<arma::urowvec>(associatedParticle->get_emission_matrix().n_cols);
		arma::urowvec new_sequence = arma::zeros<arma::urowvec>(associatedParticle->MAXDEPTH);
		
		bool clockwise = central_vargen() < 0.5;
		base_pseudo_generator_type new_generator(unirandoms());
		bool old_path_found = false;
		
		//~ std::cout << "\nAlt: "; std::cout.flush();
		//~ for (arma::uword i=0; i < current_path_ref->get<2>().n_elem; ++i ) std::cout << current_path_ref->get<2>()[i] << " ";
		//~ std::cout << "\nNeu: "; std::cout.flush();
		
		IDRefSequenceCountTuple proposal = *current_path_ref;
		
		bool success = recursive_tree_search( proposal, 0, new_sequence, false, current_genotype, 1, clockwise, old_path_found, new_generator);
		
		if (success) 
		{
			double likelihood_difference;
			if (!CollapsedVersion)
			{
				double likelihood_star = associatedParticle->single_path_likelihood(proposal.get<3>());
				double likelihood_old = associatedParticle->single_path_likelihood(current_path_ref->get<3>());
				likelihood_difference = likelihood_star - likelihood_old;
			}
			else {
				likelihood_difference = associatedParticle->collapsed_likelihood_difference(current_path_ref->get<3>(), proposal.get<3>());
			}
			if (log(central_vargen()) < likelihood_difference) 
			{
				accepted = true;
				if (CollapsedVersion) associatedParticle->update_countmatrix(current_path_ref->get<3>(), proposal.get<3>());
				(*current_path_ref) = proposal;
			}
		}
		else
			throw(std::runtime_error("Some weird error occured!"));
		
		return accepted;
		//~ for (arma::uword i=0; i < current_path_ref->get<2>().n_elem; ++i ) std::cout << current_path_ref->get<2>()[i] << " ";
		//~ std::cout << " !\n"; std::cout.flush();
	}
	
	
	void associateParticle(HMMparticle *Particle)
	{
		associatedParticle = Particle;
	}
};
