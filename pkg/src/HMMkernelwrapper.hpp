/*
 * HMMkernelwrapper.hpp
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

/* 
 * This class is to wrap the Markov path walking kernels nicely. 
 * The general idea here is to use multithreading, but to encapsulate this conceptionally
 */

#include <string>
#include <iostream>
#include <algorithm>

#pragma once
#include <boost/utility.hpp>
#include <boost/scoped_array.hpp> 
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>
#include <boost/thread.hpp> 
#include <boost/bind.hpp>
#include "HMMparticle.hpp"


// A base class that monitors the performance

class kernel_monitor_class
{
	arma::urowvec r_uses, r_moves, r_proposals, ref_affected;
	arma::rowvec r_affected;
	std::string kernel_name;
	
	protected:
	
	void registerUses(arma::uword temp_ref) { ++r_uses[temp_ref]; }
	void registerMoves(arma::uword temp_ref) { ++r_moves[temp_ref]; }
	void registerProposals(arma::uword temp_ref) { ++r_proposals[temp_ref]; }
	
	arma::uvec affected;
	
	void init_affected(arma::uword n_elements)
	{
		affected = arma::zeros<arma::uvec>(n_elements);
	}
	
	void evaluate_affected(arma::uword temp_ref)
	{
		r_affected[temp_ref] += double(accu(affected > 0)) / double(affected.n_elem);
		++ref_affected[temp_ref];	
	}
	
	void registerAffected(arma::uword element_index)
	{
		affected[element_index] += 1.0;
	}
	
	
	
	public:
	
	kernel_monitor_class(arma::uword temperature_levels, std::string myName) :
	  kernel_name(myName)
	{
		using namespace arma;
		r_moves = zeros<urowvec>(temperature_levels);
		r_proposals = zeros<urowvec>(temperature_levels);
		r_uses =  zeros<urowvec>(temperature_levels);
		r_affected = zeros<rowvec>(temperature_levels);
		ref_affected  = zeros<urowvec>(temperature_levels);
		
	}
	
	
	
	void print_monitor()
	{
		Rcpp::Rcout << "\nKernel Monitor for " << kernel_name << " Kernel:\n";
		for (arma::uword i = 0; i < r_uses.n_elem; ++i) 
			Rcpp::Rcout << "Temperature " << i << ": " << round(double(r_proposals[i])/double(r_uses[i])*10000.0)/100.0 << "% proposals"
			                                   << ", from these, " << round(double(r_moves[i])/double(r_proposals[i])*10000.0)/100.0 << "% are accepted. "
			                                   << " these affected " << round(r_affected[i] / double(ref_affected[i])*10000.0)/100.0 << "% of data points.\n";
		Rcpp::Rcout << "\n"; Rcpp::Rcout.flush();
	}
	
};



//**************************************************************************************************************************************************************

/* The class KernelType typically implements a method called "step" with parameters
 * IDRefSequenceCountTuple current_target 
 * boost::random::mt19937& random_generator
 * 
 * Another method is MCMCkernel.associateParticle(myParticle) that extracts information needed by the proposal mechanism
 */

//**************************************************************************************************************************************************************

template <typename KernelType, bool CollapsedVersion = false> class HMMkernelwrapper : public boost::noncopyable, public kernel_monitor_class
{
	unsigned n_threads; 							// number of threads
	boost::scoped_array<boost::thread> producers;	// takes up threads
	bool finishedThread;						    // to finalize
	bool update_thread_alive;					    // to make it go
	unsigned at_work;								// the number of sequences still processed
	boost::mutex updateMutex;					    // used in combination with the condition variable
	boost::condition_variable_any updateCondition;  // to wake up threads if neccessary
	
	HMMparticle::sequence_vector_iterator_type run_iterator, final_run_iterator;
	
	HMMparticle *current_particle;
	KernelType MCMCkernel;
		
    //*********************************************************************************************************************************************************		
	void cycling_thread(arma::uword rand_seed)
	{
		boost::random::mt19937 random_generator(rand_seed);
		while (!finishedThread) 
		{
			// first: wait until next round
			boost::unique_lock<boost::mutex> countDownLock(updateMutex);
			while (!finishedThread && !update_thread_alive) updateCondition.wait(countDownLock);
		
		    if (run_iterator == final_run_iterator)
				update_thread_alive = false;
			else
			{
				HMMparticle::sequence_vector_iterator_type aktuell = run_iterator;
				++run_iterator;
				++at_work;
				registerUses(current_particle->get_lambda_ref());
				registerProposals(current_particle->get_lambda_ref());
				
				//~ std::cout << "-> ";
				//~ for (arma::urowvec::const_iterator it = aktuell->get<2>().begin(); it != aktuell->get<2>().end(); ++it) std::cout << " " << (*it);
				//~ std::cout.flush();
				
				if (!CollapsedVersion) countDownLock.unlock();   // it's a little bit useless to use threads then ... but easy to write a program
		    
				if (MCMCkernel.step(aktuell, random_generator, CollapsedVersion)) {
					registerMoves(current_particle->get_lambda_ref());
					registerAffected(aktuell->get<0>());
				}
				
				//~ std::cout << "=>";
				//~ for (arma::urowvec::const_iterator it = aktuell->get<2>().begin(); it != aktuell->get<2>().end(); ++it) std::cout << " " << (*it);
				//~ std::cout << "\n"; std::cout.flush();
				
				
				if (!CollapsedVersion) countDownLock.lock();
				--at_work;
			}	 		
			updateCondition.notify_all();
		}
	}
	
	//*********************************************************************************************************************************************************	
	
	public:
	
	HMMkernelwrapper(arma::uword rand_seed, arma::uword temp_levels = 1, arma::uword param = 0) : MCMCkernel(param), 
		kernel_monitor_class(temp_levels, KernelType::KernelName() )
	{
		// initiate a set of threads that works through the random tree list to simulate new states
		n_threads = boost::thread::hardware_concurrency();
		producers.reset(new boost::thread[n_threads]);
		finishedThread = false;
		at_work = 0;
		update_thread_alive = false;
    	
		for (unsigned i=0; i < n_threads; i++) {
			producers[i] = boost::thread(boost::bind(&HMMkernelwrapper::cycling_thread, this, rand_seed+i));
		}
	}
	
	
	//*********************************************************************************************************************************************************	
	
	~HMMkernelwrapper()
	{
		finishedThread = true;
		updateCondition.notify_all();
		for (unsigned i = 0; i < n_threads; i++) producers[i].join();
	}

	//*********************************************************************************************************************************************************	
	
	void apply_kernel(HMMparticle *myParticle) 
	{
		// parallel simulation!
		boost::unique_lock<boost::mutex> countDownLock(updateMutex);   // first ... try to get lock
		current_particle = myParticle;
		update_thread_alive = true;
		at_work = 0;
		run_iterator = myParticle->begin();
		final_run_iterator = myParticle->end();
		MCMCkernel.associateParticle(myParticle);
		init_affected(myParticle->size_of_path_vector());
		
		//~ std::cout << "Start ";std::cout.flush();
		// wake up the other threads	
		updateCondition.notify_all();
		while (update_thread_alive || at_work > 0) updateCondition.wait(countDownLock);  // release lock and wait
		//~ std::cout << "End\n ";std::cout.flush();
		current_particle->summarize_Markov_paths();
		evaluate_affected(current_particle->get_lambda_ref());
	} 
};


//**************************************************************************************************************************************************************
//**************************************************************************************************************************************************************
//**************************************************************************************************************************************************************

class IncompleteGibbsKernel
{
	// just a dummy for specialization 
};

//**************************************************************************************************************************************************************


template <bool CollapsedVersion> class HMMkernelwrapper<IncompleteGibbsKernel, CollapsedVersion> : boost::noncopyable, public kernel_monitor_class
{
	
	arma::uword MAXDEPTH;
	HMMparticle *current_particle;
	
	unsigned n_threads; 							// number of threads
	boost::scoped_array<boost::thread> producers;	// takes up threads
	bool finishedThread;						    // to finalize
	unsigned at_work;								// the number of sequences still processed
	boost::mutex updateMutex;					    // used in combination with the condition variable
	boost::condition_variable_any updateCondition;  // to wake up threads if neccessary
	
	unsigned countdown_start;				        // value to reset the countdown to
	unsigned countdown;						    // number of tries
	
	arma::mat generator_transition_matrix;			// needed to generate new paths
	
	/* ****************************************************************************************************************
	 * 
	 * name: simulate_markov_sequence
	 * @param: transition_matrix - a matrix with transition probabilities
	 * @param: rand_gen - stores a random generator reference
	 * @return: a tuple including sequence, genotype and validity 
	 * 
	 */
	 
	boost::mutex coutMutex;	
	 
	HMMparticle::IDRefSequenceCountTuple produce_random_sequence(boost::random::mt19937& rand_gen)
	{
		using namespace arma;
		
		urowvec sequence(current_particle->MAXDEPTH);
		sequence[0] = 0;
		
		urowvec sim_genotype = zeros<urowvec>(current_particle->get_emission_matrix().n_cols);
		umat transitSave = zeros<umat>(generator_transition_matrix.n_rows, generator_transition_matrix.n_cols);
		uword refGenotype = 0;
		uword endstate = generator_transition_matrix.n_rows-1;
		bool validity = true;
		
		uword j = 0, state = 0;
		bool ran_twice = false;
		while (state != endstate && j < current_particle->MAXDEPTH - 2) 
		{
			// calculate new state
			rowvec transition_probabilities = generator_transition_matrix(state, span::all);
			boost::random::discrete_distribution<> diskrete(transition_probabilities.begin(), transition_probabilities.end());
			boost::variate_generator<boost::random::mt19937&, boost::random::discrete_distribution<> > diskrete_randoms(rand_gen, diskrete); 	
			uword newstate = diskrete_randoms();
			
			++j;
			sequence[j] = newstate;
			transitSave(state, newstate) = transitSave(state, newstate) + 1;
		    state = newstate;
			// add emission to resulting genotype
		    sim_genotype = sim_genotype + current_particle->get_emission_matrix()(state,span::all);
		    
		    if (!ran_twice && state == endstate)   // the HMM is simulated twice!
		    {
				++j;
				sequence[j] = 0;
				state = 0;
				ran_twice = true;
			}
			
		}
	    
	    if (j == current_particle->MAXDEPTH-1 && state != endstate) validity = false;
	    
	    HMMparticle::IDRefSequenceCountTuple result(0, 0, sequence.subvec(0, j), transitSave, validity);
	    if (validity) current_particle->get_observed_data().get_ref(result, sim_genotype);
		
		//~ coutMutex.lock();
	    //~ std::cout << "\n>";
	    //~ for (arma::uword ix = 0; ix < sim_genotype.n_elem; ++ix) std::cout << sim_genotype[ix] << " ";
	    //~ 
	    //~ std::cout << "\t>>";
	    //~ for (arma::uword ix = 0; ix < result.get<2>().n_elem; ++ix) std::cout << result.get<2>()[ix] << " ";
	    //~ 
	    //~ 
	    //~ std::cout.flush();
	    //~ coutMutex.unlock();
	    
		
		return result;
	}	
	
	//**********************************************************************************************************************************************************
	
		
	void cycling_thread(arma::uword rand_seed)
	{
		boost::random::mt19937 random_generator(rand_seed);
		
		boost::random::uniform_real_distribution<> pseudo_random_dist(0.0, 1.0);
	    boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > pseudo_randoms(random_generator, pseudo_random_dist);
		
		while (!finishedThread) 
		{
			// first: wait until next round
			boost::unique_lock<boost::mutex> countDownLock(updateMutex);
			while (!finishedThread && countdown == 0) updateCondition.wait(countDownLock);
		    
			if (countdown > 0)
			{	
				++at_work;
				--countdown;
				registerUses(current_particle->get_lambda_ref());
				
				countDownLock.unlock();
				HMMparticle::IDRefSequenceCountTuple proposal = produce_random_sequence(random_generator);
				
				
				if (proposal.get<4>())
				{
					// identify possible targets
					arma::vec preorder(current_particle->size());
		            for (arma::vec::iterator oit = preorder.begin(); oit != preorder.end(); oit++) *oit = pseudo_randoms();
					// and find the order to traverse
					arma::uvec order = arma::sort_index(preorder, 0);
					
					HMMparticle::sequence_vector_iterator_type it = current_particle->begin();
					arma::uvec::const_iterator oit = order.begin();
					while (oit != order.end() && (it + (*oit))->get<1>() != proposal.get<1>() ) ++oit;
					
					// then use the proposal
					countDownLock.lock();
					
					if (oit != order.end()) 
					{
						proposal.get<0>() = (it + (*oit))->get<0>();
							
						// we need to determine whether we can replace it
						double likelihood_difference;
						if (CollapsedVersion)
						{
							likelihood_difference = current_particle->collapsed_likelihood_difference((it + (*oit))->get<3>(), proposal.get<3>());
						} 
						else {
							double new_likelihood = current_particle->single_path_likelihood(proposal.get<3>());
							double old_likelihood = current_particle->single_path_likelihood((it + (*oit))->get<3>());
							likelihood_difference = new_likelihood - old_likelihood;
						}
					
						double new_gen_likelihood = current_particle->any_single_path_likelihood( proposal.get<3>(), generator_transition_matrix, 1.0);
						double old_gen_likelihood = current_particle->any_single_path_likelihood( (it + (*oit))->get<3>(), generator_transition_matrix, 1.0);
						
						registerProposals(current_particle->get_lambda_ref());
						
						if ( log(pseudo_randoms()) < likelihood_difference - (new_gen_likelihood - old_gen_likelihood) )
						{
							*(it + (*oit)) = proposal;	
							registerMoves(current_particle->get_lambda_ref());	
							registerAffected( (it + (*oit))->get<0>());				
						}								
						
					
					}	
				} 
				else 	
				{
					countDownLock.lock();
				}
					
				--at_work;
			}	 		
			updateCondition.notify_all();
		}
	}
	
	
	//**********************************************************************************************************************************************************
	
	public:
	
	HMMkernelwrapper(arma::uword rand_seed, arma::uword temp_levels, arma::uword param = 0) : countdown_start(param), 
		kernel_monitor_class(temp_levels, "Incomplete Gibbs Sampler")
	{
		// initiate a set of threads that works through the random tree list to simulate new states
		n_threads = boost::thread::hardware_concurrency();
		producers.reset(new boost::thread[n_threads]);
		finishedThread = false;
		at_work = 0;
		countdown = 0;
    	
		for (unsigned i=0; i < n_threads; i++) {
			producers[i] = boost::thread(boost::bind(&HMMkernelwrapper::cycling_thread, this, rand_seed+i));
		}
	}
	
	
	//*********************************************************************************************************************************************************	
	
	~HMMkernelwrapper()
	{
		finishedThread = true;
		updateCondition.notify_all();
		for (unsigned i = 0; i < n_threads; i++) producers[i].join();
	}

	
	//**********************************************************************************************************************************************************
	
	
	void apply_kernel(HMMparticle *myParticle) 
	{
		// parallel simulation!
		boost::unique_lock<boost::mutex> countDownLock(updateMutex);   // first ... try to get lock
		// Rcpp::Rcout << "#1";Rcpp::Rcout.flush();
		current_particle = myParticle;
		// generator_transition_matrix = current_particle->get_tempered_random_matrix();  // Alternative 1
		generator_transition_matrix = current_particle->get_transition_matrix();	   // Alternative 2
		// generator_transition_matrix = current_particle->get_any_random_matrix();    // Alternative 3
		at_work = 0;
		countdown = countdown_start;
		init_affected(myParticle->size_of_path_vector());
		
		// wake up the other threads	
		updateCondition.notify_all();
		// Rcpp::Rcout << "#2";Rcpp::Rcout.flush();
		while (countdown > 0 || at_work > 0) updateCondition.wait(countDownLock);  // release lock and wait
		//~ Rcpp::Rcout << "#3";Rcpp::Rcout.flush();
		current_particle->summarize_Markov_paths();
		evaluate_affected(current_particle->get_lambda_ref());
	}
};

//**********************************************************************************************************************************************************
//**********************************************************************************************************************************************************
//**********************************************************************************************************************************************************
//**********************************************************************************************************************************************************
//**********************************************************************************************************************************************************
//**********************************************************************************************************************************************************



//**********************************************************************************************************************************************************
	
class SquirrelSwarmKernel
{
	typedef int squirrel_swarm_dummy_type;

};

//**************************************************************************************************************************************************************

template <bool CollapsedVersion> class HMMkernelwrapper<SquirrelSwarmKernel, CollapsedVersion> : boost::noncopyable, public kernel_monitor_class
{
	typedef HMMparticle::sequence_vector_type sequence_vector_type;  // transfer typedef to this class
	
	sequence_vector_type proposal_vector;
	HMMparticle::sequence_vector_iterator_type run_iterator, final_run_iterator;
	
	boost::mutex countmatrix_mutex;
	arma::umat countmatrix_summary;
	arma::uword proposal_counter;
	double current_proposal_threshold;
	
	arma::uvec affect_temp;
		
	arma::uword MAXDEPTH;
	HMMparticle *associatedParticle;
	
	unsigned n_threads; 							// number of threads
	boost::scoped_array<boost::thread> producers;	// takes up threads
	bool finishedThread;						    // to finalize
	unsigned at_work;								// the number of sequences still processed
	bool update_thread_alive;					    // to make it go	
	boost::mutex updateMutex;					    // used in combination with the condition variable
	boost::condition_variable_any updateCondition;  // to wake up threads if neccessary
	
	arma::uword maximum_jumplength;
	arma::uword parameter;
	
	typedef BasicTypes::IDRefSequenceCountTuple IDRefSequenceCountTuple;
	typedef boost::random::taus88 base_pseudo_generator_type;
	
	
	boost::random::mt19937 central_random_generator;
	
	/* **************************************************************************************************************** */
	
	void set_individual_parameters (arma::uword individual_jumplength)
	{
		maximum_jumplength = individual_jumplength;
	}
	
	/* **************************************************************************************************************** */
		
	bool recursive_tree_search(IDRefSequenceCountTuple& target,
		arma::uword current_state,              	 	// state of the system before recursion
		arma::urowvec& current_sequence,			    // the sequence that will be transferred into the target
		bool ran_twice,                					// the Markov chain must run twice to simulate two chromosomes
		arma::urowvec& current_genotype,             	// genotype at the time before recusion
		arma::uword depth,                    			// depth of tree
		bool clockwise, 								// direction of tree-search
		bool& old_path_found,							// a switch for the first one
		base_pseudo_generator_type new_generator,	// pseudo_random generator, we use boost::random::taus88 because of its small size
		arma::uword& jumplength							// the length of the jump
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
				return recursive_tree_search(target, 0, current_sequence, true, current_genotype, depth+1, clockwise, old_path_found, new_generator, jumplength);
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
						
						// first check, whether we can save it:
						if (jumplength > 0) {
							--jumplength;
							return false;
						}
						else {
							
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
				           clockwise, old_path_found, new_generator, jumplength);
				current_genotype = current_genotype - associatedParticle->get_emission_matrix()(order[walker], arma::span::all);
			}
			walker++;	
			if (depth == 1 && walker >= width) walker = 0;  // only with the first one re-do again and again until finished becomes true!
			
		} while (!finished && walker < width);
	
		return finished;
	}	
	
	
	/* **************************************************************************************************************** */
	
	void step(HMMparticle::sequence_vector_iterator_type current_path_ref, boost::random::mt19937& random_generator)
	{		
		// define random number generators
		boost::uniform_int<arma::uword> unidist(0, std::numeric_limits<arma::uword>::max());
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_int<arma::uword> > unirandoms(random_generator, unidist);
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_real<> > central_vargen(random_generator, boost::uniform_real<>(0.0, 1.0));
	
		arma::urowvec current_genotype = arma::zeros<arma::urowvec>(associatedParticle->get_emission_matrix().n_cols);
		arma::urowvec new_sequence = arma::zeros<arma::urowvec>(associatedParticle->MAXDEPTH);
		
		boost::uniform_int<arma::uword> jumpdist(1, maximum_jumplength);
		boost::random::variate_generator<boost::random::mt19937&, boost::uniform_int<arma::uword> > jumplength(random_generator, jumpdist);
		
		// initialize tree
		bool clockwise = central_vargen() < 0.5;
		base_pseudo_generator_type new_generator(unirandoms());
		bool old_path_found = false;
		arma::uword aimed_jump = jumplength();
		
		IDRefSequenceCountTuple proposal = *current_path_ref;
		
		bool success = recursive_tree_search( proposal, 0, new_sequence, false, current_genotype, 1, clockwise, old_path_found, new_generator, aimed_jump);
		
		if (success)
		{
			countmatrix_mutex.lock();
			countmatrix_summary = (countmatrix_summary + proposal.get<3>()) - current_path_ref->get<3>();
			(*current_path_ref) = proposal;
			countmatrix_mutex.unlock();
		}
		else
			throw(std::runtime_error("Some weird error occured!"));
	}
		
	
	//**********************************************************************************************************************************************************		
	void cycling_thread(arma::uword rand_seed)
	{
		boost::random::mt19937 random_generator(rand_seed);
		boost::random::uniform_real_distribution<> random_dist(0.0, 1.0);
	    boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > uniform(random_generator, random_dist);
	    		
		while (!finishedThread) 
		{
			// first: wait until next round
			boost::unique_lock<boost::mutex> countDownLock(updateMutex);
			while (!finishedThread && !update_thread_alive) updateCondition.wait(countDownLock);
		
		    if (run_iterator == proposal_vector.end())
				update_thread_alive = false;
			else
			{
				HMMparticle::sequence_vector_iterator_type aktuell = run_iterator;
				countmatrix_summary = countmatrix_summary - aktuell->get<3>();
				++run_iterator;
				++at_work;
				
				countDownLock.unlock();   
		    
				if ( uniform() < current_proposal_threshold ) 
				{
					++proposal_counter;
					step(aktuell, random_generator);
					++affect_temp[aktuell->get<0>()];
				}
				
				countDownLock.lock();
				countmatrix_summary = countmatrix_summary + aktuell->get<3>();
			
				--at_work;
			}	 		
			updateCondition.notify_all();
		}
	}
	
	//**********************************************************************************************************************************************************	
	
	
	
	
	//**********************************************************************************************************************************************************
	
	public:
	
	HMMkernelwrapper(arma::uword rand_seed, arma::uword temp_levels = 1, arma::uword param = 0) : parameter(param),
		kernel_monitor_class(temp_levels, "SquirrelSwarmKernel"), central_random_generator(rand_seed)
		
	{
		set_individual_parameters (100);
		// initiate a set of threads that works through the random tree list to simulate new states
		n_threads = boost::thread::hardware_concurrency();
		
		producers.reset(new boost::thread[n_threads]);
		finishedThread = false;
		at_work = 0;
    	update_thread_alive = false;
    	
		for (unsigned i=0; i < n_threads; i++) {
			producers[i] = boost::thread(boost::bind(&HMMkernelwrapper::cycling_thread, this, rand_seed+i));
		}
		
	}
	
	
	//*********************************************************************************************************************************************************	
	
	~HMMkernelwrapper()
	{
		finishedThread = true;
		updateCondition.notify_all();
		for (unsigned i = 0; i < n_threads; i++) producers[i].join();
	}


	
	//**********************************************************************************************************************************************************
	
	
	void apply_kernel(HMMparticle *myParticle) 
	{
		boost::unique_lock<boost::mutex> countDownLock(updateMutex);   // first ... try to get lock
		boost::random::uniform_real_distribution<> random_dist(0.0, 1.0);
	    boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > uniform(central_random_generator, random_dist);
				
		associatedParticle = myParticle;		
		proposal_vector.resize(myParticle->size_of_path_vector());
		std::copy(myParticle->begin(), myParticle->end(), proposal_vector.begin() );
		
		countmatrix_summary = myParticle->get_transition_counts();
		if (myParticle->get_lambda() == 0.0) 
			current_proposal_threshold = 1.0;
		else
			current_proposal_threshold = 1.0 / double(myParticle->size_of_path_vector()) / myParticle->get_lambda();
		
		init_affected(associatedParticle->size_of_path_vector());
		//~ std::cout << "\n"; std::cout.flush();		
		
		for (arma::uword i = 0; i < parameter; ++i)
		{
			//~ std::cout << i << " "; std::cout.flush();
			affect_temp = arma::zeros<arma::uvec>(myParticle->size_of_path_vector());
			update_thread_alive = true;
		    at_work = 0;
		    proposal_counter = 0;
		    run_iterator = proposal_vector.begin();
		
		    updateCondition.notify_all();
            //~ std::cout << " ==> "; std::cout.flush();
			while (update_thread_alive || at_work > 0) updateCondition.wait(countDownLock);  // release lock and wait
			//~ std::cout << " <== "; std::cout.flush();			
			registerUses(associatedParticle->get_lambda_ref());
			registerProposals(associatedParticle->get_lambda_ref());
			
			double likelihood, likelihood_star;
			
			if (CollapsedVersion) {
				//~ std::cout << " Collapsed "; std::cout.flush();			
				likelihood_star = myParticle->collapsed_likelihood(countmatrix_summary);
				likelihood = myParticle->collapsed_likelihood(myParticle->get_transition_counts());
			} 
			else {
				//~ std::cout << " NonCollapsed "; std::cout.flush();			
				likelihood_star = myParticle->countmatrix_likelihood(countmatrix_summary);
				likelihood = myParticle->countmatrix_likelihood();
			}
			
			//~ std::cout << " MC move "; std::cout.flush();			
			
			if ( log(uniform()) < likelihood_star - likelihood )
			{
				//~ std::cout << " copy forward "; std::cout.flush();			
				std::copy(proposal_vector.begin(), proposal_vector.end(), myParticle->begin());
				//~ std::cout << " set transition forward "; std::cout.flush();			
				associatedParticle->set_transition_counts(countmatrix_summary);
				//~ std::cout << " Register forward "; std::cout.flush();			
				registerMoves(associatedParticle->get_lambda_ref());
				//~ std::cout << " Register forward affected"; std::cout.flush();			
				//~ affect_temp.print("AFT"); std::cout.flush();
				for (arma::uword jui = 0; jui < affect_temp.n_elem; ++jui) if (affect_temp[jui]>0) registerAffected(jui);
			} 
			else 	{
				//~ std::cout << " set transition backward "; std::cout.flush();			
				countmatrix_summary = myParticle->get_transition_counts();
				
				//~ std::cout << " copy backward "; std::cout.flush();			
				std::copy(myParticle->begin(), myParticle->end(), proposal_vector.begin() );
				//~ std::cout << " no move "; std::cout.flush();			
			}
		}
		
		//~ std::cout << ":) "; std::cout.flush();		
		// copy back
		myParticle->summarize_Markov_paths();
		evaluate_affected(associatedParticle->get_lambda_ref());
		
	}
		
};




