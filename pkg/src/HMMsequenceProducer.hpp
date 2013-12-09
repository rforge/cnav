/*
 * HMMsequenceProducer.hpp
 * 
 * Copyright 2013 Andreas Recke <andreas@Persephone>
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
 * 
 * 
 */
 
#pragma once
 
#include <iostream>
#define ARMA_DONT_USE_BLAS
#include <Rcpp.h>
#include <RcppArmadillo.h>

#include <vector>
#include <queue>
#include <algorithm>
#include <utility>

#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/random/uniform_real.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>

#include <boost/container/container_fwd.hpp>
#include <boost/container/flat_map.hpp>

#include <boost/thread.hpp> 
#include <boost/bind.hpp>

#include <boost/scoped_array.hpp> 
#include <boost/range/algorithm.hpp>

#include <boost/asio.hpp>

#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"
#include "HMMtransitionMatrix.hpp"


// This class is meant to separate the process of simulation sequences from
// the actual Gibbs sampling process.
// Improves the possibility to test that algorithm

class HMMsequenceProducer 
{
    const arma::uword MAXIMUM_SEQUENCE_LENGTH;   // to limit the length of Markov sequences
    
    HMMtransitionMatrix transitionData;
    
	arma::umat transition_counts;				 // containes the realizations
	arma::uword countDownCounter;				 // for the parallel process
	
    HMMdataSet observed_data;
    
    double percentage;				 // number of sequences that is simulate randomly
    //~ bool allow_collect;
    
    //~ typedef boost::container::flat_multimap<double, BasicTypes::SequenceReferenceTuple> multimap_type;
    //~ multimap_type realizations;  
    //~ 
    arma::uvec genotype_realizations_count;
    // this set stores a lot of data of realizations ... for the approximation

    // Revision 12/2013 - current states of realizations
    typedef boost::container::vector<BasicTypes::SequenceReferenceTuple> sequence_state_vector_type;  
    sequence_state_vector_type current_sequence_states;
    arma::uvec unchanged_states_vector;
    arma::umat summarize_current_states();
    void completely_random_move(boost::container::vector<BasicTypes::SequenceReferenceTuple>::iterator target, BasicTypes::base_generator_type& rand_gen);    
    
    arma::uword exchanges_resimulation, exchanges_treesearch;    
    
    // A random generator for the main thread
    BasicTypes::base_generator_type common_rgen;

	// Methods that parallely generate Markov sequences
	boost::scoped_array<boost::thread> producers;	
	boost::mutex simulationMutex, realization_protection_mutex, countDownMutex;	
	boost::condition_variable_any countDownCondition;
	bool finishedThread, finishedProduction;
	unsigned n_threads;
	arma::uword fraction;
	arma::mat common_transition_matrix;
	const arma::uword MAXCOUNT_TRIALS_FOR_SEQUENCES;
  
	// Functions to calculate likelihood values with log mathematics
	//~ double logSum (double logFirst, double logSecond);
    //~ double logSequenceProbability (const arma::umat& sequenceTransits, const arma::mat& transition_matrix);
    //~ arma::vec log_to_linear_probs (const arma::vec& log_probs);
    arma::uword select_state(const arma::rowvec& row_parameters, BasicTypes::base_generator_type& rand_gen);

    // Functions to generate sequence solutions for given genotypes
    //~ void add_approximation(arma::uword ref, arma::uword counts);
        
    // *** recursively and randomly searches a solution for a genotype. Started by construct_genotype
    BasicTypes::SequenceReferenceTuple recursive_search_genotype(const arma::urowvec& target_genotype, 
				arma::uword depth, arma::uword state, bool ran_twice, 
				arma::urowvec& simGenotype, arma::urowvec& simSequence,
				arma::umat& transition_counter, 
				bool& finished, BasicTypes::base_generator_type& rgen);
				
	//~ // *** generates a number of solutions			
	//~ void construct_genotype(arma::uword refGenotype, arma::uword how_many, BasicTypes::base_generator_type& rgen);
        //~ 
    //~ // *** this generates a number of exact solutions ... for unbiased sampling
    //~ void construct_exact_sequences();
    //~ void exact_recursive_search_genotype(const arma::uword ref,
                                          //~ arma::uword depth, arma::uword state, bool ran_twice, 
                                          //~ arma::urowvec& simGenotype, arma::urowvec& simSequence,
                                          //~ arma::umat& transition_counter);

  //~ 
    //~ // Storage functions
    //~ double hashValue(const arma::urowvec& sequence);
    //~ void push_realization(const BasicTypes::SequenceReferenceTuple& realization_element);
        //~ 
    // Functions to do the parallel generation of sequence realizations
    // Strategy: a number of sequences is generated and those which fit 
    // are stored. The rest is approximated. The amount of approximated
    // sequences is monitored
    
    // *** central post office for threads
    void count_down_realizations(const BasicTypes::SequenceReferenceTuple& ref, BasicTypes::base_generator_type& rgen);
    
    // *** post office full check
    bool count_down_finished();
    
    // *** central function for each thread
    void produce_realizations(arma::uword rseed);  
        
    //*** interrupt services to allow Ctrl-C
    void watch_ctrlc() { io_service.run(); }
    boost::thread *ctrlc_thread;
    boost::asio::io_service io_service; 
    boost::asio::signal_set signals;
    void interrupt_handler(const boost::system::error_code& error, int signal_number);
    volatile bool interrupted;
        
  
public:
	HMMsequenceProducer(HMMdataSet observed, HMMtransitionMatrix initTransitions, arma::uword rseed, bool exact, bool collect,
	                    double i_percentage = 0.1, unsigned preparation = 100, unsigned max_sequence_length = 1000, arma::uword max_simulation_repetitions=30000);
	
	~HMMsequenceProducer();  // destructor
		
	// Function to simulate a sequence based upon a given transition matrix
    BasicTypes::SequenceReferenceTuple produce_random_sequence(const arma::mat& transition_matrix, BasicTypes::base_generator_type& rand_gen);
    		
	void simulate_transition_counts(arma::uword& exch_resamp, arma::uword& exch_tree);
	// returns the fraction of data that needed to be approximated
	
	void print_realizations_count();
	
    HMMtransitionMatrix& get_transition_instance();
    HMMdataSet& get_observations_instance();
    
    double get_naive_marginal_likelihood();
    
    //~ arma::uword get_number_of_prepared_realizations();
    
    bool system_interrupted();

};
