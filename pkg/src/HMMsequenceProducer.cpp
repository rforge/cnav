/*
 * HMMsequenceProducer.cpp
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
 
#include <iostream>

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
#include <boost/asio.hpp>

#include <boost/scoped_array.hpp> 

#include <boost/range/algorithm.hpp>

#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMtransitionMatrix.hpp"

 
HMMsequenceProducer::HMMsequenceProducer(HMMdataSet observed, HMMtransitionMatrix initTransitions, arma::uword rseed,
    double i_percentage, unsigned preparation, unsigned max_sequence_length, arma::uword max_simulation_repetitions) : 
      MAXIMUM_SEQUENCE_LENGTH(max_sequence_length), observed_data(observed), 
      MAXCOUNT_TRIALS_FOR_SEQUENCES(max_simulation_repetitions),
      transitionData(initTransitions), percentage(i_percentage), common_rgen(rseed),
      signals(io_service, SIGINT, SIGTERM) // Construct a signal set registered for process termination.
{
	finishedThread = false;
	finishedProduction = true;
	
	interrupted = false;
	
	// Start an asynchronous wait for one of the signals to occur.
    signals.async_wait(boost::bind(&HMMsequenceProducer::interrupt_handler, this, boost::asio::placeholders::error, boost::asio::placeholders::signal_number));
	
	ctrlc_thread = new boost::thread(boost::bind(&HMMsequenceProducer::watch_ctrlc, this));
	
	n_threads = boost::thread::hardware_concurrency();
	countDownCounter = 0;	
	fraction = 0;	
	producers.reset(new boost::thread[n_threads]);
	
	genotype_realizations_count = arma::zeros<arma::uvec>(observed_data.get_ref_count());
	
	boost::random::uniform_int_distribution<arma::uword> uword_dist;
    boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_int_distribution<arma::uword> > 
      init_rand(common_rgen, uword_dist);		
		
	for (unsigned i=0; i < n_threads; i++) {
		arma::uword rxseed = init_rand();
		producers[i] = boost::thread(boost::bind(&HMMsequenceProducer::produce_realizations, this, rxseed));
	}
	
	// prepare the list of sequences for genotypes, try "preparation" times 
	try {
		for (arma::uword i = 0; i < observed_data.get_ref_count(); i++)
	    {
		  construct_genotype(i, preparation, common_rgen);	
	    }
    } catch(Rcpp::exception& ex)
    {
		if (!interrupted) throw ex;  // otherwise do nothing (will be controlled later)
	}
	
}

HMMsequenceProducer::~HMMsequenceProducer()
{
	finishedThread = true;
	
	countDownCondition.notify_all();
	for (unsigned i = 0; i < n_threads; i++) producers[i].join();
	
	signals.cancel();
	ctrlc_thread->join();
	delete ctrlc_thread;
	
}

//******

void HMMsequenceProducer::interrupt_handler(const boost::system::error_code& error, int signal_number)
{
	if (error) {
		
		if (signal_number == SIGINT || signal_number == SIGTERM) interrupted = true;
	}	
	
}


//******

arma::uword HMMsequenceProducer::simulate_transition_counts(double& approx_amount)
{
	// parallel simulation!
	boost::unique_lock<boost::mutex> countDownLock(countDownMutex);   // first ... try to get lock
	countDownList = observed_data.random_draw(common_rgen);
	transition_counts = arma::zeros<arma::umat>(transitionData.n_states(), transitionData.n_states());
	
	finishedProduction = false;
	countDownCounter = 0;  // restart generator
	fraction = 0;
	
	// Restart production
	countDownCondition.notify_all();  // wake up threads
	countDownCondition.wait(countDownLock);  // release lock
	
	approx_amount = fraction / double(observed_data.n_individuals());
	
	// Gather rest
	arma::uword ref = 0;
	arma::uvec::const_iterator countDownIter = countDownList.begin();
	for (; countDownIter != countDownList.end() && !interrupted; countDownIter++, ref++) 
		if (*countDownIter > 0) {
			add_approximation(ref, *countDownIter);
		}
	
	transitionData.set_transition_counts(transition_counts);
	return countDownCounter;
}

// calculate probabilities
double HMMsequenceProducer::logSequenceProbability (const arma::umat& sequenceTransits, const arma::mat& transition_matrix)
{
	using namespace arma;
	
	double summe = 0.0;
	umat::const_iterator seqIter = sequenceTransits.begin();
	mat::const_iterator  transIter = transition_matrix.begin();
	for (;seqIter != sequenceTransits.end(); seqIter++, transIter++) 
	  if (*transIter > 0.0) summe += log(*transIter) * (*seqIter);
	return summe;
}

// for calculation of the likelihood
double HMMsequenceProducer::logSum (double logFirst, double logSecond)
{
	double bigger;
	if (logFirst > logSecond) bigger = logFirst-30; else bigger = logSecond-30;
	return log(exp(logFirst-bigger) + exp(logSecond-bigger)) + bigger;
}


arma::vec HMMsequenceProducer::log_to_linear_probs (const arma::vec& log_probs)
{
	using namespace arma;
	vec re_log_probs = exp(log_probs - log_probs.max() + 20.0);  // reverse log to linear, take care of numeric limits
	// adding 50 should allow optimal representation to all values
	re_log_probs = re_log_probs / accu(re_log_probs);
	return re_log_probs;
}	


// add_approximation
void HMMsequenceProducer::add_approximation(arma::uword ref, arma::uword counts)
{
	multimap_type::const_iterator iter = realizations.begin();
	
	arma::vec probs(genotype_realizations_count[ref]);
	arma::uvec indices(genotype_realizations_count[ref]);
		
	arma::uword xcount = 0;
		
	for (arma::uword poscount = 0; iter != realizations.end(); iter++, poscount++) 
		if (iter->second.get<1>() == ref) 
	{
		if (interrupted) throw( std::runtime_error("Interrupted ...") );
		probs[xcount] = logSequenceProbability(iter->second.get<2>(), transitionData.get_transition_matrix());
		indices[xcount] = poscount;
		xcount++;
	}
	
	probs = log_to_linear_probs(probs);
	
	boost::random::discrete_distribution<> diskrete(probs.begin(), probs.end());
    boost::variate_generator<BasicTypes::base_generator_type&, boost::random::discrete_distribution<> > diskrete_randoms(common_rgen, diskrete); 	
	
	arma::umat sumTransits = arma::zeros<arma::umat>(transitionData.n_states(), transitionData.n_states());
	for (arma::uword i = 0; i < counts; i++) {
		arma::uword selektor = diskrete_randoms();
		BasicTypes::SequenceReferenceTuple current_realization = (realizations.begin() + (indices[selektor]))->second;
		sumTransits = sumTransits + current_realization.get<2>();
	}
	
	transition_counts = transition_counts + sumTransits;
}



//****************************************************************************

arma::uword HMMsequenceProducer::select_state(const arma::rowvec& row_parameters, BasicTypes::base_generator_type& rand_gen)
{
	boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_real_distribution<> > test_randoms(rand_gen, test_dist);
	
	double rnum = test_randoms(), summe = row_parameters[0];
	arma::uword index = 0;
	
	while (index < row_parameters.n_elem && rnum >= summe) 
	{
		index++;
		summe += row_parameters[index];
	}
	
	return index;
}


/* ****************************************************************************************************************
 * 
 * name: hashValue
 * @param: transition_matrix - a matrix with transition probabilities
 * @param: rand_gen - stores a random generator reference
 * @return: a tuple including sequence, genotype and validity 
 * 
 */

double HMMsequenceProducer::hashValue(const arma::urowvec& sequence)
{
	using namespace arma;
	double hashVal = 0.0;
	urowvec::const_iterator iter = sequence.begin();
	for (;iter != sequence.end(); iter++) 
		hashVal = fmod(hashVal * double(transitionData.n_states()) + M_PI * double(*iter), 1.0);
		
	return hashVal;
}


/* ****************************************************************************************************************
 * 
 * name: simulate_markov_sequence
 * @param: transition_matrix - a matrix with transition probabilities
 * @param: rand_gen - stores a random generator reference
 * @return: a tuple including sequence, genotype and validity 
 * 
 */
 
BasicTypes::SequenceReferenceTuple HMMsequenceProducer::produce_random_sequence(
  const arma::mat& transition_matrix, 
  BasicTypes::base_generator_type& rand_gen
)
{
	using namespace arma;
	
	urowvec sequence(MAXIMUM_SEQUENCE_LENGTH);
	sequence[0] = 0;
	
	urowvec sim_genotype = zeros<urowvec>(transitionData.get_emission_matrix().n_cols);
	umat transitSave = zeros<umat>(transition_matrix.n_rows, transition_matrix.n_cols);
	uword refGenotype = 0;
	bool validity = false;
	
	uword j = 0, state = 0;
	bool ran_twice = false;
	while (state != transitionData.get_endstate() && j < MAXIMUM_SEQUENCE_LENGTH) 
	{
		// calculate new state
		uword newstate = select_state(transition_matrix(state, span::all),rand_gen);
		transitSave(state, newstate) = transitSave(state, newstate) + 1;
	    state = newstate;
		// add emission to resulting genotype
	    sim_genotype = sim_genotype + transitionData.get_emission_matrix()(state,span::all);
	    
	    if (!ran_twice && state==transitionData.get_endstate())   // the HMM is simulated twice!
	    {
			state = 0;
			ran_twice = true;
		}
	    j++;
	    sequence[j] = state;
	}
    urowvec shortened_sequence(j);
    std::copy(sequence.begin(), sequence.begin()+j, shortened_sequence.begin());
	
	validity = j < MAXIMUM_SEQUENCE_LENGTH;
		
	BasicTypes::SequenceReferenceTuple result(shortened_sequence,refGenotype,transitSave,validity);
	
	if (validity) observed_data.get_ref(result, sim_genotype);
	return result;
}	


//*****************

// This function generates a random sequence for genotypes that difficult to get by chance
	// To avoid infinite runs and infinite numbers of sequences ... this function just returns 1 sequence

    BasicTypes::SequenceReferenceTuple HMMsequenceProducer::recursive_search_genotype(const arma::urowvec& target_genotype, 
                                                        arma::uword depth, arma::uword state, bool ran_twice, 
                                                        arma::urowvec& simGenotype, arma::urowvec& simSequence,
                                                        arma::umat& transition_counter, 
                                                        bool& finished, BasicTypes::base_generator_type& rgen)
    {
		if (depth >= MAXIMUM_SEQUENCE_LENGTH || arma::accu(simGenotype > target_genotype) > 0) 
		{
			BasicTypes::SequenceReferenceTuple result(simSequence,0,transition_counter,false);
			
			return result;
	    } 
	    else if (state == transitionData.get_endstate() && ran_twice) 
		{
			finished = arma::accu(simGenotype != target_genotype) == 0;
			
			arma::urowvec shortened_sequence(depth);
			std::copy(simSequence.begin(), simSequence.begin()+depth, shortened_sequence.begin() );
			
			BasicTypes::SequenceReferenceTuple result(shortened_sequence,0,transition_counter, finished);
			return result;
	    } 
	    else if (state == transitionData.get_endstate() && !ran_twice)
	    {
			simSequence[depth] = 0;
			simGenotype = simGenotype + transitionData.get_emission_matrix()(0,arma::span::all);
			BasicTypes::SequenceReferenceTuple result = recursive_search_genotype(target_genotype, depth+1, 0, true, 
			                                   simGenotype, simSequence, transition_counter, finished, rgen);
			if (!finished) {
				simGenotype = simGenotype - transitionData.get_emission_matrix()(0,arma::span::all);
			}
			return result;
			                                   
		} 
		else 
		{
			boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
		    boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_real_distribution<> > 
		      test_randoms(rgen, test_dist);
			
			arma::rowvec to_sort(transitionData.n_states());
			for (arma::rowvec::iterator kiter = to_sort.begin(); kiter != to_sort.end(); kiter++) *kiter = test_randoms();
			
			arma::urowvec order = arma::sort_index(to_sort);
		    	
			BasicTypes::SequenceReferenceTuple result;
			result.get<3>() = false;
			
			arma::uword i = 0;
			while (i < order.n_elem && !finished)
			{
				if (transitionData.get_transition_graph()(state, order[i]) > 0.0) 
				{
					arma::uword newstate = order[i];
					simSequence[depth] = newstate;
			        simGenotype = simGenotype + transitionData.get_emission_matrix()(newstate,arma::span::all);
			        transition_counter(state,newstate) = transition_counter(state,newstate) + 1;
			
					result = recursive_search_genotype(target_genotype, depth+1, newstate, ran_twice, 
					         simGenotype, simSequence, transition_counter, finished, rgen);
					         
					if (!finished) {
						simGenotype = simGenotype - transitionData.get_emission_matrix()(newstate,arma::span::all);
			            transition_counter(state,newstate) = transition_counter(state,newstate) - 1;
			            simSequence[depth] = 0;
			        }
				}
				i++;
			}
			return result; 
		}	
    }
 
    // wrapper for the above function
    void HMMsequenceProducer::construct_genotype(arma::uword refGenotype, arma::uword how_many, BasicTypes::base_generator_type& rgen)
    {
		using namespace arma;
		for (uword i = 0; i < how_many; i++) {
			
			bool finished = false;
			urowvec simGenotype = zeros<urowvec>(transitionData.get_emission_matrix().n_cols);
		    urowvec simSequence = zeros<urowvec>(MAXIMUM_SEQUENCE_LENGTH);
		    umat transition_counter = zeros<umat>(transitionData.n_states(), transitionData.n_states());
		
			BasicTypes::SequenceReferenceTuple result;
			
			result = recursive_search_genotype(observed_data.get_genotype(refGenotype), 1, 0, false, 
		                                       simGenotype, simSequence, transition_counter, finished, rgen);
		    result.get<1>() = refGenotype;
			push_realization(result);
		}
		
	}	          



/* ****************************************************************************************************************
 * 
 * name: push_realization
 * 
 * This function is just a starter for parallel simulation of Markov chains
 * 
 * @param: genotypes - contains the genotypes in a matrix
 * @return: void. Results are stored in the genotype list of this class
 * 
 */
 
void HMMsequenceProducer::push_realization(const BasicTypes::SequenceReferenceTuple& realization_element)
{
	boost::unique_lock<boost::mutex> lock(realization_protection_mutex);
	// check whether element is valid
	if (realization_element.get<3>()) {
		// zero, calculate hashValue;
		double hashVal = hashValue(realization_element.get<0>());
		
		// first, search for duplicates by sequence
		multimap_type::const_iterator iter = realizations.lower_bound(hashVal), iter_end = realizations.upper_bound(hashVal);
		bool duplicated = false;
		while (!duplicated && iter != iter_end)
		{
			duplicated = arma::accu(realization_element.get<0>() != iter->second.get<0>()) == 0;
			iter++;
		}
		
		if (!duplicated) {
			realizations.insert(multimap_type::value_type(hashVal, realization_element));
			genotype_realizations_count[realization_element.get<1>()] += 1;
		}
	}
}



/* ****************************************************************************************************************
 * 
 * name: start_generation_cycle
 * 
 * This function is just a starter for parallel simulation of Markov chains
 * 
 * @param: genotypes - contains the genotypes in a matrix
 * @return: void. Results are stored in the genotype list of this class
 * 
 */
 
void HMMsequenceProducer::count_down_realizations(const BasicTypes::SequenceReferenceTuple& ref)
{
	boost::unique_lock<boost::mutex> real_lock(simulationMutex);  // lock Mutex
	
	countDownCounter ++;
	if (ref.get<3>()) 
	{
		if (countDownList[ref.get<1>()] > 0) 
		{	
			fraction++;
			countDownList[ref.get<1>()] --;   // count down
			transition_counts = transition_counts + ref.get<2>();  // count up
			// push_realization(ref); // remove this, because it causes skewing!
		}
	}		 
}

//**********

bool HMMsequenceProducer::count_down_finished()
{
		
	boost::unique_lock<boost::mutex> real_lock(simulationMutex);  // lock Mutex
		
	if (!finishedProduction) 
	{
		finishedProduction = !(double(fraction) < double(observed_data.n_individuals())*percentage && countDownCounter < MAXCOUNT_TRIALS_FOR_SEQUENCES); 
		return finishedProduction;
	}
	else
		return true;
	// restrict simulation
}

//**********

void HMMsequenceProducer::produce_realizations(arma::uword rseed)
{
	BasicTypes::base_generator_type cycle_generator(rseed);
		
	while (!finishedThread && !interrupted) 
	{
		// first: wait until next round
		boost::unique_lock<boost::mutex> countDownLock(countDownMutex);
		while (!finishedThread && count_down_finished() && !interrupted) countDownCondition.wait(countDownLock);
		// push until finished
		while (!finishedThread && !count_down_finished() && !interrupted) 
		{
			BasicTypes::SequenceReferenceTuple product = produce_random_sequence(transitionData.get_transition_matrix(), cycle_generator);
			count_down_realizations(product);
		}
		countDownCondition.notify_all();
	}
}

//**********

void HMMsequenceProducer::print_realizations_count()
{
	arma::trans(genotype_realizations_count).print("Number of realizations per reference:");
}

//********


HMMtransitionMatrix& HMMsequenceProducer::get_transition_instance()
{
	return transitionData;
}

//*******

HMMdataSet& HMMsequenceProducer::get_observations_instance()
{
	return observed_data;
}

//*******

bool HMMsequenceProducer::system_interrupted()
{
	return interrupted;
}

//***

arma::vec HMMsequenceProducer::get_naive_marginal_likelihood(arma::uword n_samp)
{
	return observed_data.naive_marginal_likelihood(n_samp, common_rgen);
}
