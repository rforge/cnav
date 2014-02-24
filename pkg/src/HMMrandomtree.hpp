#include <iostream>
#define ARMA_DONT_USE_BLAS
//#include <Rcpp.h>
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>

#include <boost/thread.hpp> 
#include "BasicTypes.hpp"
#include "HMMtransitionMatrix.hpp"

// concept: for every target_genotype, more than one possible solution might exist. 
// These can be determined by an exhaustive tree search that tries all possible Markov sequences.
// Given a certain order of branches (e.g. state numbers), there will be an order 
// in which the solutions are found.

// However, if the order of branches is determined by a pseudo random number, the order by which
// solutions are found changes. 

// Given a certain sequence, we may find the sequence before and after in a pseudo-randomly governed tree search.

// the strategy is as follows
// 1: starting with a random number, the algorithm finds the position in a tree search
// 2: then, depending on the direction (after = clockwise versus before = counter-clockwise), the next solution is
//    determined
// 3: the new sequence is accepted by the Metropolis-Hastings rule

// Given a flat distribution of sequence solutions, this algorithm should deliver all solutions with equal probability.
// But other likelihood functions may be used in the Metropolis-Hastings rule, e.g. tempered likelihood function
// However: the efficiency of the algorithm may drop severely when the likelihood function highly prefers a certain
// sequence. This has to be evaluated


class HMMrandomtree
{	
	arma::uword MAXDEPTH;
	arma::uword my_path_sampling_repetitions;
	bool i_use_collapsed_sampler;

	typedef boost::random::taus88 base_pseudo_generator_type;
	HMMtransitionMatrix& MarkovModel;  // save it as a pointer! Do not destroy this
	BasicTypes::base_generator_type central_rgen;
	//~ boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_real<> > central_vargen;
	boost::mutex *silencer_mutex;
	
	// tuple: current temperature, current sequence, current counting matrix
	typedef boost::tuple<arma::uword, arma::uvec, arma::umat> sequence_and_counts_pair_type;   
	typedef boost::container::vector<sequence_and_counts_pair_type> current_state_vector_type;  
	typedef current_state_vector_type::iterator current_state_vector_iterator_type;
    current_state_vector_type current_states;
	
	arma::urowvec target_genotype;
	// arma::uword acceptances, tries;
	arma::uword reference; // saves a handle for the observed data
		
	void accept_new_sequence(current_state_vector_iterator_type entry, arma::uvec new_sequence, bool randomly_generated);
		
	bool recursive_tree_search(
	    current_state_vector_iterator_type entry,
	    arma::uword current_state, bool ran_twice,              
	    arma::urowvec& current_genotype, arma::uvec& new_sequence,       
	    bool clockwise, bool& old_path_found,           
        arma::uword depth, base_pseudo_generator_type new_generator,
        bool initial = false);		
		
		
	arma::mat sequence_generator_transitions;
	double generator_likelihood(const arma::umat& counts, const arma::mat& transitions);
		
public:
	
	HMMrandomtree(HMMtransitionMatrix& initModel, const arma::urowvec& my_genotype, arma::uword myreference, 
	                boost::mutex& silencer, arma::uword rseed = 42, arma::uword iMAXDEPTH = 1000,
	                arma::uword path_sampling_repetitions = 1, 
	                bool use_collapsed_sampler = false);
	                
	HMMrandomtree(const HMMrandomtree& obj); // copy constructor
	
	HMMrandomtree& operator=(const HMMrandomtree& rhs);
	
	// void set_sequence(arma::uvec new_sequence);
	
	// this function walks every particle anew
	void walk_state_sequences();
	
	// as it is only effective at lambda near 1, just use it for the highest temperature
	void random_sequence_generator(arma::uword n_samples);
	
	// const double get_acceptance_rate() const;
	arma::ucube get_counts();
	
	void forced_temperature_swap(arma::uword levelA, arma::uword levelB);
	
	// arma::urowvec get_sequence();
	// const double get_current_likelihood() const;
	// const double current_sequence_hashvalue() const;
	// const arma::uword get_reference() const;
};
