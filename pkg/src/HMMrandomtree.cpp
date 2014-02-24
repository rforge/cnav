#include "HMMrandomtree.hpp" 
 

HMMrandomtree::HMMrandomtree(HMMtransitionMatrix& initModel, const arma::urowvec& my_genotype, arma::uword myreference, 
	               boost::mutex& silencer,  arma::uword rseed, arma::uword iMAXDEPTH,
	               arma::uword path_sampling_repetitions, bool use_collapsed_sampler) :
  MarkovModel(initModel),
  target_genotype(my_genotype),
  central_rgen(rseed),
  MAXDEPTH(iMAXDEPTH),
  reference(myreference),
  silencer_mutex(&silencer),
  current_states(initModel.n_temperatures() )
{	
	my_path_sampling_repetitions = path_sampling_repetitions;
	i_use_collapsed_sampler = use_collapsed_sampler;
	
	
	boost::uniform_int<arma::uword> unidist(0, std::numeric_limits<arma::uword>::max());
	boost::random::variate_generator<BasicTypes::base_generator_type, boost::uniform_int<arma::uword> > unirandoms(central_rgen, unidist);
	
	arma::uword particle = 0;

	for (current_state_vector_iterator_type siter = current_states.begin(); siter != current_states.end(); ++particle, ++siter )
	{
		siter->get<0>() = particle;  // set particle number as reference
		
	    arma::urowvec current_genotype = arma::zeros<arma::urowvec>(MarkovModel.get_emission_matrix().n_cols);
		arma::uvec new_sequence = arma::zeros<arma::uvec>(MAXDEPTH);
		bool clockwise = true;
		base_pseudo_generator_type new_generator(unirandoms());
		bool old_path_found = false;
		
		bool success = recursive_tree_search(siter, 0, false, current_genotype, new_sequence, clockwise, old_path_found, 1, new_generator, true);
		
		if (!success) throw(std::runtime_error("Some weird error occured early!")); 
		
	}
	
}
 
 // copy constructor
 HMMrandomtree::HMMrandomtree(const HMMrandomtree& obj) : // copy constructor
   MAXDEPTH(obj.MAXDEPTH) ,
   MarkovModel(obj.MarkovModel),
   central_rgen(obj.central_rgen),
   //~ current_state_sequence(obj.current_state_sequence),
   //~ current_state_counts(obj.current_state_counts),
   target_genotype(obj.target_genotype),
   //~ acceptances(obj.acceptances),
   //~ tries(obj.tries),
   //~ current_likelihood(obj.current_likelihood),
   reference(obj.reference),
   current_states(obj.current_states.size())
{
	std::copy(obj.current_states.begin(), obj.current_states.end(), current_states.begin());
	silencer_mutex = obj.silencer_mutex;
	my_path_sampling_repetitions = obj.my_path_sampling_repetitions;
	i_use_collapsed_sampler = obj.i_use_collapsed_sampler;
}
 

// and operator=
HMMrandomtree& HMMrandomtree::HMMrandomtree::operator=(const HMMrandomtree& rhs)
{
	if (this != &rhs)  //oder if (*this != rhs)
	{
		/* kopiere elementweise, oder:*/
		HMMrandomtree tmp(rhs); //Copy-Konstruktor
		
		boost::swap(tmp.central_rgen, central_rgen);
		target_genotype        = rhs.target_genotype;
		reference              = rhs.reference;
		
		silencer_mutex = tmp.silencer_mutex;
		std::copy(tmp.current_states.begin(), tmp.current_states.end(), current_states.begin());
	}
	return *this; //Referenz auf das Objekt selbst zurückgeben
} 
 
 
/* This function is to search for a sequence to produce a certain genotype
 * 
 * name: recursive_tree_search
 * @param: see below
 * @return: just, whether something has been found
 */
bool HMMrandomtree::recursive_tree_search(
   current_state_vector_iterator_type entry,
   arma::uword current_state,              	 		// state of the system before recursion
   bool ran_twice,                					// the Markov chain must run twice to simulate two chromosomes
   arma::urowvec& current_genotype,             	// genotype at the time before recusion
   arma::uvec& new_sequence,                 		// sequence before recursion
   bool clockwise,                           		// direction of tree-search
   bool& old_path_found,                 			// indicates whether we still search for the old sequence
   arma::uword depth,                    			// depth of tree 
   base_pseudo_generator_type new_generator,		// pseudo_random generator, we use boost::random::taus88 because of its small size
   bool initial)
{
	if (depth > MAXDEPTH || arma::accu(current_genotype > target_genotype) > 0) 
	{
		return false;
	} 
	 else 
	{
		// second chromosome
		if (current_state == MarkovModel.get_endstate() && !ran_twice) 
		{
			new_sequence[depth] = 0;
			return recursive_tree_search(entry, 0, true, current_genotype, new_sequence, clockwise, old_path_found, depth+1, new_generator, initial);
		}
		
		// final state reached?
		if (current_state == MarkovModel.get_endstate() && ran_twice) 
		{
			bool finished = arma::accu(current_genotype != target_genotype) == 0;
						
			if (finished && initial) {
				entry->get<1>() = new_sequence.subvec(0, depth-1);	
				entry->get<2>() = arma::zeros<arma::umat>(MarkovModel.get_transition_graph().n_rows, MarkovModel.get_transition_graph().n_cols);
		        for (arma::uvec::const_iterator it = entry->get<1>().begin(); (it+1) != entry->get<1>().end(); it++)
			        if (*it != MarkovModel.get_endstate()) entry->get<2>()(*it, *(it+1)) += 1;
				
				return true;
			}
			
			if (finished && !old_path_found) 
			{
				old_path_found = true;
				return false;
			} 
			else 
			{
				if (finished) accept_new_sequence(entry, new_sequence.subvec(0,depth-1), false);
				return finished;
			}
		}
		 
		// third alternative, the interesting part
		
		// define a random number generator
	    boost::random::uniform_real_distribution<> pseudo_random_dist(0.0, 1.0);
	    boost::random::variate_generator<base_pseudo_generator_type&, boost::random::uniform_real_distribution<> > pseudo_randoms(new_generator, pseudo_random_dist);
		
		// fill a vector
		arma::vec preorder(MarkovModel.get_transition_graph().n_cols);
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
		if (!initial && !old_path_found) {
			while (order[walker] != entry->get<1>()[depth] && walker < MarkovModel.get_transition_graph().n_cols) ++walker;
			
			if (order[walker] != entry->get<1>()[depth] ) 
				throw(std::runtime_error("This does not work!")); // emergency, to prevent infinite loops
			
		}
		
		do {
			if (MarkovModel.get_transition_graph()(current_state, order[walker]) > 0) 
			{
				new_sequence[depth] = order[walker];
				current_genotype = current_genotype + MarkovModel.get_emission_matrix()(order[walker], arma::span::all);
				finished = recursive_tree_search(entry, order[walker], ran_twice, current_genotype, new_sequence, clockwise, old_path_found, depth+1, new_generator, initial);
				current_genotype = current_genotype - MarkovModel.get_emission_matrix()(order[walker], arma::span::all);
			}
			walker++;
			if (initial && !finished) 
			{ 
				if (walker >= MarkovModel.get_transition_graph().n_cols && depth == 1) 
				{
				   	Rcpp::Rcout << "\nNo Solution!\n"; Rcpp::Rcout.flush();
			    }
			}
			if (!initial) if (depth == 1 && walker >= MarkovModel.get_transition_graph().n_cols) walker = 0;  // only with the first one re-do again and again until finished becomes true!
						
		} while (!finished && walker < MarkovModel.get_transition_graph().n_cols);
	
		return finished;
	}	
}



/* This function is a wrapper for the recursive search
 * 
 * name: walk_state_sequence
 * @param: rand_seed: just to start the random number generator
 * @return: the resulting sequence
 */
void HMMrandomtree::walk_state_sequences()
{
	// define random number generators
	boost::uniform_int<arma::uword> unidist(0, std::numeric_limits<arma::uword>::max());
	boost::random::variate_generator<BasicTypes::base_generator_type, boost::uniform_int<arma::uword> > unirandoms(central_rgen, unidist);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_real<> > central_vargen(central_rgen, boost::uniform_real<>(0.0, 1.0));
	
	for (current_state_vector_iterator_type siter = current_states.begin(); siter != current_states.end(); ++siter )
	{
		for (arma::uword rep = 0; rep < my_path_sampling_repetitions; ++rep)
		{
			bool success;
			arma::urowvec current_genotype = arma::zeros<arma::urowvec>(MarkovModel.get_emission_matrix().n_cols);
			arma::uvec new_sequence = arma::zeros<arma::uvec>(MAXDEPTH);
		
			bool clockwise = central_vargen() < 0.5;
			base_pseudo_generator_type new_generator(unirandoms());
			bool old_path_found = false;
		
		    success = recursive_tree_search(siter, 0, false, current_genotype, new_sequence, clockwise, old_path_found, 1, new_generator);
			if (!success) throw(std::runtime_error("Some weird error occured!"));
		}
	}
}


/* This function is a wrapper for the recursive search
 * 
 * name: random_sequence_generator
 * @param: n_samples the number of samples that are simulated to find a new one
 * @return: the resulting sequence
 */
void HMMrandomtree::random_sequence_generator(arma::uword n_samples)
{
	arma::urowvec current_genotype;
	arma::uvec new_sequence = arma::zeros<arma::uvec>(MAXDEPTH);
	
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_real<> > central_vargen(central_rgen, boost::uniform_real<>(0.0, 1.0));
	
	// search for the interesting iterator position
	
	for (current_state_vector_iterator_type temp_iterator = current_states.begin(); temp_iterator != current_states.end(); ++temp_iterator)
	{
		// first: set a matrix (might be optimized, if it works)
		sequence_generator_transitions = MarkovModel.get_transition_matrix(temp_iterator->get<0>());
		for (arma::uword zeile = 0; zeile < MarkovModel.get_transition_graph().n_rows-1; ++zeile)
		{
			double intersum = 0.0;
			for (arma::uword spalte = 0; spalte < MarkovModel.get_transition_graph().n_cols; ++spalte)
				if ( sequence_generator_transitions(zeile,spalte) > 0 )
				{
				    sequence_generator_transitions(zeile,spalte) = 
					    pow(sequence_generator_transitions(zeile,spalte), MarkovModel.get_particle_temperature(temp_iterator->get<0>()));
					intersum += sequence_generator_transitions(zeile,spalte);
				}
			sequence_generator_transitions(zeile, arma::span::all) = sequence_generator_transitions(zeile, arma::span::all) / intersum;
		}
		
		// then: go on
		
		arma::uword ii = 0;
		bool success = false;
		
		while ( ++ii <= n_samples && !success)
		{	
			arma::uword depth = 1;
			arma::uword state = 0;
			bool ran_twice = false; 
			success = false;
			bool failure = false;
			current_genotype = arma::zeros<arma::urowvec>(MarkovModel.get_emission_matrix().n_cols);
			
			while (depth < MAXDEPTH && !success && !failure) 
			{
				double zufall = central_vargen();
				double summe = sequence_generator_transitions(state,0);
				arma::uword newstate = 0;
				while (newstate < MarkovModel.get_transition_graph().n_cols-1 && summe < zufall) 
				{
					newstate++;
					summe += sequence_generator_transitions(state,newstate);
				}
				
				state = newstate;
				new_sequence[depth] = state;
				current_genotype = current_genotype + MarkovModel.get_emission_matrix()(state, arma::span::all);
				
				if (state == MarkovModel.get_endstate() && !ran_twice)
			    {
					depth++;
					new_sequence[depth] = 0;
					state = 0;
					ran_twice = true;
				}
				
				arma::urowvec::const_iterator it1 = current_genotype.begin(), it2 = target_genotype.begin();
				success = state == MarkovModel.get_endstate();
				failure = false;
				while (it1 != current_genotype.end() && success && !failure) 
				{
					success = success && (*it1) == (*it2);
					failure = failure || (*it1) > (*it2);
					it1++; it2++;
				}
				depth++;
			}
			
			if (success) accept_new_sequence(temp_iterator, new_sequence.subvec(0,depth-1), true);
			
		}
	}
		
}


/* This function is to check whether a sequence is actually accepted
 * 
 * name: accept_new_sequence
 * @param: arma::uvec new_sequence
 * @return: nothing
 */
void HMMrandomtree::accept_new_sequence(current_state_vector_iterator_type entry, arma::uvec new_sequence, bool randomly_generated)
{
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::uniform_real<> > central_vargen(central_rgen, boost::uniform_real<>(0.0, 1.0));
	
	arma::umat xcounts = arma::zeros<arma::umat>(MarkovModel.get_transition_graph().n_rows, MarkovModel.get_transition_graph().n_cols);
	for (arma::uvec::const_iterator it = new_sequence.begin(); (it+1) != new_sequence.end(); ++it)
		if (*it != MarkovModel.get_endstate()) xcounts(*it, *(it+1)) += 1;

	double decision_likelihood;
	//~ tries++;  
	
    silencer_mutex->lock();
	if (randomly_generated)   // Gibbs sampling!!! decision_likelihood === 1
	{
		if (i_use_collapsed_sampler) 
			decision_likelihood = MarkovModel.collapsed_likelihood_difference(entry->get<2>(), xcounts, entry->get<0>()) 
		//~ decision_likelihood = MarkovModel.likelihood(xcounts, entry->get<0>()) - MarkovModel.likelihood(entry->get<2>(), entry->get<0>()) 
		                      - generator_likelihood(xcounts, sequence_generator_transitions) 
		                      + generator_likelihood(entry->get<2>(), sequence_generator_transitions);
		else
		   decision_likelihood = MarkovModel.likelihood(xcounts, entry->get<0>()) - MarkovModel.likelihood(entry->get<2>(), entry->get<0>()) 
		                      - generator_likelihood(xcounts, sequence_generator_transitions) 
		                      + generator_likelihood(entry->get<2>(), sequence_generator_transitions);
		
		// decision_likelihood = 1.0; 
		                     //~ likelihood_star + MarkovModel.likelihood(current_state_counts, false) - 
		                      //~ current_likelihood - MarkovModel.likelihood(xcounts, false);
	} else 
	{
		if (i_use_collapsed_sampler) 
			decision_likelihood = MarkovModel.collapsed_likelihood_difference(entry->get<2>(), xcounts, entry->get<0>());
		else
		    decision_likelihood = MarkovModel.likelihood(xcounts, entry->get<0>()) - MarkovModel.likelihood(entry->get<2>(), entry->get<0>()) ;
	}
	
	if ( log(central_vargen()) < decision_likelihood ) 
	{
		entry->get<1>() = new_sequence;	
        MarkovModel.set_transition_counts(entry->get<0>(), (MarkovModel.get_transition_counts(entry->get<0>()) + xcounts) - entry->get<2>());
      
        entry->get<2>() = xcounts;
	} 
	
	silencer_mutex->unlock();
}


/*
 * 
 * name: get_counts
 * @param: nothing
 * @return: the counting matrix
 */
arma::ucube HMMrandomtree::get_counts()
{
	arma::ucube result(MarkovModel.get_transition_graph().n_rows, MarkovModel.get_transition_graph().n_cols, current_states.size());
	arma::uword ilevel = 0;
	
	for (current_state_vector_iterator_type siter = current_states.begin(); siter != current_states.end(); ++siter, ++ilevel )
	{
		result.slice(ilevel) = siter->get<2>();
	}
	
	return result;
}




/*
 * 
 * name: generator_likelihood
 * @param: counts - a counting matrix
 * @param: transitions - a transition matrix
 * @return: the likelihood
 */
double HMMrandomtree::generator_likelihood(const arma::umat& counts, const arma::mat& transitions)
{
	double summe =  0.0;
	arma::umat::const_iterator citer = counts.begin();
	arma::mat::const_iterator piter = transitions.begin();
	for (; citer != counts.end(); ++citer, ++piter)
		if (*piter > 0) summe+= log(*piter) * double(*citer);
		
	return summe;
}
