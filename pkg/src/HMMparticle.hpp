/*
 * HMMparticle.hpp
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
 * This class implements a single sampling particle. I contains a transition matrix, the lambda value (temperature) and the Markov paths.
 * Two particles can exchange their temperature levels.
 */

#pragma once
#include <iostream>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"


class HMMparticle {
	
	
	public:
	// general typedefs
	                     // ID       , ref,        , sequence,     count matrix,  validity flag
	typedef BasicTypes::IDRefSequenceCountTuple IDRefSequenceCountTuple;
    typedef boost::container::vector<IDRefSequenceCountTuple> sequence_vector_type;
	typedef sequence_vector_type::iterator sequence_vector_iterator_type;

	arma::uword MAXDEPTH;

	private:
	// random generator
	boost::random::mt19937 central_generator;

	// the interesting data
	arma::uword lambda_ref;
	double lambda;
	arma::mat transition_matrix, transition_matrix_prior;
	arma::umat transition_counts;
	arma::umat emission_matrix;
	sequence_vector_type transition_sequences;
	arma::uword my_running_number;
	
	// cross-references
	HMMdataSet* observed_data;
	
	// Private method definitions
	
    //*********************************************************************************************************************************************************	
	/* This function is to initialize a sequence for a certain genotype
	 * 
	 * name: recursive_tree_search
	 * @param: see below
     * @return: just, that something has been found
	 */
	 
	bool recursive_tree_search(IDRefSequenceCountTuple& target,
		arma::uword current_state,              	 	// state of the system before recursion
		bool ran_twice,                					// the Markov chain must run twice to simulate two chromosomes
		arma::urowvec& current_genotype,             	// genotype at the time before recusion
		arma::uword depth                    			// depth of tree 
    )
	{
		if (depth > MAXDEPTH || arma::accu(current_genotype > observed_data->get_genotype(target.get<1>())) > 0) 
		{
			// no solution found or possible to find
			return false;
		} 
		else if (current_state == transition_matrix_prior.n_rows-1)
		{
			// an end has been reached
			if (!ran_twice) 
			{
				// switch second chromosome
				target.get<2>()[depth] = 0;
			    return recursive_tree_search(target, 0, true, current_genotype, depth+1);
			} 
			else {
				// the end is reached, we can finalize
				if (arma::accu(current_genotype != observed_data->get_genotype(target.get<1>())) == 0) 
				{
					// everything is okay, we have a sequence
					target.get<2>() = target.get<2>().subvec(0, depth-1);  // truncate to relevant
					target.get<3>() = arma::zeros<arma::umat>(transition_matrix_prior.n_rows, transition_matrix_prior.n_cols); // generate countmatrix
					
					for (arma::uvec::const_iterator it = target.get<2>().begin(); (it+1) != target.get<2>().end(); ++it)
						if (*it != transition_matrix_prior.n_rows-1) target.get<3>()(*it, *(it+1)) += 1;
				
					return true;
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
	    boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > 
			pseudo_randoms(central_generator, pseudo_random_dist);
		
		// fill a vector
		arma::vec preorder(transition_matrix_prior.n_cols);
		for (arma::vec::iterator oit = preorder.begin(); oit != preorder.end(); oit++) *oit = pseudo_randoms();
		// and find the order to traverse
		arma::uvec order = arma::sort_index(preorder, 0);
			
		arma::uword walker = 0;
		bool finished = false;
		
		do {
			if (transition_matrix_prior(current_state, order[walker]) > 0) 
			{
				target.get<2>()[depth] = order[walker];
				current_genotype = current_genotype + emission_matrix(order[walker], arma::span::all);
				finished = recursive_tree_search(target, order[walker], ran_twice, current_genotype, depth+1);
				current_genotype = current_genotype - emission_matrix(order[walker], arma::span::all);
			}
			walker++;	
		} while (!finished && walker < transition_matrix_prior.n_cols);
	
		return finished;
	}	




	//*********************************************************************************************************************************************************
	public:
	
		
	HMMparticle(HMMdataSet& my_observed_data, 
				  arma::mat prior, 
				  arma::umat i_emission_matrix,	
				  arma::uword i_lambda_ref, double i_lambda, 
				  arma::uword particle_number, 
				  arma::uword random_seed = 42,
				  arma::uword max_length = 1000) : 
		transition_matrix_prior(prior),
		emission_matrix(i_emission_matrix),
		
		lambda (i_lambda), lambda_ref (i_lambda_ref),
		central_generator(random_seed),
		MAXDEPTH(max_length),
		my_running_number(particle_number)
	{
		//~ Rcpp::Rcout << "a" << particle_number << " RS= " << random_seed << " ";Rcpp::Rcout.flush();
		observed_data = &my_observed_data;
		
		//~ Rcpp::Rcout << "b lambda =" << i_lambda << " ";Rcpp::Rcout.flush();
		for (arma::uword i = 0; i < observed_data->n_genotypes(); ++i)
		{
			IDRefSequenceCountTuple current_target(i,
			                                          observed_data->get_genotype_refs()[i],
			                                          arma::zeros<arma::urowvec>(MAXDEPTH),
			                                          arma::zeros<arma::umat>(transition_matrix_prior.n_rows, transition_matrix_prior.n_cols),
			                                          true  
			                                         );
			arma::urowvec current_genotype = arma::zeros<arma::urowvec>(emission_matrix.n_cols);												
            if ( recursive_tree_search(current_target, 0, false, current_genotype, 1) )
            {
				//~ Rcpp::Rcout << ">";Rcpp::Rcout.flush();
				
	            //~ for (arma::uword ix = 0; ix < current_target.get<2>().n_elem; ++ix) std::cout << current_target.get<2>()[ix] << " ";
	            //~ std::cout << "\n";std::cout.flush();
	            
				transition_sequences.push_back(current_target);
			}
            else throw "No solution for genotype!\n";
		}
		//~ Rcpp::Rcout << "c";Rcpp::Rcout.flush();

		summarize_Markov_paths();
		random_matrix();
		//~ Rcpp::Rcout << "d";Rcpp::Rcout.flush();
		//~ transition_matrix.print("TM=");
	}
	

	HMMparticle(const HMMparticle& obj) :
	  MAXDEPTH (obj.MAXDEPTH),
	  central_generator (obj.central_generator),
	  lambda_ref(obj.lambda_ref),
	  lambda(obj.lambda),
	  transition_matrix(obj.transition_matrix),
	  transition_matrix_prior(obj.transition_matrix_prior),
	  transition_counts(obj.transition_counts),
	  emission_matrix(obj.emission_matrix),
	  transition_sequences(obj.transition_sequences),
	  my_running_number(obj.my_running_number),
	  observed_data(obj.observed_data)
	{
	}
	
	HMMparticle& operator=( const HMMparticle& rhs ) 
	{
		HMMparticle obj(rhs);
		
		// Now, swap the data members with the temporary:
		std::swap(MAXDEPTH, obj.MAXDEPTH);
	    std::swap(central_generator, obj.central_generator);
	    std::swap(lambda_ref, obj.lambda_ref);
	    std::swap(lambda, obj.lambda);
	    std::swap(transition_matrix, obj.transition_matrix);
	    std::swap(transition_matrix_prior, obj.transition_matrix_prior);
	    std::swap(emission_matrix, obj.emission_matrix);
	    std::swap(transition_sequences, obj.transition_sequences);
	    std::swap(transition_counts, obj.transition_counts);
	    std::swap(my_running_number, obj.my_running_number);
	    std::swap(observed_data, obj.observed_data);
	    	
		return *this;
     }
	
	//*********************************************************************************************************************************************************
	
	void summarize_Markov_paths()
	{
		using namespace arma;
					
		transition_counts = zeros<umat>(transition_matrix_prior.n_rows, transition_matrix_prior.n_cols);
		
		//~ Rcpp::Rcout << "!Filled\n"; Rcpp::Rcout.flush();
		sequence_vector_iterator_type it = transition_sequences.begin();
		
		for(; it != transition_sequences.end(); ++it) 
		{
			transition_counts = transition_counts + it->get<3>();
			//~ Rcpp::Rcout << "."; Rcpp::Rcout.flush();
		}
		//~ transition_counts.print("Tcounts=");
		
	}

	
	
	//**********************************************************************************************************************************************************	
	
	// generates a realization of the random matrix
	void random_matrix()
	{	
		using namespace arma;
		
		
		transition_matrix = zeros<mat>(transition_matrix_prior.n_rows, transition_matrix_prior.n_cols);
		
		for (uword i = 0; i < transition_matrix_prior.n_rows-1; i++)
		{
			for (uword j = 0; j < transition_matrix_prior.n_cols; j++) 
				if (transition_matrix_prior(i,j) > 0.0)
				{
					boost::gamma_distribution<> mygamma(transition_matrix_prior(i,j) + lambda*double(transition_counts(i,j)));
			        boost::variate_generator<boost::random::mt19937&, boost::gamma_distribution<> > random_gamma(central_generator, mygamma);
					transition_matrix(i,j) = random_gamma();
			    }	
			
			transition_matrix(i, span::all) = transition_matrix(i, span::all) / accu(transition_matrix(i, span::all));
		}
	}
	
	
	//**********************************************************************************************************************************************************	
	
	// generates a realization of the random matrix
	arma::mat get_any_random_matrix()
	{	
		using namespace arma;
		
		mat any_transition_matrix = zeros<mat>(transition_matrix_prior.n_rows, transition_matrix_prior.n_cols);
		
		for (uword i = 0; i < transition_matrix_prior.n_rows-1; ++i)
		{
			for (uword j = 0; j < transition_matrix_prior.n_cols; ++j) 
				if (transition_matrix_prior(i,j) > 0.0)
				{
					boost::gamma_distribution<> mygamma(transition_matrix_prior(i,j) + lambda*double(transition_counts(i,j)));
			        boost::variate_generator<boost::random::mt19937&, boost::gamma_distribution<> > random_gamma(central_generator, mygamma);
					any_transition_matrix(i,j) = random_gamma();
			    }	
			
			any_transition_matrix(i, span::all) = any_transition_matrix(i, span::all) / accu(any_transition_matrix(i, span::all));
		}
		
		return any_transition_matrix;
	}
	
	//**********************************************************************************************************************************************************	

    arma::mat get_tempered_random_matrix()
    {
		// maybe this works better ... 
		arma::mat any_transition_matrix = arma::zeros<arma::mat>(transition_matrix.n_rows, transition_matrix.n_cols);
		arma::mat::iterator it1 = transition_matrix.begin(), it2 = any_transition_matrix.begin();
		for (; it1 != transition_matrix.end(); ++it1, ++it2 ) 
			if ( (*it1) > 0) (*it2) = pow(*it1, lambda); else (*it2) = 0;
		
		for (arma::uword i = 0; i < any_transition_matrix.n_rows-1; ++i)
			any_transition_matrix(i, arma::span::all) = any_transition_matrix(i, arma::span::all) / arma::accu(any_transition_matrix(i, arma::span::all));
			
		return any_transition_matrix;
	}

	//*********************************************************************************************************************************************************	
	
	// unnormalized density of the transition matrix
	double relative_countmatrix_likelihood()
	{
		double sum = 0.0;
		using namespace arma;
		
		umat::iterator cmit = transition_counts.begin();
		mat::iterator tmit = transition_matrix.begin();
		mat::iterator pcmit = transition_matrix_prior.begin();
		
		for (; cmit != transition_counts.end(); ++cmit, ++tmit, ++pcmit) if (*pcmit > 0.0) sum += log(*tmit) * (double(*cmit) - 1.0);
		
		return sum;
	}
	//*********************************************************************************************************************************************************	
	
	// unnormalized density of the transition matrix
	double countmatrix_likelihood()
	{
		double sum = 0.0;
		using namespace arma;
		
		umat::iterator cmit = transition_counts.begin();
		mat::iterator tmit = transition_matrix.begin();
		mat::iterator pcmit = transition_matrix_prior.begin();
		
		for (; cmit != transition_counts.end(); ++cmit, ++tmit, ++pcmit) if (*pcmit > 0.0) sum += log(*tmit) * ((*pcmit) + lambda*double(*cmit) - 1.0);
		
		return sum;
	}
	
	//*********************************************************************************************************************************************************	
	
	// unnormalized density of the transition matrix
	double countmatrix_likelihood(const arma::umat& testcounts)
	{
		double sum = 0.0;
		using namespace arma;
		
		umat::const_iterator cmit = testcounts.begin();
		mat::iterator tmit = transition_matrix.begin();
		mat::iterator pcmit = transition_matrix_prior.begin();
		
		for (; cmit != testcounts.end(); ++cmit, ++tmit, ++pcmit) if (*pcmit > 0.0) sum += log(*tmit) * ((*pcmit) + lambda*double(*cmit) - 1.0);
		
		return sum;
	}
	
	//*********************************************************************************************************************************************************		
	
	double collapsed_likelihood(const arma::umat& testcounts)
	{
		using namespace arma;
		double summe = 0.0;
		
		
		for (uword zeile = 0; zeile < testcounts.n_rows-1; ++zeile)
		{
			double psumme = 0.0;
			for (uword spalte = 0; spalte < testcounts.n_cols; ++spalte) if (transition_matrix_prior(zeile,spalte) > 0) 
			{
				psumme += 1.0 + lambda*double(testcounts(zeile,spalte));
				summe  += boost::math::lgamma(1.0 + lambda*double(testcounts(zeile,spalte)));
			}
			summe -= boost::math::lgamma(psumme);	
		}
		
		
		return summe;
	}
	
	//*********************************************************************************************************************************************************		
	
	double full_collapsed_likelihood()
	{
		using namespace arma;
		double summe = 0.0;
		
		for (uword zeile = 0; zeile < transition_counts.n_rows-1; ++zeile)
		{
			double psumme = 0.0;
			for (uword spalte = 0; spalte < transition_counts.n_cols; ++spalte)	if (transition_matrix_prior(zeile,spalte) > 0) 
			{
				psumme += transition_matrix_prior(zeile,spalte) + lambda*double(transition_counts(zeile,spalte));
				summe  += boost::math::lgamma(transition_matrix_prior(zeile,spalte) + lambda*double(transition_counts(zeile,spalte)));
			}
			summe -= boost::math::lgamma(psumme);	
		}
		
		return summe;
	}
	
	//*********************************************************************************************************************************************************	
	
	double full_untempered_collapsed_likelihood()
	{
		using namespace arma;
		double summe = 0.0;
		
		for (uword zeile = 0; zeile < transition_counts.n_rows-1; ++zeile)
		{
			double psumme = 0.0;
			for (uword spalte = 0; spalte < transition_counts.n_cols; ++spalte)	if (transition_matrix_prior(zeile,spalte) > 0) 
			{
				psumme += transition_matrix_prior(zeile,spalte) + double(transition_counts(zeile,spalte));
				summe  += boost::math::lgamma(transition_matrix_prior(zeile,spalte) + double(transition_counts(zeile,spalte)));
			}
			summe -= boost::math::lgamma(psumme);	
		}
		
		return summe;
	}
	
	
	//*********************************************************************************************************************************************************		
	
	void matprint(arma::umat x)
	{
		std::cout << "\n";
		for (unsigned i = 0; i < x.n_rows; ++i)
		{
			for (unsigned j=0; j < x.n_cols; ++j)
			{
				std::cout << x(i,j) << " ";
			}
			std::cout << "\n";
		}
		std::cout.flush();
	}
	
	double collapsed_likelihood_difference(const arma::umat& oldpath, const arma::umat& newpath)
	{
		using namespace arma;
		double result;
		umat newcounts = (transition_counts + newpath) - oldpath;
		try {
		
			result = collapsed_likelihood(newcounts) - collapsed_likelihood(transition_counts);
		}
		catch (std::exception& e)
		{
			std::cout << "oldpath = ";
			matprint(oldpath);
			std::cout << "Newpath = ";
			matprint(newpath);
			std::cout << "Old=";
			matprint(transition_counts);
			std::cout << ("New=");
			matprint(newcounts);
			
			summarize_Markov_paths();
			std::cout << "ReOld=";
			matprint(transition_counts);
			
			update_countmatrix(oldpath, newpath);
			std::cout << "UpdOld=";
			matprint(transition_counts);
						
			throw e;
		}
		return result;
	}
	
	
	//*********************************************************************************************************************************************************	
	
	double collapsed_likelihood_absolute(const arma::umat& oldpath, const arma::umat& newpath)
	{
		using namespace arma;
		double result;
		umat newcounts = (transition_counts + newpath) - oldpath;
		result = collapsed_likelihood(newcounts);
		
		return result;
	}
	
	
	/* ****************************************************************************************************************
	 * 
	 * name: hashValue
	 * @param: transition_matrix - a matrix with transition probabilities
	 * @param: rand_gen - stores a random generator reference
	 * @return: a tuple including sequence, genotype and validity 
	 * 
	 */
	
	double hashValue(sequence_vector_iterator_type entry)
	{
		using namespace arma;
		double hashVal = 0.0;
		urowvec::const_iterator iter = entry->get<2>().begin();
		for (;iter != entry->get<2>().end(); ++iter) 
			hashVal = fmod(hashVal * double(entry->get<3>().n_rows) + M_PI * double(*iter), 1.0);
			
		return hashVal;
	}

	
	//******************************************************************************************************************************************
	
	arma::vec current_sequences_hash_value_vector()
	{
		using namespace arma;
		sequence_vector_iterator_type iter = begin();
		vec result(size_of_path_vector());
		vec::iterator output_it = result.begin();
		for(; iter != end(); ++iter, ++output_it) (*output_it) = hashValue(iter);
		return result;
	}
	
	
	//******************************************************************************************************************************************
	
	arma::vec current_sequences_likelihood_vector()
	{
		using namespace arma;
		sequence_vector_iterator_type iter = begin();
		vec result(size_of_path_vector());
		vec::iterator output_it = result.begin();
		for(; iter != end(); ++iter, ++output_it) (*output_it) = single_path_likelihood(iter->get<3>(), 1.0);
		return result;
	}
	
	
	
	//*********************************************************************************************************************************************************	
	
	void update_countmatrix(const arma::umat& oldpath, const arma::umat& newpath)
	{
		transition_counts = (transition_counts + newpath) - oldpath;
	}
	
	//*********************************************************************************************************************************************************	
	
	double single_path_likelihood(const arma::umat& pathcounter, double calc_lambda)
	{
		double sum = 0.0;
		using namespace arma;
      
		umat::const_iterator cmit = pathcounter.begin();
        
		mat::const_iterator tmit = transition_matrix.begin();
		mat::const_iterator pcmit = transition_matrix_prior.begin();
		
		for (; cmit != pathcounter.end(); ++cmit, ++tmit, ++pcmit) 
		{
			if ((*pcmit) > 0.0) sum += log((*tmit)) * calc_lambda*double((*cmit));
		}
		//~ std::cout << "ü " << sum << " ü";std::cout.flush();
		
		return sum;
	}
		
	//*********************************************************************************************************************************************************	
	
	double single_path_likelihood(const arma::umat& pathcounter)
	{
		return single_path_likelihood(pathcounter, get_lambda());
	}	
		
		
	//*********************************************************************************************************************************************************	
	
	double any_single_path_likelihood(const arma::umat& pathcounter, const arma::mat& any_transition_matrix, double calc_lambda)
	{
		double sum = 0.0;
		using namespace arma;
      
		umat::const_iterator cmit = pathcounter.begin();
        
		mat::const_iterator tmit = any_transition_matrix.begin();
		mat::const_iterator pcmit = transition_matrix_prior.begin();
		
		for (; cmit != pathcounter.end(); ++cmit, ++tmit, ++pcmit) 
		{
			if ((*pcmit) > 0.0) sum += log((*tmit)) * calc_lambda*double((*cmit));
		}
		//~ std::cout << "ü " << sum << " ü";std::cout.flush();
		
		return sum;
	}
	
	//*********************************************************************************************************************************************************	
	
	arma::rowvec get_transition_matrix_as_rowvector()
	{
		arma::rowvec result(transition_matrix.n_elem);
		std::copy(transition_matrix.begin(), transition_matrix.end(), result.begin());
		return result;
	}
	
	
	//*********************************************************************************************************************************************************	
	
	const arma::uword size()
	{
		return transition_sequences.size();
	}
		
	const arma::mat& get_transition_matrix() 
	{
		return transition_matrix;
	}
	
	const arma::mat& get_transition_matrix_prior() 
	{
		return transition_matrix_prior;
	}
	
	const arma::umat& get_transition_counts()
	{
		return transition_counts;
	}
	
	void set_transition_counts(const arma::umat& set_counts)
	{
		transition_counts = set_counts;
	}
	
	
	const double get_lambda() 
	{
		return lambda;
	}
	
	const arma::uword get_lambda_ref()
	{
		return lambda_ref;
	}
	
	const arma::uword get_my_running_number()
	{
		return my_running_number;
	}
	
	const arma::umat& get_emission_matrix() 
	{
		return emission_matrix;
	}
	
	HMMdataSet& get_observed_data() 
	{
		return (*observed_data);
	}
	
	sequence_vector_iterator_type begin() 
	{
		return transition_sequences.begin();
	}
	
	sequence_vector_iterator_type end() 
	{
		return transition_sequences.end();
	}
	
	arma::uword size_of_path_vector()
	{
		return transition_sequences.size();
	}
	
	
	friend void exchange_temperatures(HMMparticle& A, HMMparticle& B, bool collapsedVersion);
};

