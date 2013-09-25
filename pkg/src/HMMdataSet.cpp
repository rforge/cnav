/*
 * HMMdataSet.cpp
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

#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include "BasicTypes.hpp"
#include "HMMdataSet.hpp"
 
#include <iostream>


#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple.hpp>

#include <boost/math/distributions/beta.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/tuple/tuple.hpp> 
#include <boost/random/uniform_real.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>

 
HMMdataSet::HMMdataSet(arma::umat init_genotypes,  arma::uvec init_individuals,  arma::vec init_probabilities)
{
	using namespace arma;
	
	individuals = init_individuals;  // expects sorted list of individuals!
	id_weights  = init_probabilities; // must sum up to 1.0 for each individual
	
	number_of_individuals = round(accu(id_weights));
	
	uword elements = init_genotypes.n_rows;
	genotype_refs = linspace<uvec>(0, elements-1, elements);
	
	uvec result = ones<uvec>(init_genotypes.n_rows);
	genotype_list = umat(elements, init_genotypes.n_cols);
	uword store_position = 0;
	
	uvec::iterator res_iter = result.begin();
	uvec::iterator ref_iter = genotype_refs.begin();
	
	if (init_genotypes.n_rows == 0) {
		Rcpp::Rcout << "No genotypes in data set\n"; Rcpp::Rcout.flush();
		throw Rcpp::exception("No genotypes in data set","HMMdataSet.cpp",61);
	}
	
	for (uword i = 0; res_iter != result.end(); res_iter++, i++, ref_iter++) if (*res_iter > 0)
	{   
		uvec::iterator res_iter2 = res_iter+1;
		uvec::iterator ref_iter2 = ref_iter+1;
		*ref_iter = store_position;
		
		for (uword j = i+1; res_iter2 != result.end(); res_iter2++, ref_iter2++, j++) if (*res_iter2 > 0)
		{
			if (accu(init_genotypes(i, span::all) != init_genotypes(j, span::all)) == 0) 
			{
				*res_iter2 -= 1;
				*res_iter += 1;
				*ref_iter2 = store_position;
			}
		}
		
		genotype_list(store_position, span::all) = init_genotypes(i, span::all);
		store_position++;
	}
	
	// now, the list is compressed
	genotype_list = genotype_list(span(0, store_position-1), span::all);
		
}
		
arma::uword HMMdataSet::get_ref_count()
{
	return genotype_list.n_rows;
}

arma::urowvec HMMdataSet::get_genotype(arma::uword ref)
{
	return genotype_list(ref, arma::span::all);
}

void HMMdataSet::get_ref(BasicTypes::SequenceReferenceTuple& tuple, arma::urowvec genotype)
{
	using namespace arma;
	
	uword index = 0;
	bool found = false;
	
	while (!found && index < genotype_list.n_rows) 
	{
		uword spalte = 0;
		found = true;
		while (spalte < genotype_list.n_cols && found) 
		{
			found = genotype(spalte) == genotype_list(index, spalte);
			spalte++;
		}
		if (!found) index++;
	}
	
	tuple.get<3>() = found;
	if (found) tuple.get<1>() = index;
}

arma::uvec HMMdataSet::random_draw(BasicTypes::base_generator_type& rand_gen)
{
	using namespace arma;
	
	uvec::const_iterator id_iter = individuals.begin();
	uword start_id = *id_iter;
		
	vec::const_iterator weights_iter = id_weights.begin();
	uvec::const_iterator ref_iter = genotype_refs.begin();
	
	// Prepare result data
	uvec countlist = zeros<uvec>(genotype_list.n_rows);
	
	// Prepare random generator
	boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
	boost::random::variate_generator<BasicTypes::base_generator_type&, boost::random::uniform_real_distribution<> > 
	   test_randoms(rand_gen, test_dist);
	
	// Use property of being sorted
	
	while (id_iter != individuals.end())
	{
		// draw a random number
		double rnum = test_randoms(), sum = 0.0;
		bool found;
		uword current_id = *id_iter;
		
		// search it
		do {
			sum += *weights_iter;
			found = rnum <= sum;
		     		     
		    if (!found) {
				weights_iter++;
				id_iter++;
				ref_iter++;
			}
		} while (!found && id_iter != individuals.end());	
		
		// count up the list
		if (found) countlist[*ref_iter]+=1;
		
		if (!found) {
		   throw Rcpp::exception("Sum of weights does not equal 1","HMMdataSet.cpp",167);
		}
	
	    // and seach until the next individual or end
	    while (id_iter != individuals.end() && *id_iter == current_id) 
	    {	
			weights_iter++;
			id_iter++;
			ref_iter++;
		}
	}	
	
	return countlist;
}	

arma::uword HMMdataSet::n_individuals() const
{
	return number_of_individuals;
}	


/*
 * 
 * name: calculate_likelihood
 * @param: const arma::vec& probabilities - probability for each genotype in the reference list
 * @return: accumulated log-likelihood
 * 
 */
double HMMdataSet::calculate_likelihood(const arma::vec& probabilities)
{
	using namespace arma;
	uvec::const_iterator id_iter = individuals.begin();
	uword start_id = *id_iter;
		
	vec::const_iterator weights_iter = id_weights.begin();
	uvec::const_iterator ref_iter = genotype_refs.begin();	
	
	double likelihood_result = 0;
	
	while (id_iter != individuals.end()) 
	{
		double partial_sum = 0;
		uword current_id = *id_iter;
	
	    // Accumulation algorithm makes use of a specialized internal loop
		while (id_iter != individuals.end() & *id_iter == current_id) 
		{
			partial_sum += probabilities[*ref_iter] * (*weights_iter);
			id_iter++;
			weights_iter++;
			ref_iter++;
		}
		
        likelihood_result += log(partial_sum);
	}
	
	return likelihood_result;
}

arma::vec HMMdataSet::naive_marginal_likelihood(arma::uword Nsamp, BasicTypes::base_generator_type& rand_gen)
{
	using namespace arma;
	
	vec result(Nsamp);
	
	for (uword i = 0; i < Nsamp; i++) {
		vec alpha = 0.5 + conv_to<vec>::from(random_draw(rand_gen));
				
		double summe = 0.0;
		vec::const_iterator iter = alpha.begin();
		for (uword j = 0; iter != alpha.end(); iter++) summe += boost::math::lgamma(*iter);
		summe -= boost::math::lgamma(arma::accu(alpha));
		
		result[i] = summe;
	}
	
	return result;
			
}
