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

 
HMMdataSet::HMMdataSet(arma::umat init_genotypes)
{
	using namespace arma;
	
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
	
	// and the corresponding individuals are sorted for speed purposes
	genotype_refs = sort(genotype_refs);
		
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
	vec alpha = zeros<vec>(get_ref_count());
	uvec::const_iterator ref_iter = genotype_refs.begin();	
	
	double likelihood_result = 0;
	for (;ref_iter != genotype_refs.end(); ref_iter++) 
	{
		likelihood_result += log(probabilities[*ref_iter]);
		alpha[*ref_iter] += 1.0;
	}
	
	likelihood_result += boost::math::lgamma(accu(alpha) + 1);   // (x1+x2+x3+...)! = n!
	for (vec::const_iterator iter = alpha.begin(); iter != alpha.end(); iter++) likelihood_result -= boost::math::lgamma(*iter + 1);  // x_j!
	
	return likelihood_result;
}

double HMMdataSet::naive_marginal_likelihood(double prior)
{
	using namespace arma;
	
	// initialize 
	double result = 0;
	vec alpha = zeros<vec>(get_ref_count());
	
	// count number of genotypes in each slot of alpha
	uvec::const_iterator ref_iter = genotype_refs.begin();	
	for (;ref_iter != genotype_refs.end(); ref_iter++) alpha[*ref_iter] += 1.0;
	
	// calculate exact marginal likelihood
	double summe = 0.0;
	vec::const_iterator iter = alpha.begin();
	
	// first: add likelihood function 
	summe += boost::math::lgamma(accu(alpha) + 1);   // (x1+x2+x3+...)! = n!
	for (; iter != alpha.end(); iter++) summe -= boost::math::lgamma(*iter + 1);  // x_j!
	
	// then: add prior
	summe += boost::math::lgamma(0.5 * double(alpha.n_elem));
	summe -= double(alpha.n_elem) * boost::math::lgamma(0.5);
	
	// at last: substract posterior
	summe -= boost::math::lgamma(accu(alpha + 0.5));   // (x1+x2+x3+...)! = n!
	for (; iter != alpha.end(); iter++) summe += boost::math::lgamma(*iter + 0.5); 
	
	return result;
}


const arma::uvec& HMMdataSet::get_genotype_refs() const
{
	// necessary for 
	return genotype_refs;
}

const arma::uword HMMdataSet::n_individuals() const
{
	return genotype_refs.n_elem;
}
	

