/*
 * HMMdataSet.hpp
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
 
#include "BasicTypes.hpp"

#pragma once
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple.hpp>


class HMMdataSet {
	
	arma::umat genotype_list;                    // stores the unique observed genotypes
    
    arma::uword number_of_individuals;
    arma::uvec individuals;                       // stores, which genotype refs are for which individual. Must be sorted
    arma::uvec genotype_refs;					  // stores the genotype refs to the genotype_list
    arma::vec  id_weights;                        // stores the probability weights for weighted data
    
public:
    HMMdataSet(arma::umat init_genotypes,  arma::uvec init_individuals,  arma::vec init_probabilities);
    
    arma::uvec random_draw(BasicTypes::base_generator_type& rand_gen);
    
    arma::uword get_ref_count();
    arma::urowvec get_genotype(arma::uword ref);
    void get_ref(BasicTypes::SequenceReferenceTuple& tuple, arma::urowvec genotype);
    
    arma::uword n_individuals() const;
    
    double calculate_likelihood(const arma::vec& probabilities);
    
    arma::vec naive_marginal_likelihood(arma::uword Nsamp, BasicTypes::base_generator_type& rand_gen);
};
