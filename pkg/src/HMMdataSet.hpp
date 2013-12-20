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
	
	arma::umat genotype_list;                    // stores the unique observed genotypes - a sorted list!!!
    arma::uvec genotype_refs;					 // stores the genotype refs to the genotype_list
    
public:
    HMMdataSet(arma::umat init_genotypes);
    
    arma::uword get_ref_count();
    const arma::uword n_individuals() const;
    
    const arma::uvec& get_genotype_refs() const;
    arma::urowvec get_genotype(arma::uword ref);
    void get_ref(BasicTypes::SequenceReferenceTuple& tuple, arma::urowvec genotype);
    
    double get_log_multinomial_coefficient();
    
    double calculate_likelihood(const arma::vec& probabilities);
    double naive_marginal_likelihood(double prior = 0.5);
};
