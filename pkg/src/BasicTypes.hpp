/*
 * BasicTypes.hpp
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

/*
 * This header describes some basic and types the Gibbs sampler 
 * needs
*/

#pragma once
#include <iostream>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/tuple/tuple.hpp>

namespace BasicTypes {
	// it's used everywhere
	typedef boost::random::mt19937 base_generator_type;
	
	// Tuple: sequence, reference to genotype list, transition counts, validity
   	typedef boost::tuple<arma::urowvec, arma::uword, arma::umat, bool> SequenceReferenceTuple;
   	
   	typedef boost::tuple<arma::uword, arma::uword, arma::urowvec, arma::umat, bool> IDRefSequenceCountTuple;
};
	
