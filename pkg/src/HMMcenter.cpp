/*
 * HMMcenter.cpp
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
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMgibbs.hpp"


RcppExport SEXP HMMsamplier(SEXP genotypes,            // a matrix with n cols and m rows, containing genotypes for each pseudoindividual
                            SEXP individuals,          // an integer vector containing the IDs
                            SEXP weights,              // a numeric vector containing the probability weights for each pseudoindividual
                            SEXP transition_matrix,    // a transition (numeric) matrix for generation of haplotypes
                            SEXP emission_matrix,      // the corresponding emissions as an integer matrix
                            SEXP temperatures,         // a numeric vector containing temperatures between 0..1
                            SEXP percentage,           // for sampling (see documentation)
                            SEXP 
                            SEXP burnin, 
                            SEXP mc, 
                            SEXP seed)
{
	BEGIN_RCPP
	
    using namespace Rcpp; 

    IntegerMatrix iGenotypes(genotypes);
    arma::imat ia_genotypes(iGenotypes.begin(), iGenotypes.rows(), iGenotypes.cols(), true);
    arma::umat ua_genotypes = arma::conv_to<arma::umat>::from(ia_genotypes);
    
    IntegerVector iIDs(individuals);
    arma::ivec ia_ids(iIDs.begin(), iIDs.size(), true);
    arma::uvec ua_ids = arma::conv_to<arma::uvec>::from(ia_ids);
    
    NumericVector iWeights(weights);
    arma::vec na_weights(iWeights.begin(), iWeights.size(), true);
    
    NumericVector iTemps(temperatures);
    arma::vec na_temps(iTemps.begin(), iTemps.size(), true);
    	
    arma::uword randseed = as<arma::uword>(seed);
    BasicTypes::base_generator_type	rgen(randseed);
    
    arma::uword imc = as<arma::uword>(mc),
                iburn = as<arma::uword>(burnin);
    
    double a_percentage = as<double>(percentage);
    
    NumericMatrix iTransitionMatrix(transition_matrix);
    arma::mat na_transition(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    arma::umat ua_graph = na_transition > 0.0;
    
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    	    	
	HMMdataSet dataTest(ua_genotypes, ua_ids, na_weights);
	
	Gibbs_Sampling Runner(dataTest, na_temps, ua_graph, ua_Emission, a_percentage, 100, 1000, randseed);
	std::cout << "\nRun!!!\n"; std::cout.flush();
	arma::mat runResult = Runner.run(iburn, imc);
	
	return wrap(runResult);
	END_RCPP
}
