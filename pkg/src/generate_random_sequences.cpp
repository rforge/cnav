#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

RcppExport SEXP HMMgenerateDiplotype(SEXP transition_matrix, SEXP emission_matrix, SEXP count, SEXP random_seed, SEXP max_seq_len)
{
	BEGIN_RCPP
	using namespace Rcpp;
	
	NumericMatrix nTransitionMatrix(transition_matrix);
    arma::mat n_graph(nTransitionMatrix.begin(), nTransitionMatrix.rows(), nTransitionMatrix.cols(), true);
        
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    
	arma::uword n_count = as<arma::uword>(count),
	            rseed = as<arma::uword>(random_seed),
	            maxlen = as<arma::uword>(max_seq_len),
	            res_count = 0;
	            
	arma::imat result = arma::zeros<arma::imat>(n_count, ia_Emission.n_cols);
	arma::uvec res_counts = arma::zeros<arma::uvec>(n_count);
	
	typedef boost::random::mt19937 base_generator_type;
	base_generator_type rgen(rseed);
    boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
	boost::random::variate_generator<base_generator_type&, boost::random::uniform_real_distribution<> > test_randoms(rgen, test_dist);
			
	for (arma::uword i = 0; i < n_count; i++)
	{
		arma::irowvec sim_genotype = arma::zeros<arma::irowvec>(ia_Emission.n_cols);
		arma::uword j = 0, state = 0;
		bool ran_twice = false;
	
		while (state != n_graph.n_rows-1 && j < maxlen) 
		{
			// calculate new state
			double rnum = test_randoms(), summe = n_graph(state, 0);
            arma::uword newstate = 0;
	
	        while (newstate < n_graph.n_elem-1 && rnum >= summe) 
			{
				newstate++;
				summe += n_graph(state, newstate);
			}
			
		    state = newstate;
			// add emission to resulting genotype
		    sim_genotype = sim_genotype + ia_Emission(state,arma::span::all);
		    
		    if (!ran_twice && state==n_graph.n_rows-1)   // the HMM is simulated twice!
		    {
				state = 0;
				ran_twice = true;
			}
		    j++;
		}
		
		if (j < maxlen) // only valid sequences are accepted
		{
			arma::uword gindex = 0;
			while (res_counts[gindex] > 0 && arma::accu(result(gindex, arma::span::all) != sim_genotype)>0 && gindex < n_count) gindex++;
			 
		    if (res_counts[gindex] == 0) {
				res_counts[gindex] = 1;
				result(gindex, arma::span::all) = sim_genotype;
				res_count++;
			} else {
				res_counts[gindex] += 1;
				
			}
		}
	}
	
	result = result(arma::span(0,res_count-1), arma::span::all);
	res_counts = res_counts.subvec(0, res_count-1);
	arma::vec frequencies = arma::conv_to<arma::vec>::from(res_counts);
	frequencies = frequencies / sum(frequencies);  // MLE estimator simple an efficient	
		
	List retval = List::create(_("Genotypes") = wrap(result),
	                           _("Counts") = wrap(res_counts),
	                           _("Frequency.MLE") = wrap(frequencies));
	
	return retval;
	END_RCPP
}	 



RcppExport SEXP HMMgenerateHaplotype(SEXP transition_matrix, SEXP emission_matrix, SEXP count, SEXP random_seed, SEXP max_seq_len)
{
	BEGIN_RCPP
	using namespace Rcpp;
	
	NumericMatrix nTransitionMatrix(transition_matrix);
    arma::mat n_graph(nTransitionMatrix.begin(), nTransitionMatrix.rows(), nTransitionMatrix.cols(), true);
        
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    
	arma::uword n_count = as<arma::uword>(count),
	            rseed = as<arma::uword>(random_seed),
	            maxlen = as<arma::uword>(max_seq_len),
	            res_count = 0;
	            
	arma::imat result = arma::zeros<arma::imat>(n_count, ia_Emission.n_cols);
	arma::uvec res_counts = arma::zeros<arma::uvec>(n_count);
	
	typedef boost::random::mt19937 base_generator_type;
	base_generator_type rgen(rseed);
    boost::random::uniform_real_distribution<> test_dist(0.0, 1.0);
	boost::random::variate_generator<base_generator_type&, boost::random::uniform_real_distribution<> > test_randoms(rgen, test_dist);
			
	for (arma::uword i = 0; i < n_count; i++)
	{
		arma::irowvec sim_genotype = arma::zeros<arma::irowvec>(ia_Emission.n_cols);
		arma::uword j = 0, state = 0;
		
		while (state != n_graph.n_rows-1 && j < maxlen) 
		{
			// calculate new state
			double rnum = test_randoms(), summe = n_graph(state, 0);
            arma::uword newstate = 0;
	
	        while (newstate < n_graph.n_elem-1 && rnum >= summe) 
			{
				newstate++;
				summe += n_graph(state, newstate);
			}
			
		    state = newstate;
			// add emission to resulting genotype
		    sim_genotype = sim_genotype + ia_Emission(state,arma::span::all);
		    
		    j++;
		}
		
		if (j < maxlen) // only valid sequences are accepted
		{
			arma::uword gindex = 0;
			while (res_counts[gindex] > 0 && arma::accu(result(gindex, arma::span::all) != sim_genotype)>0 && gindex < n_count) gindex++;
			 
		    if (res_counts[gindex] == 0) {
				res_counts[gindex] = 1;
				result(gindex, arma::span::all) = sim_genotype;
				res_count++;
			} else {
				res_counts[gindex] += 1;
				
			}
		}
	}
	
	result = result(arma::span(0,res_count-1), arma::span::all);
	res_counts = res_counts.subvec(0, res_count-1);
	arma::vec frequencies = arma::conv_to<arma::vec>::from(res_counts);
	frequencies = frequencies / sum(frequencies);  // MLE estimator simple an efficient	
		
	List retval = List::create(_("Haplotypes") = wrap(result),
	                           _("Counts") = wrap(res_counts),
	                           _("Frequency.MLE") = wrap(frequencies));
	
	return retval;
	END_RCPP
}	 
