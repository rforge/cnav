#include <iostream>
#include <iomanip>  
#include <cmath>
#define ARMA_DONT_USE_BLAS
#include <RcppArmadillo.h>

#include <boost/random/mersenne_twister.hpp>
#include "HMMdataSet.hpp"
#include "HMMsequenceProducer.hpp"
#include "HMMgibbs.hpp"

#include <boost/lexical_cast.hpp>

RcppExport SEXP HMMinterface(SEXP genotypes, SEXP transition_matrix, SEXP emission_matrix, 
                             SEXP temperatures, 
                             SEXP r_how_many_sequence_tries, 
                             SEXP r_maxsequence_length,
                             SEXP path_sampling_repetitions,
                             SEXP internal_sampling,
                             SEXP n_swappings,
                             SEXP collapsed_sampling,
                             SEXP burnin, SEXP mc, 
                             SEXP chib_samples,
                             SEXP seed)
{
	BEGIN_RCPP
	
    using namespace Rcpp;

    IntegerMatrix iGenotypes(genotypes);
    arma::imat ia_genotypes(iGenotypes.begin(), iGenotypes.rows(), iGenotypes.cols(), true);
    arma::umat ua_genotypes = arma::conv_to<arma::umat>::from(ia_genotypes);

    NumericVector iTemps(temperatures);
    arma::vec na_temps(iTemps.begin(), iTemps.size(), true);
        	
    arma::uword randseed = as<arma::uword>(seed);
    BasicTypes::base_generator_type	rgen(randseed);
    
    arma::uword imc = as<arma::uword>(mc),
                iinternal = as<arma::uword>(internal_sampling),
                iburn = as<arma::uword>(burnin),
                how_many_sequence_tries = as<arma::uword>(r_how_many_sequence_tries),
                maxseqlength = as<arma::uword>(r_maxsequence_length),
                iswappings = as<arma::uword>(n_swappings),
                irepetitions = as<arma::uword>(path_sampling_repetitions),
                ichib_samples = as<arma::uword>(chib_samples);
    
    bool bcollapsed = as<bool>(collapsed_sampling);
    
    IntegerMatrix iTransitionMatrix(transition_matrix);
    arma::imat ia_graph(iTransitionMatrix.begin(), iTransitionMatrix.rows(), iTransitionMatrix.cols(), true);
    arma::umat ua_graph = arma::conv_to<arma::umat>::from(ia_graph);
    
    IntegerMatrix iEmission(emission_matrix);
    arma::imat ia_Emission(iEmission.begin(), iEmission.rows(), iEmission.cols(), true);
    arma::umat ua_Emission = arma::conv_to<arma::umat>::from(ia_Emission);
    
    Rcpp::Rcout << "\n**** Starting ***** \n"; Rcpp::Rcout.flush();
    	    	
	HMMdataSet dataTest(ua_genotypes);
	
	//~ Rcpp::Rcout << "\nInit Runner ... "; Rcpp::Rcout.flush();
	Gibbs_Sampling Runner(dataTest, na_temps, ua_graph, ua_Emission, maxseqlength, 
	                        how_many_sequence_tries, irepetitions, iinternal, iswappings, bcollapsed, randseed);

    //~ Rcpp::Rcout << "\nStart sampler ... "; Rcpp::Rcout.flush();	                        
	arma::cube runResult = Runner.run(iburn, imc);
	
	//~ Rcpp::Rcout << "\nSampler returned ..."; Rcpp::Rcout.flush();	  
	// now:: extract the interesting data
	//~ arma::cube  particle_temperature_indices = runResult.subcube(arma::span(0), arma::span::all, arma::span::all);
	//~ arma::cube mc_data = runResult.subcube(arma::span(1, runResult.n_rows-1), arma::span::all, arma::span::all);
	
	//~ Rcpp::Rcout << " A "; Rcpp::Rcout.flush();	  
	NumericMatrix rcpp_particle_temperature_indices(runResult.n_slices, runResult.n_cols);
	for (arma::uword islice = 0; islice < runResult.n_slices; ++islice )
		for (arma::uword icol = 0; icol < runResult.n_cols; ++icol )
			rcpp_particle_temperature_indices(islice, icol) = runResult(0, icol, islice);
	
	//~ Rcpp::Rcout << " B "; Rcpp::Rcout.flush();	 
	//~ runResult.print("runResult"); 
	NumericVector rcpp_mc_result(Dimension(runResult.n_rows-1,runResult.n_cols,runResult.n_slices));
	//~ Rcpp::Rcout << " Initialized "; Rcpp::Rcout.flush();	 
	for (arma::uword islice = 0; islice < runResult.n_slices; ++islice)
		for (arma::uword irow = 1; irow < runResult.n_rows; ++irow )
			for (arma::uword icol = 0; icol < runResult.n_cols; ++icol )
	           rcpp_mc_result[irow - 1 + icol*(runResult.n_rows-1) + islice*(runResult.n_rows-1)*runResult.n_cols] = 
		           runResult(irow,icol,islice);
	           
	//~ std::fill(rcpp_mc_result.begin(), rcpp_mc_result.end(), 0.0);
	// rcpp_mc_result.attr("dimnames") = 
	
	//~ Rcpp::Rcout << " size Rcpp " << rcpp_mc_result.size() << "\n"; Rcpp::Rcout.flush();	 
	//~ Rcpp::Rcout << " C "; Rcpp::Rcout.flush();	  
	//**************
	
	arma::mat jumpingLikelihoods = Runner.get_temperature_likelihoods();
	NumericMatrix rcpp_jumpingLikelihoods(jumpingLikelihoods.n_rows, jumpingLikelihoods.n_cols);
	std::copy(jumpingLikelihoods.begin(), jumpingLikelihoods.end(), rcpp_jumpingLikelihoods.begin());
	
	//~ // just calculate some marginal likelihoods
	arma::uword number_of_samples = (imc < ichib_samples)? imc : ichib_samples;
	arma::mat marlik = arma::zeros<arma::mat>(number_of_samples, 4);
	arma::uvec indexlist = arma::shuffle(arma::linspace<arma::uvec>(0,imc-1,imc));
	
	Rcpp::Rcout << "\nCalculating marginal likelihood\n>";
	arma::mat marginal_calculation_points = arma::zeros<arma::mat>(number_of_samples, runResult.n_rows-1);
	
	for (arma::uword i = 0; i < number_of_samples; i++)
	{
		arma::uword sampleIndex = indexlist[i];
		
		marginal_calculation_points(i,arma::span::all) = arma::trans(runResult.slice(sampleIndex).col(0).subvec(1,runResult.n_rows-1));
		marlik(i, arma::span::all) = Runner.get_Chib_marginal_likelihood(marginal_calculation_points(i,arma::span::all));
		Rcpp::Rcout << "."; Rcpp::Rcout.flush();
	}
	Rcpp::Rcout << "<\n";
	
	NumericMatrix rcpp_marginal_calculation_points(marginal_calculation_points.n_rows, marginal_calculation_points.n_cols);
	std::copy(marginal_calculation_points.begin(), marginal_calculation_points.end(), rcpp_marginal_calculation_points.begin());
	
	// Don't know how to do it better
	NumericMatrix chibResult(marlik.n_rows, marlik.n_cols);
	for (unsigned a = 0; a < marlik.n_rows; a++) for (unsigned b = 0; b < marlik.n_cols; b++) chibResult(a,b) = marlik(a,b);
	
	CharacterVector cvec(number_of_samples);
	for (unsigned icv = 0; icv < number_of_samples; icv++) {
		cvec[icv] = boost::lexical_cast<std::string>(icv);
	}
	
	List dimnms = List::create(
	  cvec, CharacterVector::create("Chib.Marginal.Likelihood", "Point.Likelihood", "Point.Prior.Density","Point.Posterior.Density"));
	chibResult.attr("dimnames") = dimnms;
	
	Rcpp::Rcout << "\nCalculation naive likelihood for comparison\n"; 
	//~ Rcpp::Rcout << " D "; Rcpp::Rcout.flush();	  
	double naiveMarlik = Runner.get_naive_marginal_likelihood();
	//~ Rcpp::Rcout << " E "; Rcpp::Rcout.flush();	  
	
	//*********** output of likelihoods **************
	
	arma::vec likelist = marlik.col(0);
	double chibML = log(arma::as_scalar(arma::mean(exp(likelist - likelist.max())))) + likelist.max();
	
	Rcpp::Rcout << "Current Chib marginal likelihood = " << std::setprecision(3) << chibML << "\n";
	Rcpp::Rcout << "Naive model marginal likelihood  = " << std::setprecision(3) << naiveMarlik << "\n";
	
	//************************************************
	
	
	
	List HMMresult = List::create( _("temperature.indices") = rcpp_particle_temperature_indices,
	                               _("mc.samples.transition.matrix") = rcpp_mc_result,
	                               _("chib.marginal.likelihoods") = chibResult,
	                               _("chib.estimation.points") = rcpp_marginal_calculation_points,
	                               _("naive.dirichlet.marginal.likelihoods") = naiveMarlik,
	                               _("transition.graph") = iTransitionMatrix,
	                               _("emission.matrix") = iEmission,
	                               _("n.samples") = imc,
	                               _("jumping.probabilities") = jumpingLikelihoods
	                               );
	                            
	
	Rcpp::Rcout << "\n**** Finished ***** \n"; Rcpp::Rcout.flush();
	
	return HMMresult;
	END_RCPP
}
 
