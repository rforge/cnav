#include "HMMparticle.hpp"

void exchange_temperatures(HMMparticle& A, HMMparticle& B, bool collapsedVersion = false)
{
	double old_likelihood, new_likelihood;
	if (collapsedVersion)
		old_likelihood = A.full_collapsed_likelihood() + B.full_collapsed_likelihood();
	else 
		old_likelihood = A.countmatrix_likelihood() + B.countmatrix_likelihood();
	
	// just exchange both temperatures
	arma::uword swap_lambda_ref = A.lambda_ref; A.lambda_ref = B.lambda_ref; B.lambda_ref = swap_lambda_ref;
	double swap_lambda = A.lambda; A.lambda = B.lambda; B.lambda = swap_lambda;
	
	if (collapsedVersion)
		new_likelihood = A.full_collapsed_likelihood() + B.full_collapsed_likelihood();
	else 
		new_likelihood = A.countmatrix_likelihood() + B.countmatrix_likelihood();
	
	boost::random::uniform_real_distribution<> pseudo_random_dist(0.0, 1.0);
	boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > 
		pseudo_randoms(A.central_generator, pseudo_random_dist);
	
	if (log(pseudo_randoms()) >= new_likelihood - old_likelihood) {
		// swap back, if not accepted
		swap_lambda_ref = A.lambda_ref; A.lambda_ref = B.lambda_ref; B.lambda_ref = swap_lambda_ref;
	    swap_lambda = A.lambda; A.lambda = B.lambda; B.lambda = swap_lambda;
	}
	
} 
