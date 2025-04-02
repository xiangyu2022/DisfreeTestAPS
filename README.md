# Distribution-free goodness-of-fit test for angular power spectrum models 

This repository contains code, simulation results, and tutorials demonstrating how to apply distribution-free goodness-of-fit tests to angular power spectrum models, as described in the following companion articles.
- **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

- **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.

## Code and simulation results 
There are two folders that detail the simulations and figures produced for the above two papers. For both of the cases, we consider the data are generated through the model 

$$\hat{A}$$

The codes for generating the simulations plots takes a relatively long time to run for simulation, so we provide our simulation results 
[here for PRL paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRL/PRL_Simulation_Result) and [here for PRD paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRD/PRD_Simulation_Result_for_Fig1-2) for your reference. 


## Tutorials on performing distribution-free tests for your models using the provided codes

Here is a step-by-step tutorial on applying distribution-free goodness-of-fit tests to your own model. You will need: 

- Your data, denoted $y$; 

- The variance-covariance matrix of your data, denoted $Sig$; 

- The postulated function of interest, denoted $postfunc$.

**Step 1: Estimate parameters via Generalized Least Squares (GLS).** Obtain the parameter estimates by minimizing the generalized least squares objective. This can be done by:

<pre>import numpy as np 
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from autograd import jacobian

Sig_inv_sqrt = np.linalg.inv(sqrtm(Sig))
def optim_func(pars):
    diff = np.matrix(y - postfunc(pars))
    return diff @ Sig_inv @ diff.T
res = minimize(optim_func, np.repeat(0,len(pars)), method='nelder-mead')
</pre>

**Step 2: Obtain the residuals and the standardized gradient $\mu_{\theta}$. ** 

<pre>  
residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
 #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
def M_theta_func(theta):
    theta = np.array(theta)
    return Sig_inv_sqrt @ jacobian(postulated_function)(theta)
M_theta = M_theta_func(res.x)
R_n = M_theta.T @ M_theta
R_n_invsq = sqrtm(np.linalg.inv(R_n))
mu_theta = M_theta @ R_n_invsq 
</pre>

<pre>  
# Creating the r_1 and r_2， can be generalized to r_p.
r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)

# Obtain the transformation, ehat. Up here
U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
</pre>

Here, you need to prepare for your own suppose you are given a set of data, its variance-covariance matrix, and the models of interest for testing. Now, the first task to do is to estimate the unknown parameters of the model of interest. This part corresponds to solving 


## Implementation of the codes 

The codes for generating the simulations can be found [here](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_Simulation.py). 


These simulation results can be used to directly generate the plots in our paper through [the R file](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_plots.R).

For technical inquiries, please reach out to Xiangyu Zhang at zhan6004@umn.edu.

## References
[1] **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

[2] **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.
