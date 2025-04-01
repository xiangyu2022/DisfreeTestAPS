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

Here we provide a step-by-step tutorial on how we apply the distribution-free goodness-of-fit tests to your own model. You need to have your data, denoted as $y$, its variance-covariance matrix, denoted as $Sigma$, and the postulated model that is of interest. 
As the first step, we obtain the solve the estimators by minimizing the generalized least squares. This can be achieved by 

<pre> ```python 
def optim_func(pars):
    diff = np.matrix(y - postulated_function(pars))
    return diff @ Sig_inv @ diff.T
res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead')
``` </pre>


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
