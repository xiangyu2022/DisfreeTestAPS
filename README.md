# Distribution-free goodness-of-fit test for angular power spectrum models 

This repository contains code, simulation results, and tutorials demonstrating how to apply distribution-free goodness-of-fit tests to angular power spectrum models, as described in the following companion articles.
- **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

- **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.

## Code and simulation results 
There are two folders that detail the simulations and figures produced for the two previously discussed papers. The *codes_PRL* contains the codes for the simulation described in Section III. In this case, we consider the data are generaterd through 

# README

For example, let $\mathbf{C}^{M_1}(\boldsymbol{\theta})$ and $\mathbf{C}^{M_2}(\boldsymbol{\theta})$ be two candidate models for $ E[\widehat{\mathbf{C}}] $ with components

$$
C^{M_1}_{\ell}(x_i,\boldsymbol{\theta}) &= \theta_0 + \theta_1 \ell + \theta_2 x_i,\\
C^{M_2}_{\ell}(x_i,\boldsymbol{\theta}) &= \exp\{\theta_0 + \theta_1 x_i + \theta_2 x_i \ell\}.
$$

The \(x_i\) values considered correspond to \(n=100\) evenly spaced points on the interval \([0,1]\), and \(\ell = 1, \dots, 5\). The true value of the parameter \(\boldsymbol{\theta} = (\theta_0, \theta_1, \theta_2)\) is \((5, 2, 4)\). In the numerical experiments conducted, \(\boldsymbol{\theta}\) is treated as unknown and is estimated as in [\[optim\]](#optim). 

For each of the two models in [\[eqn:models\]](#eqnmodels), we consider two possible distributions for the corresponding estimator of the power spectrum:

$$
\begin{aligned}
\widehat{\mathbf{C}}_{1,k}(x_i) &\sim \mathcal{N}\Bigl(\mathbf{C}^{M_k}(x_i,\boldsymbol{\theta}), \mathbf{\Sigma}(x_i)\Bigr),\\
\widehat{\mathbf{C}}_{2,k}(x_i) &\sim T_6\Bigl(\mathbf{C}^{M_k}(x_i,\boldsymbol{\theta}), \mathbf{\Sigma}(x_i)\Bigr).
\end{aligned}
$$

That is, for each \(x_i\), \(\widehat{\mathbf{C}}_{1,k}(x_i)\) and \(\widehat{\mathbf{C}}_{2,k}(x_i)\) are independent random vectors following, respectively, a multivariate normal and a multivariate \(t\)-distribution with six degrees of freedom. The mean vectors, \(\mathbf{C}^{M_k}(x_i,\boldsymbol{\theta})\), have components \(C^{M_k}_{\ell}(x_i,\boldsymbol{\theta})\) as defined above for \(k=1,2\). The covariance matrices \(\mathbf{\Sigma}(x_i)\) are generated from a Wishart distribution with 10 degrees of freedom.



The codes for generating the simulations plots takes a relatively long time to run for simulation, so we provide our simulation results 
[here for PRL paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRL/PRL_Simulation_Result) and [here for PRD paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRD/PRD_Simulation_Result_for_Fig1-2) for your reference. 


## Tutorials on performing distribution-free tests for your models
Here, suppose you are given a set of data and a postulated model for it, and 


A R 


## Implementation of the codes 

The codes for generating the simulations can be found [here](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_Simulation.py). 


These simulation results can be used to directly generate the plots in our paper through [the R file](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_plots.R).

For technical inquiries, please reach out to Xiangyu Zhang at zhan6004@umn.edu.

## References
- **Algeri, S. et al. (2024+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

- **Zhang, X. et al. (2024+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.
