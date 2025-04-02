# Distribution-free goodness-of-fit test for angular power spectrum models 

This repository contains code, simulation results, and tutorials demonstrating how to apply distribution-free goodness-of-fit tests to angular power spectrum models, as described in the following companion articles.
- **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

- **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.

## Code and simulation results 
Two folders detail the simulations and figures produced for the submitted PRL and PRD papers. Specifically, *codes_PRD* includes
- The file *Section3_Simulation.py* shows the simulation studies considering four combinations of data-generating models shown in Section III.B.1 and Section III.C.1 shown in [2]; 
- The file *PRD_Simulation_Result_for_Fig1-2* saves the simulation results above, and it has been used together with *Section3_Drawing_Fig1-2.R* to draw Figures 1-2 in [2];
- The file *Section3_Power&TypeIerror.py* shows the statistical properties (specifically, the power and the type-I error) of the proposed distribution-free test, as described in Section III.C.2 in [2]; 
- The file *Section4_Realdata_Analysis.py* with *v7* and *v8* shows the 

The folder *codes_PRL* includes 


The codes for generating the simulations plots takes a relatively long time to run for simulation, so we provide our simulation results 
[here for PRL paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRL/PRL_Simulation_Result) and [here for PRD paper](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/tree/main/Codes_PRD/PRD_Simulation_Result_for_Fig1-2) for your reference. 

## Implementation of the codes 

The codes for generating the simulations can be found [here](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_Simulation.py). 


These simulation results can be used to directly generate the plots in our paper through [the R file](https://github.com/xiangyu2022/Distfree_Test_SGWB_Models/blob/main/Codes_PRL/PRL_plots.R).

For technical inquiries, please reach out to Xiangyu Zhang at zhan6004@umn.edu.


## Tutorials on performing distribution-free tests for your models

Here is a step-by-step tutorial on applying distribution-free goodness-of-fit tests to your own model. You will need: 

- Your data, denoted $y$, of length N; 

- The variance-covariance matrix of your data, denoted $Sig$; 

- The postulated function of interest, denoted $postfunc$, with p unknown parameters.

**Step 1: Estimate parameters via minimizing Generalized Least Squares (GLS).** 
```python
import numpy as np 
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from autograd import jacobian

Sig_inv_sqrt = np.linalg.inv(sqrtm(Sig))
def optim_func(pars):
    diff = np.matrix(y - postfunc(pars))
    return diff @ Sig_inv @ diff.T
res = minimize(optim_func, np.repeat(0,len(pars)), method='nelder-mead')
```


**Step 2: Construct p orthonormal vectors (below is an example with p=3), each of length N.** Notice that the code here is just one way to construct such vectors; for more details, please refer to Section III.2 of [2].
<pre>r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]
r2 = r2/np.linalg.norm(r2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)
# r4 = r2**3 - np.inner(r1,r2**3)*r1 - np.inner(r2,r2**3)*r2 - np.inner(r3,r2**3)*r3
# r4 = r4/np.linalg.norm(r4)
# r5 = r2**4 - np.inner(r1,r2**4)*r1 - np.inner(r2,r2**4)*r2 - np.inner(r3,r2**4)*r3 - np.inner(r4,r2**4)*r4
# r5 = r5/np.linalg.norm(r5)
# ... 
</pre>

**Step 3: Obtain the residuals and the K2-transformed residuals.** For a detailed introduction to the K2 transformation, see Section III.2 of [2].
<pre>residuals = Sig_inv_sqrt @ (y - postfunc(res.x))
M_theta = Sig_inv_sqrt @ jacobian(postfunc)(res.x) 
R_n = M_theta.T @ M_theta
R_n_invsq = sqrtm(np.linalg.inv(R_n))
mu_theta = M_theta @ R_n_invsq 

# Creating the vectors of $r_j$ that is orthogonal to mu_theta. 
r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)

# Make the operators U_mu1_r1 U_mu2_r2_tilde U_mu3_r3_tilde implemented on the residuals 
U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
</pre>

**Step 4: Obtain the test statistics using the K2-transformed residuals.** In this case, we consider the counterparts of the Kolmogorov-Smirnov statistic and the Cramér-von Mises statistic in the context of regression. 
<pre>
ks = max(abs(np.cumsum(1/np.sqrt(N)*ehat_sum)))
cvm = sum((np.cumsum(1/np.sqrt(N)*ehat_sum))**2)
</pre>

**Step 5: Simulate the limiting null distribution of the test statistics and compute the p-value.** 
<pre>
B = 100000
KS = []
CVM = []
for b in range(B):
    e = np.random.normal(0, scale=1, size=N)
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS.append(max(abs(np.cumsum(1/np.sqrt(N)*(ehat_lim_sum)))))
    CVM.append(sum((np.cumsum(1/np.sqrt(N)*(ehat_lim_sum)))**2))

pval_KS, pval_CVM = (sum(KS>=ks)+1)/(B+1), (sum(CVM>=cvm)+1)/(B+1)
</pre>


## References
[1] **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

[2] **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.
