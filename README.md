# Distribution-free goodness-of-fit test for angular power spectrum models 

This repository contains code, simulation results, and tutorials demonstrating how to apply distribution-free goodness-of-fit tests to angular power spectrum models, as described in the following companion articles.
- **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

- **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.


## Code and Simulation Results

Two folders—`codes_PRD` and `codes_PRL`—contain the code for simulation and figure-generation scripts for the submitted PRD and PRL papers, respectively.

### Folder: `codes_PRD`

- **`Section3_Simulation.py`**  
  Implements simulation studies examining four combinations of data-generating models, as described in Sections III.B.1 and III.C.1 of [2].

- **`PRD_Simulation_Result_for_Fig1-2`**  
  Stores the simulation results (which require substantial computation time). These results, together with `Section3_Drawing_Fig1-2.R`, were used to create Figures 1–2 in [2].

- **`Section3_Power&TypeIerror.py`**  
  Evaluates the statistical properties (power and type-I error) of the proposed distribution-free test, as detailed in Section III.C.2 of [2].

- **`Section4_Realdata_Analysis.py` **  
  Contains the real data analysis described in Section IV of [2].

### Folder: `codes_PRL`

- **`PRL_Simulation.py`**  
  Implements simulation studies involving four combinations of data-generating models, as discussed in Section III of [1].

- **`PRL_Simulation_Result`**  
  Stores the simulation results. Together with `PRL_plots.R`, it was used to produce Figure 1 in Section 3 of [1].

---

## Tutorials on Performing Distribution-Free Tests for Your Models

Below is a step-by-step tutorial demonstrating how to apply distribution-free goodness-of-fit tests to your own model. You will need:

- Your data, denoted y, of length N
- The variance-covariance matrix of your data, denoted Sigma
- A postulated function of interest, denoted \(postfunc\), with p unknown parameters. 

---

### Step 1: Estimate Parameters via Generalized Least Squares (GLS)

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

### Step 2: Construct p Orthonormal Vectors
Below is an example for p=3, each vector of length N. Note that this code is just one example of how to construct such vectors. For more details, see Section III.2 of [2].
```python
r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]
r2 = r2/np.linalg.norm(r2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)
# r4 = r2**3 - np.inner(r1,r2**3)*r1 - np.inner(r2,r2**3)*r2 - np.inner(r3,r2**3)*r3
# r4 = r4/np.linalg.norm(r4)
# r5 = r2**4 - np.inner(r1,r2**4)*r1 - np.inner(r2,r2**4)*r2 - np.inner(r3,r2**4)*r3 - np.inner(r4,r2**4)*r4
# r5 = r5/np.linalg.norm(r5)
# ... 
```

### Step 3: Obtain Residuals and K2-Transformed Residuals
For a detailed introduction to the K2 transformation, see Section III.2 of [2].

```python
residuals = Sig_inv_sqrt @ (y - postfunc(res.x))
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
```

### Step 4: Compute Test Statistics
Here, we consider the Kolmogorov–Smirnov and Cramér–von Mises statistics in the context of regression. 
```python
ks = max(abs(np.cumsum(1/np.sqrt(N)*ehat_sum)))
cvm = sum((np.cumsum(1/np.sqrt(N)*ehat_sum))**2)
```
### Step 5: Simulate the Limiting Null Distribution and Compute p-Values

```python
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
```

For any technical inquiry, please contact **Xiangyu Zhang** at [zhan6004@umn.edu](mailto:zhan6004@umn.edu).


## References
[1] **Algeri, S. et al. (2025+).**
  *A Distribution-Free Approach to Testing Models for Angular Power Spectra*  
  submitted to *Physical Review Letters*.

[2] **Zhang, X. et al. (2025+).**
  *On validating Angular Power Spectral Models for the Stochastic Gravitational-Wave Background Without Distributional Assumptions*  
  submitted to *Physical Review D*.
