#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary packages
import pandas as pd
import autograd.numpy as np
import random
import matplotlib.pyplot as plt
from autograd import jacobian
from scipy.optimize import minimize
from scipy.linalg import sqrtm, block_diag
from scipy.stats import multivariate_normal, multivariate_t, wishart


# In[2]:


def C_M1(pars):
    return pars[0] + pars[1] * l + pars[2]* X

def C_M2(pars):
    return np.exp(pars[0] + pars[1]* X + pars[2]* X * l)

# def C_M2(pars):
#     return (pars[0] + pars[1] * l + pars[2]* l**2) * X**(2/3)

def optim_func(y, C_M, pars):
    diff = np.matrix(y - C_M(pars))
    return float(diff @ Sig_inv @ diff.T)

def gls(y, C_M):
    res_C = minimize(lambda pars: optim_func(y, C_M, pars), par, method='Nelder-Mead')
    return res_C.x


# In[3]:


# Obtain the residuals, M_theta and R_n. 
def sphered_residuals_func(y, C_M):
    theta = gls(y,C_M)
    return Sig_inv_sqrt @ (y - C_M(theta))

def mu_theta_func (y, C_M):
    theta = gls(y, C_M)
    M_theta = Sig_inv_sqrt @ jacobian(C_M)(theta)
    R_n_invsq = sqrtm(np.linalg.inv(M_theta.T @ M_theta))
    return M_theta @ R_n_invsq 

def rotated_residuals_func (y, C_M):
    mu_theta = mu_theta_func(y, C_M)
    residuals = sphered_residuals_func(y, C_M)
    
    r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    
#     U1r4 = r4 - np.inner(mu_theta[:,0]-r1, r4)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
#     U2U1r4 = U1r4 - np.inner(mu_theta[:,1]-r2_tilde, U1r4)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
#     r4_tilde = U2U1r4 - np.inner(mu_theta[:,2]-r3_tilde, U2U1r4)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    
#     # Obtain the transformation, ehat. Up here
#     U_mu4_r4_res = residuals    - np.inner(mu_theta[:,3]-r4_tilde, residuals)   /(1-np.inner(mu_theta[:,3],r4_tilde))*(mu_theta[:,3]-r4_tilde)
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # new_ehat = ehat.reshape(round(N/L),L).sum(axis=1)/np.sqrt(N)
    return ehat

def Test_stat(res, stat):
    sum_res = res.reshape(n,L).sum(axis=1)/np.sqrt(n)
    if stat == "KS":
        return max(abs(np.cumsum(sum_res)))
    elif stat == "CVM":
        return sum((np.cumsum(sum_res))**2)
    else:
        error("stat should be specified as either 'KS', or 'CVM.'")


# In[6]:


np.random.seed(1)
N = 500
par = np.array([5, 2, 4], dtype=float)
L = 5
n = N//L
l = np.tile(np.linspace(1,5,num=L),n)
Sigmas = (wishart.rvs(df=10, size=n, scale=np.identity(L)))
# Sigmas = np.tile(np.identity(L),(n,1,1))
Sigma = block_diag(*Sigmas)
Sig_inv = np.linalg.inv(Sigma)
Sig_inv_sqrt = sqrtm(Sig_inv)
X = np.repeat(np.linspace(0,1,num=n),L)
r1 = np.repeat(1/np.sqrt(N),N)

r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]/np.sqrt(1-1/N**2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)

# r4 = r2**3 - np.inner(r1,r2**3)*r1 - np.inner(r2,r2**3)*r2 - np.inner(r3,r2**3)*r3
# r4 = r4/np.linalg.norm(r4)
# print(gls(y1,C_M1),gls(y2,C_M2),gls(y3,C_M1),gls(y4,C_M2))


# In[7]:


B = 10000
KS_res_y1, KS_res_y2, KS_res_y3, KS_res_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
KS_rotated_y1, KS_rotated_y2, KS_rotated_y3, KS_rotated_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
CVM_res_y1, CVM_res_y2, CVM_res_y3, CVM_res_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
CVM_rotated_y1, CVM_rotated_y2, CVM_rotated_y3, CVM_rotated_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
y3_nonflat, y4_nonflat = np.zeros([N//L,L]), np.zeros([N//L,L])
for j in range(0,B,1):
    for i in range(n):
        y3_nonflat[i:] = (multivariate_t.rvs(df=6,loc=C_M1(par)[L*i:L*i+L],size=1,shape=Sigmas[i]*2/3))
        y4_nonflat[i:] = (multivariate_t.rvs(df=6,loc=C_M2(par)[L*i:L*i+L],size=1,shape=Sigmas[i]*2/3))
    y1, y2 = multivariate_normal.rvs(mean=C_M1(par),size=1,cov=Sigma), multivariate_normal.rvs(mean=C_M2(par),size=1,cov=Sigma)
    y3, y4 = y3_nonflat.flatten(), y4_nonflat.flatten() 

    res1, res2 = sphered_residuals_func(y1, C_M1), sphered_residuals_func(y2, C_M2)
    res3, res4 = sphered_residuals_func(y3, C_M1), sphered_residuals_func(y4, C_M2)
    rotated_res1, rotated_res2 = rotated_residuals_func(y1, C_M1), rotated_residuals_func(y2, C_M2)
    rotated_res3, rotated_res4 = rotated_residuals_func(y3, C_M1), rotated_residuals_func(y4, C_M2)

    KS_res_y1[j],  KS_res_y2[j],  KS_res_y3[j],  KS_res_y4[j]  = Test_stat(res1,"KS"),  Test_stat(res2,"KS"),   Test_stat(res3,"KS"),  Test_stat(res4,"KS")    
    CVM_res_y1[j], CVM_res_y2[j], CVM_res_y3[j], CVM_res_y4[j] = Test_stat(res1,"CVM"), Test_stat(res2,"CVM"),  Test_stat(res3,"CVM"), Test_stat(res4,"CVM")    
    KS_rotated_y1[j], KS_rotated_y2[j], KS_rotated_y3[j], KS_rotated_y4[j]     = Test_stat(rotated_res1,"KS"),  Test_stat(rotated_res2,"KS"),  Test_stat(rotated_res3,"KS"),  Test_stat(rotated_res4,"KS") 
    CVM_rotated_y1[j], CVM_rotated_y2[j], CVM_rotated_y3[j], CVM_rotated_y4[j] = Test_stat(rotated_res1,"CVM"), Test_stat(rotated_res2,"CVM"), Test_stat(rotated_res3,"CVM"), Test_stat(rotated_res4,"CVM")
    print(j)


# In[8]:


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

probs=np.arange(0,1,0.002)

plt.scatter(np.quantile(a = KS_res_y1, q = probs, method='closest_observation'),np.quantile(a = KS_res_y2, q = probs, method='closest_observation'))
abline(1,0)
plt.show()
plt.scatter(np.quantile(a = KS_rotated_y1, q = probs, method='closest_observation'),np.quantile(a = KS_rotated_y2, q = probs, method='closest_observation'))
abline(1,0)
plt.show()

plt.scatter(np.quantile(a = KS_res_y3, q = probs, method='closest_observation'),np.quantile(a = KS_res_y4, q = probs, method='closest_observation'))
abline(1,0)
plt.show()

plt.scatter(np.quantile(a = KS_rotated_y1, q = probs, method='closest_observation'),np.quantile(a = KS_rotated_y3, q = probs, method='closest_observation'))
abline(1,0)
plt.show()

plt.scatter(np.quantile(a = KS_rotated_y1, q = probs, method='closest_observation'),np.quantile(a = KS_rotated_y4, q = probs, method='closest_observation'))
abline(1,0)
plt.show()


# In[9]:


np.savetxt("KS_res_y1_L15_df6_100.csv", KS_res_y1, delimiter=",")
np.savetxt("KS_res_y2_L15_df6_100.csv", KS_res_y2, delimiter=",")
np.savetxt("KS_res_y3_L15_df6_100.csv", KS_res_y3, delimiter=",")
np.savetxt("KS_res_y4_L15_df6_100.csv", KS_res_y4, delimiter=",")
np.savetxt("CVM_res_y1_L15_df6_100.csv", CVM_res_y1, delimiter=",")
np.savetxt("CVM_res_y2_L15_df6_100.csv", CVM_res_y2, delimiter=",")
np.savetxt("CVM_res_y3_L15_df6_100.csv", CVM_res_y3, delimiter=",")
np.savetxt("CVM_res_y4_L15_df6_100.csv", CVM_res_y4, delimiter=",")

np.savetxt("KS_rotated_y1_L15_df6_100.csv", KS_rotated_y1, delimiter=",")
np.savetxt("KS_rotated_y2_L15_df6_100.csv", KS_rotated_y2, delimiter=",")
np.savetxt("KS_rotated_y3_L15_df6_100.csv", KS_rotated_y3, delimiter=",")
np.savetxt("KS_rotated_y4_L15_df6_100.csv", KS_rotated_y4, delimiter=",")
np.savetxt("CVM_rotated_y1_L15_df6_100.csv", CVM_rotated_y1, delimiter=",")
np.savetxt("CVM_rotated_y2_L15_df6_100.csv", CVM_rotated_y2, delimiter=",")
np.savetxt("CVM_rotated_y3_L15_df6_100.csv", CVM_rotated_y3, delimiter=",")
np.savetxt("CVM_rotated_y4_L15_df6_100.csv", CVM_rotated_y4, delimiter=",")


# ### This is how to generate the asymptotic distribution in the plot:

# In[10]:


np.random.seed(0)
N = 500
par = np.array([5, 2, 4], dtype=float)
L = 5
n_ind = N//L
n=n_ind
l = np.tile(np.linspace(1,5,num=L),n)
# Sigmas = (wishart.rvs(df=20, size=n, scale=np.identity(L)))
Sigmas = np.tile(np.identity(L),(n,1,1))
Sigma = block_diag(*Sigmas)
Sig_inv = np.linalg.inv(Sigma)
Sig_inv_sqrt = sqrtm(Sig_inv)
X = np.repeat(np.linspace(0,1,num=n),L)
r1 = np.repeat(1/np.sqrt(N),N)

r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]/np.sqrt(1-1/N**2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)

B = 10000
KS100, CVM100 = [], []
#AD = []
for b in range(B):
    e = np.random.normal(0, scale=1, size=N)
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS100.append(max(abs(np.cumsum(1/np.sqrt(n_ind)*(ehat_lim_sum)))))
    CVM100.append(sum((np.cumsum(1/np.sqrt(n_ind)*(ehat_lim_sum)))**2))
    print(b)


# In[11]:


np.savetxt("KS_N500_L15_boot_asym.csv", KS100, delimiter=",")
np.savetxt("CVM_N500_L15_boot_asym.csv", CVM100, delimiter=",")

