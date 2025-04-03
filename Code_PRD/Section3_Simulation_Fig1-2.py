#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load necessary packages
import pandas as pd
import autograd.numpy as np
import random
import time
import matplotlib.pyplot as plt
from autograd import jacobian
from scipy.optimize import minimize
from scipy.linalg import sqrtm, block_diag
from scipy.stats import multivariate_normal, multivariate_t, wishart


# In[3]:


def C_M1(pars):
    return (pars[0] + pars[1] * l + pars[2] * l**2) * X**(2/3)

def C_M2(pars):
    return np.exp(pars[0] + pars[1] * l  + pars[2] * l**2 )*X**(2/3)

def optim_func(y, C_M, pars):
    diff = np.matrix(y - C_M(pars))
    return float(diff @ Sig_inv @ diff.T)

def gls(y, C_M):
    res_C = minimize(lambda pars: optim_func(y, C_M, pars), par, method='Nelder-Mead')
    return res_C.x


# In[4]:


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
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # new_ehat = ehat.reshape(round(N/L),L).sum(axis=1)/np.sqrt(N)
    return ehat

def Test_stat(res, stat):
    ## sum_res = res.reshape(round(N/L),L).sum(axis=1)/np.sqrt(N)
    if stat == "KS":
        return max(abs(np.cumsum(res/np.sqrt(N))))
    elif stat == "CVM":
        return sum((np.cumsum(res/np.sqrt(N)))**2)/N
    else:
        error("stat should be specified as either 'KS', or 'CVM.'")


# In[201]:


np.random.seed(1)
N = 960
par = np.array([4, -3, 0.4], dtype=float)
L = 8
n = N//L
l = np.tile(np.linspace(1,8,num=8),n)
X = np.repeat(np.linspace(30,170,num=8),n)
Sigmas = (wishart.rvs(df=10, size=N//L, scale=np.identity(L)))
Sigma = block_diag(*Sigmas)
Sig_inv = np.linalg.inv(Sigma)
Sig_sqrt = sqrtm(Sigma)
Sig_inv_sqrt = sqrtm(Sig_inv)
r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]/np.sqrt(1-1/N**2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)


# In[202]:


B = 10000
KS_res_y1, KS_res_y2, KS_res_y3, KS_res_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
KS_rotated_y1, KS_rotated_y2, KS_rotated_y3, KS_rotated_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
CVM_res_y1, CVM_res_y2, CVM_res_y3, CVM_res_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
CVM_rotated_y1, CVM_rotated_y2, CVM_rotated_y3, CVM_rotated_y4 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
y3_nonflat, y4_nonflat = np.zeros([N//L,L]), np.zeros([N//L,L])


# In[206]:


start_time = time.time()
for j in range(0,B,1):
    y1, y2 = multivariate_normal.rvs(mean=C_M1(par),size=1,cov=Sigma), multivariate_normal.rvs(mean=C_M2(par),size=1,cov=Sigma)
    y3 = Sig_sqrt @ np.random.laplace(loc=0, scale=1/np.sqrt(2), size=N) + C_M1(par)
    y4 = Sig_sqrt @ np.random.laplace(loc=0, scale=1/np.sqrt(2), size=N) + C_M2(par)

    res1, res2 = sphered_residuals_func(y1, C_M1), sphered_residuals_func(y2, C_M2)
    res3, res4 = sphered_residuals_func(y3, C_M1), sphered_residuals_func(y4, C_M2)
    rotated_res1, rotated_res2 = rotated_residuals_func(y1, C_M1), rotated_residuals_func(y2, C_M2)
    rotated_res3, rotated_res4 = rotated_residuals_func(y3, C_M1), rotated_residuals_func(y4, C_M2)

    KS_res_y1[j],  KS_res_y2[j],  KS_res_y3[j],  KS_res_y4[j]  = Test_stat(res1,"KS"),  Test_stat(res2,"KS"),   Test_stat(res3,"KS"),  Test_stat(res4,"KS")    
    CVM_res_y1[j], CVM_res_y2[j], CVM_res_y3[j], CVM_res_y4[j] = Test_stat(res1,"CVM"), Test_stat(res2,"CVM"),  Test_stat(res3,"CVM"), Test_stat(res4,"CVM")    
    KS_rotated_y1[j], KS_rotated_y2[j], KS_rotated_y3[j], KS_rotated_y4[j]     = Test_stat(rotated_res1,"KS"),  Test_stat(rotated_res2,"KS"),  Test_stat(rotated_res3,"KS"),  Test_stat(rotated_res4,"KS") 
    CVM_rotated_y1[j], CVM_rotated_y2[j], CVM_rotated_y3[j], CVM_rotated_y4[j] = Test_stat(rotated_res1,"CVM"), Test_stat(rotated_res2,"CVM"), Test_stat(rotated_res3,"CVM"), Test_stat(rotated_res4,"CVM")
    print(j)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# In[207]:


elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# In[208]:


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
plt.scatter(np.quantile(a = KS_rotated_y3, q = probs, method='closest_observation'),np.quantile(a = KS_rotated_y4, q = probs, method='closest_observation'))
abline(1,0)
plt.show()

plt.scatter(np.quantile(a = CVM_res_y1, q = probs, method='closest_observation'),np.quantile(a = CVM_res_y2, q = probs, method='closest_observation'))
abline(1,0)
plt.show()
plt.scatter(np.quantile(a = CVM_rotated_y1, q = probs, method='closest_observation'),np.quantile(a = CVM_rotated_y2, q = probs, method='closest_observation'))
abline(1,0)
plt.show()

plt.scatter(np.quantile(a = CVM_res_y3, q = probs, method='closest_observation'),np.quantile(a = CVM_res_y4, q = probs, method='closest_observation'))
abline(1,0)
plt.show()
plt.scatter(np.quantile(a = CVM_rotated_y3, q = probs, method='closest_observation'),np.quantile(a = CVM_rotated_y4, q = probs, method='closest_observation'))
abline(1,0)
plt.show()


# In[1]:


N = 960
L = 8
n = N//L
l = np.tile(np.linspace(1,8,num=8),n)
X = np.repeat(np.linspace(30,170,num=8),n)
B = 10000
KS_asy, CVM_asy = [], []
start_time2 = time.time()
for b in range(B):
    e = np.random.normal(0, scale=1, size=N)
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    #ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS_asy.append(max(abs(np.cumsum(ehat_lim/np.sqrt(N)))))
    CVM_asy.append(sum((np.cumsum(ehat_lim/np.sqrt(N)))**2)/N)
    print(b)
end_time2 = time.time()


# In[210]:


elapsed_time = end_time2 - start_time2
print(f"Elapsed time: {elapsed_time} seconds")


# In[211]:


np.savetxt("KS_res_y1_nosum_960_new_L.csv", KS_res_y1, delimiter=",")
np.savetxt("KS_res_y2_nosum_960_new_L.csv", KS_res_y2, delimiter=",")
np.savetxt("KS_res_y3_nosum_960_new_L.csv", KS_res_y3, delimiter=",")
np.savetxt("KS_res_y4_nosum_960_new_L.csv", KS_res_y4, delimiter=",")
np.savetxt("CVM_res_y1_nosum_960_new_L.csv", CVM_res_y1, delimiter=",")
np.savetxt("CVM_res_y2_nosum_960_new_L.csv", CVM_res_y2, delimiter=",")
np.savetxt("CVM_res_y3_nosum_960_new_L.csv", CVM_res_y3, delimiter=",")
np.savetxt("CVM_res_y4_nosum_960_new_L.csv", CVM_res_y4, delimiter=",")


np.savetxt("KS_rotated_y1_nosum_960_new_L.csv", KS_rotated_y1, delimiter=",")
np.savetxt("KS_rotated_y2_nosum_960_new_L.csv", KS_rotated_y2, delimiter=",")
np.savetxt("KS_rotated_y3_nosum_960_new_L.csv", KS_rotated_y3, delimiter=",")
np.savetxt("KS_rotated_y4_nosum_960_new_L.csv", KS_rotated_y4, delimiter=",")
np.savetxt("CVM_rotated_y1_nosum_960_new_L.csv", CVM_rotated_y1, delimiter=",")
np.savetxt("CVM_rotated_y2_nosum_960_new_L.csv", CVM_rotated_y2, delimiter=",")
np.savetxt("CVM_rotated_y3_nosum_960_new_L.csv", CVM_rotated_y3, delimiter=",")
np.savetxt("CVM_rotated_y4_nosum_960_new_L.csv", CVM_rotated_y4, delimiter=",")


# In[212]:


np.savetxt("KS_boot_asym1_new_L.csv", KS_asy, delimiter=",")
np.savetxt("CVM_boot_asym1_new_L.csv", CVM_asy, delimiter=",")

