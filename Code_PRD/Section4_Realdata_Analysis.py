#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:27:04 2024

@author: zxysmacbook
"""


import numpy as np 
from scipy.linalg import sqrtm

y = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Null/A_hat_v7.txt")
post_func = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Null/A_theta_hat_3_parameters_v7.txt")
Sig = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Null/Sigma_v7.txt")
Jacob = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Null/jacobian_3_parameters_v7.txt")
Sig_inv_sqrt = np.linalg.inv(sqrtm(Sig))

np.random.seed(1234)
N, n_ind = 960, 120

residuals = Sig_inv_sqrt @ (y - post_func)
M_theta =  Sig_inv_sqrt @ Jacob
mu_theta = M_theta @ np.linalg.inv(sqrtm(M_theta.T @ M_theta))

# Creating the r_1 and r_2， can be generalized to r_p.
r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]
r2 = r2/np.linalg.norm(r2)
r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)
U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/( 1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)

# Obtain the transformation, ehat. Up here
U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)

# Obtain the Komogorov-Smirnov statistic
ks = max(abs(np.cumsum(1/np.sqrt(n_ind)*ehat_sum)))
cvm = sum((np.cumsum(1/np.sqrt(n_ind)*ehat_sum))**2)
# Bootstrap with transformation
B = 100000
KS = []
CVM = []
for b in range(B):
    e = np.random.normal(0, scale=1, size=N) 
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS.append(max(abs(np.cumsum(1/np.sqrt(n_ind)*ehat_lim_sum))))
    CVM.append(sum((np.cumsum(1/np.sqrt(n_ind)*ehat_lim_sum))**2))
    print(b)
print((sum(KS>=ks)+1)/(B+1))
print((sum(CVM>=cvm)+1)/(B+1))


np.random.seed(1234)
y = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Alternative/A_hat_linear_v8.txt")
post_func = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Alternative/A_theta_hat_3_parameters_linear_v8.txt")
Sig = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Alternative/Sigma_linear_v8.txt")
Jacob = np.loadtxt("/Users/zxysmacbook/Downloads/Sec4_Alternative/jacobian_3_parameters_linear_v8.txt")
Sig_inv_sqrt = np.linalg.inv(sqrtm(Sig))

residuals = Sig_inv_sqrt @ (y - post_func)
M_theta =  Sig_inv_sqrt @ Jacob
mu_theta = M_theta @ np.linalg.inv(sqrtm(M_theta.T @ M_theta))

# Creating the r_1 and r_2， can be generalized to r_p.
r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]
r2 = r2/np.linalg.norm(r2)
r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)
U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/( 1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)

# Obtain the transformation, ehat. Up here
U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)

# Obtain the Komogorov-Smirnov statistic
ks = max(abs(np.cumsum(1/np.sqrt(n_ind)*ehat_sum)))
cvm = sum((np.cumsum(1/np.sqrt(n_ind)*ehat_sum))**2)
# Bootstrap with transformation
B = 100000
KS = []
CVM = []
for b in range(B):
    e = np.random.normal(0, scale=1, size=N) 
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS.append(max(abs(np.cumsum(1/np.sqrt(n_ind)*ehat_lim_sum))))
    CVM.append(sum((np.cumsum(1/np.sqrt(n_ind)*ehat_lim_sum))**2))
    print(b)
print((sum(KS>=ks)+1)/(B+1))
print((sum(CVM>=cvm)+1)/(B+1))




