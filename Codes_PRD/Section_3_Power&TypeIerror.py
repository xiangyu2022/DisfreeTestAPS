#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:00:27 2023

@author: zxysmacbook
"""

import numpy as np 
from scipy.optimize import minimize
from scipy.linalg import sqrtm,block_diag
from autograd import jacobian
from scipy.stats import multivariate_normal
# from scipy.stats import multivariate_t
from scipy.stats import wishart

np.random.seed(1)
# Data generation using the simple linear regression. This will be replaced with the true (X,y) when applied to physical model. 
N = 960
par = np.array([4, -3, 0.4], dtype=float)
L = 8
n = N//L
l = np.tile(np.linspace(1,8,num=8),n)
X = np.repeat(np.linspace(30,170,num=8),n)

Sigmas = (wishart.rvs(df=10, size=n, scale=np.identity(L)))
Sigma = block_diag(*Sigmas)
Sig_inv = np.linalg.inv(Sigma)
Sig_inv_sqrt = sqrtm(Sig_inv)

r1 = np.repeat(1/np.sqrt(N),N)
r2 = [np.sqrt(12/N)*(n/N-(N+1)/(2*N)) for n in (range(1,N+1))]/np.sqrt(1-1/N**2)
r2 = r2/np.linalg.norm(r2)
r3 = r2**2 - np.inner(r1,r2**2)*r1 - np.inner(r2,r2**2)*r2
r3 = r3/np.linalg.norm(r3)

def C_M1(pars):
    return (pars[0] + pars[1] * l + pars[2] * l**2) * X**(2/3)

y_nonflat = np.zeros([n, L])
for i in range(n):
    y_nonflat[i:] = (multivariate_normal.rvs(mean=C_M1(par)[L*i:L*i+L],size=1,cov=Sigmas[i]))
y = y_nonflat.flatten()

# Postulated function, can be changed to any physical models
def postulated_function(pars):
    return ((pars[0] + pars[1] * l + pars[2]* l**2) * X**(2/3) )


# We solve the estimators by minimizing the generalized least squares. 
def optim_func(pars):
    diff = np.matrix(y - postulated_function(pars))
    return diff @ Sig_inv @ diff.T
res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead', options={'xatol': 1e-6, 'disp': True, 'maxiter':200})


# Obtain the residuals, M_theta and R_n. 
residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
 #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
def M_theta_func(theta):
    theta = np.array(theta)
    return Sig_inv_sqrt @ jacobian(postulated_function)(theta)

M_theta = M_theta_func(res.x)

R_n = M_theta.T @ M_theta
R_n_invsq = sqrtm(np.linalg.inv(R_n))
mu_theta = M_theta @ R_n_invsq 


# Creating the r_1 and r_2， can be generalized to r_p.
r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)

# Obtain the transformation, ehat. Up here
U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
# ehat_sum = ehat
# ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)

# Obtain the Komogorov-Smirnov statistic
ks = max(abs(np.cumsum(ehat/np.sqrt(N))))
cvm = sum((np.cumsum(ehat/np.sqrt(N)))**2)/N
# Bootstrap with transformation
B = 100000
KS = []
CVM = []
#AD = []
for b in range(B):
    e = np.random.normal(0, scale=1, size=N)
    ehat_lim = e - np.inner(r1,e)*r1 -  np.inner(r2,e)*r2 -  np.inner(r3,e)*r3
    # ehat_lim_sum =  ehat_lim.reshape(n_ind,-1).sum(axis=1)
    KS.append(max(abs(np.cumsum(ehat_lim/np.sqrt(N)))))
    CVM.append(sum((np.cumsum(ehat_lim/np.sqrt(N))**2))/N)
    print(b)

print((sum(KS>=ks)+1)/(B+1))
print((sum(CVM>=cvm)+1)/(B+1))




pval_ks_y1=[]; pval_cvm_y1=[]
for j in range(10000):
    y_nonflat = np.zeros([n, L])
    for i in range(n):
        y_nonflat[i:] = (multivariate_normal.rvs(mean=C_M1(par)[L*i:L*i+L],size=1,cov=Sigmas[i]))
    y = y_nonflat.flatten()
    
    # We solve the estimators by minimizing the generalized least squares. 
    def optim_func(pars):
        diff = np.matrix(y - postulated_function(pars))
        return diff @ Sig_inv @ diff.T
    res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead', options={'xatol': 1e-6, 'disp': True, 'maxiter':500})
    
    
    # Obtain the residuals, M_theta and R_n. 
    residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
     #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
    def M_theta_func(theta):
        theta = np.array(theta)
        return Sig_inv_sqrt @ jacobian(postulated_function)(theta)
    
    M_theta = M_theta_func(res.x)
    
    R_n = M_theta.T @ M_theta
    R_n_invsq = sqrtm(np.linalg.inv(R_n))
    mu_theta = M_theta @ R_n_invsq 
    
    r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
  
    # Obtain the transformation, ehat. Up here
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)
    
    # Obtain the Komogorov-Smirnov statistic
    ks_y1 = max(abs(np.cumsum(ehat/np.sqrt(N))))
    cvm_y1 = sum((np.cumsum(ehat/np.sqrt(N)))**2)/N
    pval_ks_y1.append((sum(KS>=ks_y1)+1)/(len(KS)+1))
    pval_cvm_y1.append((sum(CVM>=cvm_y1)+1)/(len(CVM)+1))
    # Bootstrap with transformation
    print(j)

print(np.mean(np.array(pval_ks_y1)<0.1))
print(np.mean(np.array(pval_cvm_y1)<0.1))
print(np.mean(np.array(pval_ks_y1)<0.05))
print(np.mean(np.array(pval_cvm_y1)<0.05))
print(np.mean(np.array(pval_ks_y1)<0.01))
print(np.mean(np.array(pval_cvm_y1)<0.01))
print(np.mean(np.array(pval_ks_y1)<0.001))
print(np.mean(np.array(pval_cvm_y1)<0.001))




pval_ks_y2=[]; pval_cvm_y2=[]
for j in range(10000):
    y_nonflat = np.zeros([n, L])
    
    for i in range(n):
        y_nonflat[i:] = (multivariate_normal.rvs(mean=((4 - 3 * np.exp(l * 0.23)) *X**(2/3) )[L*i:L*(i+1)],size=1,
                                            cov=Sigmas[i]))
    y = y_nonflat.flatten()
    
    # We solve the estimators by minimizing the generalized least squares. 
    def optim_func(pars):
        diff = np.matrix(y - postulated_function(pars))
        return diff @ Sig_inv @ diff.T
    res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead', options={'xatol': 1e-6, 'disp': True, 'maxiter':500})
    
    
    # Obtain the residuals, M_theta and R_n. 
    residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
     #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
    def M_theta_func(theta):
        theta = np.array(theta)
        return Sig_inv_sqrt @ jacobian(postulated_function)(theta)
    
    M_theta = M_theta_func(res.x)
    
    R_n = M_theta.T @ M_theta
    R_n_invsq = sqrtm(np.linalg.inv(R_n))
    mu_theta = M_theta @ R_n_invsq 
    
    r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)    
    U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    
    # Obtain the transformation, ehat. Up here
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)
    # ehat_sum = ehat
    
    # Obtain the Komogorov-Smirnov statistic
    ks_y2 = max(abs(np.cumsum(ehat/np.sqrt(N))))
    cvm_y2 = sum((np.cumsum(ehat/np.sqrt(N)))**2)/N
    pval_ks_y2.append((sum(KS>=ks_y2)+1)/(B+1))
    pval_cvm_y2.append((sum(CVM>=cvm_y2)+1)/(B+1))
    # Bootstrap with transformation
    print(j)


print(np.mean(np.array(pval_ks_y2)<0.1))
print(np.mean(np.array(pval_cvm_y2)<0.1))
print(np.mean(np.array(pval_ks_y2)<0.05))
print(np.mean(np.array(pval_cvm_y2)<0.05))
print(np.mean(np.array(pval_ks_y2)<0.01))
print(np.mean(np.array(pval_cvm_y2)<0.01))
print(np.mean(np.array(pval_ks_y2)<0.001))
print(np.mean(np.array(pval_cvm_y2)<0.001))


pval_ks_y3=[]; pval_cvm_y3=[]
for j in range(10000):
    y_nonflat = np.zeros([n, L])
    
    for i in range(n):
        y_nonflat[i:] = (multivariate_normal.rvs(mean=((4 - 3 * (l ** 2.07)) *X**(2/3) )[L*i:L*(i+1)],size=1,cov=Sigmas[i]))
    y = y_nonflat.flatten()
    
    # We solve the estimators by minimizing the generalized least squares. 
    def optim_func(pars):
        diff = np.matrix(y - postulated_function(pars))
        return diff @ Sig_inv @ diff.T
    res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead', options={'xatol': 1e-6, 'disp': True, 'maxiter':500})
    
    
    # Obtain the residuals, M_theta and R_n. 
    residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
     #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
    def M_theta_func(theta):
        theta = np.array(theta)
        return Sig_inv_sqrt @ jacobian(postulated_function)(theta)
    
    M_theta = M_theta_func(res.x)
    
    R_n = M_theta.T @ M_theta
    R_n_invsq = sqrtm(np.linalg.inv(R_n))
    mu_theta = M_theta @ R_n_invsq 
    
    
    # Creating the r_1 and r_2， can be generalized to r_p.
    r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    
    # Obtain the transformation, ehat. Up here
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)
    # ehat_sum = ehat
    ks_y3 = max(abs(np.cumsum(ehat/np.sqrt(N))))
    cvm_y3 = sum((np.cumsum(ehat/np.sqrt(N)))**2)/N
    pval_ks_y3.append((sum(KS>=ks_y3)+1)/(B+1))
    pval_cvm_y3.append((sum(CVM>=cvm_y3)+1)/(B+1))
    # Bootstrap with transformation
    print(j)

print(np.mean(np.array(pval_ks_y3)<0.1))
print(np.mean(np.array(pval_cvm_y3)<0.1))
print(np.mean(np.array(pval_ks_y3)<0.05))
print(np.mean(np.array(pval_cvm_y3)<0.05))
print(np.mean(np.array(pval_ks_y3)<0.01))
print(np.mean(np.array(pval_cvm_y3)<0.01))
print(np.mean(np.array(pval_ks_y3)<0.001))
print(np.mean(np.array(pval_cvm_y3)<0.001))
     


pval_ks_y4=[]; pval_cvm_y4=[]
for j in range(10000):
    y_nonflat = np.zeros([n, L])
    
    for i in range(n):
        y_nonflat[i:] = (multivariate_normal.rvs(mean=((4 - 3 * l + 0.4 * l**2 + 0.02 *l**3) *X**(2/3) )[L*i:L*(i+1)],size=1,cov=Sigmas[i]))
    y = y_nonflat.flatten()
    
    # We solve the estimators by minimizing the generalized least squares. 
    def optim_func(pars):
        diff = np.matrix(y - postulated_function(pars))
        return diff @ Sig_inv @ diff.T
    res = minimize(optim_func, np.repeat(0,len(par)), method='nelder-mead', options={'xatol': 1e-6, 'disp': True, 'maxiter':500})
    
    
    # Obtain the residuals, M_theta and R_n. 
    residuals = Sig_inv_sqrt @ (y - postulated_function(res.x))
     #Sig_inv_sqrt @ np.vstack((np.repeat(1,N),X)).T
    def M_theta_func(theta):
        theta = np.array(theta)
        return Sig_inv_sqrt @ jacobian(postulated_function)(theta)
    
    M_theta = M_theta_func(res.x)
    
    R_n = M_theta.T @ M_theta
    R_n_invsq = sqrtm(np.linalg.inv(R_n))
    mu_theta = M_theta @ R_n_invsq 
    
    
    # Creating the r_1 and r_2， can be generalized to r_p.
    r2_tilde = r2 - np.inner(mu_theta[:,0]-r1, r2)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    U1r3 = r3 - np.inner(mu_theta[:,0]-r1, r3)/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    r3_tilde = U1r3 - np.inner(mu_theta[:,1]-r2_tilde, U1r3)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    
    # Obtain the transformation, ehat. Up here
    U_mu3_r3_res = residuals - np.inner(mu_theta[:,2]-r3_tilde, residuals)/(1-np.inner(mu_theta[:,2],r3_tilde))*(mu_theta[:,2]-r3_tilde)
    U_mu2_r2_res = U_mu3_r3_res - np.inner(mu_theta[:,1]-r2_tilde, U_mu3_r3_res)/(1-np.inner(mu_theta[:,1],r2_tilde))*(mu_theta[:,1]-r2_tilde)
    ehat = U_mu2_r2_res - np.inner(mu_theta[:,0]-r1, U_mu2_r2_res )/(1-np.inner(mu_theta[:,0],r1))*(mu_theta[:,0]-r1)
    # ehat_sum = ehat.reshape(n_ind,-1).sum(axis=1)
    # ehat_sum = ehat
    ks_y4 = max(abs(np.cumsum(ehat/np.sqrt(N))))
    cvm_y4 = sum((np.cumsum(ehat/np.sqrt(N)))**2)/N
    pval_ks_y4.append((sum(KS>=ks_y4)+1)/(B+1))
    pval_cvm_y4.append((sum(CVM>=cvm_y4)+1)/(B+1))
    # Bootstrap with transformation
    print(j)

print(np.mean(np.array(pval_ks_y4)<0.1))
print(np.mean(np.array(pval_cvm_y4)<0.1))
print(np.mean(np.array(pval_ks_y4)<0.05))
print(np.mean(np.array(pval_cvm_y4)<0.05))
print(np.mean(np.array(pval_ks_y4)<0.01))
print(np.mean(np.array(pval_cvm_y4)<0.01))
print(np.mean(np.array(pval_ks_y4)<0.001))
print(np.mean(np.array(pval_cvm_y4)<0.001))