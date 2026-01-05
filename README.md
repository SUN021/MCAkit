# MCAkit
Code for “MCA: High-Dimensional Modal Component Analysis towards the Mode”

## Main Files

MM.m---estimate factor model with modal component analysis(MCA) method and minorization-maximization algorithm

MCA_FN.m---estimate the number of MCA factors

AM_Gaussian.m---estimate a factor model using MCA method with the Alternating Maximization (AM) algorithm and a Gaussian kernel

AM_higerorder.m---estimate a factor model using MCA method with the AM algorithm and a higher-order kernel

VARF.m---estimate the asymtotic variance of MCA factors

VARL.m---estimate the asymtotic variance of MCA factoe-loading

IQR.m---estimate factor model with quantile factor analysis(QFA) method 

QFA_FN.m---estimate the number of QFA factors

PCA.m---estimate factor model with princile component analysis(MCA) method 

PCA_FN.m---estimate the number of PCA factors

HPCA.m---estimate factor model with huber princle component analysis(HPCA) method

HPCA_FN.m---estimate the number of HPCA factors

AR_forecast_new.m---perform autoregressive regression

AR_DI_forecast_new.m---perform factor-augumented autoregressive regression

## Other Files

EM.m---perform EM iteration in modal regression

mdreg.m---perform modal regression

rq_fnm.m---perform quantile regression

myrlm.m---perform robust linear regression with Huber loss

Robust_sigma.m---compute robust standard deviation

SC.m---Stopping criterion

Psi_Type.m---compute weight for Huber regression

kernel.m---caluculate the value of Gaussian kernel

kernel1.m---caluculate the first derivative of Gaussian kernel

kernel2.m---caluculate the second derivative of Gaussian kernel

kernel_gaussian.m---calculate the value of high-order Gaussian kernel

tratio.m---calculate the trace-ratio statistic of estimated matrix

normalize.m---normalize the factor and loading

FLregularize.m---regularize the factor and loading

mixture_normal.m---generate random variables from the mixtural-normal distribution

skewt_azzalini.m---generate random variables from the Azzalini skew-t distribution




