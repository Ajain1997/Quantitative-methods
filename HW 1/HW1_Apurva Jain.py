# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:40:14 2024

@author: rkshj
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
ff3 = pd.read_csv("ff3.csv")
ind = pd.read_csv("Industry_Portfolios.csv")

ff3.index = pd.to_datetime(ff3['Date'])
ff3 = (ff3.drop(['Date'],axis=1)).to_period("M")

ind.index = pd.to_datetime(ind['Date'])
ind = (ind.drop(['Date'],axis=1)).to_period("M")

regs = {}
for i in ind.columns:
    regs[i] = sm.OLS(endog=(ind[i]-ff3['RF']),exog=sm.add_constant(ff3[['Mkt-RF','SMB','HML']])).fit(cov_type='HC0')
print("Question 1")
print(" In their groundbreaking 1993 paper, Fama and French introduced two supplementary factors, Small Minus Big (SMB) \
and High Minus Low (HML), as an extension of the Capital Asset Pricing Model (CAPM). SMB and HML are derived from the\
returns of six distinct portfolios created by the intersection of two key characteristics: size (market capitalization) \
and book-to-market equity (B/M) ratio. The portfolios based on size include Small (S), Medium (M), and Big (B), while those\
based on B/M ratio consist of High (H), Medium (M), and Low (L).")
print("Question 2")
print("The alpha and betas for Food are\n", regs['Food '].params)
print("The alpha and betas for Retail are\n", regs['Rtail'].params)
print("The alpha and betas for Books are\n", regs['Books'].params)
print("The alpha and betas for Telecom are\n", regs['Telcm'].params)
print("The alpha and betas for Automobile are\n", regs['Autos'].params)

print("Question 3")
round(regs['Food '].pvalues,3)
round(regs['Rtail'].pvalues,3)
round(regs['Books'].pvalues,3)
round(regs['Telcm'].pvalues,3)
round(regs['Autos'].pvalues,3)

print("The values which are significant at 5% levels for food are alpha, MKT-RF and SMB.")
print("The values which are significant at 5% levels for Retail are alpha, Mkt-RF and HML")
print("The values which are significant at 5% levels for Books are alpha, Mkt-RF, SMB and HML")
print("The values which are significant at 5% levels for Telcm are Mkt-RF and SMB")
print("The values which are significant at 5% levels for Autos are Mkt-RF and HML")

print("Question 4")
print("Alpha represents the intercept term in the regression equation,\
A significant alpha indicates that the asset's return cannot be entirely explained \
by the selected factors. In other words, the asset has some unique or idiosyncratic \
risk or return not captured by the chosen factors.\
Beta coefficients serve as indicators of an asset's responsiveness or exposure to changes in the returns of corresponding factors.\
When the beta is both significant and positive, especially if it exceeds 1, it implies that the asset exhibits higher volatility \
compared to the market. On the other hand, a significant and positive beta within the range of 0 to 1 suggests that the asset is \
less volatile than the market. In cases where the beta is significant and negative, the asset tends to move counter to the direction\
of the factor")

print("Question 5a: Industry returns are not exposed to Market returns")

res = pd.DataFrame(index= ff3.index,columns=ind.columns,data=np.nan)

for i in ind.columns:
    res.loc[:,i]= regs[i].resid
sigma_unc = (1/res.shape[0]) * (res.T @ res)

regsc = {}
for i in ind.columns:
    regsc[i] = sm.OLS(endog=(ind[i]-ff3['RF']),exog=sm.add_constant(ff3[['SMB','HML']])).fit(cov_type='HC0')
    
resc = pd.DataFrame(index= ff3.index,columns=ind.columns,data=np.nan)

for i in ind.columns:
    resc.loc[:,i]= regsc[i].resid
sigma_c_Mkt = (1/resc.shape[0]) * (resc.T @ resc)

lrstat_Mkt = ff3.shape[0]*((np.linalg.slogdet(sigma_c_Mkt)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_Mkt = round(1 - sct.chi2.cdf(lrstat_Mkt,df = ind.shape[1]),3)

if pval_Con_Mkt < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that Mkt-RF does not explain covariation of industry return"%pval_Con_Mkt)
else:
    print("Fail to reject the null hypothesis")

print("Question 5b: Industry returns are not exposed to SMB")
regsc_SMB = {}
for i in ind.columns:
    regsc_SMB[i] = sm.OLS(endog=(ind[i]-ff3['RF']),exog=sm.add_constant(ff3[['Mkt-RF','HML']])).fit(cov_type='HC0')
    
resc_SMB = pd.DataFrame(index= ff3.index,columns=ind.columns,data=np.nan)

for i in ind.columns:
    resc_SMB.loc[:,i]= regsc_SMB[i].resid
sigma_c_SMB = (1/resc_SMB.shape[0]) * (resc_SMB.T @ resc_SMB)

lrstat_SMB = ff3.shape[0]*((np.linalg.slogdet(sigma_c_SMB)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_SMB = round(1 - sct.chi2.cdf(lrstat_SMB,df = ind.shape[1]),3)

if pval_Con_SMB < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that SMB does not explain covariation of industry return"%pval_Con_SMB)
else:
    print("Fail to reject the null hypothesis")

print("Question 5c: Industry returns are not exposed to HML")
regsc_HML = {}
for i in ind.columns:
    regsc_HML[i] = sm.OLS(endog=(ind[i]-ff3['RF']),exog=sm.add_constant(ff3[['Mkt-RF','SMB']])).fit(cov_type='HC0')
    
resc_HML = pd.DataFrame(index= ff3.index,columns=ind.columns,data=np.nan)

for i in ind.columns:
    resc_HML.loc[:,i]= regsc_HML[i].resid
sigma_c_HML = (1/resc_HML.shape[0]) * (resc_HML.T @ resc_HML)

lrstat_HML = ff3.shape[0]*((np.linalg.slogdet(sigma_c_HML)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_HML = round(1 - sct.chi2.cdf(lrstat_HML,df = ind.shape[1]),3)

if pval_Con_HML < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that HML does not explain covariation of industry return"%pval_Con_HML)
else:
    print("Fail to reject the null hypothesis")

print("Question 6")
regsc_APT = {}
for i in ind.columns:
    regsc_APT[i] = sm.OLS(endog=(ind[i]-ff3['RF']),exog=(ff3[['Mkt-RF','SMB','HML']])).fit(cov_type='HC0')
    
resc_APT = pd.DataFrame(index= ff3.index,columns=ind.columns,data=np.nan)

for i in ind.columns:
    resc_APT.loc[:,i]= regsc_APT[i].resid
sigma_c_APT = (1/resc_APT.shape[0]) * (resc_APT.T @ resc_APT)

lrstat_APT = ff3.shape[0]*((np.linalg.slogdet(sigma_c_APT)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_APT = round(1 - sct.chi2.cdf(lrstat_APT,df = ind.shape[1]),3)

if pval_Con_APT < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that APT does not hold true."%pval_Con_APT)
else:
    print("Fail to reject the null hypothesis")

print("Question 7")
#Finite-sample adjusted LR statistic.
t = 1115
n = 30
k = 3

finite_sample_adjustment = t - n/2 - k - 1
# LR statistic for industry returns not exposed to market returns
lrstat_Mkt_finite = finite_sample_adjustment*((np.linalg.slogdet(sigma_c_Mkt)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_Mkt_finite = round(1 - sct.chi2.cdf(lrstat_Mkt_finite,df = ind.shape[1]),3)

# LR statistic for industry returns not exposed to SMB
lrstat_SMB_finite = finite_sample_adjustment*((np.linalg.slogdet(sigma_c_SMB)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_SMB_finite = round(1 - sct.chi2.cdf(lrstat_SMB_finite,df = ind.shape[1]),3)

# LR statistic for industry returns not exposed to HML
lrstat_HML_finite = finite_sample_adjustment*((np.linalg.slogdet(sigma_c_HML)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_HML_finite = round(1 - sct.chi2.cdf(lrstat_HML_finite,df = ind.shape[1]),3)

# LR statistic for APT hypothesis
lrstat_APT_finite = finite_sample_adjustment*((np.linalg.slogdet(sigma_c_APT)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval_Con_APT_finite = round(1 - sct.chi2.cdf(lrstat_APT_finite,df = ind.shape[1]),3)

print("The finite-sample adjustment changed all the 4 lr stat. However,the pvalue of \
these LR stat is significant at 5% levels and hence we can reject the null hypothesis.")

print("Question 8")

ff3_new = ff3[ff3.index >='2000-01']
ind_new = ind[ind.index >='2000-01']
#unconstrained regression
regs_new = {}
for i in ind_new.columns:
    regs_new[i] = sm.OLS(endog=(ind_new[i]-ff3_new['RF']),exog=sm.add_constant(ff3_new[['Mkt-RF','SMB','HML']])).fit(cov_type='HC0')

res_new = pd.DataFrame(index= ff3_new.index,columns=ind_new.columns,data=np.nan)

for i in ind_new.columns:
    res_new.loc[:,i]= regs_new[i].resid
sigma_unc_new = (1/res_new.shape[0]) * (res_new.T @ res_new)

# Question 5a: industry returns not exposed to market returns
regsc_new = {}
for i in ind_new.columns:
    regsc_new[i] = sm.OLS(endog=(ind_new[i]-ff3_new['RF']),exog=sm.add_constant(ff3_new[['SMB','HML']])).fit(cov_type='HC0')
    
resc_new = pd.DataFrame(index= ff3_new.index,columns=ind_new.columns,data=np.nan)

for i in ind_new.columns:
    resc_new.loc[:,i]= regsc_new[i].resid
sigma_c_Mkt_new = (1/resc_new.shape[0]) * (resc_new.T @ resc_new)

lrstat_Mkt_new = ff3_new.shape[0]*((np.linalg.slogdet(sigma_c_Mkt_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_Mkt_new = round(1 - sct.chi2.cdf(lrstat_Mkt_new,df = ind_new.shape[1]),3)

if pval_Con_Mkt_new < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that Mkt-RF does not explain covariation of industry return"%pval_Con_Mkt_new)
else:
    print("Fail to reject the null hypothesis")

#Question 5b:Industry returns are not exposed to SMB
regsc_SMB_new = {}
for i in ind_new.columns:
    regsc_SMB_new[i] = sm.OLS(endog=(ind_new[i]-ff3_new['RF']),exog=sm.add_constant(ff3_new[['Mkt-RF','HML']])).fit(cov_type='HC0')
    
resc_SMB_new = pd.DataFrame(index= ff3_new.index,columns=ind_new.columns,data=np.nan)

for i in ind_new.columns:
    resc_SMB_new.loc[:,i]= regsc_SMB_new[i].resid
sigma_c_SMB_new = (1/resc_SMB_new.shape[0]) * (resc_SMB_new.T @ resc_SMB_new)

lrstat_SMB_new = ff3_new.shape[0]*((np.linalg.slogdet(sigma_c_SMB_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_SMB_new = round(1 - sct.chi2.cdf(lrstat_SMB_new,df = ind_new.shape[1]),3)

if pval_Con_SMB_new < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that SMB does not explain covariation of industry return"%pval_Con_SMB_new)
else:
    print("Fail to reject the null hypothesis")

#Question 5c: Industry returns are not exposed to HML
regsc_HML_new = {}
for i in ind_new.columns:
    regsc_HML_new[i] = sm.OLS(endog=(ind_new[i]-ff3_new['RF']),exog=sm.add_constant(ff3_new[['Mkt-RF','SMB']])).fit(cov_type='HC0')
    
resc_HML_new = pd.DataFrame(index= ff3_new.index,columns=ind_new.columns,data=np.nan)

for i in ind_new.columns:
    resc_HML_new.loc[:,i]= regsc_HML_new[i].resid
sigma_c_HML_new = (1/resc_HML_new.shape[0]) * (resc_HML_new.T @ resc_HML_new)

lrstat_HML_new = ff3_new.shape[0]*((np.linalg.slogdet(sigma_c_HML_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_HML_new = round(1 - sct.chi2.cdf(lrstat_HML_new,df = ind_new.shape[1]),3)

if pval_Con_HML_new < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that HML does not explain covariation of industry return"%pval_Con_HML_new)
else:
    print("Fail to reject the null hypothesis")

#question 6: APT Hypothesis
regsc_APT_new = {}
for i in ind_new.columns:
    regsc_APT_new[i] = sm.OLS(endog=(ind_new[i]-ff3_new['RF']),exog=(ff3_new[['Mkt-RF','SMB','HML']])).fit(cov_type='HC0')
    
resc_APT_new = pd.DataFrame(index= ff3_new.index,columns=ind_new.columns,data=np.nan)

for i in ind_new.columns:
    resc_APT_new.loc[:,i]= regsc_APT_new[i].resid
sigma_c_APT_new = (1/resc_APT_new.shape[0]) * (resc_APT_new.T @ resc_APT_new)

lrstat_APT_new = ff3_new.shape[0]*((np.linalg.slogdet(sigma_c_APT_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_APT_new = round(1 - sct.chi2.cdf(lrstat_APT_new,df = ind_new.shape[1]),3)

if pval_Con_APT_new < 0.05:
    print("The Pval is %f hence we reject the null hypothesis that APT does not hold true."%pval_Con_APT_new)
else:
    print("Fail to reject the null hypothesis")

#Question 7: Finite-sample adjusted LR statistic.
t = 275
n = 30
k = 3

finite_sample_adjustment_new = t - n/2 - k - 1
# LR statistic for industry returns not exposed to market returns
lrstat_Mkt_finite_new = finite_sample_adjustment_new*((np.linalg.slogdet(sigma_c_Mkt_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_Mkt_finite_new = round(1 - sct.chi2.cdf(lrstat_Mkt_finite_new,df = ind_new.shape[1]),3)

# LR statistic for industry returns not exposed to SMB
lrstat_SMB_finite_new = finite_sample_adjustment_new*((np.linalg.slogdet(sigma_c_SMB_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_SMB_finite_new = round(1 - sct.chi2.cdf(lrstat_SMB_finite_new,df = ind_new.shape[1]),3)

# LR statistic for industry returns not exposed to HML
lrstat_HML_finite_new = finite_sample_adjustment_new*((np.linalg.slogdet(sigma_c_HML_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_HML_finite_new = round(1 - sct.chi2.cdf(lrstat_HML_finite_new,df = ind_new.shape[1]),3)

# LR statistic for APT hypothesis
lrstat_APT_finite_new = finite_sample_adjustment_new*((np.linalg.slogdet(sigma_c_APT_new)[1]) - (np.linalg.slogdet(sigma_unc_new)[1]))
pval_Con_APT_finite_new = round(1 - sct.chi2.cdf(lrstat_APT_finite_new,df = ind_new.shape[1]),3)

print("The finite-sample adjustment changed all the 4 lr stat. However,the pvalue of \
these LR stat is significant at 5% levels and hence we can reject the null hypothesis.")
