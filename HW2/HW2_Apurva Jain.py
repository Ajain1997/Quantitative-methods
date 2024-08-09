# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:06:52 2024

@author: rkshj
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as pdr
import scipy.stats as sct

ffdata= pd.read_pickle("AK_ffdata.pkl")
rets = pd.read_pickle("AK_rets.pkl")
macro = pdr.DataReader(['GDPC1','CPIAUCSL'],'fred',start='1930-1')

macro.loc[:, 'GDPC1'] = macro.loc[:,'GDPC1'].ffill(limit=2)
macro.loc[:, ['RealGDP', 'Inflation']] = macro.pct_change(12).values
macro = macro[['RealGDP', 'Inflation']]
macro = macro.to_period('M')

print("Question 1")
print()
data = macro.join([rets, ffdata])
data = data.dropna()

regs_results = pd.DataFrame(index = rets.columns, columns = ['Intercept','Real GDP coefficient','Inflation coefficient'])

regs_unc = {}
for i in rets.columns:
    regs_unc[i] = sm.OLS(
        endog=data[i]-data['RF'],
        exog=sm.add_constant(data[macro.columns]),
        missing='drop'
        ).fit(cov_type='HC0')
    intercept = f"{regs_unc[i].params['const'] :0.4f}"
    Real_GDP = f"{regs_unc[i].params['RealGDP'] :0.4f}"
    Inflation = f"{regs_unc[i].params['Inflation'] :0.4f}"
    if regs_unc[i].pvalues[0] <0.05:
        intercept = intercept + '**'
    if regs_unc[i].pvalues[1] < 0.05:
        Real_GDP = Real_GDP + '**'
    if regs_unc[i].pvalues[2] < 0.05:
        Inflation = Inflation + '**'
    regs_results.loc[i,'Intercept'] = intercept
    regs_results.loc[i,'Real GDP coefficient'] = Real_GDP
    regs_results.loc[i,'Inflation coefficient'] = Inflation

print(" The below table shows the coefficients of Intercept, Real GDP and Inflation. The values which are significant at 5% level of significance are denoted by '**'.\n",regs_results)
print()
print("Question 2: Inflation does not explain the covariation of industry returns")
print()
regs_con = {}
for i in rets.columns:
    regs_con[i] = sm.OLS( endog=data[i]-data['RF'],exog=sm.add_constant(data['RealGDP']),missing='drop').fit(cov_type='HC0')

resd_unc = pd.DataFrame(index = data.index, columns=rets.columns,data=np.nan)

for i in rets.columns:
    resd_unc.loc[:,i]= regs_unc[i].resid

sigma_unc = (1/resd_unc.shape[0]) * (resd_unc.T @ resd_unc)

resd_con = pd.DataFrame(index = data.index, columns = rets.columns, data = np.nan)

for i in rets.columns:
    resd_con.loc[:,i]= regs_con[i].resid

sigma_con = (1/resd_con.shape[0]) * (resd_con.T @ resd_con)

lrstat = data.shape[0]*((np.linalg.slogdet(sigma_con)[1]) - (np.linalg.slogdet(sigma_unc)[1]))
pval = 1 - sct.chi2.cdf(lrstat,df = rets.shape[1])

print("The pvalue of the LR statistics is %0.3f and it is less than 5%% significant levels thus we can reject the null hypothesis that inflation does not explain the covariation of industry returns." %pval)
print()
print("Question 3")
print()
B_old = pd.DataFrame(index=rets.columns,columns=macro.columns,data=np.nan)

for i in regs_unc.keys():
    B_old.loc[i, :] = regs_unc[i].params[macro.columns].values

Sigma_old = pd.DataFrame(index=rets.columns,columns=rets.columns,data= ((resd_unc.T @ resd_unc)/resd_unc.shape[0]).values)

Sigma_unc = Sigma_old.copy()

gamma_old = pd.Series(index = macro.columns,data=0.)

small_tolerance = 1e-10
maxiteration = 1e4
iteration=0
paramdiff = 1
r_bar = (data[rets.columns]-data[['RF']].values).mean()
R = (data[rets.columns]-data[['RF']].values)
f_bar = data[macro.columns].mean()
def myinv(Sig): 
    Sigout = pd.DataFrame(index=Sig.index, columns=Sig.columns,data = np.linalg.inv(Sig))
    return Sigout

while ((paramdiff > small_tolerance) & (iteration < maxiteration)):
    gamma_new = (myinv( B_old.T @ myinv(Sigma_old) @ B_old)@ (B_old.T @ myinv(Sigma_old) @ (r_bar - B_old@f_bar)))
    F_tilde = data[macro.columns]+gamma_new
    B_new = (R.T @ F_tilde) @ myinv(F_tilde.T@F_tilde)
    Sigma_new = (R - F_tilde@B_new.T).T @ (R - F_tilde@B_new.T) / R.shape[0]
    
    paramdiff = np.max(((gamma_new-gamma_old).abs().max(),(B_new-B_old).abs().max().max(),(Sigma_new-Sigma_old).abs().max().max()))
   
    gamma_old = gamma_new
    B_old = B_new
    Sigma_old = Sigma_new
    
    iteration+=1
    
LR_of_APT = (np.linalg.slogdet(Sigma_new)[1] - np.linalg.slogdet(Sigma_unc)[1]) * data.shape[0]
pval_of_LR_of_APT = 1 - sct.chi2.cdf(LR_of_APT,df = 28)

price_of_inflation_growth_risk = gamma_new[1] + np.mean(data['Inflation'])
price_of_GDP_growth_risk = gamma_new[0] + np.mean(data['RealGDP'])

print(" The pvalue of the LR statitics is %0.4f which is insignificant at 5%% level of significance and implies that we fail to reject the null hypothesis stating that Alpha is Zero."%(pval_of_LR_of_APT))
print()
print("The price of Inflation and GDP growth risk are %0.4f and %0.4f" %(price_of_inflation_growth_risk,price_of_GDP_growth_risk))
print()
print("The pvalue of LR statistics is %0.4f which is insignificant at 5%% level of significance and therefore we fail to reject the null hypothesis meaning the APT hypothesis holds. The p value shows that the expected returns are mainly affected by the systematic factors i.e. GDP and Inflation and not by idiosyncratic returns. The expected returns are determined as the coefficient of systematic factors i.e. GDP and Inflation multiplied by their price risk " %(pval_of_LR_of_APT))
print()
print("Question 4")
print()
print("The constraint model null hypothesis as done in Question 2 states that Inflation does not explain the covariation of industry returns, however the pvalue of LR stat suggests to reject the null hypothesis and therefore we should use the unconstrained model as done in Question 3.In Question 3 we are testing the APT hypothesis and based on the p value we fail to reject the null hypotheis implying that the APT hypothesis holds true. Therefore, we should not change the model in question 3 as the expected returns can be determined by the systematic fators and does not have implied idiosyncratic returns.")

print("Bonus Question")
print()
macro_1 = pdr.DataReader(['VIXCLS'],'fred',start='1930-1')

macro_1 = macro_1.resample('M').mean()
macro_1 = macro_1.to_period('M')

data_2 = macro_1.join([data])
data_2 = data_2.dropna()

regs_non_tradeable = {}
regs_non_tradeable_results = pd.DataFrame(index = rets.columns, columns = ['Intercept','Real GDP coefficient','Inflation coefficient','VIX coefficient','Market Excess Return'])
for i in rets.columns:
        regs_non_tradeable[i] = sm.OLS(endog=data_2[i]-data_2['RF'],exog=sm.add_constant(data_2[['RealGDP','Inflation','VIXCLS','Mkt-RF']]),missing='drop').fit(cov_type='HC0')

res_nontradeable_unc = pd.DataFrame(index= data_2.index,columns=rets.columns,data=np.nan)

for i in rets.columns:
    res_nontradeable_unc.loc[:,i]= regs_non_tradeable[i].resid

sigma_non_tradeable_unc = (1/res_nontradeable_unc.shape[0]) * (res_nontradeable_unc.T @ res_nontradeable_unc)

B_nontradeable_old = pd.DataFrame(index=rets.columns,columns=['RealGDP','VIXCLS','Inflation','Mkt-RF'],data=np.nan)

for i in regs_non_tradeable.keys():
    B_nontradeable_old.loc[i, :] = regs_non_tradeable[i].params[['RealGDP','VIXCLS','Inflation','Mkt-RF']].values

sigma_nontradeable_old = pd.DataFrame(index=rets.columns,columns=rets.columns,data= ((res_nontradeable_unc.T @ res_nontradeable_unc)/res_nontradeable_unc.shape[0]).values)

sigma_nontradeable_unc = sigma_nontradeable_old.copy()

gamma_nontradeable_old = pd.Series(index =['RealGDP','VIXCLS','Inflation','Mkt-RF'] ,data=0.)

small_tolerance_nontradeable = 1e-10
maxiteration_nontradeable = 1e4
iteration_nontradeable=0
paramdiff_nontradeable = 1
r_bar_new = (data_2[rets.columns]-data_2[['RF']].values).mean()
R_new = (data_2[rets.columns]-data_2[['RF']].values)
f_bar_new = data_2[['RealGDP','VIXCLS','Inflation','Mkt-RF']].mean()
def myinv(Sig_new): 
    Sigout_new = pd.DataFrame(index=Sig_new.index, columns=Sig_new.columns,data = np.linalg.inv(Sig_new))
    return Sigout_new

while ((paramdiff_nontradeable > small_tolerance_nontradeable) & (iteration_nontradeable < maxiteration_nontradeable)):
    gamma_nontradeable_new = (myinv( B_nontradeable_old.T @ myinv(sigma_nontradeable_old) @ B_nontradeable_old)@ (B_nontradeable_old.T @ myinv(sigma_nontradeable_old) @ (r_bar_new - B_nontradeable_old@f_bar_new)))
    F_tilde_new = data_2[['RealGDP','VIXCLS','Inflation','Mkt-RF']]+gamma_nontradeable_new
    B_nontradeable_new = (R_new.T @ F_tilde_new) @ myinv(F_tilde_new.T@F_tilde_new)
    Sigma_nontradeable_new = (R_new - F_tilde_new@B_nontradeable_new.T).T @ (R_new - F_tilde_new@B_nontradeable_new.T) / R_new.shape[0]
    
    paramdiff_nontradeable = np.max(((gamma_nontradeable_new-gamma_nontradeable_old).abs().max(),(B_nontradeable_new-B_nontradeable_old).abs().max().max(),(Sigma_nontradeable_new-sigma_nontradeable_old).abs().max().max()))
   
    gamma_nontradeable_old = gamma_nontradeable_new.copy()
    B_nontradeable_old = B_nontradeable_new.copy()
    Sigma_nontradeable_old = Sigma_nontradeable_new.copy()
    
    iteration_nontradeable+=1
    
LR_of_APT_nontradeable = (np.linalg.slogdet(Sigma_nontradeable_new)[1] - np.linalg.slogdet(sigma_nontradeable_unc)[1]) * data_2.shape[0]
pval_of_LR_of_APT_nontradeable = 1 - sct.chi2.cdf(LR_of_APT_nontradeable,df = 28)

        
regs_tradeable_unc ={}
regs_tradeable_unc_results = pd.DataFrame(index = rets.columns, columns = ['Intercept','SMB coefficient','HML coefficient','Market Excess Return'])
for i in rets.columns:
        regs_tradeable_unc[i] = sm.OLS(endog=data_2[i]-data_2['RF'],exog=sm.add_constant(data_2[['SMB','HML','Mkt-RF']]),missing='drop').fit(cov_type='HC0')
    
resd_tradeable_unc = pd.DataFrame(index = data_2.index, columns=rets.columns,data=np.nan)

for i in rets.columns:
    resd_tradeable_unc.loc[:,i]= regs_tradeable_unc[i].resid

sigma_tradeable_unc = (1/resd_tradeable_unc.shape[0]) * (resd_tradeable_unc.T @ resd_tradeable_unc)
    

regs_tradeable_con = {}
for i in rets.columns:
    regs_tradeable_con[i] = sm.OLS(endog=data_2[i]-data_2['RF'],exog=data_2[['SMB','HML','Mkt-RF']],missing='drop').fit(cov_type='HC0')

resd_tradeable_con = pd.DataFrame(index = data_2.index, columns = rets.columns, data = np.nan)

for i in rets.columns:
    resd_tradeable_con.loc[:,i]= regs_tradeable_con[i].resid

sigma_tradeable_con = (1/resd_tradeable_con.shape[0]) * (resd_tradeable_con.T @ resd_tradeable_con)

lrstat_of_APT_tradeable = data_2.shape[0]*((np.linalg.slogdet(sigma_tradeable_con)[1]) - (np.linalg.slogdet(sigma_tradeable_unc)[1]))
pval_of_lrstat_of_tradeable = 1 - sct.chi2.cdf(lrstat_of_APT_tradeable,df = rets.shape[1])

print("The pvalue of LR statistic for non tradeable model, including VIX and market excess return is %0.4f which is insignificant at 5%% level of significance.Therefore, we fail to reject the null hypothesis and APT hypothesis holds true.The pvalue of fama french 3 factor tradeeable model is %0.4f which is significant at 5%% level of significance therefore we reject the null hypothesis thus stating that the APT hypothesis does not hold true. Between the two models, we should use non tradeable model which includes the VIX and market excess return as the APT hypothesis is true and it suggests that the expected returns are a function of coefficient of the systematic factors and their price risk andthere is no idiosyncratic returns." %(pval_of_LR_of_APT_nontradeable,pval_of_lrstat_of_tradeable))

        











