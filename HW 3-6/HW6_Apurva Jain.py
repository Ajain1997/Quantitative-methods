# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:35:02 2024

@author: rkshj
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
ff3 = pd.read_csv("FF3.csv")
rets = pd.read_csv("Industry_Portfolios.csv")
macro = pd.read_csv("macro.csv")

ff3.index = pd.to_datetime(ff3['Date'])
ff3 = (ff3.drop(['Date'],axis=1)).to_period("M")

rets.index = pd.to_datetime(rets['Date'])
rets = (rets.drop(['Date'],axis=1)).to_period("M")

macro.index = pd.to_datetime(macro['sasdate'])
macro = (macro.drop(['sasdate'],axis=1)).to_period("M")

data = macro.join((ff3,rets), how = 'inner')
std_data = (data[macro.columns]-data[macro.columns].mean())/data[macro.columns].std(ddof=0)

in_sample_period = std_data.loc[:'1999-12'].index

out_of_sample_period = std_data.loc['2000-1':'2019-12'].index

rets_data = pd.DataFrame(index = data.index, columns = rets.columns, data = np.nan)
for i in rets.columns:
    rets_data[i] = data[i] - data['RF'] 

def opt_ptf(mu, sigmainv, gamma=10):
    w_risky = (1/gamma) * sigmainv @ mu
    return w_risky

print("Direct Approach")

in_sample_elastic = {}
for i in rets.columns:
    in_sample_elastic[i] = skl.ElasticNet(alpha = 0.005).fit(std_data.shift(1).loc[in_sample_period].iloc[1:,:], data.loc[in_sample_period, i].iloc[1:])

rets_predict = {}
for i in rets.columns:
    rets_predict[i] = in_sample_elastic[i].predict(std_data.shift(1).loc[out_of_sample_period])       

mu_direct = pd.DataFrame(index = out_of_sample_period, columns = rets.columns, data = np.nan)
for i in rets.columns:
    mu_direct[i] = rets_predict[i]

resid_in_sample = pd.DataFrame(index = in_sample_period, columns = rets.columns, data = np.nan)
for i in rets.columns:
    resid_in_sample[i] = in_sample_elastic[i].predict(std_data.loc[in_sample_period])

resid_direct = data.loc[in_sample_period,rets.columns] - resid_in_sample
sigma_direct = (1/resid_direct.shape[0])*(resid_direct.T @ resid_direct)
sigmainv_direct = pd.DataFrame(index = sigma_direct.index , columns = sigma_direct.columns, data = np.linalg.inv(sigma_direct))

ind_wt_direct = mu_direct.apply(lambda x: opt_ptf( mu=x,sigmainv = sigmainv_direct), axis=1)
ind_ptf_ret_direct = (ind_wt_direct * data.loc[out_of_sample_period,rets.columns]).sum(axis=1) # need to change to in-same-Data

risk_free_weight_direct = 1 - ind_wt_direct.sum(axis = 1)
risk_free_return_direct = (risk_free_weight_direct * data['RF'].loc[out_of_sample_period])

total_ptf_wt_direct = ind_wt_direct.sum(axis = 1) + risk_free_weight_direct
total_ptf_ret_direct = ind_ptf_ret_direct + risk_free_return_direct

excess_portfolio_return_direct = total_ptf_ret_direct - data['RF'].loc[out_of_sample_period]
print("Factor Model")

ind_regs = sm.OLS(endog = rets_data.loc[in_sample_period], exog = data.loc[in_sample_period,['Mkt-RF','SMB','HML']],missing = 'drop').fit()
beta_coef_df = pd.DataFrame(index = ['Mkt-RF','SMB','HML'],columns = rets.columns, data = ind_regs.params.values)
sigma_df = pd.DataFrame(index = in_sample_period, columns = rets.columns, data = ind_regs.resid)

sigma_e = 1/sigma_df.shape[0] * (sigma_df.T @ sigma_df)
sigma_e_diag = pd.DataFrame(index = sigma_e.index, columns = sigma_e.columns, data = np.diag(np.diag(sigma_e)))

in_sample_factor_forecast = {}
for i in ['Mkt-RF','SMB','HML']:
    in_sample_factor_forecast[i] = skl.ElasticNet(alpha = 0.005).fit(std_data.shift(1).loc[in_sample_period].iloc[1:,:], data.loc[in_sample_period, i].iloc[1:])

in_sample_factor = pd.DataFrame(index = in_sample_period, columns =['Mkt-RF','SMB','HML'], data = np.nan )
for i in ['Mkt-RF','SMB','HML']:
    in_sample_factor[i] = in_sample_factor_forecast[i].predict(std_data.loc[in_sample_period])

out_of_sample_factor_forecast = {}
for i in ['Mkt-RF','SMB','HML']:
    out_of_sample_factor_forecast[i] = in_sample_factor_forecast[i].predict(std_data.shift(1).loc[out_of_sample_period])

out_of_sample_factor = pd.DataFrame(index = out_of_sample_period, columns =['Mkt-RF','SMB','HML'], data = np.nan )
for i in ['Mkt-RF','SMB','HML']:
    out_of_sample_factor[i] = out_of_sample_factor_forecast[i]

mu_fac_model = out_of_sample_factor @ beta_coef_df

resid_fac = data[['Mkt-RF','SMB','HML']].shift(1).loc[in_sample_period] - in_sample_factor
resid_fac = resid_fac.dropna()

sigma_fac = 1/resid_fac.shape[0] * (resid_fac.T@resid_fac)
sigma_fac = pd.DataFrame(index = sigma_fac.index, columns = sigma_fac.columns,data = sigma_fac)

sigma_fac_model = sigma_e + beta_coef_df.T @ sigma_fac @ beta_coef_df
sigmainv_fac_model = pd.DataFrame(index = sigma_fac_model.index, columns = sigma_fac_model.columns,data = np.linalg.inv(sigma_fac_model))

ind_wt_fac = mu_fac_model.apply(lambda x: opt_ptf( mu=x,sigmainv = sigmainv_fac_model), axis=1)
ind_ptf_ret_fac = (ind_wt_fac * data.loc[out_of_sample_period,rets.columns]).sum(axis=1)

risk_free_weight_fac = 1 - ind_wt_fac.sum(axis = 1)
risk_free_return_fac = (risk_free_weight_fac * data['RF'].loc[out_of_sample_period])

total_ptf_wt_fac = ind_wt_fac.sum(axis = 1) + risk_free_weight_fac
total_ptf_ret_fac = ind_ptf_ret_fac + risk_free_return_fac

excess_portfolio_return_fac = total_ptf_ret_fac - data['RF'].loc[out_of_sample_period]
print()
print("Question 1")
min_ptf_ret_direct = excess_portfolio_return_direct.min() * 100
min_ptf_ret_fac = excess_portfolio_return_fac.min() * 100
print()
print("The minimum portfolio return for the direct approach is %0.4f%%" %(min_ptf_ret_direct))
print("The minimum portfolio return for the factor model approach is %0.4f%%" %(min_ptf_ret_fac))
print()
print("Question 2")

mean_ptf_ret_direct = excess_portfolio_return_direct.mean() * 100 * 12
mean_ptf_ret_fac = excess_portfolio_return_fac.mean() * 100 * 12
print()
print("The annualised mean portfolio return for the direct approach is %0.4f%%" %(mean_ptf_ret_direct))
print("The annualised mean portfolio return for the factor model approach is %0.4f%%" %(mean_ptf_ret_fac))
print()
print("Question 3")
max_ptf_ret_direct = excess_portfolio_return_direct.max() * 100
max_ptf_ret_fac = excess_portfolio_return_fac.max() * 100
print()
print("The maximum portfolio return for the direct approach is %0.4f%%" %(max_ptf_ret_direct))
print("The maximum portfolio return for the factor model approach is %0.4f%%" %(max_ptf_ret_fac))
print()
print("Question 4")
excess_market_return_mean_direct = excess_portfolio_return_direct.mean()
excess_market_return_std_direct = np.std(excess_portfolio_return_direct)
sharpe_direct = excess_market_return_mean_direct/excess_market_return_std_direct * np.sqrt(12)


excess_market_return_mean_fac = excess_portfolio_return_fac.mean()
excess_market_return_std_fac = np.std(excess_portfolio_return_fac)
sharpe_fac = excess_market_return_mean_fac/excess_market_return_std_fac * np.sqrt(12)

print("The sharpe ratio for the direct approach is %0.4f" %(sharpe_direct))
print("The sharpe ratio for the factor model approach is %0.4f" %(sharpe_fac))
print()
print("Question 5")
skew_direct = sct.skew(excess_portfolio_return_direct)
skew_fac = sct.skew(excess_portfolio_return_fac)
print("The skewness for the direct approach is %0.4f" %(skew_direct))
print("The skewness for the factor model approach is %0.4f" %(skew_fac))
print()
print("Question 6")

ind_names = ind_wt_direct.columns
ind_wt_direct[ind_names] = ind_wt_direct[ind_names].apply(lambda x: pd.to_numeric(x, errors='coerce'))
minCol_direct = ind_wt_direct[ind_names].idxmin(axis = 1)
min_return_direct = pd.Series(excess_portfolio_return_direct).idxmin()
min_ind_wt_direct = minCol_direct[minCol_direct.index == min_return_direct]
min_wt_ind_direct = ind_wt_direct.loc[min_return_direct,min_ind_wt_direct].values

ind_wt_fac[ind_names] = ind_wt_fac[ind_names].apply(lambda x: pd.to_numeric(x, errors='coerce'))
minCol_fac = ind_wt_fac[ind_names].idxmin(axis = 1)
min_return_fac = pd.Series(excess_portfolio_return_fac).idxmin()
min_ind_wt_fac = minCol_fac[minCol_fac.index == min_return_fac]
min_wt_ind_fac = ind_wt_fac.loc[min_return_fac,min_ind_wt_fac].values


print()
print("The minimum industry weight for direct approach is %0.4f for" %(min_wt_ind_direct),min_ind_wt_direct[0])
print("The minimum industry weight for factor model approach is %0.4f for" %(min_wt_ind_fac),min_ind_wt_fac[0])
print()
print("Question 7")
ind_names = ind_wt_direct.columns
ind_wt_direct[ind_names] = ind_wt_direct[ind_names].apply(lambda x: pd.to_numeric(x, errors='coerce'))
maxCol_direct = ind_wt_direct[ind_names].idxmax(axis = 1)
max_return_direct = pd.Series(excess_portfolio_return_direct).idxmax()
max_ind_wt_direct = maxCol_direct[maxCol_direct.index == max_return_direct]
max_wt_ind_direct = ind_wt_direct.loc[max_return_direct,max_ind_wt_direct].values

ind_wt_fac[ind_names] = ind_wt_fac[ind_names].apply(lambda x: pd.to_numeric(x, errors='coerce'))
maxCol_fac = ind_wt_fac[ind_names].idxmax(axis = 1)
max_return_fac = pd.Series(excess_portfolio_return_fac).idxmax()
max_ind_wt_fac = maxCol_fac[maxCol_fac.index == max_return_fac]
max_wt_ind_fac = ind_wt_fac.loc[max_return_fac,max_ind_wt_fac].values

print()
print("The maximum industry weight for direct approach is %0.4f for " %(max_wt_ind_direct),max_ind_wt_direct[0])
print("The maximum industry weight for factor model approach is %0.4f" %(max_wt_ind_fac),max_ind_wt_fac[0])
print()
print("Question 8") 
plt.figure(figsize = (10,8))
excess_portfolio_return_direct.plot(label = 'Direct Excess Returns')
excess_portfolio_return_fac.plot(label = 'Factor Model Excess Returns')
plt.legend()
print("When thinking about how to predict investment returns, it's important to think about the costs of buying and selling investments. One way is to directly look at how much money you expect to make from each investment and how much of your total investment is in each one. This method usually means you don't have to buy and sell investments as often, so you don't spend as much on fees. But, it might not take full advantage of spreading out your investments to reduce risk or picking up on patterns in different types of investments. Another way is to use models that look at bigger trends in the market, like how certain types of investments tend to move together. These models might need more buying and selling, which can cost more in fees. But, they can also help protect your investments during tough times, like the 2008 financial crisis. When deciding which method to use, consider your tolerance for risk and your investment objectives.")