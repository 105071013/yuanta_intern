#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dask.dataframe as dd
import time
import pandas as pd
import numpy as np
from scipy.stats import mstats
from progressbar import *
import matplotlib.pyplot as plt
import os 
import gc



# In[2]:


if not os.path.exists('./Stats_Detail_New/'):
    os.makedirs('./Stats_Detail_New/')
if not os.path.exists('./Weight/'):
    os.makedirs('./Weight/')
if not os.path.exists('./Graph_new/'):
    os.makedirs('./Graph_new/')
if not os.path.exists('./stats/'):
    os.makedirs('./stats/')


# In[3]:


def select_top10_ind(data):
    data_MV = data[data.MV_Select == 1]

    select_ind=data_MV.groupby('Ind')['Code'].count().sort_values(ascending=False) 
    select_ind = select_ind[select_ind>4]
    select_ind = select_ind.index[0:10].tolist()

    return [ ((x in select_ind)&( y in data_MV['Code'].tolist()))*1 for x,y in zip(data['Ind'],data['Code']) ]





# # 二、計算損益統計數據


def performance_stats(equity,E_RF ):#equity 權益list
    equity = pd.DataFrame(equity)
    #年化收益
    CAGR  = ((equity.iloc[-1]/equity.iloc[0]-1)*(250/len(equity)))[0]
    #年化標準差
    STD_Y = (equity.pct_change().std()*np.sqrt(250))[0]
    #夏普值
    Sharp_ratio = (CAGR-E_RF)/STD_Y
    #最大回撤MDD 回撤率mdd
    equity = pd.DataFrame(equity)
    Cum_R=equity.values/(equity.head(1).values)
    Cum_R=pd.DataFrame(Cum_R)
    D = Cum_R.cummax() - Cum_R
    MDD = D.max()*initial_account
    d = D / (D +Cum_R)
    mdd = d.max()
    #NP/MDD
    NP = pd.DataFrame(equity.iloc[-1]-equity.iloc[0])
    NP_MDD = NP/MDD

    #勝率
    R = equity.pct_change()
    Odd = R[R>0].count()/len(equity)
    #偏態
    skew = equity.pct_change().skew()
    return CAGR,STD_Y,Sharp_ratio,mdd,NP_MDD,Odd,skew

Factor_args = {}

#主要策略
Factor_args['weight_method']     = ['eq','tri','eq_ind','tri_ind','tri_ind_cap']
#考慮做多、放空指數
Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_short_index' for x in Factor_args['weight_method'] ]+[ x+'_long_index' for x in Factor_args['weight_method'] ]
#值加權
Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_value' for x in Factor_args['weight_method'] ]


factor_all  = ['main_N_13','for_N_13','invest_N_13']
long_big_all = [1,1,1]
for k in range(len(factor_all)):
    factor = factor_all[k]
    long_big = long_big_all[k]
    initial_account = 1000000000
    # In[ ]:
    Factors_list = []
    LongBig_list = []
    CAGR_list = []
    STD_Y_list = []
    Sharp_ratio_list = []
    mdd_list = []
    NP_MDD_list = []
    Odd_list = []
    skew_list = []
    Tornover_list = []
    method_list = Factor_args['weight_method']
    for i in range(len(method_list)):
        method = method_list[i]

        temp = pd.read_csv('./Stats_Detail_New/multiway/factor/'+factor+'_Stats_Detail_New_'+method+'.txt')

        x = np.arange(len(temp))
        plt.plot(temp['Equity_'+method],label='Equity_'+method)

        date = temp['Date']

        plt.xticks(x[::300],labels=date[::300],rotation=45)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(factor)

        
        df_RM=pd.read_excel('Data/3.Rm_Rf/2007-201909Rm.xlsx')
        df_RM.columns = df_RM.iloc[3,]
        df_RM=df_RM.drop([0,1,2,3])
        df_RM=df_RM.dropna(axis=1).reset_index(drop=True)
        df_RM.columns = ['Date','大盤指數']
        df_RM['Date'] = [datetime.datetime.strftime(x,format='%Y%m%d') for x in df_RM['Date'] ]
        df_RM=df_RM.sort_values('Date').drop_duplicates()
        df_RM['Rm'] = df_RM['大盤指數'].pct_change() 



        df_RF=pd.read_excel('Data/3.Rm_Rf/2009-201909Rf.xlsx')
        df_RF.columns = df_RF.iloc[3,]
        df_RF=df_RF.drop([0,1,2,3])
        df_RF=df_RF.dropna(axis=1).reset_index(drop=True)
        df_RF.columns = ['Date','Rf']
        df_RF['Date'] = [datetime.datetime.strftime(x,format='%Y%m%d') for x in df_RF['Date'] ]
        df_RF=df_RF.sort_values('Date').drop_duplicates()
        E_RF = df_RF['Rf'].mean()/100
        

        equity = pd.DataFrame(temp['Equity_'+method])
        CAGR,STD_Y,Sharp_ratio,mdd,NP_MDD,Odd,skew = performance_stats(equity,E_RF=E_RF)
        CAGR_list.append(CAGR)
        STD_Y_list.append(STD_Y)
        Sharp_ratio_list.append(Sharp_ratio)
        mdd_list.append(mdd[0])
        NP_MDD_list.append(NP_MDD[0][0])
        Odd_list.append(Odd[0])
        skew_list.append(skew[0])
        Factors_list.append( factor)
        LongBig_list.append( long_big)
        Tornover_list.append(np.mean([x for x in temp['Turnover_'+method] if np.isnan(x)==False]))
    plt.savefig('./Graph_new/'+factor+'.jpg', bbox_inches='tight')
    plt.show()
    
    df_factor_stat = pd.DataFrame()
    df_factor_stat['Factors']       = Factors_list
    df_factor_stat['LongBig']       = LongBig_list
    df_factor_stat['Weight_Method']       = method_list                        
    df_factor_stat['Return_annual'] = CAGR_list
    df_factor_stat['Std_annual'] = STD_Y_list
    df_factor_stat['Sharp']         = Sharp_ratio_list
    df_factor_stat['mdd']        = mdd_list
    df_factor_stat['NP_MDD']     = NP_MDD_list
    df_factor_stat['Odds']          = Odd_list
    df_factor_stat['Turnover']          = Tornover_list
    df_factor_stat['Skew']          = skew_list
    df_factor_stat.round(4)
    df_factor_stat.to_csv('./Stats_New/'+factor+".csv",index=False)
    df_factor_stat
    print(factor + 'OK')

# In[ ]:




