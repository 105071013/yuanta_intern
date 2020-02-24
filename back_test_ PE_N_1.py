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


# In[4]:


Factor_args = {}

Factor_args['weight_method']     = ['tri']
#Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_short_index' for x in Factor_args['weight_method'] ]+[ x+'_long_index' for x in Factor_args['weight_method'] ]
tir_list = [ x for x in  Factor_args['weight_method']  if 'tri' in x ]
Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_value' for x in tir_list ]

Main_Factor_Name = 'PE'
Main_Factor_Long = 0
periods = 1

Factor_args['upper_lower_perc_eq'] =  { "upper" : 0.1,"lower" : 0.9 }
Factor_args['upper_lower_perc_tri'] =  { "upper" : 0.5,"lower" : 0.5 }
Factor_args[Main_Factor_Name] = { "factors" : [],
                                  "longbig" : [],
                                  "periods" : [] }

Factor_args[Main_Factor_Name]['factors'] = [Main_Factor_Name ]
Factor_args[Main_Factor_Name]['longbig'] = [Main_Factor_Long]*len(Factor_args[Main_Factor_Name]['factors']) #1是多大的
Factor_args[Main_Factor_Name]['periods'] = [periods]


# In[5]:


reader = pd.read_csv('./Data/output/df_merge_1.txt', chunksize=10 ** 6)
df_merge_final = pd.concat([x for x in reader], ignore_index=True)
df_merge_final.count()


# In[6]:


factor = Factor_args[Main_Factor_Name]['factors'][0]
periods = Factor_args[Main_Factor_Name]['periods'][0]

reader = pd.read_csv('./Data/output/df_merge_2.txt', chunksize=10 ** 6)
df_factor = pd.concat([x for x in reader], ignore_index=True)
df_merge_final        = pd.merge(left=df_merge_final,right=df_factor[['Date','Code',factor]],on=['Date','Code'],how='left')
df_merge_final.count()
del df_factor


# In[7]:


df_merge_final['CAP_t-1'] = df_merge_final.groupby('Code')['CAP'].shift(1)


# 計算T-1Factor,T-2
factor_list = Factor_args[Main_Factor_Name]['factors']
for factor in factor_list :
    df_merge_final[factor+'_t-1']= df_merge_final.groupby('Code')[factor].shift(1)
    df_merge_final[factor+'_t-2']= df_merge_final.groupby('Code')[factor].shift(2)


# shift 收盤價方便計算投組損益
df_merge_final['Close_t+1']= df_merge_final.groupby('Code')['Close'].shift(-1)
df_merge_final['Close_t-1']= df_merge_final.groupby('Code')['Close'].shift(1)
# 調整開始時間
date_begin = np.array(df_merge_final['Date'].unique())[2]
df_merge_final = df_merge_final[df_merge_final['Date']>=date_begin]


df_merge_final['Volumn_a_select'] = (df_merge_final['Volumn_a_mean'] >=50000000)*1


# 剔除Close , factor t-1 t-2 為na的資料   
df_merge_final=df_merge_final[pd.isna(df_merge_final['Close'])==False]
df_merge_final=df_merge_final[pd.isna(df_merge_final['CAP_t-1'])==False]
df_merge_final=df_merge_final[pd.isna(df_merge_final[factor+'_t-1'])==False]


# In[8]:


first_date = np.array(df_merge_final.groupby('Year')['Date'].min())
# 每年市值篩選 貼標籤
df_Year                = df_merge_final[ [x in first_date for x in df_merge_final['Date'] ] ]
df_Year                = df_Year.sort_values(['Code','Year']).reset_index(drop=True)
temp                   = df_Year[df_Year['Volumn_a_select']==1]  
print(temp.groupby('Year')['Code'].count())
temp['MV_t-1_Rank']    = temp.groupby('Year')['CAP_t-1'].rank(ascending=False)
temp['MV_Select']      = (temp['MV_t-1_Rank'] < 201)*1
temp                   = temp[['Code','Year','MV_Select','Volumn_a_select','CAP_t-1','Ind']]
temp_m                  = temp[temp['MV_Select']==1] 
print(temp_m.groupby('Year')['Code'].count())
# 標注每年200檔是值最大的個股，數量前10大產業
temp_m  = temp_m.sort_values(['Year','Ind','Code']).reset_index(drop=True)
Ind_select  = temp_m.groupby('Year').apply(lambda x : select_top10_ind(x) ).tolist()
Ind_select  = [ x for y in Ind_select for x in y ]
temp_m['Ind_Select'] = Ind_select
# 計算10產業市值權重
temp_i= temp_m [temp_m.Ind_Select ==1]
print(temp_i.groupby('Year')['Code'].count())
Ind_Cap_Weight=temp_i.groupby(['Year','Ind'])['CAP_t-1'].sum()/temp_i.groupby(['Year'])['CAP_t-1'].sum()
temp_m["Ind_CAP_Weight"] = [ Ind_Cap_Weight[x][y] if z ==1 else 0 for x,y,z in 
                                 zip(temp_m["Year"],temp_m["Ind"],temp_m["Ind_Select"]) ]

temp.groupby('Year')['Code'].count()
df_merge_Year = pd.merge(left=df_merge_final,right=temp_m[['Year','Code','MV_Select','Ind_Select','Ind_CAP_Weight']]
                         ,on=['Year','Code'],how='left')

del df_merge_final, temp_m
gc.collect()

# # 一、計算權重並輸出

# In[9]:

def Cal_Amount(df,factor,weight_method):
    df[factor+'_Nstock_'+weight_method] =  [ initial_account/2*x/y/1000 for x,y in zip(df[factor+'_Weight_'+weight_method],df['Close'])  ]
    df[factor+'_Nstock_'+weight_method] = df[factor+'_Nstock_'+weight_method].apply(np.ceil)
    df[factor+'_Amount_'+weight_method] = df[factor+'_Nstock_'+weight_method]*df['Close']*1000    
    return df[factor+'_Nstock_'+weight_method],df[factor+'_Amount_'+weight_method]

def Cal_Profit(df,factor,weight_method):
    df[factor+'_Profit_'+weight_method] = ((df['Close_t+1']-df['Close']) * df[factor+'_Nstock_'+weight_method] )*1000

    if   'short_index'in weight_method:
        
        Rm = df['Rm'].mode()
        Profit = df[factor+'_Profit_'+weight_method].sum()-initial_account*Rm/2
    elif  'long_index'in weight_method:  
        
        Rm = df['Rm'].mode()
        Profit = df[factor+'_Profit_'+weight_method].sum()+initial_account*Rm/2
    else:
        Profit = df[factor+'_Profit_'+weight_method].sum()
    return Profit

def Cal_Cost(df,factor,weight_method):

    holdsell_shortmore =df['Close']*(df[factor+'_Nstock_'+weight_method]- df[factor+'_Nstock_'+weight_method+'_t-1'])
    holdsell_shortmore = abs(holdsell_shortmore[holdsell_shortmore<0]) #現在持股絕對值 - 過去持股絕對值 <0 代表賣出股漂
    df[factor+'_Tax_'+weight_method] = holdsell_shortmore
    trade_tax = np.sum(holdsell_shortmore*0.003*1000)
    
    return  trade_tax 

def Cal_Turnover(df,factor,weight_method):

    Turnover =np.sum(  abs( df['Close']*(df[factor+'_Nstock_'+weight_method]- df[factor+'_Nstock_'+weight_method+'_t-1'])
                               )  )*1000

    return Turnover



def cal_3group_weight_tri(df,factor,longbig,short_index=False,long_index=False,value=False):

    length     =    len(df)

    
    if value == True: #因子值加權
        df[factor+'_t-1'] = winsorize(df[factor+'_t-1'])
        df[factor+'_t-1'] = min_max(df[factor+'_t-1'])
        median     =    df[factor+'_t-1'].median()
        
        upper      =    df[df[factor+'_t-1']>=median][factor+'_t-1']
        lower      =    df[df[factor+'_t-1']< median][factor+'_t-1']
        large_Rank      =    upper    
        small_Rank      =    1 - lower
        print(small_Rank)
        print(large_Rank)

    else:             #因子值排序加權
        length     =    len(df)
        middle     =    np.ceil(length /2)
        
        upper      =    df[df[factor+'_Rank']<=middle][factor+'_Rank'] 
        lower      =    df[df[factor+'_Rank']> middle][factor+'_Rank']
        large_Rank =    middle +1 - upper                              #數值最大的逆著排     100 99 98 #最大是一
        small_Rank =    lower     - middle                             #因子數值最小的順著排   1  2  3   #最小是一
   
    if longbig == 1: #買因子大 賣因子小
        if   short_index ==True:#買大因子 且放空指數 故因子小的權重為0
            large_Rank =    large_Rank/large_Rank.sum()  
            small_Rank =    small_Rank/small_Rank.sum() * 0 
        elif long_index ==True:#賣小因子 且做多指數 故因子大的權重為0
            large_Rank =    large_Rank/large_Rank.sum() * 0
            small_Rank =    small_Rank/small_Rank.sum()              
        else:
            large_Rank =    large_Rank/large_Rank.sum()  
            small_Rank =    small_Rank/small_Rank.sum()             
        return pd.concat([large_Rank,-small_Rank])
    else :          #賣因子大 買因子小
        if short_index ==True: #買小因子 且放空指數 故因子大的權重為0
            large_Rank =    large_Rank/large_Rank.sum() * 0 
            small_Rank =    small_Rank/small_Rank.sum() 
        elif long_index ==True:#賣大因子 且做多指數 故因子小的權重為0
            large_Rank =    large_Rank/large_Rank.sum() 
            small_Rank =    small_Rank/small_Rank.sum() * 0 
        else:
            large_Rank =    large_Rank/large_Rank.sum()  
            small_Rank =    small_Rank/small_Rank.sum()  
        return pd.concat([-large_Rank,small_Rank])


# In[10]:


def out_of_group(data,n):
    gt1=pd.qcut(data[factor+'_Rank_t-1'], n, labels=False)
    gt2=pd.qcut(data[factor+'_Rank'], n, labels=False)
    return ( gt1!=gt2 )*1


# In[11]:


def last_trade_date(date_list,df_Date):
    A=np.array(date_list)
    target=np.array(df_Date)
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    left = np.where(df_Date==date_list[0],np.NaN,left)
    right = A[idx]
    return left


# # Tri

# In[12]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        
        df_less1[need_col_less1+[factor+'_Rank_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank','Close_t+1']]
        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)
        df_Final = pd.concat([df_Final,df_MV_m],axis=0)
df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


# # 十組固定

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri_10g'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)
        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        

        df_less1[need_col_less1+[factor+'_Rank_t-1','Close_t+1_less1',factor+'_10g_t-1']] = df_less1[need_col+[factor+'_Rank','Close_t+1',factor+'_10g']]
        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1',factor+'_10g_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)
          
        df_MV_m['out_of_group'] = df_MV_m[factor+'_10g'] != df_MV_m[factor+'_10g_t-1']

        all_invest=df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]]).abs().sum()
        
        if (abs(all_invest - 2*initial_account) <= 2000000000 ):
            df_MV_m[need_col[0]] = df_MV_m[need_col[0]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[0]])
            df_MV_m[need_col[1]] = df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]])
            df_MV_m[need_col[2]] = df_MV_m[need_col[2]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[2]])
            

        else:
            print('Over_the_threshold!!!'+str(Date)+" " + str(abs(all_invest - 2*initial_account)))
               
            pass
        df_MV_m  = df_MV_m[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g',factor+'_10g_t-1']
                         +need_col+need_col_less1]
        df_Final = df_Final[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g' ,factor+'_10g_t-1']
                         +need_col+need_col_less1]
        print(np.sum(abs(df_MV_m[factor+'_Amount_tri_10g'] - df_MV_m[factor+'_Amount_tri_10g_t-1'])) )
        df_Final = pd.concat([df_Final,df_MV_m],axis=0)
df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


# # 產業百分比 - 普通

# In[ ]:


def winsorize(s):
    if  sum(pd.isna(s)==True)==len(s):

        return s
    else:   
        return mstats.winsorize(s, limits=[0.01, 0.01])

def min_max(s):
    if sum(pd.isna(s)==True)==len(s):
        return s
    else:   
        return (s-np.min(s))/   (np.max(s)-np.min(s))
def stardize(s):
    if sum(pd.isna(s)==True)==len(s):
        return s
    else:   
        return (s-np.mean(s))/  np.std(s)  

def min_max_win(df,factor):
    df[factor+'_t-1'] = winsorize(df[factor+'_t-1'])
    df[factor+'_t-1'] = min_max(df[factor+'_t-1'])
    return df[factor+'_t-1']


def standardize_win(df,factor):
    df[factor+'_t-1'] = winsorize(df[factor+'_t-1'])
    df[factor+'_t-1'] = stardize(df[factor+'_t-1'])
    return df[factor+'_t-1']


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri_ind%'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)
        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : min_max_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : min_max_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_less1[need_col_less1+[factor+'_Rank_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank','Close_t+1']]
        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)
        df_Final = pd.concat([df_Final,df_MV_m],axis=0)        

df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


# # 產業百分 - 十組固定

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri_ind%_10g'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : min_max_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : min_max_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)
        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_less1[need_col_less1+[factor+'_Rank_t-1',factor+'_10g_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank',factor+'_10g','Close_t+1']]

        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1',factor+'_10g_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)      
        df_MV_m['out_of_group'] = df_MV_m[factor+'_10g'] != df_MV_m[factor+'_10g_t-1']

        all_invest=df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]]).abs().sum()
        
        if (abs(all_invest - 2*initial_account) <= 2000000000 ):
            df_MV_m[need_col[0]] = df_MV_m[need_col[0]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[0]])
            df_MV_m[need_col[1]] = df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]])
            df_MV_m[need_col[2]] = df_MV_m[need_col[2]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[2]])
            

        else:
            print('Over_the_threshold!!!'+str(Date)+" " + str(abs(all_invest - 2*initial_account)))
               
            pass
        df_MV_m  = df_MV_m[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g',factor+'_10g_t-1']
                         +need_col+need_col_less1]
        df_Final = df_Final[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g' ,factor+'_10g_t-1']
                         +need_col+need_col_less1]
        print(np.sum(abs(df_MV_m[factor+'_Amount_'+weight_method] - df_MV_m[factor+'_Amount_'+weight_method+'_t-1'])) )
        df_Final = pd.concat([df_Final,df_MV_m],axis=0)
df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


# # 產業標準化 - 普通

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri_indstd'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)
        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : standardize_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : standardize_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1

        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_less1[need_col_less1+[factor+'_Rank_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank','Close_t+1']]

        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)
    

        df_Final = pd.concat([df_Final,df_MV_m],axis=0)
df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


# # 產業標準化 - 固定十組

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
weight_method   = 'tri_indstd_10g'
initial_account = 1000000000
df_Final = pd.DataFrame()
need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
need_col          = [factor+x+weight_method for x in need_col ]
need_col_less1    = [x+'_t-1' for x in need_col ]
progress = ProgressBar()
for i in progress(range(len(Date_list))):
    if i ==0:
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : standardize_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)

        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        df_Final = df_MV

    else:
        df_less1 = df_MV
        Date = Date_list[i]
        df_MV = df_merge_Year[df_merge_Year.Date==Date ]
        df_MV = df_MV[df_MV.MV_Select==1 ]
        df_MV = df_MV.sort_values(factor+'_t-1',ascending=False)

        #去極值以及對產業內因子標準化至0~1
        df_MV[factor+'_t-1_%'] = df_MV.groupby('Ind').apply(lambda x : standardize_win(x,factor)).reset_index(level=0,drop=True)
        #有太多1跟0 在用原先factor排序
        df_MV = df_MV.sort_values([factor+'_t-1_%',factor+'_t-1'],ascending=False)
        df_MV = df_MV[pd.isnull(df_MV[factor+'_t-1_%'] )==False]
        df_MV[factor+'_Rank'] = np.arange(len(df_MV))+1
        
        df_MV[factor+'_10g']=pd.qcut(df_MV[factor+'_Rank'], 10, labels=False)
        
        df_MV[factor+'_Weight_'+weight_method] = cal_3group_weight_tri(df_MV,factor,longbig,short_index=False,long_index=False,value=False) 
        df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_10g'
                       ,factor+'_Weight_'+weight_method]]
        df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
        
        df_less1[need_col_less1+[factor+'_Rank_t-1',factor+'_10g_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank',factor+'_10g','Close_t+1']]
        df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1',factor+'_10g_t-1','Close_t+1_less1']]
                           ,on=['Code'],how='outer')
        df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
        df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
        df_MV_m = df_MV_m.fillna(0)
 
        
        df_MV_m['out_of_group'] = df_MV_m[factor+'_10g'] != df_MV_m[factor+'_10g_t-1']

        all_invest=df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]]).abs().sum()
        
        if (abs(all_invest - 2*initial_account) <= 2000000000 ):
            df_MV_m[need_col[0]] = df_MV_m[need_col[0]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[0]])
            df_MV_m[need_col[1]] = df_MV_m[need_col[1]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[1]])
            df_MV_m[need_col[2]] = df_MV_m[need_col[2]].where(df_MV_m['out_of_group']==1,df_MV_m[need_col_less1[2]])
            

        else:
            print('Over_the_threshold!!!'+str(Date)+" " + str(abs(all_invest - 2*initial_account)))
               
            pass
        df_MV_m  = df_MV_m[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g',factor+'_10g_t-1']
                         +need_col+need_col_less1]
        df_Final = df_Final[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank',factor+'_Rank_t-1'
                            ,factor+'_10g' ,factor+'_10g_t-1']
                         +need_col+need_col_less1]
        print(np.sum(abs(df_MV_m[factor+'_Amount_'+weight_method] - df_MV_m[factor+'_Amount_'+weight_method+'_t-1'])) )
        df_Final = pd.concat([df_Final,df_MV_m],axis=0)
df_Final.to_csv('./Weight/'+factor+'_Weight_'+weight_method+'.txt',index=False)
#計算損益

df_performance = df_Final.reset_index(drop=True)
#del df_MV,temp


df_Equity_Trunover_Tax           = pd.DataFrame()
df_Equity_Trunover_Tax[['Date','Profit_'+weight_method]] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Profit(x,factor ,weight_method)).reset_index()
df_Equity_Trunover_Tax['Tax_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Cost(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Turnover_'+weight_method] = df_performance.groupby('Date').apply(lambda x : 
                                                                        Cal_Turnover(x,factor ,weight_method)).reset_index(drop=True)
df_Equity_Trunover_Tax['Equity_'+weight_method]   = initial_account + df_Equity_Trunover_Tax['Profit_'+weight_method].cumsum() - df_Equity_Trunover_Tax['Tax_'+weight_method].cumsum()
df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)


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


# In[ ]:


for f in range(len(factor_list)):
    factor  = factor_list[f]

    df_tri = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri'+'.txt')
    df_tri_10g = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri_10g'+'.txt')
    df_tri_ind_perc = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri_ind%'+'.txt')
    df_tri_ind_perc_10g = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri_ind%_10g'+'.txt')
    df_tri_ind_std = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri_indstd'+'.txt')
    df_tri_ind_std_10g = pd.read_csv('./Stats_Detail_New/'+factor+'_Stats_Detail_New_'+'tri_indstd_10g'+'.txt')
    x = np.arange(len(df_tri))
    plt.plot(df_tri['Equity_'+'tri'],label='Equity_'+'tri')
    plt.plot(df_tri_10g['Equity_'+'tri_10g'],label='Equity_'+'tri_10g')
    plt.plot(df_tri_ind_perc['Equity_'+'tri_ind%'],label='Equity_'+'tri_ind%')
    plt.plot(df_tri_ind_perc_10g['Equity_'+'tri_ind%_10g'],label='Equity_'+'tri_ind%_10g')
    plt.plot(df_tri_ind_std['Equity_'+'tri_indstd'],label='Equity_'+'tri_indstd')
    plt.plot(df_tri_ind_std_10g['Equity_'+'tri_indstd_10g'],label='Equity_'+'tri_indstd_10g')
    date = df_tri['Date']

    plt.xticks(x[::300],labels=date[::300],rotation=45)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(factor+'_tri_weight_extend')
    plt.savefig('./Graph_new/'+factor+'_tri_weight_extend.jpg', bbox_inches='tight')
    plt.show()
    
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

    Factors_list = []
    LongBig_list = []
    Weight_Method_list = ['tri','tri_10g','tri_ind%','tri_ind%_10g',
                          'tri_indstd','tri_indstd_10g']
    CAGR_list = []
    STD_Y_list = []
    Sharp_ratio_list = []
    mdd_list = []
    NP_MDD_list = []
    Odd_list = []
    skew_list = []
    Tornover_list = []

    data_list = [df_tri,df_tri_10g,df_tri_ind_perc,df_tri_ind_perc_10g,df_tri_ind_std,df_tri_ind_std_10g]

    for i in range(len(data_list)):

        equity = pd.DataFrame(data_list[i]['Equity_'+Weight_Method_list[i]])
        CAGR,STD_Y,Sharp_ratio,mdd,NP_MDD,Odd,skew = performance_stats(equity,E_RF=E_RF)
        CAGR_list.append(CAGR)
        STD_Y_list.append(STD_Y)
        Sharp_ratio_list.append(Sharp_ratio)
        mdd_list.append(mdd[0])
        NP_MDD_list.append(NP_MDD[0][0])
        Odd_list.append(Odd[0])
        skew_list.append(skew[0])
        Factors_list.append( Factor_args[Main_Factor_Name]['factors'][0])
        LongBig_list.append( Factor_args[Main_Factor_Name]['longbig'][0])
        Tornover_list.append(np.mean([x for x in data_list[i]['Turnover_'+Weight_Method_list[i]] if np.isnan(x)==False]))


    df_factor_stat = pd.DataFrame()
    df_factor_stat['Factors']       = Factors_list
    df_factor_stat['LongBig']       = LongBig_list
    df_factor_stat['Weight_Method']       = Weight_Method_list                         
    df_factor_stat['Return_annual'] = CAGR_list
    df_factor_stat['Std_annual'] = STD_Y_list
    df_factor_stat['Sharp']         = Sharp_ratio_list
    df_factor_stat['mdd']        = mdd_list
    df_factor_stat['NP_MDD']     = NP_MDD_list
    df_factor_stat['Odds']          = Odd_list
    df_factor_stat['Turnover']          = Tornover_list
    df_factor_stat['Skew']          = skew_list
    df_factor_stat.round(4)
    df_factor_stat.to_csv('./stats/'+factor+".csv",index=False)
    df_factor_stat


# In[ ]:




