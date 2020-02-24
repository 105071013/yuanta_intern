import dask.dataframe as dd
import time
import pandas as pd
import numpy as np
from scipy.stats import mstats
from progressbar import *
import matplotlib.pyplot as plt
import os 
import gc


# In[73]:


if not os.path.exists('./Stats_Detail_New/'):
    os.makedirs('./Stats_Detail_New/')
if not os.path.exists('./Stats_Detail_New/multiway'):
    os.makedirs('./Stats_Detail_New/multiway')    
    
if not os.path.exists('./Weight/multiway'):
    os.makedirs('./Weight/multiway')
if not os.path.exists('./Graph_new/multiway'):
    os.makedirs('./Graph_new/multiway')
if not os.path.exists('./stats/multiway'):
    os.makedirs('./stats/multiway')


# In[74]:


def select_top10_ind(data):
    data_MV = data[data.MV_Select == 1]

    select_ind=data_MV.groupby('Ind')['Code'].count().sort_values(ascending=False) 
    select_ind = select_ind[select_ind>4]
    select_ind = select_ind.index[0:10].tolist()

    return [ ((x in select_ind)&( y in data_MV['Code'].tolist()))*1 for x,y in zip(data['Ind'],data['Code']) ]


# In[75]:


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


# In[76]:


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


# # 計算權重程式區

# In[ ]:


def cal_3group_weight_tri(df,factor,upper_pec,lower_pec,longbig,short_index=False,long_index=False,value=False):
    df = df.sort_values(factor+'_t-1',ascending=False)
    df[factor+'_Rank'] = np.arange(len(df))+1
    length     =    len(df)
    middle     =    np.ceil(length /2)
    if value == True: #因子值加權
        df[factor+'_t-1'] = winsorize(df[factor+'_t-1'])
        df[factor+'_t-1'] = min_max(df[factor+'_t-1'])
        
        large_Rank =    df[df[factor+'_Rank']<=middle][factor+'_t-1']
        small_Rank =    df[df[factor+'_Rank']> middle][factor+'_t-1']
        #都為正數
        large_Rank =    large_Rank    
        small_Rank =    1 - small_Rank

    else:             #因子值排序加權
        
        large_Rank =    df[df[factor+'_Rank']<=middle][factor+'_Rank'] 
        small_Rank =    df[df[factor+'_Rank']> middle][factor+'_Rank']
        #都為正數
        large_Rank =    middle +1  - large_Rank                              #數值最大的逆著排     100 99 98 #最大是一
        small_Rank =    small_Rank - middle                             #因子數值最小的順著排   1  2  3   #最小是一
   
    if longbig == 1: #買因子大 賣因子小
        if   short_index ==True:#買大因子 且放空指數 故因子小的權重為0
            large_Rank =    large_Rank/( large_Rank.sum() )  
            small_Rank =    small_Rank/( small_Rank.sum() ) * 0 
        elif long_index ==True:#賣小因子 且做多指數 故因子大的權重為0
            large_Rank =    large_Rank/( large_Rank.sum() ) * 0
            small_Rank =    small_Rank/( small_Rank.sum() )              
        else:
            large_Rank =    large_Rank/( large_Rank.sum() )  
            small_Rank =    small_Rank/( small_Rank.sum() )             
        return  pd.concat([large_Rank,-small_Rank]),df[factor+'_Rank'] 
    else :          #賣因子大 買因子小
        if short_index ==True: #買小因子 且放空指數 故因子大的權重為0
            large_Rank =    large_Rank/( large_Rank.sum() ) * 0 
            small_Rank =    small_Rank/( small_Rank.sum() ) 
        elif long_index ==True:#賣大因子 且做多指數 故因子小的權重為0
            large_Rank =    large_Rank/( large_Rank.sum() ) 
            small_Rank =    small_Rank/( small_Rank.sum() ) * 0 
        else:
            large_Rank =    large_Rank/( large_Rank.sum() )  
            small_Rank =    small_Rank/( small_Rank.sum() )  
        return  pd.concat([-large_Rank,small_Rank]),df[factor+'_Rank'] 

def cal_3group_weight_eq(df,factor,upper_pec,lower_pec,longbig,ind=False,short_index=False,long_index=False,value=False):
    df = df.sort_values(factor+'_t-1',ascending=False)
    df[factor+'_Rank'] = np.arange(len(df))+1    
    length     =    len(df)
    if ind ==False : 
        upper   = int(length*upper_pec)
        N_upper = upper
        lower   = int(length*lower_pec)
        N_lower = length - lower
    else:
        upper   = 2
        lower   = len(df) - 2

    if value == True: #因子值加權
        df[factor+'_t-1'] = winsorize(df[factor+'_t-1'])
        df[factor+'_t-1'] = min_max(df[factor+'_t-1'])
        #0~1的數值
        large_Rank      =    df[df[factor+'_Rank']<=upper][factor+'_t-1']
        middle_Rank     =    df[(lower>=df[factor+'_Rank'])&(df[factor+'_Rank'] > upper) ][factor+'_t-1']*0
        small_Rank      =    df[df[factor+'_Rank']> lower][factor+'_t-1']
        large_Rank  =   large_Rank
        large_Rank  =   large_Rank/large_Rank.sum()
        middle_Rank =   middle_Rank* 0
        small_Rank  =    1 - small_Rank 
        small_Rank  =   small_Rank/small_Rank.sum()

    else :            #因子值排序加權
    
        large_Rank      =    df[df[factor+'_Rank']<=upper][factor+'_Rank']
        large_Rank      =    large_Rank/large_Rank/len(large_Rank)                                   #平均配置權重在large_rank這個分類
        middle_Rank     =    df[(lower>=df[factor+'_Rank'])&(df[factor+'_Rank'] > upper) ][factor+'_Rank']
        small_Rank      =    df[df[factor+'_Rank']> lower][factor+'_Rank']
        small_Rank      =    small_Rank/small_Rank/len(small_Rank)            
    
    if longbig == 1:#買大因子
        if  short_index == True:#買大因子 且放空指數 故因子小的權重為0
            large_Rank  = large_Rank
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank   *0    
        elif long_index == True:#賣小因子 且做多指數 故因子大的權重為0
            large_Rank  = large_Rank   *0
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank                   
        else:
            large_Rank  = large_Rank
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank
        return pd.concat([large_Rank,middle_Rank,-small_Rank]),df[factor+'_Rank']
    else :         #買小因子
        if  short_index == True:#買小因子 且放空指數 故因子大的權重為0
            large_Rank  = large_Rank   *0   
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank
        elif long_index == True:#賣大因子 且做多指數 故因子小的權重為0
            large_Rank  = large_Rank      
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank   *0         
        else:
            large_Rank  = large_Rank
            middle_Rank = middle_Rank  *0
            small_Rank  = small_Rank
        return pd.concat([-large_Rank,middle_Rank,small_Rank]),df[factor+'_Rank']


# In[155]:



def Cal_Weight(weight_method,df_MV,df_Ind,factor,longbig,Factor_args):
    #剔除少於4檔個股的產業
    less_stock_Ind=df_Ind.groupby('Ind')[factor+'_t-1'].count()[df_Ind.groupby('Ind')[factor+'_t-1'].count()>4].index.tolist()
    df_Ind = df_Ind[[x in less_stock_Ind for x in df_Ind['Ind']]]
    N_Ind_Select  = len(df_Ind['Ind'].unique()) #共有幾個產業符合標準
    #依據權重選取方法使用上下界參數
    if 'tri' in weight_method  :                  
        upper_pec,lower_pec = Factor_args['upper_lower_perc_tri']['upper'],Factor_args['upper_lower_perc_tri']['lower']
    else:
        upper_pec,lower_pec = Factor_args['upper_lower_perc_eq']['upper'],Factor_args['upper_lower_perc_eq']['lower']
    #依據權重方法計算權重
######################################################主要策略區###########################################################
    if   weight_method == 'tri':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor,upper_pec,lower_pec,longbig)
    
    elif weight_method == 'tri_ind' :
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec,
                                                                                             lower_pec,longbig)[0]    )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec,
                                                                                             lower_pec,longbig)[1]   )
        
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True)/N_Ind_Select)
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
     
    elif weight_method == 'tri_ind_cap' :
        weight = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec,
                                                                                             lower_pec,longbig)[0]         ) 
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec,
                                                                                             lower_pec,longbig)[1]         )        
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] = df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
            
    elif weight_method == 'eq':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor,upper_pec,lower_pec,longbig
                                                                                            ,ind=False)
    
    elif weight_method == 'eq_ind':
        weight         = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor,upper_pec,
                                                                                            lower_pec,longbig,ind=True)[0] )
        rank= df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor,upper_pec,
                                                                                            lower_pec,longbig,ind=True)[1] )
        
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select  
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
######################################################放空指數區###########################################################

    elif weight_method == 'tri_short_index':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor,upper_pec,lower_pec
                                                                                             ,longbig,short_index=True)##
    
    elif weight_method == 'tri_ind_short_index' :
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec
                                                                                                            ,lower_pec,longbig,short_index=True)[0] ) ##   
        rank                  = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec
                                                                                                            ,lower_pec,longbig,short_index=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
     
    elif weight_method == 'tri_ind_cap_short_index':
        weight                 = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                        ,upper_pec,lower_pec,longbig,short_index=True)[0] ) ##   
        rank                   = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                        ,upper_pec,lower_pec,longbig,short_index=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] = df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
        
        
    elif weight_method == 'eq_short_index':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor,upper_pec,lower_pec,longbig,ind=False,short_index=True) ##
    
    elif weight_method == 'eq_ind_short_index':  
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                         ,upper_pec,lower_pec,longbig,ind=True
                                                                         ,short_index=True)[0] ) ##
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                         ,upper_pec,lower_pec,longbig,ind=True
                                                                         ,short_index=True)[1] ) 
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
######################################################做多指數區###########################################################

    elif weight_method == 'tri_long_index':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor
                                                                            ,upper_pec,lower_pec,longbig,long_index=True) ##
    
    elif weight_method == 'tri_ind_long_index': 
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                            ,upper_pec,lower_pec,longbig,long_index=True)[0] )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                            ,upper_pec,lower_pec,longbig,long_index=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  =  pd.Series(rank.reset_index(level=0, drop=True))
    
    elif weight_method == 'tri_ind_cap_long_index':
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                           ,upper_pec,lower_pec,longbig,long_index=True)[0] )
        rank                  = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                           ,upper_pec,lower_pec,longbig,long_index=True)[1] )##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Rank']                  =  pd.Series(rank.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] =  df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
        
    elif weight_method == 'eq_long_index':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor,upper_pec,lower_pec,longbig,ind=False,long_index=True) ##
      
    elif weight_method == 'eq_ind_long_index':   
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                         ,upper_pec,lower_pec,longbig,ind=True
                                                                         , long_index=True)[0] )
        rank                  = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                         ,upper_pec,lower_pec,longbig,ind=True
                                                                         , long_index=True)[1] ) ##
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  =  pd.Series(rank.reset_index(level=0, drop=True))
###############################################################################################################################    
##############################################################因子值加權#################################################################       
############################################################################################################################### 
    elif weight_method == 'tri_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor,upper_pec,lower_pec,longbig,value=True)
    
    elif weight_method == 'tri_ind_value': 
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                           ,upper_pec,lower_pec,longbig,value=True)[0]         )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                           ,upper_pec,lower_pec,longbig,value=True)[1]         )   
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True)/N_Ind_Select)
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
        
    elif weight_method == 'tri_ind_cap_value' :
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                            ,upper_pec,lower_pec,longbig,value=True)[0]        )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                            ,upper_pec,lower_pec,longbig,value=True)[1]        )    
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] = df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
        
    elif weight_method == 'eq_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor,upper_pec,lower_pec,longbig,ind=False,value=True)
    
    elif weight_method == 'eq_ind_value':
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                                            ,upper_pec,lower_pec,longbig
                                                                                            ,ind=True,value=True)[0] )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                                            ,upper_pec,lower_pec,longbig
                                                                                            ,ind=True,value=True)[1] )
        df_MV[factor+'_Weight_'+weight_method] = pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select  
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
######################################################放空指數區###########################################################
    elif weight_method == 'tri_short_index_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor,upper_pec,lower_pec,longbig
                                                                                             ,short_index=True,value=True)##
    
    elif weight_method == 'tri_ind_short_index_value' :
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,short_index=True,value=True)[0] )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,short_index=True,value=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
        
    elif weight_method == 'tri_ind_cap_short_index_value':
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,short_index=True,value=True)[0] )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,short_index=True,value=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Rank']                  =  pd.Series(rank.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] =  df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
        
    elif weight_method == 'eq_short_index_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,ind=False,short_index=True,value=True) ##
    
    elif weight_method == 'eq_ind_short_index_value':  
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,ind=True,short_index=True,value=True)[0] ) ##
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,ind=True,short_index=True,value=True)[1] ) ##
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select 
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
######################################################做多指數區###########################################################
    elif weight_method == 'tri_long_index_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_tri(df_MV,factor,upper_pec,lower_pec,longbig
                                                                                             ,long_index=True,value=True) ##
    
    elif weight_method == 'tri_ind_long_index_value' :
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,long_index=True,value=True)[0])
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor,upper_pec,lower_pec,longbig
                                                                                             ,long_index=True,value=True)[1]) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
    
    elif weight_method == 'tri_ind_cap_long_index_value':
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,long_index=True,value=True)[0] ) ##
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_tri(x,factor
                                                                                             ,upper_pec,lower_pec,longbig
                                                                                             ,long_index=True,value=True)[1] ) ##   
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))
        df_MV[factor+'_Weight_'+weight_method] = df_MV[factor+'_Weight_'+weight_method]*df_MV["Ind_CAP_Weight"]
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))

    elif weight_method == 'eq_long_index_value':
        df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = cal_3group_weight_eq(df_MV,factor,upper_pec,lower_pec,longbig,
                                                                                              ind=False,long_index=True,value=True) ##
      
    elif weight_method == 'eq_ind_long_index_value':   
        weight                = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor,upper_pec,lower_pec,longbig,
                                                                                              ind=True,long_index=True,value=True)[0] )
        rank = df_Ind.groupby('Ind',as_index=False).apply(lambda x : cal_3group_weight_eq(x,factor,upper_pec,lower_pec,longbig,
                                                                                              ind=True,long_index=True,value=True)[1] ) ##
        df_MV[factor+'_Weight_'+weight_method] =  pd.Series(weight.reset_index(level=0, drop=True))/N_Ind_Select 
        df_MV[factor+'_Rank']                  = pd.Series(rank.reset_index(level=0, drop=True))
 
    return df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank']

 

# In[156]:


Factor_args = {}

#主要策略
Factor_args['weight_method']     = ['eq','tri','eq_ind','tri_ind','tri_ind_cap']
#考慮做多、放空指數
Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_short_index' for x in Factor_args['weight_method'] ]+[ x+'_long_index' for x in Factor_args['weight_method'] ]
#值加權
Factor_args['weight_method'] = Factor_args['weight_method']+[ x+'_value' for x in Factor_args['weight_method'] ]

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


# In[157]:


print(Factor_args)


# In[158]:


reader = pd.read_csv('./Data/output/df_merge_1.txt', chunksize=10 ** 6)
df_merge_final = pd.concat([x for x in reader], ignore_index=True)
df_merge_final.count()


# In[159]:


factor = Factor_args[Main_Factor_Name]['factors'][0]
periods = Factor_args[Main_Factor_Name]['periods'][0]

reader = pd.read_csv('./Data/output/df_merge_2.txt', chunksize=10 ** 6)
df_factor = pd.concat([x for x in reader], ignore_index=True)
df_merge_final        = pd.merge(left=df_merge_final,right=df_factor[['Date','Code',factor]],on=['Date','Code'],how='left')
df_merge_final.count()
del df_factor
gc.collect()


# In[160]:


# shift 市值方便篩選成分股
df_merge_final['CAP_t-1'] = df_merge_final.groupby('Code')['CAP'].shift(1)
# 計算T-1Factor,T-2
factor_list = Factor_args[Main_Factor_Name]['factors']
for factor in factor_list :
    df_merge_final[factor+'_t-1']= df_merge_final.groupby('Code')[factor].shift(1)
    df_merge_final[factor+'_t-2']= df_merge_final.groupby('Code')[factor].shift(2)

# shift 收盤價方便計算投組損益

df_merge_final['Close_t+1']= df_merge_final.groupby('Code')['Close'].shift(-1)
# 調整開始時間
date_begin = np.array(df_merge_final['Date'].unique())[2]
df_merge_final = df_merge_final[df_merge_final['Date']>=date_begin]


df_merge_final['Volumn_a_select'] = (df_merge_final['Volumn_a_mean'] >=50000000)*1


# 剔除Close , factor t-1 t-2 為na的資料   
df_merge_final=df_merge_final[pd.isna(df_merge_final['Close'])==False]
df_merge_final=df_merge_final[pd.isna(df_merge_final['CAP_t-1'])==False]
df_merge_final=df_merge_final[pd.isna(df_merge_final[factor+'_t-1'])==False]


# In[161]:


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


# In[162]:


import warnings
warnings.filterwarnings("ignore")

weight_method_list = Factor_args['weight_method']
factor_list = Factor_args[Main_Factor_Name]['factors']
longbig_list = Factor_args[Main_Factor_Name]['longbig']
Date_list = np.sort(df_merge_Year['Date'].unique())


factor  = factor_list[0]
longbig = longbig_list[0]
for k in range(len(weight_method_list)):
    weight_method   = weight_method_list[k]
    print(weight_method)
    initial_account = 1000000000
    df_Final = pd.DataFrame()
    need_col          = [  '_Nstock_' ,'_Amount_','_Weight_']
    need_col          = [factor+x+weight_method for x in need_col ]
    need_col_less1    = [x+'_t-1' for x in need_col ]
    progress = ProgressBar()
    for i in progress(range(len(Date_list)-1)):
        if i ==0:
            Date   = Date_list[i]
            df_MV  = df_merge_Year[df_merge_Year.Date==Date ]
            df_MV  = df_MV[df_MV.MV_Select==1  ]
            df_Ind = df_MV[df_MV.Ind_Select==1 ]
            
            df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = Cal_Weight(weight_method,df_MV,df_Ind,factor,longbig,Factor_args)
            df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                           ,factor+'_Weight_'+weight_method,'Rm']]
            df_MV = df_MV[pd.isna(df_MV[factor+'_Weight_'+weight_method])==False]
            df_MV = df_MV[df_MV[factor+'_Weight_'+weight_method]!=0]
            df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)
            df_Final = df_MV

        else:
            df_less1 = df_MV
            Date   = Date_list[i]
            df_MV  = df_merge_Year[df_merge_Year.Date==Date ]
            df_MV  = df_MV[df_MV.MV_Select==1  ]
            df_Ind = df_MV[df_MV.Ind_Select==1 ]
            
            df_MV[factor+'_Weight_'+weight_method],df_MV[factor+'_Rank'] = Cal_Weight(weight_method,df_MV,df_Ind,factor,longbig,Factor_args)
            df_MV = df_MV[['Code', 'Date','Close','Close_t+1',factor+'_t-1',factor+'_t-2',factor+'_Rank'
                           ,factor+'_Weight_'+weight_method,'Rm']]
            df_MV = df_MV[pd.isna(df_MV[factor+'_Weight_'+weight_method])==False]
            df_MV = df_MV[df_MV[factor+'_Weight_'+weight_method]!=0]
            df_MV[factor+'_Nstock_'+weight_method],df_MV[factor+'_Amount_'+weight_method]  = Cal_Amount(df_MV,factor,weight_method)

            df_less1[need_col_less1+[factor+'_Rank_t-1','Close_t+1_less1']] = df_less1[need_col+[factor+'_Rank','Close_t+1']]
            df_MV_m = pd.merge(left=df_MV,right=df_less1[need_col_less1+['Code',factor+'_Rank_t-1','Close_t+1_less1']]
                               ,on=['Code'],how='outer')
            df_MV_m['Date'] = df_MV_m['Date'].fillna(method='ffill')
            df_MV_m['Close']= df_MV_m['Close'].fillna(df_MV_m['Close_t+1_less1'])
            df_MV_m = df_MV_m.fillna(0)

            df_Final = pd.concat([df_Final,df_MV_m],axis=0)
    if not os.path.exists('./Stats_Detail_New/multiway/factor'):
        os.makedirs('./Stats_Detail_New/multiway/factor')
    if not os.path.exists('./Weight/multiway/factor'):
        os.makedirs('./Weight/multiway/factor')     
        
    df_Final.to_csv('./Weight/multiway/factor/'+factor+'_Weight_'+weight_method+'.txt',index=False)
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
    df_Equity_Trunover_Tax.to_csv('./Stats_Detail_New/multiway/factor/'+factor+'_Stats_Detail_New_'+weight_method+'.txt',index=False)
    gc.collect()
    