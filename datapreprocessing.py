#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc


# # 一、 匯入開高低收量市值等等基本資料

# In[ ]:





# In[2]:


# df_H = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909h.xlsx')
# df_H.iloc[3,2:len(df_H.columns)] = df_H.iloc[0,2:len(df_H.columns)]
# df_H.columns = df_H.iloc[0,:] 
# df_H = df_H.drop([0,1,2,3]).T
# df_H=df_H.reset_index()
# df_H.columns = df_H.iloc[0,:]
# df_H = df_H.drop([0,1])
# df_H_melt=pd.melt(df_H, id_vars=['基準日:最近一日'], value_vars=df_H.columns[1:len(df_H.columns)]
#        ,var_name='Code', value_name='High')
# df_H_melt = df_H_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[3]:


# df_L = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909l.xlsx')
# df_L.iloc[3,2:len(df_L.columns)] = df_L.iloc[0,2:len(df_L.columns)]
# df_L.columns = df_L.iloc[0,:] 
# df_L = df_L.drop([0,1,2,3]).T
# df_L=df_L.reset_index()
# df_L.columns = df_L.iloc[0,:]
# df_L = df_L.drop([0,1])
# df_L_melt=pd.melt(df_L, id_vars=['基準日:最近一日'], value_vars=df_L.columns[1:len(df_L.columns)]
#        ,var_name='Code', value_name='Low')
# df_L_melt = df_L_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[4]:


df_V = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909v.xlsx')
df_V.iloc[3,2:len(df_V.columns)] = df_V.iloc[0,2:len(df_V.columns)]
df_V.columns = df_V.iloc[0,:] 
df_V = df_V.drop([0,1,2,3]).T
df_V=df_V.reset_index()
df_V.columns = df_V.iloc[0,:]
df_V = df_V.drop([0,1])
df_V_melt=pd.melt(df_V, id_vars=['基準日:最近一日'], value_vars=df_V.columns[1:len(df_V.columns)]
       ,var_name='Code', value_name='Volumn')
df_V_melt = df_V_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[5]:


df_C = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909c.xlsx')
df_C.iloc[3,2:len(df_C.columns)] = df_C.iloc[0,2:len(df_C.columns)]
df_C.columns = df_C.iloc[0,:] 
df_C = df_C.drop([0,1,2,3]).T
df_C=df_C.reset_index()
df_C.columns = df_C.iloc[0,:]
df_C = df_C.drop([0,1])
df_C_melt=pd.melt(df_C, id_vars=['基準日:最近一日'], value_vars=df_C.columns[1:len(df_C.columns)]
       ,var_name='Code', value_name='Close')
df_C_melt = df_C_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[6]:


# df_C_R = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909c_r.xlsx')
# df_C_R.iloc[3,2:len(df_C_R.columns)] = df_C_R.iloc[0,2:len(df_C_R.columns)]
# df_C_R.columns = df_C_R.iloc[0,:] 
# df_C_R = df_C_R.drop([0,1,2,3]).T
# df_C_R=df_C_R.reset_index()
# df_C_R.columns = df_C_R.iloc[0,:]
# df_C_R = df_C_R.drop([0,1])
# df_C_R_melt=pd.melt(df_C_R, id_vars=['基準日:最近一日'], value_vars=df_C_R.columns[1:len(df_C_R.columns)]
#        ,var_name='Code', value_name='Close')
# df_C_R_melt = df_C_R_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[7]:


df_CAP = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909cap.xlsx')
df_CAP.iloc[3,2:len(df_CAP.columns)] = df_CAP.iloc[0,2:len(df_CAP.columns)]
df_CAP.columns = df_CAP.iloc[0,:] 
df_CAP = df_CAP.drop([0,1,2,3]).T
df_CAP=df_CAP.reset_index()
df_CAP.columns = df_CAP.iloc[0,:]
df_CAP = df_CAP.drop([0,1])
df_CAP_melt=pd.melt(df_CAP, id_vars=['基準日:最近一日'], value_vars=df_CAP.columns[1:len(df_CAP.columns)]
       ,var_name='Code', value_name='CAP')
df_CAP_melt = df_CAP_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###


# In[8]:


df_V_a = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2006-201909v_a.xlsx')
df_V_a.iloc[3,2:len(df_V_a.columns)] = df_V_a.iloc[0,2:len(df_V_a.columns)]
df_V_a.columns = df_V_a.iloc[0,:] 
df_V_a = df_V_a.drop([0,1,2,3]).T
df_V_a=df_V_a.reset_index()
df_V_a.columns = df_V_a.iloc[0,:]
df_V_a = df_V_a.drop([0,1])
df_V_a_melt=pd.melt(df_V_a, id_vars=['基準日:最近一日'], value_vars=df_V_a.columns[1:len(df_V_a.columns)]
       ,var_name='Code', value_name='Volumn_a')
df_V_a_melt = df_V_a_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

df_V_a_melt['Volumn_a_mean']   = df_V_a_melt.groupby('Code')['Volumn_a'].rolling(250,min_periods=250).mean().reset_index(level=0,drop=True)
df_V_a_melt = df_V_a_melt[df_V_a_melt['基準日:最近一日']>=df_C_melt['基準日:最近一日'][0]]
df_V_a_melt['Volumn_a_mean']   = df_V_a_melt['Volumn_a_mean']*1000

diff_Code=set(df_V_a_melt.Code.unique()) - set(df_CAP_melt.Code.unique())
df_V_a_melt = df_V_a_melt[[ x not in diff_Code for x in df_V_a_melt['Code'] ]].reset_index(drop=True)


# In[9]:


df_merge_1 = pd.DataFrame()
df_merge_1['Code']            = df_C_melt['Code']
df_merge_1['Date']            = df_C_melt['基準日:最近一日']
#df_merge_1['Open']            = df_O_melt['Open'].astype('float')
#df_merge_1['High']            = df_H_melt['High'].astype('float')
#df_merge_1['Low']             = df_L_melt['Low'].astype('float')
df_merge_1['Close']           = df_C_melt['Close'].astype('float')
#df_merge_1['Close_r']         = df_C_R_melt['Close'].astype('float')
df_merge_1['Volumn']          = df_V_melt['Volumn'].astype('float')
df_merge_1['Volumn_a_mean']        = df_V_a_melt['Volumn_a_mean'].astype('float')
df_merge_1['CAP']             = df_CAP_melt['CAP'].astype('float')


# In[10]:


del df_C_melt,df_V_melt,df_V_a_melt,df_CAP_melt


# # 匯入產業數據

# In[11]:


df_Ind_list = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909ind.xlsx',sheet_name='list')
df_Ind_list.iloc[3,2:len(df_Ind_list.columns)] = df_Ind_list.iloc[0,2:len(df_Ind_list.columns)]
df_Ind_list.columns = df_Ind_list.iloc[0,:] 
df_Ind_list = df_Ind_list.drop([0,1,2,3]).T
df_Ind_list=df_Ind_list.reset_index()
df_Ind_list.columns = df_Ind_list.iloc[0,:]
df_Ind_list = df_Ind_list.drop([0,1])
df_Ind_list['基準日:最近一日'] = np.arange(2009,2020)
df_Ind_list_melt=pd.melt(df_Ind_list, id_vars=['基準日:最近一日'], value_vars=df_Ind_list.columns[1:len(df_Ind_list.columns)]
       ,var_name='Code', value_name='Ind')
df_Ind_list_melt = df_Ind_list_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###
df_Ind_list_melt['Ind'] =df_Ind_list_melt.groupby('Code')['Ind'].fillna(method='ffill')

df_Ind_delist = pd.read_excel('./Data/1.OHLC_V_CAP_Ind/2007-201909ind.xlsx',sheet_name='delist')
df_Ind_delist = df_Ind_delist.drop([0,1,2,3])
df_Ind_delist.columns = ['Code','Name','Ind']
df_Ind_delist['基準日:最近一日'] = '2019'
df_Ind_delist = df_Ind_delist[df_Ind_list_melt.columns]

df_Ind_melt=pd.concat([df_Ind_list_melt,df_Ind_delist]).reset_index(drop=True)
df_Ind_melt.columns = ['Year','Code','Ind']


# In[12]:


#合併Ind
df_merge_1['Year'] = [ str(x)[0:4] for x in df_merge_1['Date'] ]
df_Ind_melt.Year       =  df_Ind_melt.Year.astype('str')
df_Ind_melt = df_Ind_melt.drop_duplicates(subset=['Year','Code'])
df_merge_1           =  pd.merge(left=df_merge_1,right=df_Ind_melt,on=['Year','Code'],how='left')  
df_merge_1['Ind'] = df_merge_1.groupby('Code')['Ind'].fillna(method='bfill').fillna(method='ffill')


# In[13]:


# 匯入RM
df_RM=pd.read_excel('Data/3.Rm_Rf/2007-201909Rm.xlsx')
df_RM.columns = df_RM.iloc[3,]
df_RM=df_RM.drop([0,1,2,3])
df_RM=df_RM.dropna(axis=1).reset_index(drop=True)
df_RM.columns = ['Date','大盤指數']
df_RM['Date'] = [datetime.strftime(x,format='%Y%m%d') for x in df_RM['Date'] ]
df_RM=df_RM.sort_values('Date').drop_duplicates()
df_RM['Rm'] = df_RM['大盤指數'].pct_change() 
df_merge_1.Date = df_merge_1.Date.astype('str')
df_merge_1_1           =  pd.merge(left=df_merge_1,right=df_RM,on=['Date'],how='left')


# In[14]:


if not os.path.exists('./Data/output/'):
    os.makedirs('./Data/output/')
df_merge_1_1.to_csv('./Data/output/df_merge_1.txt',index=False)
#del df_merge_1 
#gc.collect()


# In[15]:


df_merge_1_1.count()


# # 二、 匯入財務指標等CMoney現有資料

# In[16]:


df_PE = pd.read_excel('./Data/2.Cmoney_factor/2007-201909pe.xlsx')
df_PE.iloc[3,2:len(df_PE.columns)] = df_PE.iloc[0,2:len(df_PE.columns)]
df_PE.columns = df_PE.iloc[0,:] 
df_PE = df_PE.drop([0,1,2,3]).T
df_PE=df_PE.reset_index()
df_PE.columns = df_PE.iloc[0,:]
df_PE = df_PE.drop([0,1])
df_PE_melt=pd.melt(df_PE, id_vars=['基準日:最近一日'], value_vars=df_PE.columns[1:len(df_PE.columns)]
       ,var_name='Code', value_name='PE')
df_PE_melt = df_PE_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###
del df_PE


# In[17]:


df_PB = pd.read_excel('./Data/2.Cmoney_factor/2007-201909pb.xlsx')
df_PB.iloc[3,2:len(df_PB.columns)] = df_PB.iloc[0,2:len(df_PB.columns)]
df_PB.columns = df_PB.iloc[0,:] 
df_PB = df_PB.drop([0,1,2,3]).T
df_PB=df_PB.reset_index()
df_PB.columns = df_PB.iloc[0,:]
df_PB = df_PB.drop([0,1])
df_PB_melt=pd.melt(df_PB, id_vars=['基準日:最近一日'], value_vars=df_PB.columns[1:len(df_PB.columns)]
       ,var_name='Code', value_name='PB')
df_PB_melt = df_PB_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###
del df_PB


# In[18]:


df_for = pd.read_excel('./Data/2.Cmoney_factor/2007-201909for.xlsx')
df_for.iloc[3,2:len(df_for.columns)] = df_for.iloc[0,2:len(df_for.columns)]
df_for.columns = df_for.iloc[0,:] 
df_for = df_for.drop([0,1,2,3]).T
df_for=df_for.reset_index()
df_for.columns = df_for.iloc[0,:]
df_for = df_for.drop([0,1])
df_for_melt=pd.melt(df_for, id_vars=['基準日:最近一日'], value_vars=df_for.columns[1:len(df_for.columns)]
       ,var_name='Code', value_name='For')
df_for_melt = df_for_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_for_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_for_melt = df_for_melt[[ x not in diff_Code for x in df_for_melt['Code'] ]].reset_index(drop=True)

df_for_melt['For_less1'] = df_for_melt.groupby('Code')['For'].shift(1).reset_index(level=0,drop=True) 
df_for_melt['For_%']     = [ x/y-1 if y>0 else -(x/y-1) if y <0 else np.NaN for x,y in zip(df_for_melt['For'],df_for_melt['For_less1']) ]
del df_for


# In[19]:


df_rgz = pd.read_excel('./Data/2.Cmoney_factor/2007-201909rgz.xlsx')
df_rgz.iloc[3,2:len(df_rgz.columns)] = df_rgz.iloc[0,2:len(df_rgz.columns)]
df_rgz.columns = df_rgz.iloc[0,:] 
df_rgz = df_rgz.drop([0,1,2,3]).T
df_rgz=df_rgz.reset_index()
df_rgz.columns = df_rgz.iloc[0,:]
df_rgz = df_rgz.drop([0,1])
df_rgz_melt=pd.melt(df_rgz, id_vars=['基準日:最近一日'], value_vars=df_rgz.columns[1:len(df_rgz.columns)]
       ,var_name='Code', value_name='Rgz')
df_rgz_melt = df_rgz_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_rgz_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_rgz_melt = df_rgz_melt[[ x not in diff_Code for x in df_rgz_melt['Code'] ]].reset_index(drop=True)

df_rgz_melt['Rgz'] = df_rgz_melt['Rgz']/100
df_rgz_melt['Rgz_less1'] = df_rgz_melt.groupby('Code')['Rgz'].shift(1).reset_index(level=0,drop=True) 
df_rgz_melt['Rgz_%']     = [ x/y-1 if y>0 else -(x/y-1) if y <0 else np.NaN for x,y in zip(df_rgz_melt['Rgz'],df_rgz_melt['Rgz_less1']) ]
del df_rgz


# In[20]:


df_for_N = pd.read_excel('./Data/2.Cmoney_factor/2007-201909for_N.xlsx')
df_for_N.iloc[3,2:len(df_for_N.columns)] = df_for_N.iloc[0,2:len(df_for_N.columns)]
df_for_N.columns = df_for_N.iloc[0,:] 
df_for_N = df_for_N.drop([0,1,2,3]).T
df_for_N=df_for_N.reset_index()
df_for_N.columns = df_for_N.iloc[0,:]
df_for_N = df_for_N.drop([0,1])
df_for_N_melt=pd.melt(df_for_N, id_vars=['基準日:最近一日'], 
                      value_vars=df_for_N.columns[1:len(df_for_N.columns)]
                     ,var_name='Code', value_name='for_N')
df_for_N_melt = df_for_N_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_for_N_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_for_N_melt = df_for_N_melt[[ x not in diff_Code for x in df_for_N_melt['Code'] ]].reset_index(drop=True)
del df_for_N


# In[21]:


df_invest_N = pd.read_excel('./Data/2.Cmoney_factor/2007-201909invest_N.xlsx')
df_invest_N.iloc[3,2:len(df_invest_N.columns)] = df_invest_N.iloc[0,2:len(df_invest_N.columns)]
df_invest_N.columns = df_invest_N.iloc[0,:] 
df_invest_N = df_invest_N.drop([0,1,2,3]).T
df_invest_N=df_invest_N.reset_index()
df_invest_N.columns = df_invest_N.iloc[0,:]
df_invest_N = df_invest_N.drop([0,1])
df_invest_N_melt=pd.melt(df_invest_N, id_vars=['基準日:最近一日'], value_vars=df_invest_N.columns[1:len(df_invest_N.columns)]
       ,var_name='Code', value_name='invest_N')
df_invest_N_melt = df_invest_N_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_invest_N_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_invest_N_melt = df_invest_N_melt[[ x not in diff_Code for x in df_invest_N_melt['Code'] ]].reset_index(drop=True)
del df_invest_N


# In[22]:


df_main_N = pd.read_excel('./Data/2.Cmoney_factor/2007-201909main_N.xlsx')
df_main_N.iloc[3,2:len(df_main_N.columns)] = df_main_N.iloc[0,2:len(df_main_N.columns)]
df_main_N.columns = df_main_N.iloc[0,:] 
df_main_N = df_main_N.drop([0,1,2,3]).T
df_main_N=df_main_N.reset_index()
df_main_N.columns = df_main_N.iloc[0,:]
df_main_N = df_main_N.drop([0,1])
df_main_N_melt=pd.melt(df_main_N, id_vars=['基準日:最近一日'], value_vars=df_main_N.columns[1:len(df_main_N.columns)]
       ,var_name='Code', value_name='main_N')
df_main_N_melt = df_main_N_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_main_N_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_main_N_melt = df_main_N_melt[[ x not in diff_Code for x in df_main_N_melt['Code'] ]].reset_index(drop=True)
del df_main_N 


# In[23]:


df_sharecapital = pd.read_excel('./Data/2.Cmoney_factor/2007-201909sharecapital.xlsx')
df_sharecapital.iloc[3,2:len(df_sharecapital.columns)] = df_sharecapital.iloc[0,2:len(df_sharecapital.columns)]
df_sharecapital.columns = df_sharecapital.iloc[0,:] 
df_sharecapital = df_sharecapital.drop([0,1,2,3]).T
df_sharecapital=df_sharecapital.reset_index()
df_sharecapital.columns = df_sharecapital.iloc[0,:]
df_sharecapital = df_sharecapital.drop([0,1])
df_sharecapital_melt=pd.melt(df_sharecapital, id_vars=['基準日:最近一日'], value_vars=df_sharecapital.columns[1:len(df_sharecapital.columns)]
       ,var_name='Code', value_name='sharecapital')
df_sharecapital_melt = df_sharecapital_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###

diff_Code=set(df_sharecapital_melt.Code.unique()) - set(df_PE_melt.Code.unique())
df_sharecapital_melt = df_sharecapital_melt[[ x not in diff_Code for x in df_sharecapital_melt['Code'] ]].reset_index(drop=True)
del df_sharecapital 


# In[24]:


df_merge_2 = pd.DataFrame()
df_merge_2['Code'] = df_PE_melt['Code']
df_merge_2['Date'] = df_PE_melt['基準日:最近一日']
df_merge_2['PE'] = df_PE_melt['PE']
df_merge_2['PB'] = df_PB_melt['PB']
df_merge_2['For'] = df_for_melt['For']
df_merge_2['For_%'] = df_for_melt['For_%']
df_merge_2['Rgz'] = df_rgz_melt['Rgz']
df_merge_2['Rgz_%'] = df_rgz_melt['Rgz_%']
df_merge_2['for_N'] = df_for_N_melt['for_N']
df_merge_2['invest_N'] = df_invest_N_melt['invest_N']
df_merge_2['main_N'] = df_main_N_melt['main_N']
df_merge_2['sharecapital'] = df_sharecapital_melt['sharecapital']


# In[25]:


df_merge_2.to_csv('./Data/output/df_merge_2.txt',index=False)


# In[26]:


#del df_PE_melt,df_PB_melt,df_for_melt,df_rgz_melt,
del df_for_N_melt,df_invest_N_melt,df_main_N_melt
import gc
gc.collect()


# # 三、其他資料 sales vwap div

# In[27]:


df_merge_1 = pd.read_csv('./Data/output/df_merge_1.txt',engine='c')


# In[28]:


df_vwap=pd.read_excel('Data/4.Other/2007-2019vwap1.xlsx')
df_vwap=df_vwap.reset_index()
df_vwap.columns = df_vwap.iloc[2,:]
df_vwap =  df_vwap.iloc[6:,6:]
df_vwap.columns = ['Date']+list(df_vwap.columns[1:] )
df_vwap_melt=pd.melt(df_vwap, id_vars=['Date'], value_vars=df_vwap.columns[1:len(df_vwap.columns)]
       ,var_name='Code', value_name='VWAP')

df_vwap_melt['Date'] = [datetime.strftime(x,'%Y%m%d') for x in df_vwap_melt['Date'] ]
df_vwap_melt['Code'] = df_vwap_melt['Code'].str.split(" ",expand=True)[0]

df_merge_1[['Date','Code']] = df_merge_1[['Date','Code']].astype('str')
df_merge_vwap=pd.merge(left=df_merge_1[['Date','Code','Close']],right=df_vwap_melt,on=['Date','Code'],how='left')
df_merge_vwap['Over_Vwap']= df_merge_vwap['Close']/df_merge_vwap['VWAP'] - 1
df_merge_vwap=df_merge_vwap.sort_values(['Code','Date']).reset_index(drop=True)


# In[29]:


df_Sales=pd.read_excel('Data/4.Other/2007-201909sales.xlsx')
df_Sales.iloc[3,2:len(df_Sales.columns)] = df_Sales.iloc[0,2:len(df_Sales.columns)]
df_Sales.columns = df_Sales.iloc[0,:] 
df_Sales = df_Sales.drop([0,1,2,3]).T
df_Sales=df_Sales.reset_index()
df_Sales.columns = df_Sales.iloc[0,:]
df_Sales = df_Sales.drop([0,1])
df_Sales_melt=pd.melt(df_Sales, id_vars=['基準日:最近一日'], value_vars=df_Sales.columns[1:len(df_Sales.columns)]
       ,var_name='Code', value_name='Sales_M')
df_Sales_melt = df_Sales_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###
df_Sales_melt.columns = ['Key','Code','Sales_M']
df_Sales_melt['Sales_M'] = df_Sales_melt['Sales_M'].astype('float')
df_Sales_melt['Sales_M']  = df_Sales_melt['Sales_M']*1000

def make_month_key(Date):
    year= Date[0:4]
    month=Date[4:6]
    day  =Date[6:8]
    if int(month) ==1:
        if int(day) <=10:
            return str(int(year)-1)+'11'
        else:
            return str(int(year)-1)+'12'
    if int(month) ==2:
        if int(day) <=10:
            return str(int(year)-1)+'12'
        else:
            return str(int(year))+'01'
    else:
        if int(day) <=10:
            if len(str(int(month)-2)) :
                return year+'0'+str(int(month)-2)
            else:
                return year+str(int(month)-2)
            
        else:
            if len(str(int(month)-1)) :
                return year+'0'+str(int(month)-1)
            else:
                return year+str(int(month)-1)     
            
df_merge_1['Key'] = [  make_month_key(x) for x in df_merge_1['Date'] ] 
df_Sales_melt.Key = df_Sales_melt.Key.astype('str')
df_merge_sales = pd.merge(left=df_merge_1[['Date','Key','Code','CAP']],right=df_Sales_melt,on=['Key','Code'],how='left') 

df_merge_sales['PS'] = np.where(np.array(df_merge_sales['Sales_M'])==0,
                                np.NaN, np.array(df_merge_sales['CAP']*100000000/df_merge_sales['Sales_M']))
df_merge_sales = df_merge_sales[['Date','Code','PS']]
df_merge_sales = df_merge_sales.sort_values(['Code','Date']).reset_index(drop=True)


# In[30]:


df_Div=pd.read_excel('Data/4.Other/2000-2019div_Y.xlsx')
df_Div.iloc[3,2:len(df_Div.columns)] = df_Div.iloc[0,2:len(df_Div.columns)]
df_Div.columns = df_Div.iloc[0,:] 
df_Div = df_Div.drop([0,1,2,3]).T
df_Div=df_Div.reset_index()
df_Div.columns = df_Div.iloc[0,:]
df_Div = df_Div.drop([0,1])
df_Div_melt=pd.melt(df_Div, id_vars=['基準日:最近一日'], value_vars=df_Div.columns[1:len(df_Div.columns)]
       ,var_name='Code', value_name='Div_Y')
df_Div_melt = df_Div_melt.sort_values(['Code','基準日:最近一日']).reset_index(drop=True)###
df_Div_melt.columns = ['Key','Code','Div_Y']
df_Div_melt['Div_Y'] = df_Div_melt['Div_Y'].astype('float')

def make_year_key(Date):
    Date = str(Date)
    year= Date[0:4]
    month=Date[4:6]
    day  =Date[6:8]
    if int(month) <=3:
        return str(int(year)-2)
    else:
        return str(int(year)-1)
    
df_merge_1['Key'] = [  make_year_key(x) for x in df_merge_1['Date'] ] 
df_Div_melt.Key = df_Div_melt.Key.astype('str')
df_Div_melt.Code = df_Div_melt.Code.astype('str')
df_merge_1.Code  = df_merge_1.Code.astype('str')
df_merge_div = pd.merge(left=df_merge_1,right=df_Div_melt,on=['Key','Code'],how='left')  
df_merge_div['DP'] = df_merge_div['Div_Y']/df_merge_div['Close']
df_merge_div = df_merge_div[['Date','Code','DP']]
df_merge_div = df_merge_div.sort_values(['Code','Date']).reset_index(drop=True)


# In[31]:


df_merge_3 = pd.DataFrame()
df_merge_3[['Date','Code']] = df_merge_vwap[['Date','Code']]
df_merge_3['Over_Vwap']     = df_merge_vwap['Over_Vwap'] 
df_merge_3['PS']            = df_merge_sales['PS'] 
df_merge_3['DP']            = df_merge_div['DP']
df_merge_3.to_csv('./Data/output/df_merge_3.txt',index=False)


# In[32]:


df_merge_3.count()


# In[ ]:





# In[ ]:





# In[ ]:




