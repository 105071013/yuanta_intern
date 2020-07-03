import pandas as pd
from pymongo import MongoClient
import urllib.parse 


def label_func(tags,labels):
    for x in tags:
        if x in labels:
            return 1
    return 0


def data_retriever(datasource,start_idx=0,end_idx=None,labels=[],col_selector=[]):
    if(type(datasource)==dict):
        #mongodb://user:password@example.com/?authSource=the_database
        
        password = urllib.parse.quote_plus(datasource['password'])
        
        DBconn = 'mongodb://' + datasource['username'] + ':' + '%s'%(password)+ '@' + datasource['serverIP']+'/?authSource=' + datasource['authsource']
        myclient = MongoClient(DBconn)
        mydb = myclient[datasource['DB']]
        mycol = mydb[datasource['collection']]
        df = pd.DataFrame(mycol.find())
        df = df[col_selector]
        
        if('contents' in col_selector):
            df=df[df['contents']!=""]
            
        if (labels!=[]):
            df['label'] = 0
            df['label'] =df['tags'].apply(label_func,labels=labels)
        
        df=df.iloc[start_idx:end_idx]
        df = df.reset_index(drop=True)
    else:
        if(type(datasource)==str):
            df = pd.read_csv(data_source)
    return df