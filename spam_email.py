#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install manual_spellchecker
# !pip install catboost
# !pip install transformers==3.0.0
# !pip install transformers sentencepiece
# !pip install torch


# In[2]:


import pandas as pd
import numpy as np
from boto3.session import Session
from manual_spellchecker import spell_checker
import re
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch, gc



def spam_email():
    # specify GPU
    device = torch.device("cuda")


    # # S3

    # In[3]:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    def key(x,y):
        ACCESS_KEY_ID = 'AKIA3JS6MC7YH7YL6CWQ'
        SECRET_ACCESS_KEY = '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'
        REGION_NAME = 'ap-southeast-1'

        session = Session(
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY
        )

        BUCKET_NAME = 'log-for-team-data'
        s3 = session.resource("s3")
        bucket = s3.Bucket(BUCKET_NAME)

        obj = bucket.Object(key=x) # example: market/zone1/data.csv
        response = obj.get()
    #     y = vaex.from_csv(response['Body'])
    #     y = pd.read_json(response['Body'], lines=True)
        y = pd.read_csv(response['Body'])
        return y


    # In[4]:


    df=key('sorkon.csv','df')


    # # กรณีเอาเข้าแบบ json

    # In[5]:


    # df_pandas = df._source
    # dff1 = pd.json_normalize(df_pandas)
    # dff1.head()
    # dff2 = dff1[['sender_address','recp_address','return_path','event_name','directionality','message_subject','recp_count']]


    # # Filter

    # In[6]:


    df1= df.loc[df['event_name']=='RECEIVE']
    df2=df1.loc[df1['directionality']=='Incoming']


    # In[7]:


    df2['sender_address']=df2['sender_address'].str.lower()
    df2['slash_all']=df2['sender_address'].str.extract('(?P<slash_all>[^@]+$)')
    df2['recp_address_domain']=df2['recp_address'].astype(str).str.findall('(?P<recp_address_domain>@.?[\w.]+)')
    df2['recp_address_domain']=df2['recp_address_domain'].astype(str).str.replace(r'\[\W','',regex=True)
    df2['recp_address_domain']=df2['recp_address_domain'].astype(str).str.replace(r'\W\]','',regex=True)
    df2['recp_address_domain']=df2['recp_address_domain'].astype(str).str.replace(r'[@\'\s]+','',regex=True)
    df2['recp_address_domain']=df2['recp_address_domain'].str.strip()
    comparison_column = np.where(df2["slash_all"] == df2["recp_address_domain"], True, False)
    df2["equal"] = comparison_column


    # In[8]:


    df222=df2.loc[df2['slash_all']!='local.com']
    df333 =df222.loc[df222['slash_all']!='local.co.th']
    df3=df333[df333['equal'] != True]

    def merge(data, columns_df, data_to_merge):
        df = pd.merge(data, data_to_merge, how='left', on=columns_df, indicator=True)
        df = df.drop('_merge', 1)
        return df
    def check_isin(main_data, data_check, goal_column, check_column):
        df = data_check.drop_duplicates().reset_index(drop=True)
        main_data[goal_column] = main_data[check_column].isin(df[check_column])
        main_data[goal_column] = main_data[goal_column]*1
        data = main_data[main_data.columns]
        data.reset_index(drop=True)
        return data

    df3['TLD']=df3['slash_all'].str.extract('(?P<TLD>[^.]+$)')
    df3['slash_all'] = df3['slash_all'] .str.strip()
    df3['subdomain'] =df3['slash_all'].str.extract('(?P<subdomain>^[\w\d\s]+|^[\W]+\w+)')
    df3['split']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?P<sliit>\W+)')
    df3['slash']=df3['split'].str.contains('/')
    df3['subdomain1']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?P<subdomain1>[\w\d\s]+)')
    df3['split1']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?P<split1>\W+)')
    df3['subdomain2']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?:\W+)(?P<subdomain2>[\w\s\d]+)')
    df3['split2']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?:\W+)(?:[\w\s\d]+)(?P<split2>\W+)')
    # slash_num
    df3['slash_num']=df3['slash_all'].str.count('(?P<slash_num>\d)')
    df3['slash_ch']=df3['slash_all'].str.count('(?P<slash_ch>\W)')
    df3['slash_word']=df3['slash_all'].str.count('(?P<slash_word>[a-zA-Z])')
    sum_column = df3["slash_num"] + df3["slash_word"] + df3["slash_ch"]
    df3["slash_all_score"] = sum_column
    df3['slash_w']=df3['slash_all'].str.count('(?P<slash_w>\.)')
    sum1_column = df3["slash_ch"] - df3["slash_w"]
    df3["not_dot"] = sum1_column

    # two
    df8 = df3.slash_num.unique()
    df9 = pd.DataFrame(df8, columns=['slash_num'])
    df9['two'] = 0
    df9["two"] = [("1" if i <= 3 else "2" if i == 4 else "3" if i == 5 else "4" if i == 6 else "5" if i == 7 else "6" if i == 8 else "7" if i >= 9 else 0) for i in df9["slash_num"]]
    dg = merge(df3, 'slash_num', df9)

    # subdomain-num
    regex_subdomain_num = '(?P<subdomain_num>\d)'
    regex_subdomain_word = '(?P<subdomain_word>[a-zA-Z])'
    regex_subdomain_all = '(?P<subdomain_all>\w)'

    dg['subdomain_num']  = dg['subdomain'].str.count(regex_subdomain_num)
    dg['subdomain_word'] = dg['subdomain'].str.count(regex_subdomain_word)
    dg['subdomain_all']  = dg['subdomain'].str.count(regex_subdomain_all)

    # subdomain1
    dg['subdomain1_num']  = dg['subdomain1'].str.count(regex_subdomain_num) # นับว่ามีตัวเลขมากแค่ไหน
    dg['subdomain1_word'] = dg['subdomain1'].str.count(regex_subdomain_word) # นับว่ามีตัวเลขมากแค่ไหน

    # subdomain2
    dg['subdomain2_num']  = dg['subdomain2'].str.count(regex_subdomain_num) # บว่ามีตัวเลขมากแค่ไหน

    # Split
    dg['split'] = dg['split'].str.strip()

    # Defind variable
    split_dot = '(?P<split_dot>^\.$)'
    split_count = '(?P<split_count>\W)'

    # Match & Count
    dg['split'] = dg['split'].str.strip()
    dg['split_dot'] = dg['split'].str.match(split_dot)
    dg['split_count'] = dg['split'].str.count(split_count)

    #  Third condition
    dd= dg['split_dot'].map(str)+dg['split_count'].map(str)
    dd1=pd.DataFrame(dd,columns=['split_dc'])
    dd2 = pd.concat([dg,dd1],axis =1)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    dd2['third_con']= le.fit_transform(dd2['split_dc']) #หลักการของมันคือ เรียงตามตัวอักษรในที่นี้คือ bad=0, good=1
    ds3=dd2.drop(['split_dc'],axis=1)


    # # กรณีนำข้อมูลมาจาก s3

    # In[9]:


    def key(x,y):
        ACCESS_KEY_ID = 'AKIA3JS6MC7YH7YL6CWQ'
        SECRET_ACCESS_KEY = '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'
        REGION_NAME = 'ap-southeast-1'

        session = Session(
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY
        )

        BUCKET_NAME = 'log-for-team-data'
        s3 = session.resource("s3")
        bucket = s3.Bucket(BUCKET_NAME)

        obj = bucket.Object(key=x) # example: market/zone1/data.csv
        response = obj.get()
    #     y = vaex.from_csv(response['Body'])
    #     y = pd.read_json(response['Body'], lines=True)
        y = pd.read_csv(response['Body'])
        return y


    # In[10]:


    pp = key('Domain.csv','pp')


    # In[11]:


    # Brand name word
    pp.rename(columns = {"Domain":"slash_all"}, inplace = True)
    ds3['slash_all'] = ds3['slash_all'].str.strip()
    ds3['slash_all']= ds3['slash_all'].astype('string')
    pp['slash_all']= pp['slash_all'].astype('string')
    ds3['_merge'] = ds3['slash_all'].isin(pp['slash_all'])
    ds3['_merge'] = ds3['_merge']*1
    ds3.rename(columns ={'_merge':'brand_name'}, inplace=True)
    dt = ds3[ds3.columns]
    dt.reset_index(inplace=True)


    # Random word  Detection
    dt['subdomain1_d']=dt['subdomain1']
    dt["subdomain1_d"]= dt["subdomain1_d"].astype(str)
    dt['subdomain1_d']=dt['subdomain1_d'].str.strip()
    dt[['subdomain1_d']] = dt[['subdomain1_d']].fillna('N')
    ob1 = spell_checker(dt, 'subdomain1_d')
    ob1.spell_check()
    ff = ob1.get_all_errors()
    hh = pd.DataFrame(ff, columns=['subdomain1_d'])
    dt13 = check_isin(dt, hh, 'random_sub1_d', 'subdomain1_d')

    # Dictionary_check2
    dt13['dictionary_check2'] = dt13['subdomain2']
    dt13['dictionary_check2'] = dt13['dictionary_check2'].str.replace('\d+', '')
    dt13['dictionary_check2'] = dt13['dictionary_check2'].str.strip()
    dt13[['dictionary_check2']] = dt13['dictionary_check2'].fillna('N')
    sff = spell_checker(dt13,'dictionary_check2')
    sff.spell_check()
    sfff = sff.get_all_errors()
    ppd = pd.DataFrame(sfff, columns=['dictionary_check2'])
    dt15 = check_isin(dt13, ppd, 'random_dictionary_check2', 'dictionary_check2')

    # subdomain1
    dt15['not_brand_random1'] = 0
    dt15["not_brand_random1"] = [(1 if dt15["brand_name"][i] == 0 and dt15["random_sub1_d"][i] == 0 else 0) for i in dt15.index]
    #featureเพิ่มเติม
    dt15['split_dot']=dt15['split_dot']*1


    # # Malicious sender model

    # In[12]:


    xx=dt15[['not_brand_random1','random_dictionary_check2','random_sub1_d','brand_name','third_con','split_dot','two','not_dot',

        'split_count', 'subdomain2_num','subdomain1_word','subdomain1_num','subdomain_all','subdomain_word','subdomain_num','slash_word','slash_num']]
    x_test=xx.fillna(0)


    # In[13]:


    # Catboost_Result
    import pickle
    loaded_model = pickle.load(open('Catboost_Malicious__Spam.sav', 'rb'))
    y_pred =loaded_model.predict(x_test)
    dt18 = pd.DataFrame(y_pred,columns=['Suspect_sender'])
    Prob = loaded_model.predict_proba(x_test)
    dt19 = pd.DataFrame(Prob,columns=['Prob_bad','Prob_good'])
    dt20=pd.concat([dt15,dt18,dt19], axis='columns')


    # # Message_subject analysis

    # In[14]:


    dt20['message_subject']=dt20['message_subject'].fillna('No subject')
    dt20['message_subject']=dt20['message_subject'].str.lower()
    dt21=dt20.drop(dt20[dt20.message_subject.str.contains(r'^re')].index)


    # # Blacklist word

    # In[15]:


    dt21['blacklist_word']= dt21['message_subject']
    dt21['blacklist_word']= dt21['blacklist_word'].str.count(r'price|account|access|bank|client|confirm|credit|debit|information|log|notification|password|pay|recently|risk|security|service|user|urgent')
    dt21['blacklist_word1']= dt21['message_subject']
    dt21['blacklist_word1']= dt21['blacklist_word1'].str.count(r'เสนอ|วิกฤต|โปรโมชั่น|โทร|ฟรี|ราคาพิเศษ|ขอเอกสาร|รหัส|หารายได้เสริม|ขอข้อมูล|พิเศษเฉพาะคุณ|ด่วน|กรุณาตอบกลับ|รบกวนตอบกลับ|ยืนยัน|ส่งเอกสารยืนยัน')


    # In[39]:


    dt22 =dt21.reset_index()
    dt23=dt22.drop(['level_0','index'],axis=1)
    dt23


    # # BERT

    # ### Tokenization

    # In[40]:


    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    max_seq_len = 30
    tokens_test1 = tokenizer.batch_encode_plus(dt23.message_subject.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False)
    test_seq = torch.tensor(tokens_test1['input_ids'])
    test_mask = torch.tensor(tokens_test1['attention_mask'])


    # ### Get Predictions for Test Data

    # In[41]:


    class BERT_Arch(nn.Module):

        def __init__(self, bert):

          super(BERT_Arch, self).__init__()

          self.bert = bert

          # dropout layer
          self.dropout = nn.Dropout(0.1)

          # relu activation function
          self.relu =  nn.ReLU()

          # dense layer 1
          self.fc1 = nn.Linear(768,512)

          # dense layer 2 (Output layer)
          self.fc2 = nn.Linear(512,2)

          #softmax activation function
          self.softmax = nn.LogSoftmax(dim=1)

        #define the forward pass
        def forward(self, sent_id, mask):

          #pass the inputs to the model
          _, cls_hs = self.bert(sent_id, attention_mask=mask)

          x = self.fc1(cls_hs)

          x = self.relu(x)

          x = self.dropout(x)

          # output layer
          x = self.fc2(x)

          # apply softmax activation
          x = self.softmax(x)

          return x


    # In[42]:


    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    model = model.to(device)


    # ### import model

    # In[43]:


    # model_save_name = 'saved_weights.pt'
    path ='saved_weights_batch_2.pt'
    model.load_state_dict(torch.load(path))

    #กรณีเก็บไฟล์ไว้ที่อื่น
    # model_save_name = 'saved_weights.pt'
    # path = F"s3://log-for-team-data/{model_save_name}"
    # model.load_state_dict(torch.load(path))


    # In[44]:


    # get predictions for test data
    with torch.no_grad():
        prediction = model(test_seq.to(device), test_mask.to(device))
        prediction = prediction.detach().cpu().numpy()
        Suspect_message_subject = np.argmax(prediction, axis = 1)

    # In[45]:


    dt24= pd.DataFrame(prediction,columns=['Prob_0','Prob_1'])
    dt25 = pd.DataFrame(Suspect_message_subject,columns=['Suspect_message_subject'])


    # In[47]:


    frame= [dt23,dt24,dt25]
    dt26 = pd.concat(frame,axis=1)
    dt26.shape


    # # final result

    # In[26]:


    #กรณีที่มันไม่สามารถที่จะ predict subject ได้ คำตอบในช่อง Suspect_message_subjec จะออกมาเป็น NAN


    # In[49]:


    dt27 =dt26[['sender_address','recp_address','return_path','message_subject','Suspect_sender','Prob_bad','Prob_good','Suspect_message_subject','Prob_0','Prob_1','event_name','directionality','recp_count','blacklist_word','blacklist_word1']]
    print(dt27)


    # # Save data to s3

    # In[50]:


    import boto3
    import os
    from botocore.exceptions import NoCredentialsError

    '''ACCESS_KEY = 'AKIA3JS6MC7YH7YL6CWQ'
    SECRET_KEY = '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'
    bucket = 'log-for-team-data'  # Use the default bucket name'''


    '''def upload_to_aws(local_file, bucket, key):
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_KEY)
       
        try:
            with open(local_file,'rb') as f:
        # The following code uploads the data into S3 bucket to be accessed later for training
        # prefix = folder , key = body_file
                boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(key)).upload_fileobj(f)
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False
    
    
    # In[51]:
    
    '''
    #save
    dt27.to_csv('dt27.csv', index = False)
    # Upload to S3
    '''uploaded = upload_to_aws('dt27.csv', 'log-for-team-data','dt27.csv')
    
    #remove csv from jupyter notebook
    os.remove("dt27.csv")'''
    return
print(spam_email())



