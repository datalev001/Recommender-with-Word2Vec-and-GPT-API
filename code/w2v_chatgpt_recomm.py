import nltk
import numpy as np
import pandas as pd
nltk.download('punkt')  # download the tokenizer
import os
import datetime
from datetime import date, datetime, timedelta
import openai
import ast
import json
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# download data from https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
tran_df = pd.read_excel('online_retail_II.xlsx')

# define rules and clean data 1:
c1 = (tran_df['Invoice'].isnull() == False)
c2 = (tran_df['Quantity']>0)
c3 = (tran_df['Customer ID'].isnull() == False)
c4 = (tran_df['StockCode'].isnull() == False)
c5 = (tran_df['Description'].isnull() == False)
tran_df = tran_df[c1 & c2 & c3 & c4 & c5]

# define rules and clean data 2:
grp = ['Invoice', 'StockCode','Description', 'Quantity', 'InvoiceDate']
tran_df = tran_df.drop_duplicates(grp)
tran_df['transaction_date'] = tran_df['InvoiceDate'].dt.date

# define rules and clean data 3:
pro_lst = list(set(cats_top['index']))
tran_df_sel = tran_df[tran_df['Description'].isin(pro_lst)]

# convert 'trans_date' to date type
tran_df_sel['trans_date'] = pd.to_datetime(tran_df_sel['transaction_date'], format = '%Y-%m-%d')

# Create a summary dataframe for the BG/NBD model
summary_data = summary_data_from_transaction_data\
    (tran_df_sel, 'Customer ID', 
    'trans_date', 
    observation_period_end='2010-05-01')

# Fit the BG/NBD model
p = 0.3
bgf = BetaGeoFitter(penalizer_coef = p)
bgf.fit(frequency = summary_data['frequency'],
        recency = summary_data['recency'], 
        T = summary_data['T'])

# Predict the probability of a customer buying in the next 30 days
summary_data['predicted_purchase_30_days'] =\
    bgf.conditional_expected_number_of_purchases_up_to_time\
    (30, summary_data['frequency'], 
    summary_data['recency'], summary_data['T'])

# Create a data frame with 'customer_id' and 'score'
customer_scores = summary_data.reset_index()
customer_scores = customer_scores[['Customer ID', 'predicted_purchase_30_days']]

# Rename the column to 'score' as you requested
customer_scores.rename(columns = {'predicted_purchase_30_days': 'score'}, inplace=True)
    
customer_scores = customer_scores.sort_values(['score'], ascending = False)

# Print the customer scores
print(customer_scores)
customer_scores.score.quantile([0.05*j for j in range(20)])

customer_scores_top = customer_scores[customer_scores.score>0.12]
customer_scores_lst = set(customer_scores_top['Customer ID'])

tran_df_cand = tran_df_sel[tran_df_sel['Customer ID'].isin(customer_scores_lst)]
    
prodtc_desc = list(set(tran_df_cand.Description))
sm = [f'{i:03}' for i in range(len(prodtc_desc))]
names = ['product_' + str(j) for j in sm]
nm_dict = dict(zip(prodtc_desc, names))

vtran_df_cand_sum = tran_df_cand.groupby(['Customer ID', 'Description'])['Quantity'].sum().reset_index()
   
vtran_df_cand_sum['product'] = vtran_df_cand_sum['Description'].map(nm_dict)

vtran_df_cand_sum_pv = vtran_df_cand_sum.pivot(index='Customer ID',\
        columns='product', values='Quantity').reset_index()

vtran_df_cand_sum_pv = vtran_df_cand_sum_pv.fillna(0.0)

cccc = ['Customer ID', 'product_000', 'product_001', 'product_002','product_003',
        'product_004', 'product_005', 'product_006']

cols = list(vtran_df_cand_sum_pv.columns)
cols.remove('Customer ID')
vtran_df_cand_sum_pv[cols] = vtran_df_cand_sum_pv[cols] / vtran_df_cand_sum_pv[cols].max()

avg_q = vtran_df_cand_sum_pv[cols].mean()

# Initialize a new DataFrame 'D1' with the same structure as 'D'
res_df = vtran_df_cand_sum_pv.copy()

# Iterate through each column in 'D'
for column in cols:
    # Calculate the median for the current column
    avgq = avg_q[column]
    
    # Update the values in 'D1' based on your rules
    res_df[column] = res_df[column].apply(lambda x: 2 if x > avgq else (1 if x > 0 else 0))

res_df.to_excel('wv_rec.xlsx',index = False)

res_df['sumv'] = res_df[cols].sum(axis = 1)

def generate_desc(row):
    desc = []
    for col in cols:  # Exclude 'customer_id'
        if row[col] != 0:
            desc.append(f'{col} = {row[col]}')
    return ' '.join(desc)

res_df_trans = res_df[['Customer ID']].copy()
res_df_trans['desc'] = res_df.apply(generate_desc, axis=1)

def get_embedding(text):
   result = openai.Embedding.create(input=text, model="text-embedding-ada-002")
   result_text = np.array(result['data'][0]['embedding'])
   return result_text

res_df_trans['ada_embedding'] = res_df_trans['desc'].apply(get_embedding)

res_df_new = res_df_trans.copy()
res_df_new['ada_embedding'] = res_df_new['ada_embedding'].apply(lambda x: x.tolist())
res_df_new['ada_embedding'] = res_df_new['ada_embedding'].apply(json.dumps)
res_df_new.to_csv('wv_rec_new.csv', index=False)

res_df_clus = res_df_trans.copy()
v_array = np.array(res_df_clus['ada_embedding'].to_list())

# Create a DataFrame with separate columns for each element in the list
v_df = pd.DataFrame(v_array, columns=[f'v_{i}' for i in range(v_array.shape[1])])

# Concatenate the 'customer_id' and 'v' DataFrames
res_df_clus = pd.concat([res_df_clus['Customer ID'], v_df], axis=1)

# Perform K-means clustering
k = 15 # You can choose the number of clusters
kmeans = KMeans(n_clusters=k)
res_df_clus['cluster'] = kmeans.fit_predict(v_df)

# Now, the 'df' DataFrame contains the customer IDs and their assigned clusters
print(res_df_clus[['Customer ID', 'cluster']])
res_df_clus = res_df_clus[['Customer ID', 'cluster']]
A = res_df_clus['cluster'].value_counts().reset_index()
A.columns = ['segment_name', 'count']

res_df1 = pd.merge(res_df, res_df_clus, on = ['Customer ID'], how = 'inner')
buy_df_clus = res_df1.groupby('cluster')[cols].mean().reset_index()
nms = [it+ '_c' for it in cols]
buy_df_clus.columns = ['cluster'] + nms
res_df1 = pd.merge(res_df1, buy_df_clus, on = ['cluster'], how = 'inner')

p_columns = nms[:]  # Exclude 'customer_id'

# Function to find the top N column names for each row
def find_top_N_cols(row, N):
    sorted_cols = sorted(p_columns, key=lambda col: row[col], reverse=True)
    truncated_cols = [col[:-2] for col in sorted_cols]
    return truncated_cols[:N]

# Find the top 5 columns for each row and store them in new columns
res_df1['top_1'] = res_df1.apply(lambda row: find_top_N_cols(row, 1)[0], axis=1)
res_df1['top_2'] = res_df1.apply(lambda row: find_top_N_cols(row, 2)[1], axis=1)
res_df1['top_3'] = res_df1.apply(lambda row: find_top_N_cols(row, 3)[2], axis=1)
res_df1['top_4'] = res_df1.apply(lambda row: find_top_N_cols(row, 4)[3], axis=1)
res_df1['top_5'] = res_df1.apply(lambda row: find_top_N_cols(row, 5)[4], axis=1)

# Display the updated DataFrame
res_df3 = res_df1.drop_duplicates(['cluster'])

def create_recommend_list(row):
    recommend = []
    for col in cols:
        if row[col] == 0 and any(col in row['top_{}'.format(i)] for i in range(1, 6)):
            recommend.append(col)
    return recommend

# Apply the function to each row and create the 'recommend' column
res_df1['recommend'] = res_df1.apply(create_recommend_list, axis=1)
reco_df = res_df1[['Customer ID','cluster', 'recommend']]

res_df_reco = res_df1[['Customer ID','cluster', 'recommend'] + cols +\
                      ['top_1', 'top_2', 'top_3', 'top_4', 'top_5']]

res_df_reco.to_excel('reco.xlsx', index=False)    

clus_df_top = res_df3[['cluster', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5']]

exchanged_dict = {value: key for key, value in nm_dict.items()}

# Define a function to map keys to values
def map_to_values(key_column):
    return key_column.map(exchanged_dict)

# Create the new columns
clus_df_top['top_v1'] = map_to_values(clus_df_top['top_1'])
clus_df_top['top_v2'] = map_to_values(clus_df_top['top_2'])
clus_df_top['top_v3'] = map_to_values(clus_df_top['top_3'])
clus_df_top['top_v4'] = map_to_values(clus_df_top['top_4'])
clus_df_top['top_v5'] = map_to_values(clus_df_top['top_5'])

def concat_values(row):
    return ', '.join(row[6:11])  

# Create the 'products_lst' column
clus_df_top['products_lst'] = clus_df_top.apply(concat_values, axis=1)
clus_df_top.to_excel('clus_df_top.xlsx', index=False)   

notice = ', do not randomly fabricate unless you 80% know, \
          otherwise say I do not know'

def sentence_to_vector(sentence):
    response = openai.Completion.create(model = 'text-davinci-003', 
                                    prompt = sentence ,
                                    max_tokens = 300)

    result = response['choices'][0]['text']
    return result

for j in range(len(clus_df_top)):
    cluster = clus_df_top.iloc[j]['cluster']
    find_text = clus_df_top.iloc[j]['products_lst']
    print ('------------------------------------------')
    print ('Segment:' +  str(cluster))
    request0 = ' Here are the products that appears in in one customers segment: ' + find_text
    request1 = '. These customers should have some similar attributes \
                such as demongraphics or life styles, please use these products \
                they have often purchased and combine with the knowledge in \
                retail business and marketing to describe \
                the segment, specifically you need introduce the attributes of the customers \
                in this segment. Also, please explain the rationale behind buying these products \
                these products together. Note the background of this request is that \
                I want to use the segment to create customer product recommander, \
                such as collaborative filter or content-based filter ' 
    request2 = 'Here is the additional knowledge you might need to answer my question: Customers in \
               this segment might share similar attributes, \
               such as demographics or lifestyles. For example, they could be\
               individuals who appreciate home decor and have a taste for \
               unique and cozy items. They may value products that add a touch \
               of elegance and warmth to their living spaces. These customers \
               might be more likely to be homeowners or individuals who take \
               pride in decorating their homes. '
               
    request = request0 + request1 + request2 + notice           
    #print (request)            
    final_answer_chatbot = sentence_to_vector(request)
    print (final_answer_chatbot)

    
    




