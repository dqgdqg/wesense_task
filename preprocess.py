import pandas as pd
import os
from IPython import embed
import numpy as np

np.random.seed(0)

def xlsx_to_df(root, name):
    xlsx_path = os.path.join(root, name)
    df = pd.read_excel(xlsx_path, index_col=0, engine='openpyxl')

    df.index = df.index.astype(int)

    sheet_name = name.split('_')[0]
    df['sheet_name'] = sheet_name
    
    columns = df.columns.to_list()
    for i in range(len(columns)-3):
        columns[i] = '_'.join(columns[i].split('_')[1:])

    df.columns = columns

    return df

df_list = []

for root, dirs, files in os.walk("./data", topdown=False):
    for name in files:
        if name.endswith('.xlsx'):
            df = xlsx_to_df(root, name)
            df_list.append(df)

df = pd.concat(df_list, ignore_index=True).fillna(0)

# normalization

for col in df.columns[:26]:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# get labels

labels = list(set(df['CONDITION'].to_list()))
df['label'] = 0
for i in range(len(df['CONDITION'])):
    df['label'][i] = labels.index(df['CONDITION'][i])

# get train/valid/test split

train, valid, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

# Statistic on label distribution

for df in [train, valid, test]:
    label_cnt = []
    for label in labels:
        label_cnt.append((df['CONDITION'] == label).sum())
    print(label_cnt)

# save csv

train = train.to_csv('./data/train.csv')
valid = valid.to_csv('./data/valid.csv')
test = test.to_csv('./data/test.csv')

