import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('data/rotten_tomato_train.csv')
data_test = pd.read_csv('data/rotten_tomato_test.csv')

data_train = data_train[data_train['Sentiment']!=2]
data_train.loc[data_train['Sentiment']==1,'Sentiment'] = 0
data_train.loc[data_train['Sentiment'].isin([3,4]),'Sentiment'] = 1


df_bert = pd.DataFrame({
    'id': range(len(data_train)),
    'label': data_train['Sentiment'],
    'alpha': ['a']*data_train.shape[0],
    'text': data_train['Phrase'].replace(r'\n', ' ', regex=True)
})

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.1)

df_bert_test = pd.DataFrame({
    'id': range(len(data_test)),
    'text': data_test['Phrase'].replace(r'\n', ' ', regex=True)
})

df_bert_train.to_csv('data/two_cls/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/two_cls/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/two_cls/test.tsv', sep='\t', index=False, header=True)
