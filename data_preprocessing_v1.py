import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_reviews(fpath, label):
    reviews = defaultdict(list)
    with open(fpath, 'rb') as f:
        for line in f:
            reviews['Phrase'].append(line.decode('latin1').strip())
            reviews['Sentiment'].append(label)
    df = pd.DataFrame(reviews)
    return df


neg_reviews = load_reviews('data/rt-polaritydata/rt-polaritydata/rt-polarity.neg',0)
pos_reviews = load_reviews('data/rt-polaritydata/rt-polaritydata/rt-polarity.pos',1)

train = pd.concat([neg_reviews, pos_reviews])
data_train = train.sample(frac=1, replace=False).reset_index(drop=True)
data_test = train.copy()

# data_train = pd.read_csv('data/rotten_tomato_train.csv')
# data_test = pd.read_csv('data/rotten_tomato_test.csv')


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

df_bert_train.to_csv('data/train_2cls.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev_2cls.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test_2cls.tsv', sep='\t', index=False, header=True)
