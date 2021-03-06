import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


train = pd.read_csv('./data/training_variants')
test = pd.read_csv('./data/stage2_test_variants.csv')
trainx = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('data/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
train = train.drop(["ID"], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
df_variants_test = pd.read_csv('data/test_variants', usecols=['ID', 'Gene', 'Variation'])
df_text_test = pd.read_csv('data/test_text', sep='\|\|', engine='python',
                           skiprows=1, names=['ID', 'Text'])
df_variants_test['Text'] = df_text_test['Text']
df_test = df_variants_test

df_labels_test = pd.read_csv('data/stage1_solution_filtered.csv')
df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])

df_test = df_test.merge(df_labels_test[['ID', 'Class']], on='ID', how='left').drop('ID', axis=1)
df_test = df_test[df_test['Class'].notnull()]

df_stage_2_train = pd.concat([train, df_test])
df_stage_2_train.reset_index(drop=True, inplace=True)
train = df_stage_2_train


print(testx.head())

num_words = 2000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train['Text'].values)

model = load_model('lstm_model_3')
Xtest = tokenizer.texts_to_sequences(testx['Text'].values)
Xtest = pad_sequences(Xtest, maxlen=2000)
probas = model.predict(Xtest)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = testx['ID']
submission_df.head()

submission_df.to_csv('submission.csv', index=False)
