import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import scikitplot.plotters as skplt

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

train = pd.read_csv('data/training_variants')
test = pd.read_csv('data/stage2_test_variants.csv')
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


num_words = 2000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train['Text'].values)

X = tokenizer.texts_to_sequences(train['Text'].values)
X = pad_sequences(X, maxlen=2000)

embed_dim = 128

ckpt_callback = ModelCheckpoint('lstm_model_1',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

model = Sequential()
model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
model.add(LSTM(196, recurrent_dropout=0.5, dropout=0.5,return_sequences=True))
model.add(LSTM(196, recurrent_dropout=0.5, dropout=0.5))
model.add(Dense(9,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
print(model.summary())

Y = pd.get_dummies(train['Class']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
batch_size = 64
model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])

model = load_model('lstm_model_1')

probas = model.predict(X_test)
pred_indices = np.argmax(probas, axis=1)
classes = np.array(range(1, 10))
preds = classes[pred_indices]
print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))
skplt.plot_confusion_matrix(classes[np.argmax(Y_test, axis=1)], preds)


Xtest = tokenizer.texts_to_sequences(testx['Text'].values)
Xtest = pad_sequences(Xtest, maxlen=2000)

probas = model.predict(Xtest)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

#submission_df.to_csv('submission1.csv', index=False)
