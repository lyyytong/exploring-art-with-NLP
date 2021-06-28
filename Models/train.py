import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras import models
from custom_metric import F1_score
from text_processing import lemm_text, stemm_text

########### Import dataframe ##########
df  = read_csv('/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/Data/Artemis Official Data/artemis_dataset_release_v0.csv')

# Make texts lowercase
df['utterance']=df['utterance'].str.lower()

########### Lemmitize: returns the word to its lemma. E.g "is" becomes "be".
df['utterance'] = list(map(lemm_text, df['utterance']))

########### Stemming: reduces the word to its stem. E.g "cooking" becomes "cook" and "played" becomes "play".
df['utterance'] = list(map(stemm_text, df['utterance']))

########### Train-test split dataset ###########
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_data['utterance'].values
y_train = train_data['emotion'].values
X_test = test_data['utterance'].values
y_test = test_data['emotion'].values

########### Tokenize & pad data ###########
vocab_size = 36347
embedding_dim = 100
max_length = 200
trunc_type = 'pre'
padding_type = 'pre'
oov_tok = '<OOV>'
pad_tok = '<PAD>'

# For training data
# Create a tokenizer with num_words and OOV_Token attributes
tokenizer = Tokenizer(num_words=vocab_size+2, oov_token=oov_tok, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n') # remove ! and ? from filter

# Use that tokenizer to fit the training sentences we got above
tokenizer.fit_on_texts(X_train)

# Use the tokenizer we have fitted on the training sentences and create encoded sequences of index of training sentences
training_sequences = tokenizer.texts_to_sequences(X_train)

# Pad the sequences for them to be on the same length with max_len and truncating attributes
training_padded = pad_sequences(training_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

# For testing data
# Use the already fitted tokenizer above and create encoded sequences of index of test sentences
testing_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to be on the same length as max_len
testing_padded = pad_sequences(testing_sequences,
                               maxlen=max_length,
                               padding=padding_type,
                               truncating=trunc_type)

########## Save tokenizer for later use ##########
with open('/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

########### One-hot encode labels ###########

label_encoder = LabelEncoder()

# Training labels
training_labels = label_encoder.fit_transform(y_train)
training_labels = to_categorical(training_labels)

# Testing labels
testing_labels = label_encoder.transform(y_test)
testing_labels = to_categorical(testing_labels)

########### Build model ###########
model = Sequential([
        Embedding(vocab_size+2, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(128, dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(9, activation='softmax')
])

filepath = '/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/Models/'
checkpoint_callback = callbacks.ModelCheckpoint(filepath=filepath+'9_class_Stem&Lemm_BiLSTM_{epoch}.h5',
                                                save_weights_only=False,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[F1_score()])

########### Train model ###########
num_epochs = 10
history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    batch_size=32,
                    callbacks=[checkpoint_callback],
                    validation_data=(testing_padded, testing_labels))

########### Evaluate model ###########
##### Load, compile, evaluate best model from checkpoint callback
# model = models.load_model('/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/Models/9_class_Stem&Lemm_BiLSTM_4.h5',
#                           custom_objects={'F1_score': F1_score})

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=[F1_score()])

# model.evaluate(testing_padded, testing_labels)

##### Test text to sentiment
# def test_model(model, sentence):
#     text = lemm_text(sentence)
#     text = stemm_text(text)
#     sequence_test = tokenizer.texts_to_sequences([text])
#     padded_test = pad_sequences(sequence_test,
#                                 maxlen=max_length,
#                                 padding=padding_type,
#                                 truncating=trunc_type)
#     pred = model.predict(padded_test)

#     for pct, index in zip(pred[0], range(len(pred[0]))):
#         print(f'{label_encoder.inverse_transform([index])[0]}: {"%.2f%%"%pct}')

##### Confusion matrix
# from sklearn.metrics import confusion_matrix

# y_test_pred = model.predict(testing_padded)
# y_test_pred = [np.argmax(y, axis=None, out=None) for y in y_test_pred]
# y_test_pred = [label_encoder.inverse_transform([y])[0] for y in y_test_pred]

# conf_mat = confusion_matrix(y_test, y_test_pred)
# # gives
# # array([[ 4964,     0,   643,  1775,   257,   489,   307,   209,   354],
# #        [   78,   137,    28,    43,   382,    11,   337,   120,   149],
# #        [  566,     0,  7651,  4596,    73,   586,   443,   327,   345],
# #        [  891,     0,  2716, 19738,    80,   620,   267,   449,   505],
# #        [  409,     7,    84,   167,  2386,    22,   635,   368,   447],
# #        [  551,     0,  1123,  2162,    29,  3132,   232,    88,   183],
# #        [  219,     4,   272,   185,   314,    62,  6292,   730,   210],
# #        [  191,     4,   226,   517,   351,    34,  1009,  7232,   363],
# #        [  638,     1,   710,  1997,   561,   220,   546,   646,  5242]])