from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Bidirectional
from tensorflow.keras import Sequential
from custom_metric import F1_score

########## Load sentiment-analysis model ##########
filepath = '/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/Models/9_class_Stem&Lemm_BiLSTM_4.h5'
bi_lstm = models.load_model(filepath, custom_objects={'F1_score':F1_score})

bi_lstm.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[F1_score()])

labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
          'excitement', 'fear', 'sadness', 'something else']