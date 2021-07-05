from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from custom_metric import F1_score
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 200
trunc_type = 'pre'
padding_type = 'pre'

# Lemmitization & stemming functions
def lemm_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def stemm_text(text):
    tokens = word_tokenize(text)
    p_stemmer = PorterStemmer()
    tokens = [p_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load model & function to predict
bi_lstm = models.load_model("bi_lstm.h5",
          custom_objects={'F1_score':F1_score},
          compile=True)

labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
          'excitement', 'fear', 'sadness', 'something else']

def preprocess_predict(texts):
    inputs = texts.lower()
    inputs = lemm_text(inputs)
    inputs = stemm_text(inputs)
    sequence = tokenizer.texts_to_sequences([inputs])
    padded_sequence = pad_sequences(sequence,
                                    maxlen=max_length,
                                    padding=padding_type,
                                    truncating=trunc_type)

    predictions = bi_lstm.predict(padded_sequence)
    return predictions
    