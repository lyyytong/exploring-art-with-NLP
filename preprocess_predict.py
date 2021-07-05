import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Custom F1 metric
def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred>=0.5, 1., 0.)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function

class F1_score(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight('all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight('all_possible_positives', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)

    def result(self):
        precision = self.tp_count / self.all_predicted_positives
        recall = self.tp_count / self.all_possible_positives
        f1 = 2*(precision*recall)/(precision+recall)
        return f1

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
bi_lstm = tf.keras.models.load_model("bi_lstm.h5",
                                     custom_objects={'F1_score':F1_score})

labels = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
          'excitement', 'fear', 'sadness', 'something else']

def predict(texts):
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
    