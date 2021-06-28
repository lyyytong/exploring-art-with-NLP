import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

########## Functions to preprocess inputs ##########
# vocab_size = 36347
# embedding_dim = 100
max_length = 200
trunc_type = 'pre'
padding_type = 'pre'

# Load tokenizer 
with open('/Users/lysmacbookpro/Desktop/CS Work/W11-12 Final Project/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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

def preprocess_text(texts):
    inputs = texts.lower()
    inputs = lemm_text(inputs)
    inputs = stemm_text(inputs)
    sequence = tokenizer.texts_to_sequences([inputs])
    padded_sequence = pad_sequences(sequence,
                                    maxlen=max_length,
                                    padding=padding_type,
                                    truncating=trunc_type)
    return padded_sequence