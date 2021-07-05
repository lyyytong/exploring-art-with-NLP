from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

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