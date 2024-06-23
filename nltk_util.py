import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokennize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    """pass
    sentence = ["hello","how","are","you"]
    words   = ["hi","hello","i","you","bye","thank","you"]
    bog     = [ 0   ,   1 ,  0  , 1  ,  0   ,   0   , 0  ]
    """
    sentence_words = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


