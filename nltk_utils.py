import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

stemmer = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def tokenize(sentence):
    return tokenizer.tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
