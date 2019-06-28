import os

import pickle
import json

import re
import html
import emoji

from wiseling import (
    remove_dup_chars,
    remove_dup_spaces,
    insert_spaces,
    tokenize,
    remove_stopwords,
)
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorizer
IDF_PATH = os.path.dirname(os.path.realpath(__file__)) + "/vectorizer/idf.pickle"
VOCAB_PATH = os.path.dirname(os.path.realpath(__file__)) + "/vectorizer/vocab.json"

# Logistic Regression Model
LR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/model/lr.pickle"

class SentimentAnalyser:
    def __init__(self):
        self.__vector_func = TfidfVectorizer(
            preprocessor=lambda x: x, tokenizer=lambda x:x, max_features=10000
        )
        self.__vector_func.vocabulary_ = json.load(open(VOCAB_PATH, "rb"))
        self.__vector_func.idf_ = pickle.load(open(IDF_PATH, "rb"))

        self.__model = pickle.load(open(LR_PATH, 'rb'))

    def __extract_emoji(self, message: str) -> list:
        return emoji.get_emoji_regexp().findall(message)
    def __extract_hashtag(self, message: str) -> list:
        return re.findall(r'(#[^\s]+)(?:\Z|\s)', message)

    def __preprocess(self, message: str) -> str:
        thai_numbers = {'๐': '0','๑': '1', '๒': '2', '๓': '3', '๔': '4', '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'}
        message = str(message)
        for word, initial in thai_numbers.items():
            message = message.replace(word, initial)
        message = re.sub(r'http\S+|www\S+', "", message.lower())
        message = re.sub(r"([a-zA-Z0-9_.+-""]*@[a-zA-Z0-9-]+[\.[a-zA-Z0-9-.]+]?)", "", message)
        message = emoji.get_emoji_regexp().sub("", message)
        message = re.sub(r"(#[^\s]+)(?:\Z|\s)", "", message)
        message = re.sub(r"\+[0-9\+]+", " <wsplus> ", message)
        message = re.sub(r'(?:\s|\D|\A)(5{2,}[46]*5*\+*)', ' <wslaugh> ', message)
        message = html.unescape(str(message))
        message = insert_spaces(str(message))
        message = remove_dup_chars(message)
        message = re.sub(r'([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^_\`\{\|\}\~\฿\“\”])\1+', r'\1', message)
        message = re.sub(r'([\"\#\$\%\&\'\(\)\*\+\,\.\-\/\:\;\<\=\>\@\[\\\]\^_\`\{\|\}\~\฿\“\”])*', "", message)
        message = re.sub(r'^\s+', "", message)
        message = re.sub(r'\d+', "", message)
        message = remove_dup_spaces(message)
        
        return message

    def sentiment(self, message: str) -> str:
        message = str(message)
        emo = self.__extract_emoji(message)
        emo = [emoji.demojize(e) for e in emo]
        hashtag = self.__extract_hashtag(message)
        message = self.__preprocess(message)
        tokens = word_tokenize(message, engine="newmm")
        tokens = [word for word in tokens if len(word) > 1]
        tokens = tokens + emo + hashtag
        X = self.__vector_func.transform([tokens])
        y = self.__model.predict(X)
        
        return y[0]
