import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import torch
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word not in stop_words and word not in string.punctuation and len(word)]
    return [' '.join(text)]

def vectorise(text, tf):
    text = preprocess_text(text)
    #print(text)
    tfidf = tf
    vec_text = tfidf.transform(text)
    return vec_text

'''def predict(text, model, labels):
    y_pred = model.predict(text)
    label = labels[int(y_pred[0])]
    return label '''


def predict(text, model,tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=200,
        padding='max_length',
        return_token_type_ids=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0)
    outputs = model(ids, mask, token_type_ids)
    outputs = torch.sigmoid(outputs).detach().numpy()
    return outputs