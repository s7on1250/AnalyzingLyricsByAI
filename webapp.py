import streamlit as st
import base64
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import predict, vectorise
from sklearn.linear_model import LogisticRegression
from transformers import RobertaModel, RobertaTokenizer
import torch
from bs4 import BeautifulSoup
import re
import numpy as np
import string

# background
def set_page_bg(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    return


# preprocess text
punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
         '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥️', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
         '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
         '▒', '：', '¼', '⊕', '▼', '▪️', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦️', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
         '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤️', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}


def clean_text(text):
    '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = re.sub(r'\:(.*?)\:','',text)
    text = str(text).lower()    #Making Text Lowercase
    text = re.sub('\[.*?\]', '', text)
    #The next 2 lines remove html text
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text


def clean_contractions(text, mapping):
    '''Clean contraction using contraction mapping'''
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if ""+word+"" in text:
            text = text.replace(""+word+"", ""+mapping[word]+"")
    #Remove Punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text


def clean_special_chars(text, punct, mapping):
    '''Cleans special characters present(if any)'''
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def remove_space(text):
    '''Removes awkward spaces'''
    #Removes awkward spaces
    text = text.strip()
    text = text.split()
    return " ".join(text)


def text_preprocessing_pipeline(text):
    '''Cleaning and parsing the text.'''
    text = clean_text(text)
    text = clean_special_chars(text, punct, punct_mapping)
    text = remove_space(text)
    return text


# load labels 
with open('./model/labels.txt', 'r') as f:
    labels = [a for a in f.readlines()]
    f.close()

# load model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # self.l2 = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768,5)
    
    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        # output_2 = self.l2(output_1)
        output = self.fc(features)
        return output
#tf = pickle.load(open('./model/tfidf.pickle', "rb"))
#model = pickle.load(open('./model/log.pickle', "rb"))

model = BERTClass()
model.load_state_dict(torch.load('./model/model.bin'))
#model.to('cpu')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# set background
set_page_bg('bg.png')

st.title('Genre classifier')

# text input
lyrics = st.text_input("Enter lyrics:")


# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = text_preprocessing_pipeline(lyrics.title())
    #vectorised_text = vectorise(result, tf)
    #label = predict(vectorised_text, model, labels)
    # y_pred = predict(vectorised_text, model, labels)

    label = predict(result, model, tokenizer)
    # chose the class with the highest probability
    # with black text color
    st.markdown(f'<p style="color: black; font-size: 30px;">{labels[np.argmax(label)]}</p>', unsafe_allow_html=True)
    st.balloons()

