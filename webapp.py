import streamlit as st
import base64
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import predict, vectorise
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
import torch


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


# load labels 
with open('./model/labels.txt', 'r') as f:
    labels = [a for a in f.readlines()]
    f.close()

# load model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = BertModel.from_pretrained('roberta-base')
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
model.load_state_dict(torch.load('./model/model.bin'),map_location=torch.device('cpu'))
#model.to('cpu')
tokenizer = BertTokenizer.from_pretrained('roberta-base')
# set background
set_page_bg('bg.png')

st.title('Genre classifier')

# text input
lyrics = st.text_input("Enter lyrics:")


# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = lyrics.title()
    #vectorised_text = vectorise(result, tf)
    #label = predict(vectorised_text, model, labels)
    # y_pred = predict(vectorised_text, model, labels)

    label = predict(result, model, tokenizer)
    st.text(f'class: {label}')

