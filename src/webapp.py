import sys
import os
import streamlit as st
import base64
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F
model_name = 'roberta-base'
from clean_text import text_preprocessing_pipeline
sys.path.insert(0, '../src')
from utils import predict



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
with open(os.getcwd() + '/model/labels.txt', 'r') as f:
    labels = [a for a in f.readlines()]
    f.close()

# load model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l1 = torch.nn.Linear(768, 256)
        self.fc = torch.nn.Linear(768,5)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        #features = F.relu(self.l1(features))
        # output_2 = self.l2(output_1)
        output = F.softmax(self.fc(features), dim=1)
        return output

model = BERTClass()
model.load_state_dict(torch.load(os.getcwd() + '/model/model4.bin'))
#model.to('cpu')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# set background
set_page_bg(os.getcwd() + '/img/bg.png')

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
    st.markdown(f'<p style="color: black; font-size: 30px;">{labels[label]}</p>', unsafe_allow_html=True)
    st.balloons()
