import streamlit as st
import base64
from utils import *
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F
model_name = 'roberta-base'
from clean_text import text_preprocessing_pipeline
from annotated_text import annotated_text
import nltk 
from nltk.tokenize import word_tokenize

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

#tf = pickle.load(open('./model/tfidf.pickle', "rb"))
#model = pickle.load(open('./model/log.pickle', "rb"))

model = BERTClass()
model.load_state_dict(torch.load('./model/model4.bin', map_location=torch.device('cpu')))
#model.to('cpu')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# set background
set_page_bg('bg.png')

st.title('Genre classifier')

# text input
lyrics = st.text_area("Enter lyrics:", height=500)

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
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<p style="color: black; font-size: 30px;">{"Genre: "}{labels[label]}</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p style="color: black; font-size: 30px;">{"Possible title: "}{12345}</p>', unsafe_allow_html=True)
    st.balloons()

    label_str = labels[label]
    rap_list = ['nigga', 'money', 'ass']
    rock_list = ['music']
    pop_list = ['love']
    metal_list = ['death']
    rb_list = ['rb']
    text_res = []
    # text_split = word_tokenize(lyrics)
    custom_tokenizer = nltk.tokenize.RegexpTokenizer(r'\s+|\n|[\.,!\?;:\(\)]|\w+')
    text_split = custom_tokenizer.tokenize(lyrics)
    print(text_split)
    print("genre: ", label_str)
    for word in text_split:
        print("word: ", word.lower() in rap_list, label_str=='rap\n')
        if word.lower() in rap_list and label_str == 'rap\n' or \
           word.lower() in metal_list and label_str == 'metal\n' or \
           word.lower() in rock_list and label_str == 'rock\n' or \
           word.lower() in pop_list and label_str == 'pop\n' or \
           word.lower() in rb_list and label_str == 'rb':
            tup = (word, labels[label][:-1])
            print(tup)
            text_res.append(tup)
        else:
            text_res.append(word+' ')
    # text_final = ' '.join(text_res)
    print(text_res)

    st.subheader("Key words of this genre: ")
    sentence = []
    lines = []
    for elem in text_res:
        if elem == '\n ':
            if sentence:
                lines.append(sentence)
                sentence = []
        else:
            sentence.append(elem)
    for line in lines:
        annotated_text(line)