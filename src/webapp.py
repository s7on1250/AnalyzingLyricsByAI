import sys
import os
import streamlit as st
import base64
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F
model_name = 'roberta-base'
from clean_text import text_preprocessing_pipeline
from display_utils import set_page_bg, highlight_words, load_words
import nltk
sys.path.insert(0, '../src')
from utils import predict
import re
import pymorphy3
import json
import pandas as pd
import plotly.express as px


DATA = './data/top_words.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA)


# load labels 
with open(os.getcwd() + '/model/labels.txt', 'r') as f:
    labels = f.read().splitlines()
    f.close()

# draw a map
def draw_map_cases(): 
    fig = px.choropleth_mapbox(df,
                               geojson=json_locations,
                               locations='iso_code',
                               hover_data=['top_word'],
                               color_continuous_scale="Reds",
                               mapbox_style="carto-positron",
                               title="Most frequent words in country",
                               zoom=1,
                               opacity=0.5,
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

# load model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.fc = torch.nn.Linear(768,5)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output = F.softmax(self.fc(features), dim=1)
        return output

model = BERTClass()
model.load_state_dict(torch.load(os.getcwd() + '/model/model4.bin', map_location=torch.device('cpu')))
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# set background
set_page_bg(os.getcwd() + '/img/bg.png')

with open('data/countries.geo.json') as json_file:
    json_locations = json.load(json_file)
# Draw the map
df = load_data()

st.sidebar.title("Выберите функцию для отображения")

select_event = st.sidebar.selectbox('Show map', ('Жанровый классификатор', 'Интерактивная карта'))
if select_event == 'Жанровый классификатор':
    st.markdown("<h1 style='text-align: center; color: #322c2c;'>Жанровый классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color: #fe6053;'>Введите текст песни</div>", unsafe_allow_html=True)
    lyrics = st.text_area("", height=500)

    # display the name when the submit button is clicked
    # .title() is used to get the input text string
    if(st.button('Submit')):
        result = text_preprocessing_pipeline(lyrics.title())

        label = predict(result, model, tokenizer)

        genre = labels[label]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<p style="color: black; font-size: 30px;">{"Genre: "}{genre}</p>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<p style="color: black; font-size: 30px;">{"Possible title: "}{12345}</p>', unsafe_allow_html=True)
        st.balloons()


        css_style = """
            <style>
                body {
                    font-size: 16px;
                    line-height: 1.3;
                }
            </style>
        """

        st.markdown(css_style, unsafe_allow_html=True)


        # Список слов для выделения
        words_to_highlight_rap = load_words('/src/popular_rap_words.txt')
        words_to_highlight_metal = load_words('/src/popular_metal_words.txt')
        words_to_highlight_rock = load_words('/src/popular_rock_words.txt')
        words_to_highlight_pop = load_words('/src/popular_pop_words.txt')
        words_to_highlight_rb = load_words('/src/popular_rb_words.txt')
        genre_to_list = {"rap": words_to_highlight_rap, "metal": words_to_highlight_metal,
                        "rock": words_to_highlight_rock, "pop": words_to_highlight_pop,
                        "rb": words_to_highlight_rb}

        # Выделение слов в тексте
        words_to_highlight = genre_to_list[genre]
        highlighted_text = highlight_words(lyrics, words_to_highlight)

        # Отображение текста с выделенными словами
        st.subheader('Popular words for this genre: ')
        st.markdown(highlighted_text, unsafe_allow_html=True)
    

if select_event == 'Интерактивная карта':
    st.markdown("<h1 style='text-align: center; color: #322c2c;'>Самые популярные слова в треках разных стран</h1>", unsafe_allow_html=True)
    # st.title('Самые популярные слова в странах мира')
    st.plotly_chart(draw_map_cases(),
                    use_container_width=True
                    )

